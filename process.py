"""Video -> YOLO11 plate detection -> ByteTrack -> OCR + voting -> SQLite.

Usage:
    python process.py data/videos/video1.mp4 data/videos/video2.mp4 [--region auto|IN|UK|off]
                      [--imgsz all|auto|640|1280] [--conf 0.3] [--stride 1]
    python process.py --clear              # wipe all results

The Streamlit app (app.py) calls process_video() directly for uploaded videos.
"""
import argparse
import re
import sqlite3
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path

import cv2

OUT = Path("data/out")
MODEL = Path("models/plate.pt")
DB = OUT / "sightings.db"

# Detector input size. 640 finds big close-up plates, 1280 finds small distant ones.
#   "all"  = run every size in IMGSZ_CHOICES on each frame and merge -> any plate size (default)
#   "auto" = probe a few frames, then use the single best size (faster)
#   640 / 1280 / ... = fixed size
IMGSZ = "all"
IMGSZ_CHOICES = (640, 1280)
NMS_IOU = 0.5         # boxes overlapping this much across sizes = same plate
TRACKER = dict(       # ByteTrack settings (Ultralytics bytetrack.yaml defaults)
    tracker_type="bytetrack", track_high_thresh=0.25, track_low_thresh=0.1,
    new_track_thresh=0.25, track_buffer=30, match_thresh=0.8, fuse_score=True)
CONF = 0.3
VID_STRIDE = 1        # look at every Nth frame (2 = ~2x faster, slightly worse tracking)

MIN_WIDTH = 20        # px; narrower plate crops are too small to read
MIN_SHARPNESS = 40.0  # Laplacian variance; lower = blurry, skipped
OCR_EVERY = 3         # OCR every Nth sighting of a track (CPU is slow)
EDGE = 3              # px; skip plates touching the frame border (half-visible)
MAX_READS = 15        # reads kept per track (smallest-plate reads get replaced by bigger ones)
READ_SIZE_FRACTION = 0.6  # vote only with reads where the plate was >= 60% of its max width
MAX_FIXES = 3         # max letter<->digit corrections when snapping to a plate format
ALLOW = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-"
MERGE_GAP_S = 2.0       # tracks this close in time with similar text = same car
MERGE_SIMILARITY = 0.7  # 0..1 text similarity needed to merge two tracks

# "IN" / "UK" / "PK" = Indian / UK / Pakistani plates, "auto" = whichever fits best, None = no correction
PLATE_REGION = "auto"
PLATE_PATTERN = None  # optional extra regex filter, e.g. r"^[A-Z]{2}\d{2}[A-Z]{3}$"

# Plate layouts: A = letter, 9 = digit
FORMATS = {
    # Indian: AA 9(9) A(AA) 9999, e.g. DL7CD5017, HR26CO6869
    "IN": ["AA" + "9" * d + "A" * k + "9999" for d in (2, 1) for k in range(4) if 2 + d + k >= 4],
    # UK current style: AA99 AAA, e.g. GX15OGJ
    "UK": ["AA99AAA"],
    # Pakistan: AA(A) 999(9), e.g. LEA1234, KAD404, ICT123; older Punjab adds a
    # 2-digit year: LEB 07 1234
    "PK": [l + d for l in ("AA", "AAA") for d in ("999", "9999")] + ["AAA99999", "AAA999999"],
}


def clean(txt: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", txt.upper())


TO_DIGIT = {"O": "0", "Q": "0", "D": "0", "U": "0", "I": "1", "L": "4", "Z": "2",
            "S": "5", "B": "8", "G": "6", "A": "4", "T": "7"}
TO_LETTER = {v: k for k, v in TO_DIGIT.items() if k not in "QDUL"}
CHEAP = [{"O", "0"}, {"I", "1"}]  # near-identical glyphs: cheaper to swap than drop


def fit_format(text: str, region):
    """Plate text after snapping to `region`'s layout, or None."""
    res = _fit(text, region)
    return res[1] if res else None


def pick_region(texts):
    """Decide a video's plate region from all its raw reads: the region whose
    layout most reads fit cleanly (few fixes, little junk) wins."""
    def score(r):
        fits = [_fit(t, r) for t in texts]
        return sum(1 for f in fits if f and f[0] <= 2), -sum(f[0] for f in fits if f)
    return max(FORMATS, key=score) if texts else "IN"


def _fit(text: str, region):
    """(cost, text): snap an OCR read onto a plate layout, fixing letter/digit look-alikes by
    position (Z->2, L->4, 0->O ...) and dropping junk around it (e.g. the 'IND'
    or 'GB' strip). Returns None if it can't be a plate of that region."""
    if region is None:
        return 0, text
    layouts = [f for r in (FORMATS if region == "auto" else [region]) for f in FORMATS[r]]
    best = None
    for kinds in layouts:
        n = len(kinds)
        for s in range(len(text) - n + 1):
            rest, dropped = text[s:s + n], len(text) - n
            out, subs = [], 0
            for ch, kind in zip(rest, kinds):
                ok = ch.isdigit() if kind == "9" else ch.isalpha()
                if not ok:
                    new = (TO_DIGIT if kind == "9" else TO_LETTER).get(ch)
                    if new is None:
                        break
                    subs += 0.5 if {ch, new} in CHEAP else 1
                    ch = new
                out.append(ch)
            else:
                if subs > MAX_FIXES:
                    continue
                # fewest changes (substitutions + dropped junk), then fewest swaps, then earliest start
                cand = (subs + dropped, subs, s, "".join(out))
                best = min(best, cand) if best else cand
    return (best[0], best[3]) if best else None


def valid(text: str) -> bool:
    if len(text) < 4:
        return False
    return PLATE_PATTERN is None or re.match(PLATE_PATTERN, text) is not None


def sharpness(img) -> float:
    return cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()


def read_plate(reader, crop):
    """OCR a crop; join multi-segment plates (e.g. 'ABC' '123') left to right."""
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    if gray.shape[0] < 64:  # upscale small plates to ~64-80px tall, helps EasyOCR a lot
        f = min(max(2.0, 72 / gray.shape[0]), 6.0)
        gray = cv2.resize(gray, None, fx=f, fy=f, interpolation=cv2.INTER_CUBIC)
    elif gray.shape[0] > 200:  # very large close-up plates: shrink, OCR is faster and no worse
        f = 160 / gray.shape[0]
        gray = cv2.resize(gray, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
    parts = reader.readtext(gray, allowlist=ALLOW)
    if not parts:
        return None
    parts.sort(key=lambda p: min(pt[0] for pt in p[0]))
    text = clean("".join(p[1] for p in parts))  # raw; snapped to a region later
    conf = sum(p[2] for p in parts) / len(parts)
    return (text, conf) if len(text) >= 4 else None


def init_db():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "crops").mkdir(exist_ok=True)
    (OUT / "frames").mkdir(exist_ok=True)
    db = sqlite3.connect(DB)
    db.execute("""CREATE TABLE IF NOT EXISTS sightings(
        video TEXT, track_id INT, plate TEXT, conf REAL,
        first_s REAL, last_s REAL, crop TEXT, frame TEXT, video_path TEXT)""")
    return db


def load_model(path=None):
    path = Path(path or MODEL)
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Train one with train.py first.")
    from ultralytics import YOLO
    model = YOLO(str(path))
    # detect only classes named like "plate"; a model without one (e.g. the stock
    # COCO yolo11n) falls back to all of its classes
    model.plate_classes = [i for i, n in model.names.items() if "plate" in n.lower()] or None
    return model


def load_reader():
    import easyocr
    return easyocr.Reader(["en"], gpu=False)


def pick_imgsz(model, video_path, samples=12):
    """Run the detector at each candidate size on frames spread through the video
    and keep the size that finds the most confident plates."""
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    frames = []
    for k in range(samples):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(total * (k + 0.5) / samples))
        ok, f = cap.read()
        if ok:
            frames.append(f)
    cap.release()

    def score(sz):
        rs = model.predict(frames, imgsz=sz, conf=CONF, classes=model.plate_classes, verbose=False)
        return sum(float(c) for r in rs for c in r.boxes.conf if c >= 0.5)
    return max(IMGSZ_CHOICES, key=score) if frames else 640


def detect(model, frame, sizes, conf):
    """Plate boxes for one frame. With several sizes, the detector runs at each one
    (640 catches big close-up plates, 1280 small distant ones) and duplicate boxes of
    the same plate are merged with NMS. Returns an Ultralytics Boxes object."""
    import torch
    import torchvision
    from ultralytics.engine.results import Boxes
    data = torch.cat([model.predict(frame, imgsz=sz, conf=conf, classes=model.plate_classes,
                                    verbose=False)[0].boxes.data
                      for sz in sizes]).cpu()
    if len(sizes) > 1 and len(data):
        data = data[torchvision.ops.nms(data[:, :4], data[:, 4], NMS_IOU)]
    return Boxes(data, frame.shape[:2])


def new_tracker():
    from ultralytics.trackers.byte_tracker import BYTETracker
    from ultralytics.utils import IterableSimpleNamespace
    return BYTETracker(IterableSimpleNamespace(**TRACKER))


def draw(frame, tracks):
    """Proof frame: every tracked plate boxed and labelled with its track id."""
    img = frame.copy()
    for x1, y1, x2, y2, tid, score, *_ in tracks:
        p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
        cv2.rectangle(img, p1, p2, (255, 60, 0), 2)
        cv2.putText(img, f"id:{int(tid)} plate {score:.2f}", (p1[0], max(p1[1] - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 60, 0), 2)
    return img


def run(video_path: str, model, reader, db, region=PLATE_REGION, imgsz=IMGSZ,
        conf=CONF, stride=VID_STRIDE, progress=None):
    """Process one video into the DB. progress(fraction, message) is called as it goes.
    imgsz: "all" (every size in IMGSZ_CHOICES, finds any plate size), "auto" (best
    single size), or a number. Returns (vehicles saved, plate region, sizes used)."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"cannot open {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    if imgsz == "all":
        sizes = IMGSZ_CHOICES
    elif imgsz == "auto":
        if progress:
            progress(0.0, "choosing detector size...")
        sizes = (pick_imgsz(model, video_path),)
    else:
        sizes = (int(imgsz),)

    reads = defaultdict(list)  # track_id -> [(plate, conf)]
    best = {}                  # track_id -> (area, crop, frame)
    seen = {}                  # track_id -> [first_s, last_s]
    hits = defaultdict(int)    # track_id -> number of usable sightings
    tracker = new_tracker()

    i = -1
    while True:
        i += 1
        if i % stride:  # skipped frame: grab() advances without decoding
            if not cap.grab():
                break
            continue
        ok, frame = cap.read()
        if not ok:
            break
        if (i // stride) % 10 == 0 and progress:
            progress(min(i / total, 1.0), f"frame {i}/{total} · {len(seen)} plates tracked")
        tracks = tracker.update(detect(model, frame, sizes, conf).numpy(), frame)
        if not len(tracks):
            continue
        t = i / fps
        h, w = frame.shape[:2]
        frame_img = None
        for x1, y1, x2, y2, tid, *_ in tracks:
            tid = int(tid)
            x1, y1, x2, y2 = max(int(x1), 0), max(int(y1), 0), min(int(x2), w), min(int(y2), h)
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0 or crop.shape[1] < MIN_WIDTH:
                continue
            if x1 <= EDGE or y1 <= EDGE or x2 >= w - EDGE or y2 >= h - EDGE:
                continue  # plate cut off by the frame edge -> partial, misleading OCR
            seen.setdefault(tid, [t, t])[1] = t
            if sharpness(crop) < MIN_SHARPNESS:
                continue

            # Cars start far away (tiny plate) and get closer. Keep OCR going while the
            # plate is still growing, and once MAX_READS is reached swap out the
            # smallest read, so the biggest, clearest views decide the plate.
            pw = x2 - x1
            rs = reads[tid]
            if hits[tid] % OCR_EVERY == 0 and (len(rs) < MAX_READS or pw > min(r[2] for r in rs)):
                res = read_plate(reader, crop)
                if res:
                    rs.append((*res, pw))
                    if len(rs) > MAX_READS:
                        rs.remove(min(rs, key=lambda r: r[2]))
            hits[tid] += 1

            area = (x2 - x1) * (y2 - y1)
            if tid not in best or area > best[tid][0]:
                if frame_img is None:
                    frame_img = draw(frame, tracks)
                best[tid] = (area, crop.copy(), frame_img)
    cap.release()

    # only trust reads taken when the plate was near its largest size in the video
    for tid, rs in reads.items():
        top = max((r[2] for r in rs), default=0)
        reads[tid] = [(t, c) for t, c, pw in rs if pw >= READ_SIZE_FRACTION * top]

    if region == "auto":
        region = pick_region([t for rs in reads.values() for t, _ in rs])
        if progress:
            progress(1.0, f"plate format detected: {region}")
    for tid, rs in reads.items():
        reads[tid] = [(f, c) for f, c in ((fit_format(t, region), c) for t, c in rs)
                      if f and valid(f)]

    name = Path(video_path).stem
    db.execute("DELETE FROM sightings WHERE video = ?", (name,))  # re-runs replace
    vehicles = merge_tracks([
        {"tid": tid, "reads": list(rs), "best": best[tid], "span": list(seen[tid])}
        for tid, rs in reads.items() if rs and tid in best])
    for v in vehicles:
        plate, pconf = vote(v["reads"])
        tid = v["tid"]
        crop_p = (OUT / "crops" / f"{name}_{tid}.jpg").as_posix()
        frame_p = (OUT / "frames" / f"{name}_{tid}.jpg").as_posix()
        cv2.imwrite(crop_p, v["best"][1])
        cv2.imwrite(frame_p, v["best"][2])
        db.execute("INSERT INTO sightings VALUES (?,?,?,?,?,?,?,?,?)",
                   (name, tid, plate, pconf, *v["span"], crop_p, frame_p,
                    Path(video_path).as_posix()))
    db.commit()
    if progress:
        progress(1.0, f"done · {len(vehicles)} vehicles saved ({len(seen)} tracks seen)")
    return len(vehicles), region, "+".join(map(str, sizes))


def vote(rs):
    score = defaultdict(float)
    for text, c in rs:
        score[text] += c  # confidence-weighted vote
    plate = max(score, key=score.get)
    return plate, score[plate] / sum(score.values())


def merge_tracks(tracks):
    """Join tracks that are really one car: the tracker loses plates that jump
    between frames (low-fps / duplicated-frame video), so stitch consecutive
    tracks whose voted text is similar and whose time spans are close."""
    out = []
    for t in sorted(tracks, key=lambda t: t["span"][0]):
        prev = out[-1] if out else None
        if (prev and t["span"][0] - prev["span"][1] <= MERGE_GAP_S and
                SequenceMatcher(None, vote(prev["reads"])[0],
                                vote(t["reads"])[0]).ratio() >= MERGE_SIMILARITY):
            prev["reads"] += t["reads"]
            prev["span"][1] = max(prev["span"][1], t["span"][1])
            if t["best"][0] > prev["best"][0]:
                prev["best"] = t["best"]
        else:
            out.append(t)
    return out


def process_video(video_path, reader=None, model_path=None, **opts):
    """One-shot helper: fresh model (so track IDs reset), shared OCR reader."""
    reader = reader or load_reader()
    db = init_db()
    try:
        return run(str(video_path), load_model(model_path), reader, db, **opts)
    finally:
        db.close()


def delete_video(name, remove_file=False):
    """Remove one video's results (DB rows, crops, frames), optionally the video too."""
    if not DB.exists():
        return 0
    with sqlite3.connect(DB) as db:
        rows = db.execute("SELECT crop, frame, video_path FROM sightings WHERE video = ?", (name,)).fetchall()
        db.execute("DELETE FROM sightings WHERE video = ?", (name,))
    for crop, frame, _ in rows:
        Path(crop).unlink(missing_ok=True)
        Path(frame).unlink(missing_ok=True)
    if remove_file:
        files = {Path(r[2]) for r in rows if r[2]} | set(Path("data/videos").glob(name + ".*"))
        for f in files:
            f.unlink(missing_ok=True)
    return len(rows)


def clear_all(remove_videos=False):
    """Wipe every result: DB rows, all crops and proof frames. Optionally delete
    every video in data/videos/ as well."""
    n = 0
    if DB.exists():
        with sqlite3.connect(DB) as db:
            n = db.execute("SELECT COUNT(*) FROM sightings").fetchone()[0]
            db.execute("DELETE FROM sightings")
    for sub in ("crops", "frames"):
        for f in (OUT / sub).glob("*.jpg"):
            f.unlink(missing_ok=True)
    if remove_videos:
        for f in Path("data/videos").glob("*.*"):
            f.unlink(missing_ok=True)
    return n


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("videos", nargs="*")
    p.add_argument("--clear", action="store_true", help="delete all results (keeps video files)")
    p.add_argument("--region", default=PLATE_REGION, help="auto | IN | UK | off")
    p.add_argument("--imgsz", default=IMGSZ, help="all | auto | 640 | 1280 ...")
    p.add_argument("--conf", type=float, default=CONF)
    p.add_argument("--stride", type=int, default=VID_STRIDE)
    a = p.parse_args()
    if a.clear:
        print(f"Cleared {clear_all()} rows, crops and frames.")
        return
    if not a.videos:
        p.error("give at least one video, or --clear")
    region = None if a.region.lower() == "off" else (a.region if a.region == "auto" else a.region.upper())

    imgsz = a.imgsz if a.imgsz in ("all", "auto") else int(a.imgsz)
    reader = load_reader()
    for v in a.videos:
        print(f"Processing {v}")
        n, region, sz = process_video(v, reader, region=region, imgsz=imgsz, conf=a.conf, stride=a.stride,
                          progress=lambda f, m: print(f"  {m}", end="\r"))
        print(f"\n  {Path(v).stem}: {n} vehicles saved (plate format: {region}, detector sizes: {sz})")


if __name__ == "__main__":
    main()
