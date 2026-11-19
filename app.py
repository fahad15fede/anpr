import re
import sqlite3
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd
import streamlit as st

import process

DB = Path("data/out/sightings.db")
COLS = ["video", "track_id", "plate", "conf", "first_s", "last_s", "crop", "frame", "video_path"]

st.set_page_config(page_title="PlateTrace", page_icon="🚗", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&family=JetBrains+Mono:wght@700;800&display=swap');
html, body, [class*="css"], .stMarkdown, .stCaption, button, input, label {font-family:'Inter', sans-serif;}
.block-container {padding-top: 1.6rem; max-width: 1200px;}
header[data-testid="stHeader"] {background: transparent;}

/* header */
.hero {display:flex; align-items:center; justify-content:space-between; flex-wrap:wrap; gap:1rem;
       padding:1.2rem 1.5rem; border-radius:18px; margin-bottom:1rem;
       background:linear-gradient(135deg,#07160d 0%,#0d120d 55%,#1a1a06 100%); border:1px solid #1f3d2a;}
.hero h1 {margin:0; padding:0; font-size:2.2rem; font-weight:800; letter-spacing:-.02em;}
.hero h1 span {color:#00E676;}
.hero p {margin:.2rem 0 0; color:#9aa59a; font-size:.95rem;}
.stats {display:flex; gap:.7rem; flex-wrap:wrap;}
.stat {background:#0a0f0a; border:1px solid #1f3d2a; border-radius:12px; padding:.55rem 1rem; min-width:110px;}
.stat b {display:block; font-size:1.5rem; color:#00E676; font-weight:800; line-height:1.2;}
.stat small {color:#8a948a; text-transform:uppercase; letter-spacing:.08em; font-size:.68rem;}

/* tabs */
.stTabs [data-baseweb="tab-list"] {gap:.4rem;}
.stTabs [data-baseweb="tab"] {background:#0d120d; border-radius:10px 10px 0 0; padding:.5rem 1.1rem;}

/* verdicts */
.pass, .maybe, .none {border-radius:14px; padding:.9rem 1.3rem; margin:1.2rem 0 .7rem; font-weight:800;}
.pass {background:#00E676; color:#000; font-size:1.5rem; letter-spacing:.06em; box-shadow:0 0 30px #00e67633;}
.maybe{background:#FFD600; color:#000; font-size:1.15rem;}
.none {border:2px dashed #FFD600; color:#FFD600; font-weight:600;}

/* cards + plates */
.card {background:#0d120d; border:1px solid #1f3d2a; border-radius:14px; padding:1rem 1.2rem; line-height:1.9;}
.card .k {color:#8a948a; font-size:.8rem; text-transform:uppercase; letter-spacing:.08em;}
.plate{display:inline-block; background:#FFD600; color:#000; font-family:'JetBrains Mono',monospace;
       font-weight:800; font-size:1.5rem; padding:.15rem .8rem; border-radius:6px; border:3px solid #000;
       box-shadow:0 0 0 2px #FFD600; letter-spacing:.08em;}
.plate.sm {font-size:1rem; padding:.05rem .5rem; border-width:2px;}
.chips {display:flex; flex-wrap:wrap; gap:.6rem; margin:.4rem 0 1rem;}

/* plate-shaped search box */
.stTextInput input {text-transform:uppercase; font-family:'JetBrains Mono',monospace !important;
       font-size:2rem !important; letter-spacing:.2em; text-align:center; font-weight:800;
       background:#FFD600 !important; color:#000 !important; border:4px solid #000 !important;
       border-radius:10px !important; padding:.6rem !important;}
.stTextInput input::placeholder {color:#00000066;}
[data-testid="stFileUploaderDropzone"] {border:2px dashed #1f3d2a; border-radius:14px; padding:2rem; background:#07100a;}
img {border-radius:10px;}
</style>""", unsafe_allow_html=True)


@st.cache_data
def load(mtime: float) -> pd.DataFrame:  # mtime busts the cache after re-processing
    if not DB.exists():
        return pd.DataFrame(columns=COLS)
    with sqlite3.connect(DB) as con:
        return pd.read_sql("SELECT * FROM sightings ORDER BY video, first_s", con)


df = load(DB.stat().st_mtime if DB.exists() else 0.0)


def clean(s: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", s.upper())


with st.sidebar:
    st.markdown("### How it works")
    st.markdown("1. **Upload** a traffic video\n2. Plates of **any size** are detected, tracked and read\n"
                "3. **Search** a plate to see if it passed, when, and the proof frame")
    st.caption("YOLO11 · ByteTrack · EasyOCR · SQLite")

avg = f"{df.conf.mean():.0%}" if len(df) else "–"
st.markdown(f"""<div class='hero'>
  <div><h1>🚗 Plate<span>Trace</span></h1>
       <p>Type a number plate. We tell you if that vehicle passed the camera.</p></div>
  <div class='stats'>
    <div class='stat'><b>{df.video.nunique()}</b><small>Videos</small></div>
    <div class='stat'><b>{len(df)}</b><small>Vehicles</small></div>
    <div class='stat'><b>{avg}</b><small>Avg confidence</small></div>
  </div></div>""", unsafe_allow_html=True)
if df.empty:
    st.info("No videos processed yet. Upload one in the **⬆️ Upload video** tab.")
if "flash" in st.session_state:
    st.success(st.session_state.pop("flash"), icon="✅")

tab_search, tab_upload, tab_log = st.tabs(["🔎 Search", "⬆️ Upload video", "📋 All detections"])


@st.cache_resource(show_spinner="Loading OCR model (first time only)...")
def ocr_reader():
    return process.load_reader()


def show(row, label_html):
    st.markdown(label_html, unsafe_allow_html=True)
    a, b, c = st.columns([1, 2, 2])
    if Path(row.crop).exists():
        a.image(row.crop, caption="Plate")
    if Path(row.frame).exists():
        b.image(row.frame, caption="Proof frame")
    with c:
        st.markdown(f"<div class='card'><span class='plate'>{row.plate}</span><br><br>"
                    f"<span class='k'>Video</span><br><b>{row.video}</b><br>"
                    f"<span class='k'>Seen</span><br><b>{row.first_s:.1f}s – {row.last_s:.1f}s</b><br>"
                    f"<span class='k'>Read confidence</span><br><b>{row.conf:.0%}</b></div>",
                    unsafe_allow_html=True)
        vp = row.video_path if isinstance(row.video_path, str) else f"data/videos/{row.video}.mp4"
        if Path(vp).exists():
            st.video(vp, start_time=int(row.first_s))


with tab_search:
    q = clean(st.text_input("Number plate", placeholder="e.g. ABC123",
                            label_visibility="collapsed"))
    if q:
        res = df.assign(score=df.plate.apply(lambda p: SequenceMatcher(None, q, p).ratio()))
        exact = res[res.plate == q]
        near = res[(res.score >= 0.75) & (res.plate != q)].sort_values("score", ascending=False)

        if len(exact):
            for _, r in exact.iterrows():
                show(r, "<div class='pass'>✔ VEHICLE PASSED</div>")
        elif len(near):
            for _, r in near.head(3).iterrows():
                show(r, f"<div class='maybe'>≈ POSSIBLE MATCH ({r.score:.0%} similar)</div>")
        else:
            st.markdown(f"<div class='none'>NO RECORD FOUND. <b>{q}</b> was not detected "
                        "in the processed footage.</div>", unsafe_allow_html=True)
    elif not df.empty:
        st.markdown("##### Recently detected")
        recent = df.tail(12)
        cols = st.columns(6)
        for k, (_, r) in enumerate(recent.iterrows()):
            with cols[k % 6]:
                if Path(r.crop).exists():
                    st.image(r.crop, width="stretch")
                st.markdown(f"<span class='plate sm'>{r.plate}</span><br>"
                            f"<small style='color:#8a948a'>{r.video} · {r.first_s:.0f}s</small>",
                            unsafe_allow_html=True)

with tab_upload:
    st.caption("Upload traffic footage. Every plate in it is detected, read and added to the search.")
    up = st.file_uploader("Video", type=["mp4", "mov", "avi", "mkv", "m4v", "webm"],
                          label_visibility="collapsed")
    with st.expander("Settings"):
        s1, s2, s3 = st.columns(3)
        region = s1.selectbox("Plate format", ["Auto", "India", "UK", "China", "Any (no correction)"],
                              help="Auto looks at all reads in the video and picks the best-fitting format.")
        imgsz = s2.selectbox("Plate sizes", ["All sizes", "Auto (faster)", 640, 1280],
                             help="All sizes: runs the detector at 640 (big close-up plates) AND 1280 "
                                  "(small distant plates) on every frame, so any plate size is found. "
                                  "Auto: tests both on a few frames and uses the best one only (faster). "
                                  "640 / 1280: fixed size.")
        stride = s3.selectbox("Speed", [1, 2, 3], format_func=lambda n: {
            1: "Every frame (best)", 2: "Every 2nd frame (~2x faster)", 3: "Every 3rd frame (~3x faster)"}[n])
        models = sorted(p.as_posix() for p in Path("models").glob("*.pt") if p.is_file())
        model_path = st.selectbox("Detector model", models,
                                  index=models.index("models/plate.pt") if "models/plate.pt" in models else 0,
                                  help="plate.pt = your trained plate model. A model without a 'plate' "
                                       "class (e.g. yolo11n_coco.pt) uses everything it detects instead.")

    if up is not None:
        name = re.sub(r"[^A-Za-z0-9_-]+", "_", Path(up.name).stem).strip("_") or "video"
        dest = Path("data/videos") / f"{name}{Path(up.name).suffix.lower()}"
        if name in set(df.video):
            st.info(f"**{name}** was processed before. Processing again replaces its old results.")
        if st.button("▶ Process video", type="primary"):
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(up.getbuffer())
            reader = ocr_reader()
            bar = st.progress(0.0, text="Starting...")
            try:
                n, used_region, used_sz = process.process_video(
                    dest, reader, model_path=model_path,
                    region={"Auto": "auto", "India": "IN", "UK": "UK", "China": "CN"}.get(region),
                    imgsz={"All sizes": "all", "Auto (faster)": "auto"}.get(imgsz, imgsz),
                    stride=stride,
                    progress=lambda f, msg: bar.progress(f, text=msg))
            except Exception as e:  # bad/corrupt file, codec issue, ...
                bar.empty()
                st.error(f"Could not process this video: {e}")
            else:
                st.session_state.flash = (f"{name}: {n} vehicles found "
                                          f"(plate format {used_region}, detector sizes {used_sz}).")
                st.rerun()

    if not df.empty:
        st.markdown("##### Processed videos")
        summary = df.groupby("video").agg(vehicles=("plate", "size"), plates=("plate", ", ".join))
        st.dataframe(summary, column_config={"plates": st.column_config.TextColumn(width="large")})

with tab_log:
    c1, c2 = st.columns(2)
    c1.metric("Videos processed", df.video.nunique())
    c2.metric("Vehicles detected", len(df))
    pick = st.multiselect("Filter by video", sorted(df.video.unique()))
    st.dataframe(
        (df[df.video.isin(pick)] if pick else df)[["video", "plate", "conf", "first_s", "last_s"]],
        hide_index=True,
        column_config={
            "conf": st.column_config.ProgressColumn("Confidence", min_value=0, max_value=1, format="%.0f%%"),
            "first_s": st.column_config.NumberColumn("First seen (s)", format="%.1f"),
            "last_s": st.column_config.NumberColumn("Last seen (s)", format="%.1f"),
        },
    )

    st.markdown("##### 🗑 Manage data")
    d1, d2 = st.columns(2)
    with d1, st.container(border=True):
        st.markdown("**Delete one video's results**")
        victim = st.selectbox("Video", sorted(df.video.unique()), index=None,
                              placeholder="Choose a video", label_visibility="collapsed")
        rm_file = st.checkbox("Also delete the video file", key="rm_one")
        if st.button("Delete video results", disabled=victim is None):
            n = process.delete_video(victim, remove_file=rm_file)
            st.session_state.flash = f"Deleted {n} detections of {victim}" + (" and its video file." if rm_file else ".")
            st.rerun()
    with d2, st.container(border=True):
        st.markdown("**Clear everything**")
        st.caption("Removes every detection, plate crop and proof frame. Cannot be undone.")
        rm_all = st.checkbox("Also delete all video files in data/videos/", key="rm_all")
        sure = st.checkbox("Yes, I want to clear all data", key="sure")
        if st.button("Clear all data", type="primary", disabled=not sure):
            n = process.clear_all(remove_videos=rm_all)
            st.session_state.flash = f"Cleared all data ({n} detections)" + (" and all videos." if rm_all else ".")
            st.rerun()
