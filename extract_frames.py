"""Dump frames for labelling (no ffmpeg needed).
Usage: python extract_frames.py data/videos/video1.mp4 data/videos/video2.mp4 [--fps 2]
Frames land in data/frames_to_label/.
"""
import argparse
from pathlib import Path

import cv2

p = argparse.ArgumentParser()
p.add_argument("videos", nargs="+")
p.add_argument("--fps", type=float, default=2)
a = p.parse_args()

out = Path("data/frames_to_label")
out.mkdir(parents=True, exist_ok=True)
for v in a.videos:
    cap = cv2.VideoCapture(v)
    step = max(int(round((cap.get(cv2.CAP_PROP_FPS) or 25) / a.fps)), 1)
    i = n = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % step == 0:
            cv2.imwrite(str(out / f"{Path(v).stem}_{i:06d}.jpg"), frame)
            n += 1
        i += 1
    cap.release()
    print(f"{v}: {n} frames")
