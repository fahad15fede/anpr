"""Fine-tune YOLO11 on plates. Usage: python train.py [--model yolo11s.pt] [--epochs 60]"""
import argparse
import shutil
from pathlib import Path

import torch
from ultralytics import YOLO

p = argparse.ArgumentParser()
p.add_argument("--model", default="yolo11n.pt")  # yolo11s.pt if accuracy is short
p.add_argument("--epochs", type=int, default=60)
p.add_argument("--imgsz", type=int, default=960)
p.add_argument("--batch", type=int, default=16)
a = p.parse_args()

device = 0 if torch.cuda.is_available() else "cpu"
model = YOLO(a.model)
model.train(data="data/plates.yaml", epochs=a.epochs, imgsz=a.imgsz,
            batch=a.batch, device=device)

best = Path(model.trainer.best)
Path("models").mkdir(exist_ok=True)
shutil.copy(best, "models/plate.pt")
print(f"Copied {best} -> models/plate.pt")
