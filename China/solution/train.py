"""Fine-tune YOLOv8n on the rubbings. Uses the GPU if there is one (e.g. Colab), else CPU."""
import argparse
import os

import torch
from ultralytics import YOLO

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="yolo_data/data.yaml")
ap.add_argument("--epochs", type=int, default=40)
ap.add_argument("--imgsz", type=int, default=640)
ap.add_argument("--batch", type=int, default=None)
ap.add_argument("--project", default="runs")
ap.add_argument("--resume", action="store_true", help="continue from <project>/glyphs/weights/last.pt")
a = ap.parse_args()

gpu = torch.cuda.is_available()
batch = a.batch or (32 if gpu else 16)
print(f"device={'cuda:0' if gpu else 'cpu'} batch={batch}")

if a.resume:
    YOLO(os.path.join(a.project, "glyphs", "weights", "last.pt")).train(resume=True)
else:
    YOLO("yolov8n.pt").train(
        data=a.data, epochs=a.epochs, imgsz=a.imgsz, batch=batch, device=0 if gpu else "cpu",
        workers=2 if gpu else 6, project=a.project, name="glyphs", exist_ok=True, seed=0,
        plots=False, amp=gpu, patience=100,
        # no flips: stroke orientation matters for glyph identity
        fliplr=0.0, flipud=0.0, degrees=5, scale=0.4, translate=0.1,
        hsv_h=0.0, hsv_s=0.0, hsv_v=0.4, mosaic=1.0, close_mosaic=10,
    )
