"""Convert train_labels.csv to YOLO format: 8 crate glyphs + class 'other', 85/15 train/val split."""
import argparse
import os
import random
from pathlib import Path

import cv2
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--data", type=Path, default=Path("../task-china/task_china"))
ap.add_argument("--out", type=Path, default=Path("yolo_data"))
a = ap.parse_args()

labels = pd.read_csv(a.data / "train_labels.csv")
crates = pd.read_csv(a.data / "crates.csv")
class_of = {g: i for i, g in enumerate(crates.glyph_id)}
OTHER = len(class_of)
names = list(crates.glyph_id) + ["other"]

images = sorted(labels.image.unique())
random.Random(0).shuffle(images)
n_val = int(0.15 * len(images))
split = {img: ("val" if i < n_val else "train") for i, img in enumerate(images)}

for s in ("train", "val"):
    (a.out / s / "images").mkdir(parents=True, exist_ok=True)
    (a.out / s / "labels").mkdir(parents=True, exist_ok=True)

for img_name, group in labels.groupby("image"):
    s = split[img_name]
    src = (a.data / "train" / "images" / img_name).resolve()
    dst = a.out / s / "images" / img_name
    if not dst.exists():
        os.symlink(src, dst)
    H, W = cv2.imread(str(src), cv2.IMREAD_GRAYSCALE).shape
    lines = []
    for r in group.itertuples():
        x0, y0 = max(r.x, 0), max(r.y, 0)
        x1, y1 = min(r.x + r.w, W), min(r.y + r.h, H)
        if x1 <= x0 or y1 <= y0:
            continue
        c = class_of.get(r.glyph_id, OTHER)
        lines.append(f"{c} {(x0 + x1) / 2 / W:.6f} {(y0 + y1) / 2 / H:.6f} "
                     f"{(x1 - x0) / W:.6f} {(y1 - y0) / H:.6f}")
    (a.out / s / "labels" / (Path(img_name).stem + ".txt")).write_text("\n".join(lines) + "\n")

yaml = [f"path: {a.out.resolve()}", "train: train/images", "val: val/images", "names:"]
yaml += [f"  {i}: {n}" for i, n in enumerate(names)]
(a.out / "data.yaml").write_text("\n".join(yaml) + "\n")
pd.Series(split).rename("split").to_csv(a.out / "split.csv", header=True)
print(f"{len(images) - n_val} train / {n_val} val images, classes: {names}")
