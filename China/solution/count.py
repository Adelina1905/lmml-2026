"""Count each crate's glyph per set with the trained detector, write counts.csv and decode the flag."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from ultralytics import YOLO

ap = argparse.ArgumentParser()
ap.add_argument("--data", type=Path, default=Path("../task-china/task_china"))
ap.add_argument("--weights", default="best.pt")
ap.add_argument("--split", type=Path, default=Path("yolo_data/split.csv"))
ap.add_argument("--imgsz", type=int, default=640)
ap.add_argument("--out", type=Path, default=Path("counts.csv"))
a = ap.parse_args()

model = YOLO(a.weights)
labels = pd.read_csv(a.data / "train_labels.csv")
crates = pd.read_csv(a.data / "crates.csv")
glyphs = list(crates.glyph_id)


def detect(paths):
    """One (class, confidence) array per image."""
    out = []
    for i in range(0, len(paths), 32):
        for r in model.predict([str(p) for p in paths[i:i + 32]], imgsz=a.imgsz, conf=0.05,
                               iou=0.5, agnostic_nms=True, device="cpu", verbose=False):
            out.append(np.c_[r.boxes.cls.numpy(), r.boxes.conf.numpy()])
    return out


# per-glyph confidence threshold that best reproduces the true counts on validation images
split = pd.read_csv(a.split, index_col=0)["split"]
val_imgs = sorted(split[split == "val"].index)
val_dets = detect([a.data / "train" / "images" / n for n in val_imgs])

thresholds = {}
print("glyph  thr   true  pred  mean|err|/img")
for c, g in enumerate(glyphs):
    true = np.array([((labels.image == n) & (labels.glyph_id == g)).sum() for n in val_imgs])
    best = None
    for t in np.arange(0.10, 0.90, 0.025):
        pred = np.array([(d[:, 0] == c)[d[:, 1] >= t].sum() for d in val_dets])
        score = abs(pred.sum() - true.sum()) + np.abs(pred - true).sum()
        if best is None or score < best[0]:
            best = (score, t, pred)
    _, t, pred = best
    thresholds[g] = t
    print(f"{g}  {t:.3f}  {true.sum():4d}  {pred.sum():4d}  {np.abs(pred - true).mean():.3f}")

# count the crate's glyph in every set
rows, word = [], ""
for c, (crate, g) in enumerate(zip(crates.crate, crates.glyph_id)):
    per_set = []
    for set_dir in sorted((a.data / "crates" / crate).iterdir()):
        dets = detect(sorted(set_dir.glob("*.jpg")))
        n = int(sum(((d[:, 0] == c) & (d[:, 1] >= thresholds[g])).sum() for d in dets))
        per_set.append(n)
        rows.append([crate, set_dir.name, g, n])

    # decode: split at the biggest gap, high = 1, set_1 is the top bit, 1 = A
    s = sorted(per_set)
    cut = s[int(np.argmax(np.diff(s)))]
    bits = "".join("1" if n > cut else "0" for n in per_set)
    letter = chr(ord("A") + int(bits, 2) - 1)
    word += letter
    print(crate, g, per_set, bits, letter)

pd.DataFrame(rows, columns=["crate", "set", "glyph_id", "count"]).to_csv(a.out, index=False)
print("written", a.out)
print(f"FLAG: SIGMOID_{word}")
