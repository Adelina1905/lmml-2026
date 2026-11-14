import os, sys, pandas as pd
IMG_DIR = sys.argv[1] if len(sys.argv) > 1 else "data/samples/train"
TRUTH = "data/train-output.csv"          # change to your CSV's path

images = {f for _, _, fs in os.walk(IMG_DIR) for f in fs}
out = pd.read_csv("output.csv")

# Format rules
assert list(out.columns) == ["image_a", "image_b"], "wrong columns"
assert not (out.image_a == out.image_b).any(), "self-pairs found"
pairs = [tuple(sorted(p)) for p in zip(out.image_a, out.image_b)]
assert len(pairs) == len(set(pairs)), "repeated pairs found"
bad = {n for p in pairs for n in p} - images
assert not bad, f"unknown image names: {bad}"
print("Format OK")

# Score
# T = {tuple(sorted(p)) for p in zip(*[pd.read_csv(TRUTH)[c] for c in ("image_a", "image_b")])}
T = {tuple(sorted(p)) for p in zip(*[pd.read_csv(TRUTH)[c] for c in ("image_a", "image_b")]) if p[0] in images and p[1] in images}
M = set(pairs); C = T & M
print(f"true {len(T)} | found {len(M)} | correct {len(C)}")
print("recall   ", round(len(C) / max(len(T), 1), 3))
print("precision", round(len(C) / max(len(M), 1), 3))
print("missed:", sorted(T - M)[:20]); print("wrong:", sorted(M - T)[:20])