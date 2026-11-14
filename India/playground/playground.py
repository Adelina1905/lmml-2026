import os, itertools, cv2, numpy as np, imagehash, pandas as pd
from PIL import Image

IMG_DIR = "data/samples/train"
files = sorted(f for f in os.listdir(IMG_DIR) if f.lower().endswith((".jpg", ".jpeg", ".png")))



hashes = {f: imagehash.phash(Image.open(os.path.join(IMG_DIR, f))) for f in files}


# Shortlist: for each image, its 5 most similar fingerprint
candidates = set()
for a in files:
  near = sorted((b for b in files if b != a), key=lambda b: hashes[a] - hashes[b])[:5]
  for b in near:
    candidates.add(tuple(sorted((a, b))))
print(len(candidates), "pairs to check")


sift = cv2.SIFT_create()
def load(f):
  img = cv2.imread(os.path.join(IMG_DIR, f), cv2.IMREAD_GRAYSCALE)
  s = 800 / max(img.shape)
  return cv2.resize(img, None, fx=s, fy=s) if s < 1 else img

feats = {}
def get(f):
  if f not in feats:
    feats[f] = sift.detectAndCompute(load(f), None)
  return feats[f]

def agreeing_points(a, b):
    ka, da = get(a); kb, db = get(b)
    if da is None or db is None or len(ka) < 10 or len(kb) < 10: return 0
    matches = cv2.BFMatcher().knnMatch(da, db, k=2)
    good = [m for m, n in (p for p in matches if len(p) == 2) if m.distance < 0.75 * n.distance]
    if len(good) < 10: return 0
    src = np.float32([ka[m.queryIdx].pt for m in good])
    dst = np.float32([kb[m.trainIdx].pt for m in good])
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    return 0 if mask is None else (int(mask.sum()), mask.mean())

rows = []
for a, b in candidates:
    r = agreeing_points(a, b)
    inl, ratio = r if isinstance(r, tuple) else (0, 0.0)
    rows.append((a, b, inl, ratio, hashes[a] - hashes[b]))
df = pd.DataFrame(rows, columns=["image1", "image2", "inliers", "ratio", "hash_dist"])
os.makedirs("output", exist_ok=True)
df.to_csv("output/all_scores.csv", index=False)

# Decide is at least 40 points agree
THRESH = 180
dups = df[(df.inliers >= THRESH) | (df.hash_dist <= 4)]
dups[["image1", "image2"]].to_csv("output/output.csv", index=False)
print(len(dups), "duplicate pairs found")

