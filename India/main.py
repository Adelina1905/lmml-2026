"""
Twins Separated at Birth - near-duplicate photo detection.

Reads test-input.zip from the current working directory and writes output.csv
with columns image_a,image_b: one row per unordered pair of images that come
from the same source photograph.

Pipeline
  1. Decode every image (downscaled), compute RootSIFT keypoints and small
     grayscale thumbnails.
  2. Candidate retrieval: for every image take its K nearest neighbours by a
     cheap global descriptor (8x8 normalised thumbnails of the full frame and
     of a 10% centre crop, compared both ways).
  3. Verification of each candidate pair:
       - SIFT matching + Lowe ratio test + RANSAC similarity transform
       - the transform's scale, converted to original pixel units, must look
         like a "resized copy" (ratio <= ~1.60x)
       - the second image is warped onto the first and pixel agreement is
         measured (NCC of intensities and of gradient magnitudes) on the
         overlap, ignoring black rotation borders.
  4. Pairs passing all thresholds are written to output.csv.
"""
import csv
import io
import os
import sys
import time
import zipfile
import multiprocessing as mp

import numpy as np

try:
    import cv2
    cv2.setNumThreads(1)
except Exception:  # pragma: no cover
    print("OpenCV (opencv-python) is required: pip install opencv-python-headless", file=sys.stderr)
    raise

from PIL import Image

INPUT_ZIP = "test-input.zip"
OUTPUT_CSV = "output.csv"
IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff", ".gif")

SIFT_SIDE = 512       # longest side used for SIFT
CHECK_SIDE = 320      # longest side used for the pixel comparison
N_FEATURES = 600      # SIFT keypoints per image
K_CAND = 15           # neighbours per image from the global descriptor

# decision thresholds (tuned on the training set)
MIN_INLIERS = 10
MIN_NCC = 0.88        # intensity NCC after alignment
MIN_GRAD_NCC = 0.55   # gradient-magnitude NCC after alignment
MAX_LOG_SCALE = 0.47  # |log(pixel scale ratio)|, i.e. ratio <= 1.60

G = {}  # per-process globals (inherited through fork)


# ----------------------------------------------------------------------------
# image loading / features
# ----------------------------------------------------------------------------
def decode(data, target):
    """Decode image bytes to a BGR array whose longest side is <= target."""
    W0 = H0 = None
    try:
        W0, H0 = Image.open(io.BytesIO(data)).size
    except Exception:
        pass
    flag = cv2.IMREAD_COLOR
    if W0:
        for r, fl in ((8, cv2.IMREAD_REDUCED_COLOR_8), (4, cv2.IMREAD_REDUCED_COLOR_4),
                      (2, cv2.IMREAD_REDUCED_COLOR_2)):
            if max(W0, H0) / r >= target:
                flag = fl
                break
    im = cv2.imdecode(np.frombuffer(data, np.uint8), flag)
    if im is None:  # fall back to PIL for unusual formats
        pil = Image.open(io.BytesIO(data))
        pil.seek(0)
        im = cv2.cvtColor(np.array(pil.convert("RGB")), cv2.COLOR_RGB2BGR)
        W0, H0 = pil.size
    if W0 is None:
        H0, W0 = im.shape[:2]
    h, w = im.shape[:2]
    s = target / max(h, w)
    if s < 1:
        im = cv2.resize(im, (max(1, int(round(w * s))), max(1, int(round(h * s)))),
                        interpolation=cv2.INTER_AREA)
    return im, (W0, H0)


def thumb(g, crop, s=8):
    h, w = g.shape
    c = g[int(h * crop):h - int(h * crop), int(w * crop):w - int(w * crop)]
    if c.size == 0:
        c = g
    t = cv2.resize(c, (s, s), interpolation=cv2.INTER_AREA).astype(np.float32).ravel()
    t -= t.mean()
    return t / (np.linalg.norm(t) + 1e-6)


def extract(job):
    zpath, members = job
    out = []
    sift = cv2.SIFT_create(nfeatures=N_FEATURES)
    with zipfile.ZipFile(zpath) as z:
        for m in members:
            try:
                data = z.read(m)
                im, size = decode(data, SIFT_SIDE)
                g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
                kp, des = sift.detectAndCompute(g, None)
                if des is None or len(kp) == 0:
                    des = np.zeros((0, 128), np.uint8)
                    pts = np.zeros((0, 2), np.float32)
                else:
                    des = np.clip(des, 0, 255).astype(np.uint8)
                    pts = np.array([k.pt for k in kp], np.float32)
                h, w = g.shape
                s = CHECK_SIDE / max(h, w)
                gs = cv2.resize(g, (max(1, int(round(w * s))), max(1, int(round(h * s)))),
                                interpolation=cv2.INTER_AREA) if s < 1 else g.copy()
                glob = np.concatenate([thumb(g, 0.0), thumb(g, 0.1)])
                out.append((m, size, g.shape, pts, des, gs, glob, True))
            except Exception as e:  # unreadable file: keep the name, no features
                print("warning: could not read", m, e, file=sys.stderr)
                out.append((m, (1, 1), (1, 1), np.zeros((0, 2), np.float32),
                            np.zeros((0, 128), np.uint8), np.zeros((1, 1), np.uint8),
                            np.zeros(128, np.float32), False))
    return out


def rootsift(des):
    d = des.astype(np.float32)
    d /= d.sum(1, keepdims=True) + 1e-7
    return np.sqrt(d)


# ----------------------------------------------------------------------------
# pair verification
# ----------------------------------------------------------------------------
def valid_mask(g):
    """1 for real image content, 0 for black padding connected to the corners."""
    lab = (g <= 12).astype(np.uint8)
    h, w = g.shape
    ff = np.zeros((h + 2, w + 2), np.uint8)
    for (x, y) in ((0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)):
        if lab[y, x] == 1:
            cv2.floodFill(lab, ff, (x, y), 2)
    v = (lab != 2).astype(np.uint8)
    return cv2.erode(v, np.ones((3, 3), np.uint8))


def ncc(a, b, m):
    a = a[m]
    b = b[m]
    if a.size < 100:
        return -1.0
    a = a - a.mean()
    b = b - b.mean()
    return float((a * b).sum() / (np.sqrt((a * a).sum() * (b * b).sum()) + 1e-6))


def grad(x):
    return np.hypot(cv2.Sobel(x, cv2.CV_32F, 1, 0), cv2.Sobel(x, cv2.CV_32F, 0, 1))


def verify(pair):
    i, j = pair
    F = G["feat"]
    _, szi, shi, pi, di, gi, _, oki = F[i]
    _, szj, shj, pj, dj, gj, _, okj = F[j]
    r = dict(i=i, j=j, inl=0, ncc=-1.0, gncc=-1.0, lsp=9.0, rot=0.0)
    if not (oki and okj) or len(di) < 3 or len(dj) < 3:
        return r
    bf = cv2.BFMatcher(cv2.NORM_L2)
    m = bf.knnMatch(rootsift(di), rootsift(dj), k=2)
    good = [x[0] for x in m if len(x) == 2 and x[0].distance < 0.8 * x[1].distance]
    if len(good) < 4:
        return r
    src = pj[[g.trainIdx for g in good]]
    dst = pi[[g.queryIdx for g in good]]
    M, inl = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=3.0,
                                         maxIters=2000, confidence=0.995)
    if M is None:
        return r
    r["inl"] = int(inl.sum())
    sc = float(np.hypot(M[0, 0], M[1, 0]))
    r["rot"] = float(np.degrees(np.arctan2(M[1, 0], M[0, 0])))
    # scale in original pixel units (j pixels -> i pixels)
    ki = max(szi) / max(shi)
    kj = max(szj) / max(shj)
    r["lsp"] = abs(float(np.log(max(sc, 1e-6) * ki / kj)))
    if r["inl"] < MIN_INLIERS or r["lsp"] > MAX_LOG_SCALE + 0.2:
        return r
    # photometric check at CHECK_SIDE resolution
    fi = gi.shape[1] / shi[1]
    fj = gj.shape[1] / shj[1]
    Mc = M.copy()
    Mc[:, :2] *= fi / fj
    Mc[:, 2] *= fi
    h, w = gi.shape
    a = gi.astype(np.float32)
    b = cv2.warpAffine(gj.astype(np.float32), Mc, (w, h), flags=cv2.INTER_LINEAR)
    mask = (cv2.warpAffine(valid_mask(gj), Mc, (w, h), flags=cv2.INTER_NEAREST) > 0) & (valid_mask(gi) > 0)
    mask = cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    r["ov"] = float(mask.mean())
    r["ncc"] = ncc(a, b, mask)
    A1 = cv2.GaussianBlur(a, (0, 0), 1.0)
    B1 = cv2.GaussianBlur(b, (0, 0), 1.0)
    r["gncc"] = ncc(grad(A1), grad(B1), mask)
    return r


def is_dup(r):
    return (r["inl"] >= MIN_INLIERS and r["lsp"] <= MAX_LOG_SCALE
            and r["ncc"] >= MIN_NCC and r["gncc"] >= MIN_GRAD_NCC)


# ----------------------------------------------------------------------------
def main():
    t0 = time.time()
    zpath = INPUT_ZIP
    with zipfile.ZipFile(zpath) as z:
        members = [n for n in z.namelist()
                   if not n.endswith("/") and n.lower().endswith(IMG_EXT)
                   and not os.path.basename(n).startswith(".") and "__MACOSX" not in n]
    members.sort()
    print(f"{len(members)} images", flush=True)
    if len(members) < 2:
        with open(OUTPUT_CSV, "w", newline="") as f:
            csv.writer(f).writerow(["image_a", "image_b"])
        return

    nproc = max(1, os.cpu_count() or 1)
    chunks = [members[k:k + 32] for k in range(0, len(members), 32)]
    jobs = [(zpath, c) for c in chunks]
    try:
        ctx = mp.get_context("fork")
    except ValueError:
        ctx = None
    if ctx is not None and nproc > 1:
        with ctx.Pool(nproc) as pool:
            parts = pool.map(extract, jobs, chunksize=1)
    else:
        parts = [extract(jb) for jb in jobs]
    feat = [x for p in parts for x in p]
    G["feat"] = feat
    N = len(feat)
    print(f"features done in {time.time() - t0:.1f}s", flush=True)

    # candidate retrieval
    D = np.stack([f[6] for f in feat]).astype(np.float32)
    full, crop = D[:, :64], D[:, 64:]
    S = (full @ full.T + crop @ crop.T) / 2
    X = full @ crop.T
    S = np.maximum(S, np.maximum(X, X.T))
    np.fill_diagonal(S, -9)
    K = min(K_CAND, N - 1)
    nn = np.argpartition(-S, K, axis=1)[:, :K]
    cand = set()
    for i in range(N):
        for j in nn[i]:
            j = int(j)
            if i != j:
                cand.add((min(i, j), max(i, j)))
    del S, X
    cand = sorted(cand)
    print(f"{len(cand)} candidate pairs", flush=True)

    if ctx is not None and nproc > 1:
        with ctx.Pool(nproc) as pool:
            R = pool.map(verify, cand, chunksize=64)
    else:
        R = [verify(p) for p in cand]

    names = [os.path.basename(f[0]) for f in feat]
    pairs = set()
    for r in R:
        if is_dup(r):
            a, b = names[r["i"]], names[r["j"]]
            if a != b:
                pairs.add(tuple(sorted((a, b))))
    with open(OUTPUT_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image_a", "image_b"])
        for a, b in sorted(pairs):
            w.writerow([a, b])
    print(f"wrote {len(pairs)} pairs to {OUTPUT_CSV} in {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()