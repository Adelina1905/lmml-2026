"""
LMML 2026 - France - "Point par point"

Find the word hidden among the Paris listings.

The idea in short:
  * Each planted dot gets a "colour" from what a traveller looks at before
    booking: price, availability_365 and reviews_per_month.
  * Inside one letter the colour changes smoothly along the strokes. So if two
    planted dots have almost the same colour, they are also close together on
    the map. For real listings, colour and location are mostly unrelated.
  * Step 1 (separate the letters from the city): for every listing, find its
    10 nearest neighbours in colour space and take the median map distance to
    them. This score has two clear peaks: planted dots are well under 1 km,
    real listings are about 3 km.
  * Step 2 (one cluster per letter): link each planted dot to the dots that
    are close to it in both colour and location (mutual k-NN graph); every
    connected piece of the graph is one letter.
  * Step 3 (look at each letter): plot each cluster on the map. For a cleaner
    picture, each dot is moved to the mean position of its colour neighbours
    inside the same letter.

Run from inside this folder:   python solve.py
Writes: labels.csv, FLAG.txt and the files in results/
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.mixture import GaussianMixture
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data", "paris_listings.csv")
RES = os.path.join(HERE, "results")
os.makedirs(RES, exist_ok=True)

K_COLOUR = 10        # colour neighbours used for the score
K_GRAPH = 8          # neighbours in the mutual k-NN graph
GEO_SCALE_KM = 0.5   # 0.5 km on the map counts like 1 std of colour
MIN_LETTER = 80      # smaller graph pieces are attached to the closest letter
MIN_PIECE = 20       # pieces smaller than this are treated as noise

report = []
def log(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    report.append(s)

# ----------------------------------------------------------------------------
# 0. data
# ----------------------------------------------------------------------------
df = pd.read_csv(DATA)
log(f"rows={len(df)}  cols={df.shape[1]}")

# map coordinates in km (local flat projection around Paris)
lat0 = np.radians(48.86)
X_KM = (df.longitude * 111.32 * np.cos(lat0)).to_numpy()   # east
Y_KM = (df.latitude * 110.57).to_numpy()                   # north

# colour = what a traveller weighs before booking a night
colour = pd.DataFrame({
    "log_price": np.log(df.price),
    "availability_365": df.availability_365.astype(float),
    "log_reviews_per_month": np.log(df.reviews_per_month),
})
C = StandardScaler().fit_transform(colour)

# ----------------------------------------------------------------------------
# 1. separate the planted dots from the real city
# ----------------------------------------------------------------------------
_, nn_idx = NearestNeighbors(n_neighbors=K_COLOUR + 1).fit(C).kneighbors(C)
G = np.c_[X_KM, Y_KM]
geo_dist = np.linalg.norm(G[nn_idx[:, 1:]] - G[:, None, :], axis=2)
score = np.median(geo_dist, axis=1)          # km
log_score = np.log10(score + 1e-3)

# the score has two peaks; a 2-component Gaussian mixture on log10(score)
# finds the threshold between them without any hand tuning
gm = GaussianMixture(2, random_state=0).fit(log_score.reshape(-1, 1))
low = int(np.argmin(gm.means_.ravel()))
p_low = gm.predict_proba(log_score.reshape(-1, 1))[:, low]
planted = p_low > 0.95          # keep only confident planted dots
thr_km = score[planted].max()
log(f"mixture means (km): {np.round(10**gm.means_.ravel(), 3)}")
log(f"planted candidates: {planted.sum()}  (score threshold ~{thr_km:.2f} km)")

# ----------------------------------------------------------------------------
# 2. split the planted dots into letters (colour + location graph)
# ----------------------------------------------------------------------------
idx = np.where(planted)[0]
Z = np.c_[C[idx], G[idx] / GEO_SCALE_KM]
_, gi = NearestNeighbors(n_neighbors=K_GRAPH + 1).fit(Z).kneighbors(Z)
n = len(idx)
rows = np.repeat(np.arange(n), K_GRAPH)
A = csr_matrix((np.ones(rows.size), (rows, gi[:, 1:].ravel())), shape=(n, n))
A = A.multiply(A.T)                       # keep only mutual neighbours
_, comp = connected_components(A, directed=False)
sizes = pd.Series(comp).value_counts()
big = [c for c in sizes.index if sizes[c] >= MIN_LETTER]
mid = [c for c in sizes.index if MIN_PIECE <= sizes[c] < MIN_LETTER]

# attach medium pieces to the letter with the nearest dot on the map
for c in mid:
    pc = G[idx[comp == c]]
    best, bd = None, np.inf
    for b in big:
        d = np.min(np.linalg.norm(G[idx[comp == b]][:, None, :] - pc[None], axis=2))
        if d < bd:
            best, bd = b, d
    log(f"piece of {sizes[c]} dots attached to piece {best} (gap {bd:.2f} km)")
    comp[comp == c] = best

raw_label = -np.ones(len(df), int)
for c in big:
    raw_label[idx[comp == c]] = c

# ----------------------------------------------------------------------------
# 3. look at each letter
# ----------------------------------------------------------------------------
# Each cluster is one letter. The artist's order: sort the letters by their
# median price per night (the first thing a traveller weighs) - cheapest first.
clusters = sorted(big, key=lambda c: np.median(df.price.to_numpy()[raw_label == c]))
# Letters read by eye from results/03_letters.png (glyph recovered from all five
# dimensions: lat, lon, log price, availability, log reviews/month).
# Every letter is drawn with its own rotation / squash, so read each one on its own.
READ_BY_PRICE = ["N", "E", "U", "R", "O", "N"]
assert len(clusters) == len(READ_BY_PRICE), f"expected 6 letters, got {len(clusters)}"
WORD = "".join(READ_BY_PRICE)          # NEURON
FLAG = f"SIGMOID_{WORD}"

# final labels: k = position of the letter in the word, -1 = real listing
label = -np.ones(len(df), int)
letter = np.array([""] * len(df), dtype=object)
for k, c in enumerate(clusters):
    m = raw_label == c
    label[m] = k
    letter[m] = READ_BY_PRICE[k]

out = pd.DataFrame({"listing_id": df.listing_id, "label": label, "letter": letter})
out.to_csv(os.path.join(HERE, "labels.csv"), index=False)
with open(os.path.join(HERE, "FLAG.txt"), "w") as f:
    f.write(FLAG + "\n")

log("\nletters (label = position in the word):")
for k, ch in enumerate(WORD):
    m = label == k
    log(f"  label {k}  '{ch}'  dots={m.sum():4d}  centre lat={df.latitude[m].mean():.4f} "
        f"lon={df.longitude[m].mean():.4f}  median price={df.price[m].median():.0f}  "
        f"median avail={df.availability_365[m].median():.0f}  "
        f"median rpm={df.reviews_per_month[m].median():.2f}")
log(f"real city listings (label -1): {(label == -1).sum()}")
log(f"\nFLAG: {FLAG}")

# ----------------------------------------------------------------------------
# figures
# ----------------------------------------------------------------------------
def smoothed_xy(m):
    """Move each dot of one letter to the mean map position of its 10 nearest
    colour neighbours in the same letter (this removes the ~150 m privacy
    jitter)."""
    ii = np.where(m)[0]
    cc = C[ii]
    _, nb = NearestNeighbors(n_neighbors=min(10, len(ii))).fit(cc).kneighbors(cc)
    return X_KM[ii][nb].mean(1), Y_KM[ii][nb].mean(1)

pal = plt.get_cmap("tab10")

# (a) score histogram
fig, ax = plt.subplots(figsize=(9, 4))
ax.hist(log_score, bins=250, color="0.4")
ax.axvline(np.log10(thr_km), color="r", ls="--", label=f"threshold {thr_km:.2f} km")
ax.set_xlabel("log10( median map distance to the 10 nearest colour neighbours, km )")
ax.set_ylabel("listings")
ax.set_title("Two groups: planted dots (left) and the real city (right)")
ax.legend()
fig.tight_layout(); fig.savefig(os.path.join(RES, "01_score_histogram.png"), dpi=130); plt.close(fig)

# (b) overview map
fig, ax = plt.subplots(figsize=(10, 8))
ax.scatter(df.longitude, df.latitude, s=0.3, c="0.85", label="real listings (grey)")
for k, ch in enumerate(WORD):
    m = label == k
    ax.scatter(df.longitude[m], df.latitude[m], s=4, color=pal(k), label=f"{k}: {ch}")
ax.set_aspect(1 / np.cos(lat0)); ax.set_xlabel("longitude"); ax.set_ylabel("latitude")
ax.set_title("Planted dots found in Paris, coloured by cluster")
ax.legend(markerscale=4, fontsize=8, loc="lower right")
fig.tight_layout(); fig.savefig(os.path.join(RES, "02_overview_map.png"), dpi=130); plt.close(fig)

# (c) each letter: glyph recovered from all 5 dimensions (PCA of standardised
#     lat, lon, log price, availability, log reviews/month inside the cluster).
#     The 2 leading components hold the letter; the other 3 are only noise.
D5 = np.c_[X_KM, Y_KM, colour.to_numpy()]
ROT = {0: 90, 1: 270, 2: 0, 3: 90, 4: 0, 5: 180}   # display rotation (deg) per label, chosen by eye
# label 0 has tiny, 0.01-rounded reviews_per_month values, so its colour plane is noisy;
# its map shape is clean, so we draw it from the map (x=-lat, y=-lon), smoothed in 5-D
USE_MAP = {0: True}
FLIP = {0: False, 1: False, 2: False, 3: False, 4: False, 5: False}
fig, axes = plt.subplots(2, 6, figsize=(26, 9))
for k, ch in enumerate(WORD):
    m = label == k
    Zs = D5[m]; Zs = (Zs - Zs.mean(0)) / Zs.std(0)
    _, sv, vt = np.linalg.svd(Zs, full_matrices=False)
    P = Zs @ vt.T[:, :2]
    _, nb = NearestNeighbors(n_neighbors=4).fit(Zs).kneighbors(Zs)
    if USE_MAP.get(k, False):
        _, nb = NearestNeighbors(n_neighbors=8).fit(Zs).kneighbors(Zs)
        P = np.c_[-Y_KM[m], -X_KM[m]]
        P = P - P.mean(0)
    P = P[nb].mean(1)
    th = np.radians(ROT[k]); R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    P = P @ R.T
    if FLIP[k]: P[:, 0] *= -1
    a = axes[0, k]; a.scatter(df.longitude[m], df.latitude[m], s=6, color=pal(k))
    a.set_aspect(1 / np.cos(lat0)); a.set_xticks([]); a.set_yticks([])
    a.set_title(f"label {k}: raw map (lon, lat)", fontsize=11)
    a = axes[1, k]; a.scatter(P[:, 0], P[:, 1], s=8, color=pal(k))
    a.set_aspect(1); a.set_xticks([]); a.set_yticks([])
    a.set_title(f"label {k} -> '{ch}'  (5-D glyph)  sv={np.round(sv[:3]/np.sqrt(m.sum()),2)}", fontsize=10)
fig.suptitle(f"Each cluster on its own; letters ordered by median price.  Word: {WORD}", fontsize=16)
fig.tight_layout(); fig.savefig(os.path.join(RES, "03_letters.png"), dpi=110); plt.close(fig)

# (d) the whole word in one picture (smoothed, same view)
fig, ax = plt.subplots(figsize=(9, 9))
for k, ch in enumerate(WORD):
    m = label == k
    sx, sy = smoothed_xy(m)
    ax.scatter(-sy, -sx, s=6, color=pal(k), label=f"{k}: {ch}")
    ax.text((-sy).mean(), (-sx).mean(), ch, fontsize=26, weight="bold", alpha=0.35)
ax.set_aspect(1); ax.set_xticks([]); ax.set_yticks([]); ax.legend(markerscale=4)
ax.set_title("All six letters (smoothed), x = -latitude, y = -longitude")
fig.tight_layout(); fig.savefig(os.path.join(RES, "04_all_letters.png"), dpi=120); plt.close(fig)

# (e) colour space: every letter is its own shape here as well
fig, axes = plt.subplots(1, 3, figsize=(20, 6))
pairs = [("log_price", "availability_365"), ("log_price", "log_reviews_per_month"),
         ("availability_365", "log_reviews_per_month")]
for a, (p, q) in zip(axes, pairs):
    a.scatter(colour[p], colour[q], s=0.3, c="0.85")
    for k, ch in enumerate(WORD):
        m = label == k
        a.scatter(colour[p][m], colour[q][m], s=4, color=pal(k), label=ch)
    a.set_xlabel(p); a.set_ylabel(q)
axes[0].legend(markerscale=4)
fig.suptitle("Colour features: the planted dots form separate shapes, the city is spread out")
fig.tight_layout(); fig.savefig(os.path.join(RES, "05_colour_space.png"), dpi=110); plt.close(fig)

with open(os.path.join(RES, "report.txt"), "w") as f:
    f.write("\n".join(report) + "\n")
print("\nwrote labels.csv, FLAG.txt, results/*")
