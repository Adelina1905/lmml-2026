from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

root = Path(__file__).resolve().parent
d = pd.read_csv(root / "paris_listings.csv")
xy = d[["longitude", "latitude"]].to_numpy() * [np.cos(np.deg2rad(d.latitude.mean())), 1]
neighbors = NearestNeighbors(n_neighbors=31).fit(xy).kneighbors(return_distance=False)
names = d.neighbourhood.to_numpy()
agreement = (names[neighbors] == names[:, None]).mean(axis=1)
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for(ax, cutoff) in zip(axes, [0.1, 0.3, 0.5]):
    s = d[agreement < cutoff]
    ax.scatter(s.longitude, s.latitude, s=4)
    ax.set_title(f"Neighborhood disagreement: agreement < {cutoff}, n={len(s)}")
fig.tight_layout()
fig.savefig(root / "hint_neighborhood.png", dpi=140)
print("neighborhood mismatches", pd.Series(agreement).describe().to_dict(), flush=True)
features = ["price", "availability_365", "review_scores_rating"]
x = StandardScaler().fit_transform(d[features])
noise = DBSCAN(eps=0.2, min_samples=25).fit_predict(x) == -1
s = d[noise]
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for(ax, (a, b)) in zip(axes, [(features[0], features[1]), (features[0], features[2]), (features[1], features[2])]):
    ax.scatter(s[a], s[b], s=3, alpha=0.6)
    ax.set_xlabel(a)
    ax.set_ylabel(b)
fig.tight_layout()
fig.savefig(root / "hint_noise_features.png", dpi=140)
num = d.select_dtypes(include="number").drop(columns="listing_id")
z = StandardScaler().fit_transform(num.fillna(num.median()))
pc = PCA(n_components=6).fit_transform(z)
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
for(ax, (a, b)) in zip(axes.flat, [(0, 1), (0, 2), (1, 2), (2, 3), (3, 4), (4, 5)]):
    ax.scatter(pc[:, a], pc[:, b], s=1, alpha=0.25)
    ax.set_title(f"PC {a + 1} / {b + 1}")
fig.tight_layout()
fig.savefig(root / "hint_pca.png", dpi=130)
