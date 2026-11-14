from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, HDBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parent
d = pd.read_csv(ROOT / "paris_listings.csv")
sets = [["price", "review_scores_rating", "availability_365"], ["price", "review_scores_rating", "minimum_nights"], ["price", "review_scores_rating", "number_of_reviews"], ["price", "accommodates", "minimum_nights"]]
for(j, fs) in enumerate(sets):
    x = StandardScaler().fit_transform(d[fs].fillna(d[fs].median()))
    first = DBSCAN(eps=0.4, min_samples=50).fit_predict(x)
    idx = np.flatnonzero(first == -1)
    second = HDBSCAN(min_cluster_size=8, min_samples=3, copy=True).fit_predict(x[idx])
    counts = pd.Series(second).value_counts()
    groups = list(counts.index)
    print("SET", j, "NOISE", len(idx), "GROUPS", counts.to_dict(), flush=True)
    fig, axes = plt.subplots(len(groups), 4, figsize=(13, max(4, len(groups) * 2.7)), squeeze=False)
    for(row, label) in enumerate(groups):
        subset = d.iloc[idx[second == label]]
        ax = axes[row, 0]
        ax.scatter(subset.longitude, subset.latitude, s=7)
        ax.set_title(f"{label}: n={len(subset)} (geography)")
        for(ax, (a, b)) in zip(axes[row, 1:], [(0, 1), (0, 2), (1, 2)]):
            ax.scatter(subset[fs[a]], subset[fs[b]], s=7)
            ax.set_xlabel(fs[a], fontsize=8)
            ax.set_ylabel(fs[b], fontsize=8)
    fig.tight_layout()
    fig.savefig(ROOT / f"noise_layers_{j}.png", dpi=110)
    np.savez(ROOT / f"noise_layers_{j}.npz", indices=idx, labels=second)
