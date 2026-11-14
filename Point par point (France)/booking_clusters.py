from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, HDBSCAN
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
d = pd.read_csv(ROOT / "paris_listings.csv")
features = ["price", "accommodates", "minimum_nights", "review_scores_rating", "availability_365"]
x = StandardScaler().fit_transform(d[features].fillna(d[features].median()))
first = DBSCAN(eps=0.5, min_samples=25).fit_predict(x)
for(name, idx) in [("all", np.arange(len(d))), ("noise", np.flatnonzero(first == -1))]:
    labels = HDBSCAN(min_cluster_size=12, min_samples=5, copy=True).fit_predict(x[idx])
    counts = pd.Series(labels).value_counts()
    print(name, counts.to_dict(), flush=True)
    groups = list(counts.index)
    fig, axes = plt.subplots((len(groups) + 7) // 8, 8, figsize=(24, 3 * ((len(groups) + 7) // 8)), squeeze=False)
    for(ax, label) in zip(axes.flat, groups):
        s = d.iloc[idx[labels == label]]
        ax.scatter(s.longitude, s.latitude, s=3)
        ax.set_title(f"{label}: n={len(s)}")
    fig.tight_layout()
    fig.savefig(ROOT / f"booking_{name}.png", dpi=110)
    np.savez(ROOT / f"booking_{name}.npz", indices=idx, labels=labels)
