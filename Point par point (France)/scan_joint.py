from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

root = Path(__file__).resolve().parent
d = pd.read_csv(root / "paris_listings.csv")
sets = [["longitude", "latitude", "price", "review_scores_rating", "availability_365"], ["longitude", "latitude", "price", "review_scores_rating"], ["longitude", "latitude", "price", "minimum_nights", "accommodates"], ["longitude", "latitude"]]
for(j, fs) in enumerate(sets):
    x = StandardScaler().fit_transform(d[fs].fillna(d[fs].median()))
    for(eps) in ([0.1, 0.2, 0.3, 0.5] if(j != 3) else [0.02, 0.04, 0.06]):
        labels = DBSCAN(eps=eps, min_samples=8).fit_predict(x)
        counts = pd.Series(labels).value_counts()
        groups = [int(c) for c,n in counts.items() if(c != -1 and n >= 35 and n <= 2500)]
        print(j, eps, counts.head(15).to_dict(), flush=True)
        if(not groups): continue
        fig, axes = plt.subplots((len(groups) + 7) // 8, 8, figsize=(24, 3 * ((len(groups) + 7) // 8)), squeeze=False)
        for(ax, label) in zip(axes.flat, groups):
            s = d[labels == label]
            ax.scatter(s.longitude, s.latitude, s=3)
            ax.set_title(f"{label}: n={len(s)}")
        fig.tight_layout()
        fig.savefig(root / f"scan_joint_{j}_{eps}.png", dpi=100)
        np.save(root / f"scan_joint_{j}_{eps}.npy", labels)
