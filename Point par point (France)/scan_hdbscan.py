from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import HDBSCAN
from sklearn.preprocessing import StandardScaler

root = Path(__file__).resolve().parent
d = pd.read_csv(root / "paris_listings.csv")
sets = [["price", "review_scores_rating", "availability_365"], ["price", "review_scores_rating", "number_of_reviews"], ["price", "minimum_nights", "number_of_reviews", "availability_365"], ["price", "review_scores_rating", "reviews_per_month"]]
for(j, fs) in enumerate(sets):
    x = StandardScaler().fit_transform(d[fs].fillna(d[fs].median()))
    labels = HDBSCAN(min_cluster_size=20, min_samples=5).fit_predict(x)
    counts = pd.Series(labels).value_counts()
    groups = [int(c) for c,n in counts.items() if(c != -1 and n <= 2000)]
    print(j, counts.to_dict(), flush=True)
    fig, axes = plt.subplots((len(groups) + 7) // 8, 8, figsize=(24, 3 * ((len(groups) + 7) // 8)), squeeze=False)
    for(ax, label) in zip(axes.flat, groups):
        s = d[labels == label]
        ax.scatter(s.longitude, s.latitude, s=3)
        ax.set_title(f"{label}: n={len(s)}")
    fig.tight_layout()
    fig.savefig(root / f"scan_hdbscan_{j}.png", dpi=110)
    np.save(root / f"scan_hdbscan_{j}.npy", labels)
