from pathlib import Path
from itertools import combinations
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

root = Path(__file__).resolve().parent
d = pd.read_csv(root / "paris_listings.csv")
pool = ["price", "review_scores_rating", "number_of_reviews", "reviews_per_month", "availability_365", "accommodates"]
found = []
for(features) in combinations(pool, 3):
    x = StandardScaler().fit_transform(d[list(features)])
    for(eps) in [0.08, 0.15, 0.25, 0.4]:
        labels = DBSCAN(eps=eps, min_samples=12).fit_predict(x)
        counts = pd.Series(labels).value_counts()
        for(label, n) in counts.items():
            if(label == -1 or n < 45 or n > 1800): continue
            s = d[labels == label]
            compact = np.linalg.det(np.cov(s[["longitude", "latitude"]].to_numpy().T))
            found.append((compact, features, eps, int(label), n, s.index.to_numpy()))
found.sort(key=lambda item: item[0])
fig, axes = plt.subplots(5, 8, figsize=(24, 16))
for(ax, (compact, features, eps, label, n, idx)) in zip(axes.flat, found):
    s = d.loc[idx]
    ax.scatter(s.longitude, s.latitude, s=3)
    ax.set_title(f"{','.join(f[:3] for f in features)} e={eps}\nlabel={label} n={n}", fontsize=9)
    print(features, eps, label, n, compact, flush=True)
fig.tight_layout()
fig.savefig(root / "scan_clusters.png", dpi=150)
np.savez(root / "scan_clusters.npz", **{f"group_{i}": v[5] for i, v in enumerate(found[:40])})
