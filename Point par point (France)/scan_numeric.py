from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, HDBSCAN
from sklearn.preprocessing import StandardScaler

root = Path(__file__).resolve().parent
d = pd.read_csv(root / "paris_listings.csv")
numeric = [c for c in d.select_dtypes(include="number").columns if(c not in ["listing_id", "longitude", "latitude"])]
feature_sets = [numeric, ["price", "accommodates", "review_scores_rating"], ["price", "minimum_nights", "availability_365"], ["price", "number_of_reviews", "availability_365", "review_scores_rating"], ["price", "accommodates", "bedrooms", "beds", "review_scores_rating", "availability_365"]]
found = []
for(j, features) in enumerate(feature_sets):
    for(logged) in [False, True]:
        raw = d[features].fillna(d[features].median())
        if(logged): raw = np.log1p(raw)
        x = StandardScaler().fit_transform(raw)
        for(eps) in ([0.4, 0.7, 1.0, 1.4] if(len(features) > 3) else [0.05, 0.1, 0.2]):
            labels = DBSCAN(eps=eps, min_samples=8).fit_predict(x)
            counts = pd.Series(labels).value_counts()
            print(j, logged, eps, counts.head(12).to_dict(), flush=True)
            for(label, n) in counts.items():
                if(label == -1 or n < 15 or n > 1200): continue
                s = d[labels == label]
                compact = np.linalg.det(np.cov(s[["longitude", "latitude"]].to_numpy().T))
                found.append((compact, j, logged, eps, int(label), n, s.index.to_numpy()))
found.sort(key=lambda item: item[0])
fig, axes = plt.subplots(6, 8, figsize=(24, 18))
for(ax, (compact, j, logged, eps, label, n, idx)) in zip(axes.flat, found):
    s = d.loc[idx]
    ax.scatter(s.longitude, s.latitude, s=3)
    ax.set_title(f"set={j}, log={logged}, e={eps}\nlabel={label} n={n}", fontsize=9)
fig.tight_layout()
fig.savefig(root / "scan_numeric.png", dpi=130)
np.savez(root / "scan_numeric.npz", **{f"group_{i}": v[6] for i, v in enumerate(found[:48])})
