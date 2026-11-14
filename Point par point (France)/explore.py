from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
df = pd.read_csv(ROOT / "paris_listings.csv")
features = ["price", "availability_365", "review_scores_rating"]
print(df[features].value_counts().head(25).to_string())
fig, axes = plt.subplots(1, 4, figsize=(20, 5))
axes[0].scatter(df.longitude, df.latitude, s=1, alpha=0.25)
axes[0].set_title("All listings")
for(ax, feature) in zip(axes[1:], features):
    ax.scatter(df.longitude, df.latitude, c=df[feature], s=1, cmap="turbo")
    ax.set_title(feature)
fig.tight_layout()
fig.savefig(ROOT / "exploration.png", dpi=160)
plt.close(fig)
X = StandardScaler().fit_transform(df[features])
for(eps) in [0.03, 0.05, 0.08, 0.1, 0.15, 0.2]:
    labels = DBSCAN(eps=eps, min_samples=25).fit_predict(X)
    counts = pd.Series(labels).value_counts().sort_index()
    print("EPS", eps, "COUNTS", counts.to_dict())
    clusters = [int(label) for label, count in counts.items() if(label != -1 and count >= 50)]
    if(not clusters): continue
    fig, axes = plt.subplots(1, len(clusters), figsize=(max(5, len(clusters) * 3), 4), squeeze=False)
    for(ax, label) in zip(axes[0], clusters):
        subset = df[labels == label]
        ax.scatter(subset.longitude, subset.latitude, s=2)
        ax.set_title(f"{label}: n={len(subset)}")
    fig.tight_layout()
    fig.savefig(ROOT / f"exploration_dbscan_{eps}.png", dpi=130)
    plt.close(fig)
