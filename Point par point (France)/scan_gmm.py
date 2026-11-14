from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

root = Path(__file__).resolve().parent
d = pd.read_csv(root / "paris_listings.csv")
sets = [["price", "review_scores_rating", "availability_365"], ["price", "review_scores_rating", "number_of_reviews"], ["price", "review_scores_rating", "reviews_per_month"], ["price", "minimum_nights", "availability_365"]]
for(j, fs) in enumerate(sets):
    x = StandardScaler().fit_transform(d[fs].fillna(d[fs].median()))
    for(k) in [7, 12]:
        model = GaussianMixture(n_components=k, random_state=42, n_init=2).fit(x)
        labels = model.predict(x)
        print(j, k, pd.Series(labels).value_counts().to_dict(), flush=True)
        fig, axes = plt.subplots(2, (k + 1) // 2, figsize=(3 * ((k + 1) // 2), 6), squeeze=False)
        for(label, ax) in enumerate(axes.flat):
            s = d[labels == label]
            ax.scatter(s.longitude, s.latitude, s=1)
            ax.set_title(f"{label}: n={len(s)}")
        fig.tight_layout()
        fig.savefig(root / f"scan_gmm_{j}_{k}.png", dpi=120)
