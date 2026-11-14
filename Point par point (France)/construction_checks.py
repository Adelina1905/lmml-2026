from pathlib import Path
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

ROOT = Path(__file__).resolve().parent
d = pd.read_csv(ROOT / "paris_listings.csv")
cents = np.rint(d.price * 100).astype(int) % 100
common = [0, 50, 33, 67, 25, 75, 20, 40, 60, 80]
odd = ~cents.isin(common)
print("Odd price cents by minimum nights:", d[odd].minimum_nights.value_counts().head(20).to_dict(), flush=True)
print("Duplicate coordinates:", d.groupby(["latitude", "longitude"]).size().sort_values(ascending=False).head(12).to_dict(), flush=True)
print("Column precision:", {c: d[c].astype(str).str.split(".").str[-1].str.len().value_counts().to_dict() for c in ["latitude", "longitude", "price", "review_scores_rating", "reviews_per_month"]}, flush=True)
masks = {"Unusual cents, stays < 7 nights": odd & (d.minimum_nights < 7), "Unusual cents, stays < 4 nights": odd & (d.minimum_nights < 4), "Unusual cents, stays >= 28 nights": odd & (d.minimum_nights >= 28), "Unusual cents, entire rental units, short stays": odd & (d.minimum_nights < 7) & (d.property_type == "Entire rental unit")}
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
for(ax, (title, mask)) in zip(axes.flat, masks.items()):
    s = d[mask]
    ax.scatter(s.longitude, s.latitude, s=4)
    ax.set_title(f"{title}\nn={len(s)}")
fig.tight_layout()
fig.savefig(ROOT / "construction_price.png", dpi=140)
fig, axes = plt.subplots(3, 1, figsize=(14, 9))
for(ax, col) in zip(axes, ["price", "review_scores_rating", "availability_365"]):
    ax.plot(d[col].rolling(250).mean())
    ax.set_ylabel(col)
fig.tight_layout()
fig.savefig(ROOT / "construction_row_order.png", dpi=130)
