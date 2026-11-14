#!/usr/bin/env python3
"""Optional: draws results/accuracy_breakdown.png (needs matplotlib). Run after solve.py."""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
rows = list(csv.DictReader(open(HERE / "results" / "items.csv", encoding="utf-8")))
ok = lambda rs: sum(r["correct"] == "True" for r in rs)
exc = [r for r in rows if r["status"] != "kept"]
kept = [r for r in rows if r["status"] == "kept"]

labels = [f"Reported\n(all {len(rows)} items)",
          f"Leaked items\n({len(exc)} excluded)",
          f"Honest\n({len(kept)} clean items)"]
vals = [ok(rows) / len(rows), ok(exc) / len(exc), ok(kept) / len(kept)]
counts = [f"{ok(rows)}/{len(rows)}", f"{ok(exc)}/{len(exc)}", f"{ok(kept)}/{len(kept)}"]

fig, ax = plt.subplots(figsize=(7, 4.2), dpi=150)
fig.patch.set_facecolor("#fcfcfb"); ax.set_facecolor("#fcfcfb")
bars = ax.bar(labels, vals, width=0.5, color=["#86b6ef", "#184f95", "#2a78d6"])
for b, v, c in zip(bars, vals, counts):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.2%}  ({c})",
            ha="center", va="bottom", fontsize=10, color="#0b0b0b")
ax.set_ylim(0, 1.12)
ax.set_ylabel("Accuracy", color="#52514e")
ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
ax.grid(axis="y", color="#e6e5e1", linewidth=0.8); ax.set_axisbelow(True)
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
ax.spines["bottom"].set_color("#b9b8b2")
ax.tick_params(colors="#52514e")
ax.set_title("CanalQA: the score drops to 64.41% once leaked items are removed",
             fontsize=11, color="#0b0b0b", loc="left")
fig.tight_layout()
fig.savefig(HERE / "results" / "accuracy_breakdown.png", facecolor=fig.get_facecolor())
print("wrote results/accuracy_breakdown.png")
