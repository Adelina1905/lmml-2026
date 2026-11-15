#!/usr/bin/env python3
"""Charts for results/ (needs matplotlib; main.py and solve.py do not)."""
import collections, csv, os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import main
groups = sorted(main.STATS, key=lambda g: -(main.STATS[g][2] - main.STATS[g][1]) / main.STATS[g][0])
c = [main.STATS[g][1] / main.STATS[g][0] for g in groups]
e = [main.STATS[g][2] / main.STATS[g][0] for g in groups]
rows = list(csv.DictReader(open(os.path.join(HERE, "train-input.csv"), encoding="utf-8-sig")))
models = main.route([r["prompt"] for r in rows])
share = collections.Counter(); tot = collections.Counter()
for r, m in zip(rows, models):
    g = main.group(r["prompt"]); tot[g] += 1; share[g] += m == "expensive"
fig, ax = plt.subplots(figsize=(10, 5.5))
y = range(len(groups))
ax.barh([i + 0.2 for i in y], c, height=0.4, color="#9aa5b1", label="apprentice (cheap, 1 dh) right")
ax.barh([i - 0.2 for i in y], e, height=0.4, color="#2f6db5", label="master (expensive, 25 dh) right")
for i, g in enumerate(groups):
    ax.text(1.02, i, "{}/{} sent to master".format(share[g], tot[g]), va="center", fontsize=8)
ax.set_yticks(list(y)); ax.set_yticklabels(groups); ax.invert_yaxis()
ax.set_xlim(0, 1.35); ax.set_xlabel("share of training requests answered correctly")
ax.set_title("Accuracy per request group (sorted by gain = master - apprentice)")
ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.12), ncol=2, fontsize=8, frameon=False); fig.tight_layout()
fig.savefig(os.path.join(HERE, "results", "group_accuracy.png"), dpi=130)
print("saved results/group_accuracy.png")
