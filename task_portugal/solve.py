"""
LMML 2026 - Task Portugal: "The Douro Paradox" (Simpson's Paradox)

Finds the single column that, split into its natural groups, makes the
control (old method) beat the treatment (new drip irrigation) inside every
group, even though treatment wins overall.

Run from inside this folder:
    python solve.py
Outputs:
    results/report.txt          full printed report
    results/column_screen.csv   every candidate column and whether it reverses
    results/reversal_plot.png   overall vs within-group success rates
    results/distance_histogram.png  the natural gap in distance_from_porto_km
    FLAG.txt                    the flag only
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "douro_vineyard_data.csv")
RES = os.path.join(HERE, "results")
os.makedirs(RES, exist_ok=True)

lines = []


def out(s=""):
    print(s)
    lines.append(str(s))


def rates(frame, by=None):
    """Success rate and count per arm (optionally per group)."""
    keys = ["treatment"] if by is None else [by, "treatment"]
    t = frame.groupby(keys, observed=True).harvest_success.agg(["mean", "count"])
    return t.unstack("treatment") if by is not None else t


def full_reversal(frame, group_col, min_per_arm=30):
    """True if control > treatment in every group (each group has both arms)."""
    t = frame.groupby([group_col, "treatment"], observed=True).harvest_success.agg(["mean", "count"]).unstack()
    if t["count"].isna().any().any() or (t["count"] < min_per_arm).any().any():
        return False, t
    return bool((t["mean"]["control"] > t["mean"]["treatment"]).all()), t


def natural_gaps(x):
    """Gaps in the sorted unique values that are far bigger than the usual step."""
    s = np.sort(np.unique(x))
    g = np.diff(s)
    thr = max(10 * np.median(g), 0.02 * (s[-1] - s[0]))
    return [(float(s[i]), float(s[i + 1])) for i in np.where(g > thr)[0]]


df = pd.read_csv(DATA)
out(f"Rows: {len(df)}, columns: {df.shape[1]}, missing values: {int(df.isna().sum().sum())}")

# ------------------------------------------------------------------ 1. overall
out("\n=== 1. Overall success rate (pooled) ===")
ov = rates(df)
for arm in ["control", "treatment"]:
    out(f"  {arm:9s}: {ov.loc[arm, 'mean']:.4f}  ({ov.loc[arm, 'mean']*100:.2f}%)  n={int(ov.loc[arm, 'count'])}")
out(f"  -> treatment ahead by {(ov.loc['treatment','mean']-ov.loc['control','mean'])*100:+.2f} percentage points")

# ------------------------------------------------------------- 2. screen columns
out("\n=== 2. Screening every candidate column ===")
skip = {"plot_id", "treatment", "harvest_success"}
screen = []
for c in df.columns:
    if c in skip:
        continue
    if df[c].dtype == object or str(df[c].dtype) == "str" or df[c].nunique() <= 15:
        ok, t = full_reversal(df, c)
        ctrl_wins = int((t["mean"]["control"] > t["mean"]["treatment"]).sum())
        screen.append((c, "categorical (own levels)", len(t), ctrl_wins, ok))
    else:
        gaps = natural_gaps(df[c].values)
        if gaps:
            edges = [-np.inf] + [(a + b) / 2 for a, b in gaps] + [np.inf]
            g = pd.cut(df[c], edges)
            tmp = df.assign(_g=g)
            ok, t = full_reversal(tmp, "_g")
            t = t.dropna()
            ctrl_wins = int((t["mean"]["control"] > t["mean"]["treatment"]).sum())
            screen.append((c, f"numeric, natural gap(s) {gaps}", len(t), ctrl_wins, ok))
        # exhaustive single-threshold scan (both sides >= 30 plots per arm)
        x = df[c].values
        y = df.harvest_success.values
        tr = (df.treatment == "treatment").values
        hits = []
        for th in np.unique(np.quantile(x, np.linspace(0.05, 0.95, 181))):
            good = True
            for m in (x <= th, x > th):
                if (m & tr).sum() < 30 or (m & ~tr).sum() < 30 or y[m & tr].mean() >= y[m & ~tr].mean():
                    good = False
                    break
            if good:
                hits.append(round(float(th), 2))
        screen.append((c, f"numeric, any single threshold (scan of 181 cut points) -> reversing cuts: {hits}", 2, None, bool(hits)))

sc = pd.DataFrame(screen, columns=["column", "split", "n_groups", "groups_where_control_wins", "full_reversal"])
sc.to_csv(os.path.join(RES, "column_screen.csv"), index=False)
for r in sc.itertuples():
    out(f"  {r.column:24s} full_reversal={str(r.full_reversal):5s} | {r.split}"
        + ("" if pd.isna(r.groups_where_control_wins) else f" | control wins {int(r.groups_where_control_wins)}/{r.n_groups} groups"))

winners = sorted(set(sc[sc.full_reversal].column))
out(f"\nColumns producing a full reversal: {winners}")
assert winners == ["distance_from_porto_km"], winners
COL = winners[0]

# --------------------------------------------------- 3. the split of the answer
out(f"\n=== 3. Splitting {COL} into its natural groups ===")
gaps = natural_gaps(df[COL].values)
lo, hi = gaps[0]
out(f"  Natural gap: no plot lies between {lo} km and {hi} km "
    f"(every other step between neighbouring values is about 0.1 km)")
near = df[COL] <= lo
df["distance_group"] = np.where(near, f"near Porto (70-{lo:g} km)", f"far from Porto ({hi:g}-205 km)")
order = [f"near Porto (70-{lo:g} km)", f"far from Porto ({hi:g}-205 km)"]

gt = rates(df, "distance_group").loc[order]
out("")
out(f"  {'group':28s} {'control':>16s} {'treatment':>16s}   winner")
for g in order:
    c_m, t_m = gt.loc[g, ("mean", "control")], gt.loc[g, ("mean", "treatment")]
    c_n, t_n = int(gt.loc[g, ("count", "control")]), int(gt.loc[g, ("count", "treatment")])
    out(f"  {g:28s} {c_m*100:6.2f}% (n={c_n:4d}) {t_m*100:6.2f}% (n={t_n:4d})   {'CONTROL' if c_m > t_m else 'treatment'}")
    assert c_m > t_m
out(f"  {'ALL PLOTS (pooled)':28s} {ov.loc['control','mean']*100:6.2f}% (n={int(ov.loc['control','count']):4d}) "
    f"{ov.loc['treatment','mean']*100:6.2f}% (n={int(ov.loc['treatment','count']):4d})   treatment")

# ------------------------------------------------------------- 4. why it happens
out("\n=== 4. Why pooling reverses the result ===")
share = df.groupby("treatment").distance_group.value_counts(normalize=True).unstack()[order]
base = df.groupby("distance_group").harvest_success.mean()[order]
for arm in ["control", "treatment"]:
    out(f"  {arm:9s} plots: {share.loc[arm, order[0]]*100:5.1f}% near, {share.loc[arm, order[1]]*100:5.1f}% far")
out(f"  Baseline success: near {base[order[0]]*100:.1f}%, far {base[order[1]]*100:.1f}%")
out("  The new method was mostly trialled far from Porto, where vineyards succeed more often anyway,")
out("  while the old method sat mostly near Porto, where success is lower; pooling mixes this location")
out("  effect into the comparison and makes treatment look better although it loses in both groups.")

# Standardised (location-adjusted) comparison: weight each group equally by its total size
w = df.distance_group.value_counts()[order] / len(df)
adj_c = float((gt[("mean", "control")] * w).sum())
adj_t = float((gt[("mean", "treatment")] * w).sum())
out(f"\n  Location-adjusted rates (group rates weighted by group size): control {adj_c*100:.2f}%, treatment {adj_t*100:.2f}%")

# ------------------------------------------------------------------- 5. plots
fig, ax = plt.subplots(figsize=(8, 4.5))
cats = ["All plots\n(pooled)"] + [g.replace(" (", "\n(") for g in order]
cv = [ov.loc["control", "mean"]] + list(gt[("mean", "control")])
tv = [ov.loc["treatment", "mean"]] + list(gt[("mean", "treatment")])
xpos = np.arange(len(cats))
b1 = ax.bar(xpos - 0.2, np.array(cv) * 100, 0.4, label="control (old method)", color="#4C72B0")
b2 = ax.bar(xpos + 0.2, np.array(tv) * 100, 0.4, label="treatment (new drip)", color="#DD8452")
for b in list(b1) + list(b2):
    ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.4, f"{b.get_height():.1f}%", ha="center", fontsize=9)
ax.set_xticks(xpos, cats)
ax.set_ylabel("harvest_success rate (%)")
ax.set_title("Simpson's Paradox: treatment wins pooled, control wins in every distance group")
ax.set_ylim(0, 40)
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(os.path.join(RES, "reversal_plot.png"), dpi=130)

fig, ax = plt.subplots(figsize=(8, 3.5))
bins = np.arange(70, 206, 2.5)
ax.hist(df.loc[df.treatment == "control", COL], bins=bins, alpha=0.7, label="control", color="#4C72B0")
ax.hist(df.loc[df.treatment == "treatment", COL], bins=bins, alpha=0.7, label="treatment", color="#DD8452")
ax.axvspan(lo, hi, color="grey", alpha=0.2, label=f"empty gap {lo:g}-{hi:g} km")
ax.set_xlabel("distance_from_porto_km")
ax.set_ylabel("plots")
ax.set_title("Two location clusters: control mostly near Porto, treatment mostly far")
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(os.path.join(RES, "distance_histogram.png"), dpi=130)

# ------------------------------------------------------------------- 6. flag
flag = "SIGMOID_" + COL.upper()
with open(os.path.join(HERE, "FLAG.txt"), "w") as f:
    f.write(flag)
out(f"\nFLAG: {flag}")

with open(os.path.join(RES, "report.txt"), "w") as f:
    f.write("\n".join(lines) + "\n")
