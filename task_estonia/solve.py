"""
task_estonia -- "This Data Is Perfect!"
Audit of the e-Residency fraud-screen pipeline (project_pipeline.py).

Three questionable choices in the shipped pipeline:
  A. StandardScaler is fit on ALL rows before the train/test split (preprocessing leak)   -> decoy
  B. train_test_split is not stratified although labels are imbalanced (29% fraud)        -> decoy
  C. The split is done per ROW, but many applicants submitted several near-identical
     applications (same date_of_birth + citizenship). The identity columns are dropped
     before splitting, so the same person lands in train AND test and 1-NN simply
     finds that person's other application.                                               -> REAL BUG

Run:  python solve.py          (from inside this folder; pandas 2.2.2, numpy 1.26.4, sklearn 1.5.1)
Outputs go to results/ and the flag to FLAG.txt.
"""

import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import (
    GroupShuffleSplit,
    LeaveOneGroupOut,
    LeaveOneOut,
    cross_val_predict,
    train_test_split,
)
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
os.makedirs(RES, exist_ok=True)

SEED = 1137          # the seed used by the shipped pipeline
N_SEEDS = 100        # repeated splits to measure run-to-run noise
ID_COLS = ["submission_id", "date_of_birth", "citizenship", "label"]

lines = []


def log(s=""):
    print(s)
    lines.append(s)


def model():
    """The shipped model, wrapped so the scaler is fit on training rows only."""
    return make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=1))


df = pd.read_csv(os.path.join(HERE, "applications.csv"))
X = df.drop(columns=ID_COLS)
y = df["label"]
# One applicant = same date of birth + same citizenship.
applicant = df["date_of_birth"] + "|" + df["citizenship"]

# ---------------------------------------------------------------- 0. data facts
log("=" * 72)
log("0. DATA FACTS")
log("=" * 72)
sizes = applicant.value_counts()
log(f"rows (applications)          : {len(df)}")
log(f"distinct applicants (DOB+cit): {applicant.nunique()}")
log("applications per applicant   : " +
    ", ".join(f"{k}x -> {v} people" for k, v in sizes.value_counts().sort_index().items()))
log(f"rows that belong to a repeat applicant: {(applicant.map(sizes) > 1).mean():.1%}")
log(f"applicants whose label is constant across their applications: "
    f"{(df.groupby(applicant)['label'].nunique() == 1).mean():.1%}")
spread = (df.groupby(applicant)[X.columns].std().mean() / X.std()).mean()
log(f"within-applicant feature spread / overall spread (mean over 6 features): {spread:.3f}"
    "  -> resubmissions are near-copies")
Xs_all = StandardScaler().fit_transform(X)
nn_idx = NearestNeighbors(n_neighbors=2).fit(Xs_all).kneighbors(Xs_all)[1][:, 1]
log(f"nearest neighbour (excluding itself) is the SAME applicant for "
    f"{(applicant.values[nn_idx] == applicant.values).mean():.1%} of rows")
log(f"class balance: {y.mean():.1%} fraud; majority-class baseline accuracy = {1 - y.mean():.4f}")

# ------------------------------------------------- 1. reproduce the reported number
log()
log("=" * 72)
log("1. REPRODUCE THE REPORTED SCORE (exact copy of project_pipeline.py)")
log("=" * 72)


def shipped(seed, fix_scaler=False, stratify=False, group=False):
    """Shipped holdout evaluation with each questionable choice switchable on its own."""
    if group:
        tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
                      .split(X, y, applicant))
    else:
        idx = np.arange(len(df))
        tr, te = train_test_split(idx, test_size=0.2, random_state=seed,
                                  stratify=y if stratify else None)
    if fix_scaler:
        clf = model().fit(X.iloc[tr], y.iloc[tr])
        pred = clf.predict(X.iloc[te])
    else:
        Xs = StandardScaler().fit_transform(X)          # fit on all rows (shipped behaviour)
        clf = KNeighborsClassifier(n_neighbors=1).fit(Xs[tr], y.iloc[tr])
        pred = clf.predict(Xs[te])
    return accuracy_score(y.iloc[te], pred)


reported = shipped(SEED)
log(f"shipped pipeline, seed {SEED}: {reported:.4f}   (report says 0.9342)")

# Direct re-run of the original code path for full fidelity
Xs = StandardScaler().fit_transform(X)
a, b, c, d = train_test_split(Xs, y, test_size=0.2, random_state=SEED)
orig = accuracy_score(d, KNeighborsClassifier(n_neighbors=1).fit(a, c).predict(b))
assert abs(orig - reported) < 1e-12
log(f"verbatim project_pipeline.py logic       : {orig:.4f}")

# ------------------------------------------------ 2. each choice tested in isolation
log()
log("=" * 72)
log("2. FIX EACH QUESTIONABLE CHOICE IN ISOLATION (holdout 80/20)")
log("=" * 72)
variants = {
    "shipped (no fix)":                     dict(),
    "A: scaler fit on train only":          dict(fix_scaler=True),
    "B: stratified split":                  dict(stratify=True),
    "A+B: both decoys fixed":               dict(fix_scaler=True, stratify=True),
    "C: split by applicant (real fix)":     dict(group=True),
    "A+C: scaler fixed + split by applicant": dict(fix_scaler=True, group=True),
}
seed_scores = {}
for name, kw in variants.items():
    s1137 = shipped(SEED, **kw)
    many = np.array([shipped(s, **kw) for s in range(N_SEEDS)])
    seed_scores[name] = {"seed_1137": s1137, "mean": many.mean(), "std": many.std(ddof=1),
                         "min": many.min(), "max": many.max(), "all": many}
noise = seed_scores["shipped (no fix)"]["std"]
log(f"run-to-run noise of the shipped pipeline (std over {N_SEEDS} random seeds): "
    f"{noise:.4f} = {noise * 100:.2f} pp")
log()
log(f"{'variant':38s} {'seed 1137':>10s} {'mean':>8s} {'std':>7s} {'change vs shipped (mean)':>26s}")
base_mean = seed_scores["shipped (no fix)"]["mean"]
for name, s in seed_scores.items():
    delta = s["mean"] - base_mean
    log(f"{name:38s} {s['seed_1137']:10.4f} {s['mean']:8.4f} {s['std']:7.4f} "
        f"{delta * 100:+8.2f} pp ({delta / noise:+.1f} x noise)")

# paired differences on identical seeds isolate the effect of each decoy
log()
for name in ["A: scaler fit on train only", "B: stratified split"]:
    diff = seed_scores[name]["all"] - seed_scores["shipped (no fix)"]["all"]
    log(f"paired diff [{name}] vs shipped over {N_SEEDS} seeds: mean {diff.mean() * 100:+.2f} pp, "
        f"std {diff.std(ddof=1) * 100:.2f} pp")
diff = seed_scores["C: split by applicant (real fix)"]["all"] - seed_scores["shipped (no fix)"]["all"]
log(f"paired diff [C: split by applicant] vs shipped over {N_SEEDS} seeds: mean {diff.mean() * 100:+.2f} pp")

# --------------------------------------------- 3. official flag metric (leave-one-out)
log()
log("=" * 72)
log("3. FLAG METRIC: one out-of-fold prediction per row, cross_val_predict")
log("=" * 72)
pred_loo = cross_val_predict(model(), X, y, cv=LeaveOneOut())
acc_loo = (pred_loo == y).mean()
log(f"LeaveOneOut()      (row-level, still leaky)            : {acc_loo:.4f}")

pred_loo_out = cross_val_predict(KNeighborsClassifier(n_neighbors=1), Xs_all, y, cv=LeaveOneOut())
log(f"LeaveOneOut(), scaler fit on all rows (decoy A unfixed) : {(pred_loo_out == y).mean():.4f}")

logo = LeaveOneGroupOut()
pred_logo = cross_val_predict(model(), X, y, cv=logo, groups=applicant)
acc_true = (pred_logo == y).mean()
log(f"LeaveOneGroupOut() by applicant (CORRECTED)             : {acc_true:.4f}   "
    f"({logo.get_n_splits(groups=applicant)} folds = one per applicant)")

pred_logo_out = cross_val_predict(KNeighborsClassifier(n_neighbors=1), Xs_all, y,
                                  cv=logo, groups=applicant)
acc_logo_out = (pred_logo_out == y).mean()
log(f"LeaveOneGroupOut(), scaler fit on all rows (decoy A)    : {acc_logo_out:.4f}  "
    "-> same whole percent; decoy A does not matter")

# robustness: group by date of birth alone
pred_dob = cross_val_predict(model(), X, y, cv=LeaveOneGroupOut(), groups=df["date_of_birth"])
log(f"robustness: LeaveOneGroupOut grouped by DOB only        : {(pred_dob == y).mean():.4f}")

flag = f"SIGMOID_{int(round(acc_true * 100))}"
log()
log(f"TRUE ACCURACY = {acc_true:.4f}  (majority baseline {1 - y.mean():.4f})")
log(f"FLAG          = {flag}")

# ------------------------------------------------------------------ 4. save outputs
with open(os.path.join(HERE, "FLAG.txt"), "w") as f:
    f.write(flag + "\n")
with open(os.path.join(RES, "report.txt"), "w") as f:
    f.write("\n".join(lines) + "\n")
pd.DataFrame({"submission_id": df["submission_id"], "applicant": applicant, "label": y,
              "pred_leaky_loo": pred_loo, "pred_grouped_logo": pred_logo}).to_csv(
    os.path.join(RES, "oof_predictions.csv"), index=False)
metrics = {
    "reported_shipped_seed1137": reported,
    "noise_std_shipped_over_seeds": noise,
    "holdout_variants": {k: {kk: float(vv) for kk, vv in v.items() if kk != "all"}
                         for k, v in seed_scores.items()},
    "loo_leaky": float(acc_loo),
    "logo_corrected": float(acc_true),
    "logo_scaler_fit_on_all": float(acc_logo_out),
    "majority_baseline": float(1 - y.mean()),
    "flag": flag,
}
with open(os.path.join(RES, "metrics.json"), "w") as f:
    json.dump(metrics, f, indent=2)

# chart: one horizontal bar per variant (seed-averaged), 2 colours = leaky vs grouped split
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [(n, s["mean"], s["std"]) for n, s in seed_scores.items()]
    rows += [("LOO, row-level (leaky)", acc_loo, 0.0),
             ("LOGO by applicant (corrected, FLAG)", acc_true, 0.0)]
    names = [r[0] for r in rows][::-1]
    vals = [r[1] for r in rows][::-1]
    errs = [r[2] for r in rows][::-1]
    grouped = ["applicant" in n or n.startswith("C") or "A+C" in n for n in names]
    colors = ["#eb6834" if g else "#2a78d6" for g in grouped]
    fig, ax = plt.subplots(figsize=(9, 4.8), dpi=150)
    fig.patch.set_facecolor("#fcfcfb"); ax.set_facecolor("#fcfcfb")
    ax.barh(names, vals, xerr=errs, color=colors, height=0.6,
            error_kw=dict(ecolor="#52514e", lw=1, capsize=3))
    ax.axvline(1 - y.mean(), color="#8a8984", lw=1, ls="--")
    ax.text(1 - y.mean() + 0.003, len(vals) - 0.45, "majority baseline 70.9%", color="#52514e", fontsize=8)
    for i, v in enumerate(vals):
        ax.text(v + errs[i] + 0.006, i, f"{v:.1%}", va="center", fontsize=8, color="#0b0b0b")
    ax.set_xlim(0.6, 1.02); ax.set_ylim(-0.6, len(vals) - 0.1)
    ax.set_xlabel("accuracy  (holdout bars: mean ± std over 100 seeds; last two: cross_val_predict over all rows)", color="#52514e", fontsize=8)
    ax.set_title("Only splitting by applicant moves the score; the two decoys stay within noise",
                 fontsize=10, color="#0b0b0b", loc="left")
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    ax.tick_params(colors="#52514e", labelsize=8)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color="#2a78d6", label="row-level split (same person in train & test)"),
                       Patch(color="#eb6834", label="split by applicant")],
              fontsize=8, frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(RES, "decoys_vs_real_bug.png"))
    print("saved results/decoys_vs_real_bug.png")
except ImportError:
    print("matplotlib not installed -- skipping chart")
