"""
LMML 2026 - task_norway: "The Nisse's Bad Shift"

Idea
----
Labels are not wrong at random. On a "bad night" the nisse swapped two digits
(a <-> b) EVERY time he wrote them. So the errors live in one contiguous block
of sample IDs (one night) and affect exactly one pair of digits, all of them.

Pipeline
--------
1. Train a digit classifier (RBF SVM) on the train scans with their TRUE labels.
2. Predict every test scan and compare with the label on file (given_label).
3. Group samples into nights (blocks of 150 consecutive IDs, the night size seen
   in train). For every night and every digit pair (a, b) count how many samples
   labelled a are predicted b and vice versa ("swap evidence").
4. A night is "bad" if, for one pair, most of its a/b samples look swapped.
   Then swap EVERY a <-> b in that night (the nisse does it every single time),
   even where the classifier is unsure. Isolated disagreements in good nights are
   classifier mistakes, not label errors, so they are left alone.
5. Validate the whole procedure on train with leave-one-night-out (the classifier
   never sees the night it is judging) and score repairs with the task's F1.
6. Retrain the classifier on all corrected labels (train + repaired test) and
   save it as model.joblib.

Run from inside this folder:  python solve.py
"""
import os
import sys
from itertools import combinations

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.svm import SVC

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
RES = os.path.join(HERE, "results")
os.makedirs(RES, exist_ok=True)

NIGHT_SIZE = 150      # one night = 150 consecutive sample IDs (seen in train: 450..599)
MIN_RATIO = 0.5       # >= half of the night's a/b samples must look swapped
MIN_COUNT = 5         # and at least 5 of them (a real swap gives ~25-35)
PIXELS = [f"pixel_{i}" for i in range(64)]

_log = []


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    _log.append(s)


def make_model():
    # RBF-kernel SVM on the raw 0..16 pixels (deterministic, ~99% on these digits).
    # gamma="scale" = 1 / (n_features * X.var()) adapts to the pixel range, so no
    # extra scaling is needed (dividing pixels by 16 would give the identical kernel).
    return SVC(C=10, gamma="scale")


def sample_num(ids):
    return ids.str.replace("IMG_", "", regex=False).astype(int).values


def night_evidence(given, pred):
    """For one night: best swapped pair and its evidence."""
    best = None
    for a, b in combinations(range(10), 2):
        in_pair = (given == a) | (given == b)
        n = int(in_pair.sum())
        if n == 0:
            continue
        k = int(((given == a) & (pred == b)).sum() + ((given == b) & (pred == a)).sum())
        cand = (k / n, k, n, a, b)
        if best is None or cand[:2] > best[:2]:
            best = cand
    return best  # (ratio, k, n, a, b)


def detect_and_repair(nums, given, pred):
    """Return repaired labels and a per-night table."""
    nights = nums // NIGHT_SIZE
    fixed = given.copy()
    rows = []
    for nt in np.unique(nights):
        m = nights == nt
        ratio, k, n, a, b = night_evidence(given[m], pred[m])
        bad = ratio >= MIN_RATIO and k >= MIN_COUNT
        if bad:
            g = given[m]
            f = g.copy()
            f[g == a] = b
            f[g == b] = a
            fixed[m] = f
        rows.append(dict(night=int(nt), first_id=int(nums[m].min()), last_id=int(nums[m].max()),
                         rows=int(m.sum()), disagreements=int((given[m] != pred[m]).sum()),
                         best_pair=f"{a}<->{b}", swapped_looking=k, pair_samples=n,
                         ratio=round(ratio, 3), bad_night=bad))
    return fixed, pd.DataFrame(rows)


def repair_f1(given, fixed, truth):
    """Task metric: a repair counts if label was wrong and is now the true label."""
    changed = fixed != given
    wrong = given != truth
    tp = int((changed & (fixed == truth)).sum())
    fp = int(changed.sum() - tp)
    fn = int(wrong.sum() - tp)
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return dict(TP=tp, FP=fp, FN=fn, precision=round(p, 4), recall=round(r, 4), F1=round(f1, 4))


def main():
    tr = pd.read_csv(os.path.join(DATA, "train-input.csv"))
    tro = pd.read_csv(os.path.join(DATA, "train-output.csv"))
    te = pd.read_csv(os.path.join(DATA, "test-input.csv"))
    tr = tr.merge(tro, on="sample_id", how="left", validate="1:1")
    assert tr.corrected_label.notna().all()

    Xtr, ytr, gtr = tr[PIXELS].values, tr.corrected_label.values, tr.given_label.values
    ntr = sample_num(tr.sample_id)
    Xte, gte = te[PIXELS].values, te.given_label.values
    nte = sample_num(te.sample_id)

    log("=== LMML 2026 task_norway: The Nisse's Bad Shift ===")
    log(f"python {sys.version.split()[0]}")
    import sklearn
    log(f"sklearn {sklearn.__version__}, numpy {np.__version__}, pandas {pd.__version__}")
    log(f"train rows {len(tr)}, test rows {len(te)}")

    # ---------- 1. What do the train errors look like? ----------
    wrong = gtr != ytr
    log(f"\n[1] Train: {wrong.sum()} wrong labels out of {len(tr)}")
    log("given -> true pairs:", pd.Series(list(zip(gtr[wrong], ytr[wrong]))).value_counts().to_dict())
    log(f"wrong IDs span {ntr[wrong].min()}..{ntr[wrong].max()} "
        f"(nights of {NIGHT_SIZE}: {sorted(set(ntr[wrong] // NIGHT_SIZE))})")
    in_bad = (ntr // NIGHT_SIZE == 3) & np.isin(ytr, [4, 9])
    log(f"4s and 9s in night 3 (IDs 450..599): {in_bad.sum()}, of which wrong: {(in_bad & wrong).sum()}"
        f"  -> EVERY 4/9 that night is swapped, nothing else anywhere is wrong")

    # ---------- 2. Classifier quality ----------
    cv = StratifiedKFold(10, shuffle=True, random_state=0)
    acc = cross_val_score(make_model(), Xtr, ytr, cv=cv)
    log(f"\n[2] SVM 10-fold CV accuracy on train (true labels): {acc.mean():.4f} +- {acc.std():.4f}")

    # ---------- 3. Honest validation: leave-one-night-out on train ----------
    log("\n[3] Validation on train, leave-one-night-out (model never sees the night it judges)")
    nights_tr = ntr // NIGHT_SIZE
    pred_lono = np.empty_like(ytr)
    for nt in np.unique(nights_tr):
        m = nights_tr == nt
        pred_lono[m] = make_model().fit(Xtr[~m], ytr[~m]).predict(Xtr[m])
    fixed_tr, tab_tr = detect_and_repair(ntr, gtr, pred_lono)
    log(tab_tr.to_string(index=False))
    s_method = repair_f1(gtr, fixed_tr, ytr)
    s_naive = repair_f1(gtr, pred_lono, ytr)  # baseline: trust the classifier everywhere
    log(f"night-swap method   : {s_method}")
    log(f"naive 'relabel every disagreement' baseline: {s_naive}")
    tab_tr.to_csv(os.path.join(RES, "train_night_evidence.csv"), index=False)

    # ---------- 4. Test ----------
    log("\n[4] Test: model trained on all train true labels")
    model = make_model().fit(Xtr, ytr)
    pred_te = model.predict(Xte)
    fixed_te, tab_te = detect_and_repair(nte, gte, pred_te)
    log(tab_te.to_string(index=False))
    tab_te.to_csv(os.path.join(RES, "test_night_evidence.csv"), index=False)

    changed = fixed_te != gte
    log(f"\nrepairs on test: {changed.sum()}")
    log("given -> corrected:", pd.Series(list(zip(gte[changed], fixed_te[changed]))).value_counts().to_dict())
    log(f"repaired IDs span {nte[changed].min()}..{nte[changed].max()}")
    rep = pd.DataFrame(dict(sample_id=te.sample_id[changed], given_label=gte[changed],
                            corrected_label=fixed_te[changed], svm_prediction=pred_te[changed]))
    rep.to_csv(os.path.join(RES, "test_repairs.csv"), index=False)
    log(rep.to_string(index=False))

    # sanity: are the repairs independent of the exact night boundaries?
    bad_rows = tab_te[tab_te.bad_night]
    for _, r in bad_rows.iterrows():
        a, b = map(int, r.best_pair.split("<->"))
        ev = np.where(np.isin(gte, [a, b]) & (((gte == a) & (pred_te == b)) | ((gte == b) & (pred_te == a))))[0]
        lo, hi = nte[ev].min(), nte[ev].max()
        tight = (nte >= lo) & (nte <= hi) & np.isin(gte, [a, b])
        ok_before = nte[np.isin(gte, [a, b]) & (nte < lo)].max()
        ok_after = nte[np.isin(gte, [a, b]) & (nte > hi)].min()
        same = set(np.where(tight)[0]) == set(np.where(changed)[0])
        log(f"boundary check: swap evidence spans {lo}..{hi}; nearest untouched {a}/{b} labels at "
            f"{ok_before} and {ok_after} (both agree with the classifier). Repairs with the tight "
            f"evidence window instead of the 150-block are identical: {same}")

    left = (~changed) & (pred_te != gte)
    log(f"\ndisagreements left unchanged (classifier mistakes in good nights): {left.sum()}")
    log(pd.DataFrame(dict(sample_id=te.sample_id[left], given_label=gte[left],
                          svm_prediction=pred_te[left])).to_string(index=False))

    # ---------- 5. Deliverable ----------
    out = pd.DataFrame({"sample_id": te.sample_id.values, "corrected_label": fixed_te.astype(int)})
    out.to_csv(os.path.join(HERE, "test-output.csv"), index=False)
    chk = pd.read_csv(os.path.join(HERE, "test-output.csv"))
    assert list(chk.columns) == ["sample_id", "corrected_label"]
    assert len(chk) == len(te) and (chk.sample_id.values == te.sample_id.values).all()
    assert chk.corrected_label.between(0, 9).all()
    log(f"\nwrote test-output.csv: {len(chk)} rows, columns {list(chk.columns)}")

    # ---------- 6. Retrain on corrected labels and save ----------
    Xall = np.vstack([Xtr, Xte])
    yall = np.concatenate([ytr, fixed_te])
    acc_all = cross_val_score(make_model(), Xall, yall, cv=cv)
    final = make_model().fit(Xall, yall)
    joblib.dump(final, os.path.join(HERE, "model.joblib"))
    log(f"\n[6] Final model: RBF SVM trained on {len(yall)} samples with corrected labels")
    log(f"10-fold CV accuracy on corrected labels: {acc_all.mean():.4f} +- {acc_all.std():.4f}")
    acc_raw = cross_val_score(make_model(), Xall, np.concatenate([gtr, gte]), cv=cv)
    log(f"(same model on the ORIGINAL noisy labels: {acc_raw.mean():.4f})")
    log("saved model.joblib  (load: joblib.load('model.joblib').predict(pixels_0_to_16))")

    with open(os.path.join(RES, "report.txt"), "w") as f:
        f.write("\n".join(_log) + "\n")

    # ---------- plots (optional, needs matplotlib) ----------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for ax, tab, title in [(axes[0], tab_tr, "Train (leave-one-night-out)"), (axes[1], tab_te, "Test")]:
            colors = ["#d62728" if b else "#7f7f7f" for b in tab.bad_night]
            ax.bar([f"{r.first_id}-{r.last_id}" for r in tab.itertuples()], tab.ratio, color=colors)
            ax.axhline(MIN_RATIO, ls="--", c="k", lw=1)
            for i, r in enumerate(tab.itertuples()):
                ax.text(i, r.ratio + 0.02, r.best_pair, ha="center", fontsize=8)
            ax.set_ylim(0, 1.1)
            ax.set_title(title)
            ax.set_ylabel("share of best pair's samples that look swapped")
            ax.tick_params(axis="x", rotation=60, labelsize=7)
        fig.tight_layout()
        fig.savefig(os.path.join(RES, "night_swap_evidence.png"), dpi=130)

        fig, ax = plt.subplots(figsize=(12, 3))
        dis = pred_te != gte
        ax.scatter(nte[dis & ~changed], gte[dis & ~changed], c="#7f7f7f", label="disagreement left as is")
        ax.scatter(nte[changed], gte[changed], c="#d62728", label="repaired (3<->8 night)")
        for x in range(1200, 1800, NIGHT_SIZE):
            ax.axvline(x, c="k", lw=0.5, ls=":")
        ax.set_xlabel("sample number")
        ax.set_ylabel("given label")
        ax.set_yticks(range(10))
        ax.legend(loc="center right")
        ax.set_title("Test: where the classifier disagrees with the label on file")
        fig.tight_layout()
        fig.savefig(os.path.join(RES, "test_disagreements.png"), dpi=130)
        log("plots saved in results/")
    except ImportError:
        log("matplotlib not installed: skipping plots")


if __name__ == "__main__":
    main()
