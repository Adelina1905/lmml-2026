"""LMML 2026 - task_romania "Shaken, Still Standing".

Run from inside this folder:   python solve.py

What it does
  1. Loads train.csv (7,500 labelled) and sealed.csv (5,000 unlabelled).
  2. Finds the three regimes hidden in the data:
       A  low-gain records : 36 "heavy" channels are recorded at 0.25x amplitude
       B  high-gain records with c13 <= -0.60  (most of train)
       C  high-gain records with c13 >  -0.60  (most of sealed: covariate shift)
  3. Fits one expert per regime, each the model family that matches its rule:
       A  logistic regression on degree-2 polynomial features of the 12 gain-free channels
       B  3-way parity (XOR) of thresholds on c35, c43, c38
       C  plain logistic regression on all 48 channels
     Intermediate attempt (kept for comparison): B/C separated by a soft Gaussian gate on
     c13 learned with supervised EM.  Final model: the data show c13 is a HARD switch, so the
     cut is learned directly and each record goes to exactly one expert.
  4. Validates honestly: baseline single models vs. the combined models, with nested 5-fold CV
     on train and a reweighted estimate for the sealed mix of regimes.
  5. Writes predictions.csv, runs the official evaluator, saves FLAG.txt and results/.
"""
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, norm
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
RES = HERE / "results"
RES.mkdir(exist_ok=True)
CH = [f"c{j + 1:02d}" for j in range(48)]
SEED = 0
LOG = []


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.append(s)


# ============================================================ 1. load
train = pd.read_csv(HERE / "dataset" / "train.csv")
sealed = pd.read_csv(HERE / "dataset" / "sealed.csv")
X, y = train[CH].to_numpy(), train.vrancea.to_numpy()
Xs = sealed[CH].to_numpy()
ALL = np.vstack([X, Xs])                                  # unlabelled structure only, no labels used
log(f"train {X.shape}, sealed {Xs.shape}, share of vrancea=1 in train {y.mean():.3f}")

# ============================================================ 2. find the networks
# 2a. "heavy" channels: excess kurtosis ~1.1 means a scale mixture (two gains); the
#     other 12 channels are plain Gaussian (kurtosis ~0) and are not affected by the gain.
K = kurtosis(ALL, axis=0)
HEAVY = K > 0.5
GAUSS = [c for c, h in zip(CH, HEAVY) if not h]
log("Gaussian (gain-free) channels:", GAUSS)
log(f"heavy channels: {HEAVY.sum()}, kurtosis heavy ~{K[HEAVY].mean():.2f}, gaussian ~{K[~HEAVY].mean():.2f}")

CENTER = np.median(ALL, axis=0)


def gain_rms(Z):
    """RMS distance from the channel centre over the 36 heavy channels (one number per record)."""
    return np.sqrt(((Z[:, HEAVY] - CENTER[HEAVY]) ** 2).mean(1))


GAIN_CUT = 0.6
rms_all = gain_rms(ALL)
is_A_all = rms_all < GAIN_CUT
isA, isA_s = is_A_all[:7500], is_A_all[7500:]
log(f"network A (low gain): train {isA.sum()} ({isA.mean():.1%}), sealed {isA_s.sum()} ({isA_s.mean():.1%})")
log(f"gap in RMS: max A {rms_all[is_A_all].max():.3f}  <  min others {rms_all[~is_A_all].min():.3f}")
ratio = ALL[is_A_all].std(0) / ALL[~is_A_all].std(0)
log(f"std ratio A/others on heavy channels {ratio[HEAVY].mean():.3f}, on gaussian channels {ratio[~HEAVY].mean():.3f}")

I13 = CH.index("c13")
PAR = [CH.index(c) for c in ("c35", "c43", "c38")]
GI = [CH.index(c) for c in GAUSS]


# ============================================================ 3. experts
def make_A():
    return make_pipeline(StandardScaler(), PolynomialFeatures(2, include_bias=False), StandardScaler(),
                         LogisticRegression(C=0.03, max_iter=10000))


def make_C():
    return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000))


def parity(Z, th):
    """1 when an even number of (c35>t1, c43>t2, c38>t3) hold."""
    return (((Z[:, PAR] > th).sum(1)) % 2 == 0).astype(int)


def fit_parity(Z, yy, w, th):
    th = th.copy()
    for _ in range(3):                                    # coordinate search over the three thresholds
        for j in range(3):
            grid = np.quantile(Z[:, PAR[j]], np.linspace(0.05, 0.95, 361))
            sc = [np.average(parity(Z, np.r_[th[:j], v, th[j + 1:]]) == yy, weights=w) for v in grid]
            th[j] = grid[int(np.argmax(sc))]
    return th, np.average(parity(Z, th) == yy, weights=w)


def em_fit(B, yb, n_iter=30, seed=1, verbose=False):
    """Supervised EM for the B/C mixture of experts on the high-gain records.
    Latent z = network C.  Gate: Gaussian on c13 per network.  Experts: parity (B), logistic (C).
    Responsibility r_i ~ pi_k N(c13_i | mu_k, sd_k) p_k(y_i | x_i), with out-of-fold p_C."""
    c13 = B[:, I13]
    mu, sd, pi = np.array([-1.78, -0.33]), np.array([0.97, 0.97]), np.array([0.53, 0.47])   # [B, C]
    L = pi * norm.pdf(c13[:, None], mu, sd)
    r = L[:, 1] / L.sum(1)
    th = np.array([-0.15, -0.6, 0.7])
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    for it in range(n_iter):
        pC = np.zeros(len(yb))
        for a, b in cv.split(B, yb):
            m = make_C().fit(B[a], yb[a], logisticregression__sample_weight=r[a] + 1e-3)
            pC[b] = m.predict_proba(B[b])[:, 1]
        pC = np.clip(pC, 1e-3, 1 - 1e-3)
        th, accB = fit_parity(B, yb, 1 - r + 1e-3, th)
        pB = np.where(parity(B, th) == 1, accB, 1 - accB)
        lB = np.where(yb == 1, pB, 1 - pB)
        lC = np.where(yb == 1, pC, 1 - pC)
        LB = pi[0] * norm.pdf(c13, mu[0], sd[0]) * lB
        LC = pi[1] * norm.pdf(c13, mu[1], sd[1]) * lC
        r = LC / (LB + LC)
        pi = np.array([1 - r.mean(), r.mean()])
        mu = np.array([np.average(c13, weights=1 - r), np.average(c13, weights=r)])
        sd = np.sqrt([np.average((c13 - mu[0]) ** 2, weights=1 - r), np.average((c13 - mu[1]) ** 2, weights=r)])
        if verbose and (it % 5 == 0 or it == n_iter - 1):
            log(f"  EM it {it:2d}: pi(B,C)={pi.round(3)} mu={mu.round(3)} sd={sd.round(3)} "
                f"thresholds={th.round(3)} accB={accB:.4f} accC={np.average((pC > .5) == yb, weights=r):.4f}")
    modelC = make_C().fit(B, yb, logisticregression__sample_weight=r + 1e-3)
    return dict(mu=mu, sd=sd, pi=pi, th=th, accB=accB, modelC=modelC, r=r)


def prior_for(c13, mu, sd, n_iter=300):
    """Share of network C in a new set of records, by EM on c13 with mu/sd fixed."""
    w = 0.5
    for _ in range(n_iter):
        LB = (1 - w) * norm.pdf(c13, mu[0], sd[0])
        LC = w * norm.pdf(c13, mu[1], sd[1])
        w = (LC / (LB + LC)).mean()
    return w


def predict_BC(em, Z, piC):
    c13 = Z[:, I13]
    LB = (1 - piC) * norm.pdf(c13, em["mu"][0], em["sd"][0])
    LC = piC * norm.pdf(c13, em["mu"][1], em["sd"][1])
    gC = LC / (LB + LC)
    pC = em["modelC"].predict_proba(Z)[:, 1]
    pB = np.where(parity(Z, em["th"]) == 1, em["accB"], 1 - em["accB"])
    return gC * pC + (1 - gC) * pB, gC


def fit_full(Xtr, ytr, Atr, verbose=False):
    mA = make_A().fit(Xtr[Atr][:, GI], ytr[Atr])
    em = em_fit(Xtr[~Atr], ytr[~Atr], verbose=verbose)
    return mA, em


def predict_full(mA, em, Z, Az, piC):
    p = np.zeros(len(Z))
    p[Az] = mA.predict_proba(Z[Az][:, GI])[:, 1]
    p[~Az], _ = predict_BC(em, Z[~Az], piC)
    return p


# ---------- final model: c13 is a hard switch (root of a tree), not a fuzzy network marker
def switch_fit(B, yb, seed=1, verbose=False):
    """Learn the c13 cut between the parity rule (below) and the linear rule (above).
    1. fit each expert on records far from any plausible cut (c13 < -1.2 for parity, c13 > 0 for linear);
    2. out-of-fold, check which expert is right for every record;
    3. choose the cut that maximises accuracy; 4. refit both experts on their side of the cut."""
    c13 = B[:, I13]
    loB, hiC = c13 < -1.2, c13 > 0.0
    th, _ = fit_parity(B[loB], yb[loB], np.ones(loB.sum()), np.array([-0.15, -0.6, 0.7]))
    okB = parity(B, th) == yb                      # parity has only 3 thresholds: in-sample is fine
    okC = np.zeros(len(yb), bool)
    for a, b in StratifiedKFold(5, shuffle=True, random_state=seed).split(B, yb):
        aa = a[hiC[a]]
        okC[b] = make_C().fit(B[aa], yb[aa]).predict(B[b]) == yb[b]
    cand = np.unique(c13[(c13 > -1.2) & (c13 < 0.0)])
    order = np.argsort(c13)
    cs, oB, oC = c13[order], okB[order].astype(int), okC[order].astype(int)
    # accuracy if cut between cs[i-1] and cs[i]: parity on [0,i), linear on [i,n)
    accB_cum = np.r_[0, np.cumsum(oB)]
    accC_rev = np.r_[np.cumsum(oC[::-1])[::-1], 0]
    i_lo, i_hi = np.searchsorted(cs, -1.2), np.searchsorted(cs, 0.0)
    scores = accB_cum[i_lo:i_hi + 1] + accC_rev[i_lo:i_hi + 1]
    i = i_lo + int(np.argmax(scores))
    cut = (cs[i - 1] + cs[i]) / 2
    below = c13 <= cut
    th, accB = fit_parity(B[below], yb[below], np.ones(below.sum()), th)
    modelC = make_C().fit(B[~below], yb[~below])
    if verbose:
        log(f"  switch: c13 cut = {cut:.4f}  (records below {below.sum()}, above {(~below).sum()})")
        log(f"  parity thresholds c35>{th[0]:.3f}, c43>{th[1]:.3f}, c38>{th[2]:.3f}; in-sample parity acc {accB:.4f}")
    return dict(cut=cut, th=th, accB=accB, modelC=modelC)


def switch_predict(sw, Z):
    below = Z[:, I13] <= sw["cut"]
    pB = np.where(parity(Z, sw["th"]) == 1, sw["accB"], 1 - sw["accB"])
    pC = sw["modelC"].predict_proba(Z)[:, 1]
    return np.where(below, pB, pC), below


def final_fit(Xtr, ytr, Atr, verbose=False):
    mA = make_A().fit(Xtr[Atr][:, GI], ytr[Atr])
    sw = switch_fit(Xtr[~Atr], ytr[~Atr], verbose=verbose)
    return mA, sw


def final_predict(mA, sw, Z, Az):
    p = np.zeros(len(Z))
    p[Az] = mA.predict_proba(Z[Az][:, GI])[:, 1]
    p[~Az], _ = switch_predict(sw, Z[~Az])
    return p


# ============================================================ 4. validation
log("\n=== 4a. baseline single models (5-fold CV on train, all 48 channels) ===")
cv5 = StratifiedKFold(5, shuffle=True, random_state=SEED)
singles = {
    "logistic regression": lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)),
    "SVM (rbf)": lambda: make_pipeline(StandardScaler(), SVC(C=3.0)),
    "random forest": lambda: RandomForestClassifier(500, n_jobs=-1, random_state=SEED),
    "gradient boosting": lambda: HistGradientBoostingClassifier(random_state=SEED),
}
oof_single = {k: np.zeros(7500, int) for k in singles}
oof_mix = np.zeros(7500)
oof_mix_sealedprior = np.zeros(7500)
oof_final = np.zeros(7500)
iw = np.zeros(7500)
iw_switch = np.zeros(7500)
cuts = []
fold_piC = []
for f, (a, b) in enumerate(cv5.split(X, y)):
    for k, mk in singles.items():
        oof_single[k][b] = mk().fit(X[a], y[a]).predict(X[b])
    mA, em = fit_full(X[a], y[a], isA[a])
    pi_tr = em["pi"][1]
    pi_se = prior_for(Xs[~isA_s][:, I13], em["mu"], em["sd"])
    fold_piC.append((pi_tr, pi_se))
    oof_mix[b] = predict_full(mA, em, X[b], isA[b], pi_tr)
    oof_mix_sealedprior[b] = predict_full(mA, em, X[b], isA[b], pi_se)
    # importance weights: make the held-out fold look like the sealed mix of networks
    c13 = X[b][:, I13]
    dens = lambda w: (1 - w) * norm.pdf(c13, em["mu"][0], em["sd"][0]) + w * norm.pdf(c13, em["mu"][1], em["sd"][1])
    wb = np.where(isA[b], isA_s.mean() / isA.mean(),
                  (1 - isA_s.mean()) / (1 - isA.mean()) * dens(pi_se) / dens(pi_tr))
    iw[b] = wb
    # final model: hard switch on c13
    mA2, sw = final_fit(X[a], y[a], isA[a])
    oof_final[b] = final_predict(mA2, sw, X[b], isA[b])
    cuts.append(sw["cut"])
    # importance weights for the switch model: sealed share of each side of the cut, per network
    cs_tr = X[a][~isA[a]][:, I13] <= sw["cut"]
    cs_se = Xs[~isA_s][:, I13] <= sw["cut"]
    below_b = X[b][:, I13] <= sw["cut"]
    iw_switch[b] = np.where(isA[b], isA_s.mean() / isA.mean(),
                            (1 - isA_s.mean()) / (1 - isA.mean()) *
                            np.where(below_b, cs_se.mean() / cs_tr.mean(), (1 - cs_se.mean()) / (1 - cs_tr.mean())))
    log(f"  fold {f}: soft gate share of C train-part {pi_tr:.3f} sealed {pi_se:.3f} | hard switch cut {sw['cut']:.4f}")

rows = []
for k in singles:
    ok = oof_single[k] == y
    rows.append((k, ok.mean(), np.average(ok, weights=iw_switch), ok[isA].mean(), ok[~isA].mean()))
ok = (oof_mix > 0.5) == y
oks = (oof_mix_sealedprior > 0.5) == y
okf = (oof_final > 0.5) == y
rows.append(("experts + soft EM gate (train prior)", ok.mean(), np.average(ok, weights=iw_switch), ok[isA].mean(), ok[~isA].mean()))
rows.append(("experts + soft EM gate (sealed prior)", oks.mean(), np.average(oks, weights=iw_switch), oks[isA].mean(), oks[~isA].mean()))
rows.append(("FINAL experts + hard c13 switch", okf.mean(), np.average(okf, weights=iw_switch), okf[isA].mean(), okf[~isA].mean()))
log(f"c13 cut per fold: {np.round(cuts, 4)}")
tab = pd.DataFrame(rows, columns=["model", "cv_acc_train", "cv_acc_reweighted_to_sealed", "acc_net_A", "acc_net_B+C"])
log("\n=== 4b. results (nested 5-fold CV; whole pipeline refitted inside every fold) ===")
log(tab.round(4).to_string(index=False))
log(f"accuracy the evaluator needs: (n+k)/(2n) = (5000+4252)/10000 = 0.9252")
tab.to_csv(RES / "model_comparison.csv", index=False)

# per-expert view: which model family fits which network
log("\n=== 4c. which model family fits which network (5-fold CV inside each network) ===")
EM_all = em_fit(X[~isA], y[~isA], verbose=False)
rB = EM_all["r"]
hardB = rB < 0.1
hardC = rB > 0.9
blocks = {"A (low gain), 12 gaussian channels": (X[isA][:, GI], y[isA]),
          "B (c13 low, r<0.1)": (X[~isA][hardB], y[~isA][hardB]),
          "C (c13 high, r>0.9)": (X[~isA][hardC], y[~isA][hardC])}
fams = {"logistic (linear)": lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)),
        "logistic + degree-2 poly": make_A,
        "SVM rbf": lambda: make_pipeline(StandardScaler(), SVC(C=3)),
        "gradient boosting": lambda: HistGradientBoostingClassifier(random_state=SEED)}
fam_rows = []
for bn, (Zb, yb_) in blocks.items():
    row = {"network": bn, "n": len(yb_)}
    for fn, mk in fams.items():
        accs = []
        for a, b in StratifiedKFold(5, shuffle=True, random_state=SEED).split(Zb, yb_):
            accs.append((mk().fit(Zb[a], yb_[a]).predict(Zb[b]) == yb_[b]).mean())
        row[fn] = np.mean(accs)
    fam_rows.append(row)
fam = pd.DataFrame(fam_rows)
log(fam.round(4).to_string(index=False))
fam.to_csv(RES / "family_by_network.csv", index=False)

# are the remaining errors in regime A structured, or label noise?
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict
pA = cross_val_predict(make_A(), X[isA][:, GI], y[isA], cv=StratifiedKFold(10, shuffle=True, random_state=SEED),
                       method="predict_proba")[:, 1]
okA = ((pA > 0.5) == y[isA]).astype(int)
qA = cross_val_predict(HistGradientBoostingClassifier(max_depth=3, random_state=SEED), X[isA], okA,
                       cv=StratifiedKFold(5, shuffle=True, random_state=SEED), method="predict_proba")[:, 1]
log(f"regime A: 10-fold accuracy {okA.mean():.4f}; AUC of predicting its mistakes from all 48 channels "
    f"{roc_auc_score(okA, qA):.3f} (0.5 = unpredictable, i.e. label noise)")

# ============================================================ 5. final fit + sealed predictions
log("\n=== 5a. intermediate attempt: experts + soft EM gate, fitted on all training records ===")
mA, em = fit_full(X, y, isA, verbose=True)
piC_s = prior_for(Xs[~isA_s][:, I13], em["mu"], em["sd"])
log(f"gate on c13: B ~ N({em['mu'][0]:.3f}, {em['sd'][0]:.3f}),  C ~ N({em['mu'][1]:.3f}, {em['sd'][1]:.3f})")
log(f"share of network C among high-gain records: train {em['pi'][1]:.3f}  ->  sealed {piC_s:.3f}")
p_soft = predict_full(mA, em, Xs, isA_s, piC_s)
pred_soft = (p_soft > 0.5).astype(int)
pd.DataFrame({"record_id": sealed.record_id, "vrancea": pred_soft}).to_csv(RES / "predictions_soft_gate.csv", index=False)

log("\n=== 5b. FINAL: experts + hard c13 switch, fitted on all training records ===")
mA, sw = final_fit(X, y, isA, verbose=True)
p_s = final_predict(mA, sw, Xs, isA_s)
pred = (p_s > 0.5).astype(int)
_, below_s = switch_predict(sw, Xs[~isA_s])
log(f"sealed records routed: A (quadratic) {isA_s.sum()}, parity side {below_s.sum()}, linear side {(~below_s).sum()}")
log(f"predicted vrancea=1 share on sealed {pred.mean():.3f}")
okA_cv, okP_cv = okf[isA].mean(), okf[~isA & (X[:, I13] <= sw['cut'])].mean()
okL_cv = okf[~isA & (X[:, I13] > sw['cut'])].mean()
est = (isA_s.sum() * okA_cv + below_s.sum() * okP_cv + (~below_s).sum() * okL_cv) / len(Xs)
log(f"CV accuracy per branch: quadratic {okA_cv:.4f}, parity {okP_cv:.4f}, linear {okL_cv:.4f}")
log(f"expected sealed accuracy (branch CV accuracies x sealed branch sizes): {est:.4f}  -> about "
    f"{int(round((1 - est) * 5000))} errors; evaluator tolerates at most 374")

out = pd.DataFrame({"record_id": sealed.record_id, "vrancea": pred})
out.to_csv(HERE / "predictions.csv", index=False)
log("wrote predictions.csv", out.shape)

# ============================================================ 6. evaluator
res = subprocess.run([sys.executable, str(HERE / "evaluation_script.py"), str(HERE / "predictions.csv")],
                     capture_output=True, text=True)
answer = res.stdout.strip()
log("\n=== 6. evaluator says ===\n" + answer)
if answer.startswith("SIGMOID_"):
    (HERE / "FLAG.txt").write_text(answer + "\n")
rs = subprocess.run([sys.executable, str(HERE / "evaluation_script.py"), str(RES / "predictions_soft_gate.csv")],
                    capture_output=True, text=True)
log("evaluator on the soft-gate attempt:", rs.stdout.strip())
log(f"agreement final vs soft gate on sealed: {(pred_soft == pred).mean():.3f}")
# the baseline's best single model, for contrast
pb = singles["gradient boosting"]().fit(X, y).predict(Xs)
pd.DataFrame({"record_id": sealed.record_id, "vrancea": pb}).to_csv(RES / "predictions_gradient_boosting.csv", index=False)
rb = subprocess.run([sys.executable, str(HERE / "evaluation_script.py"), str(RES / "predictions_gradient_boosting.csv")],
                    capture_output=True, text=True)
log("evaluator on best single model (gradient boosting):", rb.stdout.strip())
log(f"agreement mixture vs gradient boosting on sealed: {(pb == pred).mean():.3f}")

(RES / "report.txt").write_text("\n".join(LOG) + "\n")

# ============================================================ 7. plots
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    BLUE, ORANGE, INK, MUTED = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e"
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": MUTED,
                         "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                         "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb"})

    fig, ax = plt.subplots(figsize=(7, 3.6))
    bins = np.linspace(0.15, 2.1, 70)
    ax.hist(rms_all[is_A_all], bins, color=ORANGE, label="network A (gain 0.25)", rwidth=0.9)
    ax.hist(rms_all[~is_A_all], bins, color=BLUE, label="networks B + C", rwidth=0.9)
    ax.axvline(GAIN_CUT, color=MUTED, ls="--", lw=1)
    ax.set_xlabel("RMS amplitude of the 36 heavy channels"); ax.set_ylabel("records")
    ax.set_title("Two sensor gains: a clean gap", color=INK, loc="left"); ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(RES / "01_gain_split.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 3.6))
    b2 = np.linspace(-5.5, 3.6, 60)
    ax.hist(X[~isA][:, I13], b2, density=True, color=BLUE, alpha=0.55, label="train (high gain)")
    ax.hist(Xs[~isA_s][:, I13], b2, density=True, color=ORANGE, alpha=0.55, label="sealed (high gain)")
    ax.axvline(sw["cut"], color=INK, ls="--", lw=1)
    ax.text(sw["cut"] + 0.08, 0.68, f"switch c13 = {sw['cut']:.3f}", fontsize=8, color=INK)
    ax.set_xlabel("c13"); ax.set_ylabel("density")
    ax.set_title("Covariate shift: sealed is enriched above the c13 switch", color=INK, loc="left")
    ax.legend(frameon=False, fontsize=8, loc="upper left"); fig.tight_layout()
    fig.savefig(RES / "02_c13_shift.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    t2 = tab.set_index("model")
    idx = np.arange(len(t2))
    ax.barh(idx + 0.2, t2.cv_acc_train, 0.38, color=BLUE, label="CV accuracy (train mix)")
    ax.barh(idx - 0.2, t2.cv_acc_reweighted_to_sealed, 0.38, color=ORANGE, label="reweighted to sealed mix")
    ax.axvline(0.9252, color=INK, ls="--", lw=1); ax.text(0.9262, len(t2) - 0.4, "needed 0.9252", fontsize=8, color=INK)
    ax.set_yticks(idx); ax.set_yticklabels(t2.index, fontsize=8); ax.set_xlim(0.5, 1.0)
    ax.set_xlabel("accuracy"); ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.set_title("Only the per-regime experts with the c13 switch clear the bar", color=INK, loc="left")
    fig.tight_layout(); fig.savefig(RES / "03_model_comparison.png", dpi=130); plt.close(fig)
    Bx, By = X[~isA], y[~isA]
    okB_ = parity(Bx, sw["th"]) == By
    okC_ = np.zeros(len(By), bool)
    for a, b in StratifiedKFold(5, shuffle=True, random_state=SEED).split(Bx, By):
        aa = a[Bx[a][:, I13] > sw["cut"]]
        okC_[b] = make_C().fit(Bx[aa], By[aa]).predict(Bx[b]) == By[b]
    edges = np.arange(-3.5, 2.01, 0.25)
    mids, aB, aC = [], [], []
    for lo_, hi_ in zip(edges[:-1], edges[1:]):
        m_ = (Bx[:, I13] > lo_) & (Bx[:, I13] <= hi_)
        if m_.sum() >= 15:
            mids.append((lo_ + hi_) / 2); aB.append(okB_[m_].mean()); aC.append(okC_[m_].mean())
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.plot(mids, aB, color=BLUE, lw=2, marker="o", ms=4, label="parity rule (c35, c43, c38)")
    ax.plot(mids, aC, color=ORANGE, lw=2, marker="o", ms=4, label="linear rule (logistic regression)")
    ax.axvline(sw["cut"], color=INK, ls="--", lw=1)
    ax.text(sw["cut"] + 0.05, 0.42, f"c13 = {sw['cut']:.3f}", fontsize=8, color=INK)
    ax.axhline(0.5, color=MUTED, lw=0.8, ls=":")
    ax.set_ylim(0.35, 1.02); ax.set_xlabel("c13 (high-gain records, train)"); ax.set_ylabel("accuracy in bin")
    ax.set_title("A hard switch: each rule is at chance on the other side", color=INK, loc="left")
    ax.legend(frameon=False, fontsize=8, loc="lower left"); fig.tight_layout()
    fig.savefig(RES / "04_c13_switch.png", dpi=130); plt.close(fig)
    log("plots saved to results/")
except ImportError:
    print("matplotlib not installed; skipped plots")
