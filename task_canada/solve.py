"""Two Solitudes - reproducible solution.

    python solve.py            # run from inside task_canada/ (all materials are in this folder)

Writes model.npz (W 9x4096, b 9), FLAG.txt and results/ next to this script and runs the vault on it.
Uses only: the synthetic corpus, the unlabeled sealed fingerprints, and hand-written synthetic
templates (gen.py). No outside incident records; the sealed reports are never labelled by hand.

Pipeline
  1. Parse every corpus report into header (place, date) + sentences.
  2. Generic sentences (impact boilerplate whose label mix is near uniform) are separated out;
     reports left with no hazard sentence at all (pure boilerplate) are dropped as label noise.
  3. The corpus label is the hazard of the FIRST hazard sentence; later sentences are often
     planted distractors from other hazards (e.g. an air crash inside a "winter" report).
     A sentence classifier trained on first sentences finds them; they are removed from the
     report and reused as examples of their own hazard.
  4. Rebuilt reports get a random place (own date kept) and random same-language boilerplate,
     so place names, boilerplate and template quirks stop carrying label signal.
  5. Extra synthetic reports from hand-written EN/FR templates with broader vocabulary (gen.py).
  6. Function words (EN+FR) and the digit token are masked out of the fingerprint, so language
     itself (the "two solitudes" confound in the corpus) cannot drive the decision.
  7. Logistic regression, then transductive adaptation on the unlabeled real fingerprints:
     class-balanced self-training, then label propagation over a kNN graph of the real reports,
     and a final refit on corpus + confidently propagated real reports.
"""
import csv, collections, math, random, re, sys
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression

HERE = Path(__file__).resolve().parent
TASK = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else HERE
sys.path.insert(0, str(TASK)); sys.path.insert(0, str(HERE))
from featurize import featurize, tokens, bucket          # noqa: E402  (task file, unchanged)
from open_vault import CLASSES, load_sealed, try_vault   # noqa: E402
from gen import generate                                 # noqa: E402

RES = HERE / "results"; RES.mkdir(exist_ok=True)
_LOG = open(RES / "run_log.txt", "w", encoding="utf-8")
def log(*a):
    print(*a, flush=True); print(*a, file=_LOG, flush=True)

# ------------------------------------------------------------------ language & stopwords
FR_W = ['le','la','les','des','du','une','est','sont','et','dans','au','ont','par','qui','sur']
EN_W = ['the','was','were','and','of','in','to','by','from','at','on','as','there']
def lang_of(X):
    fr = X[:, [bucket(w) for w in FR_W]].astype(bool).sum(1)
    en = X[:, [bucket(w) for w in EN_W]].astype(bool).sum(1)
    return (fr > en).astype(int)

STOP_EN = """a an the of in on at to from by for with without and or but as is was were be been being are has have had it its
this that these those there their they them he she his her we our you i which who whom whose what when where while after before
during over under into onto out up down off about above below between through across along around near than then so not no nor
also very more most some any each all both other others such only own same can could would should will may might must shall do
does did done""".split()
STOP_FR = """le la les l un une des de du d au aux et ou mais où donc or ni car à a en dans sur sous par pour avec sans chez vers entre
pendant depuis avant après près ce cet cette ces se s sa son ses leur leurs lui il ils elle elles on nous vous je qui que qu quoi
dont est sont était étaient été être ont avait avaient a y ne pas plus moins très tout tous toute toutes autre autres même aussi
puis lors quelque quelques plusieurs certains certaines comme si""".split()
STOP = set(STOP_EN) | set(STOP_FR) | {'0'}

# ------------------------------------------------------------------ parsing the synthetic corpus
HDR = [re.compile(r'^(.*?\b\d{4})\.\s+'),
       re.compile(r'^(On [^,]+, \d{4}, (?:in|near|at) [^,]+?(?: [A-Z]{2})?), '),
       re.compile(r'^(Le [^,]+ \d{4}, [^:]+): '),
       re.compile(r"^(Le [^,]+ \d{4}, (?:près de|à|dans|au|en|aux) [^,]+?), ")]
SPLIT = re.compile(r'(?<=[.!?])\s+(?=[A-ZÀ-Ý0-9])')
HP = re.compile(r'^(.*?),\s+((?:[^,]*\s)?\d{4})$')
def split_doc(t):
    for h in HDR:
        m = h.match(t)
        if m and len(m.group(1)) < 90:
            rest = t[m.end():]
            return m.group(1), (rest[0].upper() + rest[1:]) if rest else rest
    return None, t
def sents(t): return [s.strip() for s in SPLIT.split(t) if s.strip()]
def norm(s): return ' '.join(tokens(s))

def main():
    rng = random.Random(0)
    with open(TASK / "corpus.csv", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    X = featurize([r["text"] for r in rows]); y = np.array([CLASSES.index(r["label"]) for r in rows])
    Xs = load_sealed(TASK / "sealed_reports.npz")

    # stopword mask (function-word unigrams + frequent function-word bigrams)
    vocab = collections.Counter()
    for r in rows:
        tk = tokens(r["text"]); vocab.update(tk); vocab.update(a + ' ' + b for a, b in zip(tk, tk[1:]))
    mask = np.ones(4096, bool)
    for w in STOP: mask[bucket(w)] = False
    for w, c in vocab.items():
        if ' ' in w and c >= 20:
            a, b = w.split(' ')
            if a in STOP and b in STOP: mask[bucket(w)] = False

    # steps 1-2: headers, generic sentences, content-free reports
    P = []
    for r in rows:
        h, rest = split_doc(r["text"]); P.append((h, sents(rest), CLASSES.index(r["label"])))
    cnt = collections.defaultdict(collections.Counter)
    for h, ss, l in P:
        for s in ss: cnt[norm(s)][l] += 1
    gen = {s for s, c in cnt.items() if sum(c.values()) >= 8 and max(c.values()) / sum(c.values()) < 0.6}
    D = []
    for h, ss, l in P:
        cs = [s for s in ss if norm(s) not in gen]; gs = [s for s in ss if norm(s) in gen]
        txt = ' '.join(cs + gs) or (h or 'x')
        D.append(dict(h=h, cs=cs, gs=gs, y=l, lang=int(lang_of(featurize([txt]))[0])))
    log("content-free reports dropped:", sum(not d["cs"] for d in D))

    # step 3: distractor sentences via a first-sentence classifier
    fs = [(d["cs"][0], d["y"]) for d in D if d["cs"]]
    sm = LogisticRegression(C=10, max_iter=3000).fit(featurize([s for s, _ in fs]), [c for _, c in fs])
    later = [s for d in D for s in d["cs"][1:]]
    pl = sm.predict_proba(featurize(later)); k = 0
    for d in D:
        n = max(0, len(d["cs"]) - 1); d["lp"] = pl[k:k + n]; k += n

    # step 4: rebuilt, de-confounded reports
    pools = {0: {'place': [], 'g': []}, 1: {'place': [], 'g': []}}
    for d in D:
        if d["h"]:
            m = HP.match(d["h"])
            if m: pools[d["lang"]]['place'].append(m.group(1))
        pools[d["lang"]]['g'] += d["gs"]
    T, Y = [], []
    for d in D:
        if not d["cs"]: continue
        prim, dis = [d["cs"][0]], []
        for s, p in zip(d["cs"][1:], d["lp"]):
            (dis.append((s, int(p.argmax()))) if (p.argmax() != d["y"] and p.max() > 0.8) else prim.append(s))
        hd = ''
        if d["h"]:
            m = HP.match(d["h"])
        for _ in range(3):
            parts = prim + rng.sample(pools[d["lang"]]['g'], rng.randint(0, 4)); rng.shuffle(parts)
            hd = (rng.choice(pools[d["lang"]]['place']) + ', ' + m.group(2) + '. ') if (d["h"] and m) else ''
            T.append(hd + ' '.join(parts)); Y.append(d["y"])
        for s, c in dis:
            T.append(' '.join([s] + rng.sample(pools[d["lang"]]['g'], rng.randint(0, 4)))); Y.append(c)

    # step 5: hand-written template reports (with seasonal dates for the weather hazards)
    MON = ['January','February','March','April','May','June','July','August','September','October','November','December']
    MFR = ['janvier','février','mars','avril','mai','juin','juillet','août','septembre','octobre','novembre','décembre']
    SEAS = {'flood': [1,2,4,10,14,3,1,1,1,3,1,1], 'storm': [1,1,1,2,4,6,9,7,9,3,2,1],
            'winter': [10,6,4,1,.2,.1,.1,.1,.2,1,3,10], 'wildfire': [.2,.2,.3,2,7,9,10,5,1,.3,.2,.2],
            'drought': [.3,.3,.3,.5,2,7,10,8,2,.5,.3,.3]}
    rg = random.Random(0)
    for c, t in generate(300, seed=1):
        fr = any(w in tokens(t) for w in ('le', 'la', 'les', 'des', 'une'))
        mi = rg.choices(range(12), weights=SEAS.get(c, [1] * 12))[0]; yr = rg.randint(1900, 2020)
        place = rg.choice(pools[1 if fr else 0]['place'])
        if rg.random() < 0.85:
            date = (f"{rg.randint(1,28)} {MFR[mi]} {yr}" if fr else f"{MON[mi]} {rg.randint(1,28)}, {yr}") if rg.random() < 0.6 \
                else (f"{MFR[mi]} {yr}" if fr else f"{MON[mi]} {yr}")
            t = f"{place}, {date}. " + t
        T.append(t); Y.append(CLASSES.index(c))
    XA, YA = featurize(T), np.array(Y)
    log("training reports:", len(YA), np.bincount(YA))

    def fit(Xt, Yt, wt):
        m = LogisticRegression(C=10, max_iter=1000, tol=1e-3).fit(Xt * mask, Yt, sample_weight=wt)
        return m.coef_ * mask, m.intercept_
    def softmax(S): S = S - S.max(1, keepdims=True); E = np.exp(S); return E / E.sum(1, keepdims=True)

    # step 6-7a: base model + class-balanced self-training on the unlabeled real fingerprints
    W, b = fit(XA, YA, np.ones(len(YA)))
    log("base model fitted; corpus acc", ((X @ W.T + b).argmax(1) == y).mean().round(3))
    for nper in (20, 40, 60, 80, 100):
        Pr = softmax(Xs @ W.T + b); Pd = Pr.argmax(1); sel = []
        for c in range(9):
            idx = np.flatnonzero(Pd == c); sel += list(idx[np.argsort(-Pr[idx, c])][:nper])
        sel = np.array(sel)
        W, b = fit(np.vstack([XA, Xs[sel]]), np.concatenate([YA, Pd[sel]]),
                   np.concatenate([np.ones(len(YA)), np.full(len(sel), 5.0)]))
        log(" self-training round", nper, np.bincount((Xs @ W.T + b).argmax(1), minlength=9))

    # step 7b: label propagation over a kNN graph of the real reports, then refit
    df = (Xs > 0).sum(0); A = Xs * mask * np.log(len(Xs) / (1 + df))
    A /= np.linalg.norm(A, axis=1, keepdims=True) + 1e-9
    Sim = A @ A.T; np.fill_diagonal(Sim, -1)
    nn = np.argsort(-Sim, 1)[:, :10]; ns = np.maximum(np.take_along_axis(Sim, nn, 1), 0)
    Y0 = softmax(Xs @ W.T + b); F = Y0.copy()
    for _ in range(30):
        F = 0.7 * (F[nn] * ns[:, :, None]).sum(1) / (ns.sum(1, keepdims=True) + 1e-9) + 0.3 * Y0
    PL, conf = F.argmax(1), F.max(1)
    sel = np.argsort(-conf)[:int(0.8 * len(Xs))]
    W, b = fit(np.vstack([XA, Xs[sel]]), np.concatenate([YA, PL[sel]]),
               np.concatenate([np.ones(len(YA)), np.full(len(sel), 5.0)]))
    Pf = (Xs @ W.T + b).argmax(1)
    log("final predicted mix on real reports:", dict(zip(CLASSES, np.bincount(Pf, minlength=9).tolist())))

    # sanity check on hand-written fictional dev reports (not real records)
    from dev_reports import DEV
    Xd = featurize([t for _, t in DEV]); yd = np.array([CLASSES.index(l) for l, _ in DEV])
    log("accuracy on hand-written dev reports:", ((Xd @ W.T + b).argmax(1) == yd).mean().round(3))

    np.savez(HERE / "model.npz", W=W, b=b)
    mix = np.bincount(Pf, minlength=9)
    with open(RES / "prediction_mix.csv", "w") as f:
        f.write("hazard,real_reports_predicted\n"); [f.write(f"{c},{n}\n") for c, n in zip(CLASSES, mix)]
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        first = np.load(TASK / "first_model.npz"); P0 = (Xs @ first["W"].T + first["b"]).argmax(1)
        x = np.arange(9); fig, ax = plt.subplots(figsize=(9, 4))
        ax.bar(x - 0.2, np.bincount(P0, minlength=9), 0.4, label="first model", color="#9aa5b1")
        ax.bar(x + 0.2, mix, 0.4, label="final model", color="#2f6fb0")
        ax.set_xticks(x); ax.set_xticklabels(CLASSES, rotation=30); ax.set_ylabel("real reports")
        ax.set_title("Predicted hazards on the 1,369 sealed real reports"); ax.legend(); fig.tight_layout()
        fig.savefig(RES / "prediction_mix.png", dpi=120)
    except ImportError:
        pass
    words = try_vault(Xs @ W.T + b, TASK / "vault.npz", verbose=False)
    if words:
        log(f"The vault releases: SIGMOID_{words[0]}")
        (HERE / "FLAG.txt").write_text(f"SIGMOID_{words[0]}\n")
    else:
        log("The vault releases nothing for this model.")

if __name__ == "__main__":
    main()
