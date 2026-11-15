"""
LMML 2026 - task_moldova "Feather Stories"

Find the 6-letter ring of the one White Stork that is missing from ring_register.csv.

Key facts found in the data (all checked below):
  1. The 24 letters used (no U, V) form six look-alike groups of four:
     AKRX  BGSZ  CDOQ  EFLP  IJTY  HMNW
     A misread letter ALWAYS stays inside its group, so the group pattern of a
     reading identifies the bird. 24 readings match no registered ring: all of
     them are White Stork readings at the Lower Prut site (Cahul, L5819951).
  2. Each viewing SITE distorts letters in its own fixed way (a "template"):
     inside every group the site swaps letters pairwise (one of the 3 pairings,
     or none). Every observer at that site sees the same swaps, in every year,
     and a bird read away from home shows the swaps of the site where it is read.
     Only five templates exist (per group: 0 = no swap, 1/2/3 = the three pairings).
  3. So the majority reading at the Lower Prut (RHGLWQ) is the distorted word.
     The Lower Prut has no registered bird, so its template has to be inferred:
       - "opposite" letters among the minority readings show that the site
         distorts (an undistorted site almost never produces them);
       - they appear in groups AKRX and HMNW, which only the two "full"
         templates 112133 and 231212 distort;
       - between those two, a letter-level check decides: with 231212 every odd
         reading is an ordinary misread of the true letter (the true letter or
         one of its two neighbours); with 112133, 11 odd readings would have to be
         'opposite' misreads, which undistorted sites produce for only 0.2% of
         letters. (Site frequency alone would have favoured 112133.)
     Undoing 231212 on RHGLWQ gives ANSEMO.

Run from inside this folder:   python solve.py
Output: results/report.txt, results/templates.csv, results/candidates.csv,
        results/lower_prut_votes.png (if matplotlib), FLAG.txt
"""
import collections
import math
import os
import re

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
os.makedirs(RES, exist_ok=True)
lines = []


def out(*a):
    s = " ".join(str(v) for v in a)
    print(s)
    lines.append(s)


GROUPS = ["AKRX", "BGSZ", "CDOQ", "EFLP", "IJTY", "HMNW"]
GRP = {c: i for i, g in enumerate(GROUPS) for c in g}
# within each group, the element that maps a letter to its "opposite" (the pairing
# that observers' ordinary noise almost never produces; measured at undistorted sites)
OPP = [2, 3, 1, 2, 1, 2]
TEMPLATES = ["000000", "323000", "000321", "112133", "231212"]


def pos_in_group(c):
    return GROUPS[GRP[c]].index(c)


def swap(e, c):
    """Apply group element e (0..3) to letter c: index XOR e inside its group."""
    g = GROUPS[GRP[c]]
    return g[g.index(c) ^ e]


def elem(true, written):
    return pos_in_group(true) ^ pos_in_group(written)


def pattern(word):
    return "".join(str(GRP[c]) if c != "?" else "." for c in word)


obs = pd.read_csv(os.path.join(HERE, "observations.csv"))
reg = pd.read_csv(os.path.join(HERE, "ring_register.csv"))
reads = obs[obs.ring_read.notna()].copy()

# ------------------------------------------------------------------ step 1
letters_reg = set("".join(reg.ring))
letters_read = set("".join(reads.ring_read)) - {"?"}
out("[1] alphabet")
out(f"    letters in register: {len(letters_reg)}, in readings: {len(letters_read)}; "
    f"missing: {sorted(set('ABCDEFGHIJKLMNOPQRSTUVWXYZ') - letters_reg)}")
out(f"    look-alike groups: {' '.join(GROUPS)}")

reg_pat = collections.defaultdict(list)
for ring, sp in zip(reg.ring, reg.species):
    reg_pat[sp].append((pattern(ring), ring))
truth = []
for s, sp in zip(reads.ring_read, reads.species):
    rx = re.compile(pattern(s))
    m = [ring for p, ring in reg_pat[sp] if rx.fullmatch(p)]
    truth.append(m[0] if len(m) == 1 else ("NONE" if not m else None))
reads["true"] = truth
known = reads[reads.true.notna() & (reads.true != "NONE")]
unknown = reads[reads.true == "NONE"]
out(f"    readings: {len(reads)}; matched to exactly one registered ring by group pattern: "
    f"{len(known)}; ambiguous: {reads.true.isna().sum()}; matching NO registered ring: {len(unknown)}")
out("    unmatched readings by site/species: "
    + str(unknown.groupby(['locality_id', 'district', 'species']).size().to_dict()))
in_grp = sum(GRP[p] == GRP[q] for s, g in zip(known.ring_read, known.true)
             for p, q in zip(s, g) if p != "?")
tot = sum(p != "?" for s in known.ring_read for p in s)
out(f"    letters staying in their true letter's group: {in_grp}/{tot}")

# ------------------------------------------------------------------ step 2
out("\n[2] site templates (which swap each site applies in each group)")
M = collections.defaultdict(lambda: np.zeros((6, 4)))
info = {}
for _, row in known.iterrows():
    for p, q in zip(row.ring_read, row.true):
        if p != "?":
            M[row.locality_id][GRP[q], elem(q, p)] += 1
    info[row.locality_id] = (row.latitude, row.longitude, row.district)
rows = []
for loc, m in M.items():
    if m.sum() < 30:
        continue
    sc = {tp: sum(m[g, int(tp[g])] * np.log(0.7) + (m[g].sum() - m[g, int(tp[g])]) * np.log(0.1)
                  for g in range(6)) for tp in TEMPLATES}
    best = max(sc, key=sc.get)
    margin = sc[best] - sorted(sc.values())[-2]
    free = "".join(str(int(m[g].argmax())) for g in range(6))
    rows.append((loc, info[loc][2], info[loc][0], info[loc][1], int(m.sum()), free, best, round(margin, 1)))
tmpl = pd.DataFrame(rows, columns=["locality", "district", "lat", "lon", "letters", "free_fit", "template", "margin"])
tmpl.sort_values(["template", "lat"]).to_csv(os.path.join(RES, "templates.csv"), index=False)
conf = tmpl[tmpl.margin >= 15]
counts = conf.template.value_counts().reindex(TEMPLATES).fillna(0).astype(int)
out(f"    sites with >=30 read letters: {len(tmpl)}; confidently assigned: {len(conf)}")
out("    sites per template: " + ", ".join(f"{t}:{n}" for t, n in counts.items()))
out("    examples (site: template, letters): " + "; ".join(
    f"{r.locality} {r.district}: {r.template} ({r.letters})"
    for r in conf.sort_values("letters", ascending=False).head(6).itertuples()))

# consistency across observers at one site
tmap = dict(zip(conf.locality, conf.template))
home = {g: (round(a, 4), round(b, 4)) for g, a, b in zip(reg.ring, reg.ringed_latitude, reg.ringed_longitude)}
k2 = known.copy()
k2["away"] = [(round(a, 4), round(b, 4)) != home[g] for a, b, g in zip(k2.latitude, k2.longitude, k2.true)]
fol = collections.Counter()
for _, row in k2[k2.away & k2.locality_id.isin(tmap)].iterrows():
    tp = tmap[row.locality_id]
    for p, q in zip(row.ring_read, row.true):
        if p == "?" or tp[GRP[q]] == "0":
            continue
        e = elem(q, p)
        fol["follows visited site" if e == int(tp[GRP[q]]) else "true letter" if e == 0 else "other"] += 1
out(f"    birds read AWAY from home, distorted groups: {dict(fol)}  -> the template belongs to the site")
for loc in ["L8150848", "L5562838", "L10196035"]:
    sub = known[(known.locality_id == loc) & ~known.observers.str.contains(r"\|")]
    per = []
    for ob, d in sub.groupby("observers"):
        m = np.zeros((6, 4))
        for s, g in zip(d.ring_read, d.true):
            for p, q in zip(s, g):
                if p != "?":
                    m[GRP[q], elem(q, p)] += 1
        if m.sum() >= 30:
            per.append(f"{ob}:{''.join(str(int(m[g].argmax())) for g in range(6))}")
    out(f"    {loc} ({tmap.get(loc)}), per observer: {', '.join(per)}")

# ------------------------------------------------------------------ step 3
out("\n[3] Lower Prut stork")
cl = unknown.sort_values("date")
for _, row in cl.iterrows():
    out(f"      {row.date}  {row.ring_read}  {row.observers}")
R = list(cl.ring_read)
A = ""
for i in range(6):
    c = collections.Counter(r[i] for r in R if r[i] != "?")
    A += c.most_common(1)[0][0]
out(f"    majority (distorted) word: {A}   group pattern {pattern(A)}")

# noise rates measured at clean calibrated cells (site, group with one dominant swap)
clean = {}
for loc, m in M.items():
    for g in range(6):
        n = m[g].sum()
        if n >= 20 and m[g].max() / n >= 0.75:
            clean[(loc, g)] = int(m[g].argmax())
cat = {"identity": collections.Counter(), "distorted": collections.Counter()}
for _, row in known.iterrows():
    for p, q in zip(row.ring_read, row.true):
        if p == "?" or (row.locality_id, GRP[q]) not in clean:
            continue
        e = clean[(row.locality_id, GRP[q])]
        a = swap(e, q)
        d = pos_in_group(a) ^ pos_in_group(p)
        k = "apparent" if d == 0 else "opp_of_apparent" if d == OPP[GRP[q]] else "nbr_of_apparent"
        cat["identity" if e == 0 else "distorted"][k] += 1
rate = {}
for cfg, c in cat.items():
    n = sum(c.values())
    rate[cfg] = {k: c[k] / n for k in ["apparent", "opp_of_apparent", "nbr_of_apparent"]}
    out(f"    calibration, {cfg} groups: " + ", ".join(f"{k} {v:.3f}" for k, v in rate[cfg].items()))

obs_cnt = {}
for g in sorted(set(GRP[c] for c in A)):
    c = collections.Counter()
    for i, ch in enumerate(A):
        if GRP[ch] != g:
            continue
        for r in R:
            p = r[i]
            if p == "?":
                continue
            d = pos_in_group(ch) ^ pos_in_group(p)
            c["apparent" if d == 0 else "opp_of_apparent" if d == OPP[g] else "nbr_of_apparent"] += 1
    obs_cnt[g] = c
    out(f"    group {GROUPS[g]}: {dict(c)}")
n_opp = sum(c["opp_of_apparent"] for c in obs_cnt.values())
n_let = sum(sum(c.values()) for c in obs_cnt.values())
exp_id = rate["identity"]["opp_of_apparent"] * n_let
out(f"    opposite-of-majority letters: {n_opp} of {n_let}; expected if the site did not distort: {exp_id:.1f}")

# ------------------------------------------------------------------ step 4
out("\n[4] candidates: undo each template on the majority word")
prior = (counts + 0.5) / (counts + 0.5).sum()
cand = []
for tp in TEMPLATES:
    word = "".join(swap(int(tp[GRP[ch]]), ch) for ch in A)
    ll = 0.0
    for g, c in obs_cnt.items():
        r = rate["identity" if tp[g] == "0" else "distorted"]
        ll += sum(n * math.log(r[k]) for k, n in c.items())
    cand.append([tp, word, int(counts[tp]), ll])
cdf = pd.DataFrame(cand, columns=["template", "ring", "sites_with_template", "loglik"])
cdf["post_all"] = np.exp(cdf.loglik - cdf.loglik.max()) * prior.values
cdf["post_all"] /= cdf.post_all.sum()
mask = cdf.ring != A   # the undistorted reading RHGLWQ was submitted and rejected
cdf["post_given_not_majority"] = np.where(mask, cdf.post_all, 0)
cdf["post_given_not_majority"] /= cdf.post_given_not_majority.sum()
cdf = cdf.sort_values("post_given_not_majority", ascending=False)
cdf.round(4).to_csv(os.path.join(RES, "candidates.csv"), index=False)
for r in cdf.itertuples():
    out(f"    {r.template} -> {r.ring}   sites {r.sites_with_template:2d}   loglik {r.loglik:7.2f}   "
        f"posterior {r.post_all:.3f}   (excluding {A}: {r.post_given_not_majority:.3f})")
out("    -> two finalists, the 'full' templates 112133 and 231212 (distort every group).")
out("       By site frequency alone 112133 would be preferred (18 vs 8 sites).")

# ------------------------------------------------------------------ step 5
out("\n[5] letter-level check of the two finalists")
# Rule measured at undistorted sites: an ordinary misread moves the TRUE letter to one of its
# two 'neighbours' in the group, almost never to its 'opposite'.
id_opp_true = cat["identity"]["opp_of_apparent"] / sum(cat["identity"].values())
out(f"    at undistorted sites, a letter written as the OPPOSITE of the true letter: {id_opp_true:.2%} of letters")
# non-parametric P(written | true, apparent) from calibrated cells
pairW = collections.defaultdict(collections.Counter)
for _, row in known.iterrows():
    for p, q in zip(row.ring_read, row.true):
        if p == "?" or (row.locality_id, GRP[q]) not in clean:
            continue
        pairW[(q, swap(clean[(row.locality_id, GRP[q])], q))][p] += 1
final = []
for r in cdf.head(2).itertuples():
    viol, lp, expl = 0, 0.0, []
    for i, a in enumerate(A):
        T = r.ring[i]
        g = GRP[a]
        c = pairW[(T, a)]
        n = sum(c.values())
        pr = {l: (c[l] + 1) / (n + 4) for l in GROUPS[g]}
        for rd in R:
            p = rd[i]
            if p == "?" or p == a:
                continue
            d = pos_in_group(T) ^ pos_in_group(p)
            if d == OPP[g]:
                viol += 1
                expl.append(f"pos{i + 1}:{p}")
            lp += math.log(pr[p] / (1 - pr[a]))
    final.append((r.ring, r.template, viol, lp, expl))
    out(f"    {r.ring} ({r.template}): minority letters that would be OPPOSITE-of-true misreads: {viol} "
        f"{expl}; letter-pair log-lik {lp:.2f}")
final.sort(key=lambda t: (t[2], -t[3]))
winner = final[0]
out(f"    -> {winner[0]} explains every odd reading as an ordinary misread of the true letter "
    f"(true letter or its neighbour); the other finalist needs {final[1][2]} 'opposite' misreads.")

best_ring = winner[0]
runner = final[1][0]
flag = f"SIGMOID_{best_ring}"
out(f"\nFLAG: {flag}   (runner-up: SIGMOID_{runner})")
with open(os.path.join(HERE, "FLAG.txt"), "w") as f:
    f.write(flag + "\n")
with open(os.path.join(RES, "report.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 6, figsize=(14, 3.2), sharey=True)
    for i, ax in enumerate(axes):
        g = GROUPS[GRP[A[i]]]
        c = collections.Counter(r[i] for r in R)
        ax.bar(list(g) + ["?"], [c[l] for l in g] + [c["?"]],
               color=["#2a6f97" if l == A[i] else "#c0504d" if l == best_ring[i] else "#bbbbbb" for l in g] + ["#dddddd"])
        ax.set_title(f"pos {i + 1}: read {A[i]} -> {best_ring[i]}")
    axes[0].set_ylabel("readings")
    fig.suptitle(f"Lower Prut stork: majority {A} (blue) vs decoded {best_ring} (red)")
    fig.tight_layout()
    fig.savefig(os.path.join(RES, "lower_prut_votes.png"), dpi=120)
except ImportError:
    pass
