"""
LMML 2026 - task_moldova "Feather Stories"

Goal: find the six letters on the ring of the one White Stork that is NOT in
ring_register.csv, using the noisy ring readings in observations.csv.

Method
  1. Learn the observers' error model (letter confusion matrix) from readings
     that can be confidently tied to a registered bird: same species, seen
     within 5 km of where that bird was ringed, best string match.
  2. Find stork readings that no registered stork ring explains. They all
     come from one place: the Lower Prut (Cahul district, locality L5819951).
  3. Decode that cluster of readings letter by letter with maximum likelihood
     under the confusion matrix ('?' = unreadable, ignored).
  4. Check robustness: bootstrap, comparison against every registered ring,
     and a location-free search for unexplained stork readings.

Run from inside this folder:   python solve.py
Output: results/report.txt, results/position_votes.png (if matplotlib), FLAG.txt
"""
import collections
import math
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
os.makedirs(RES, exist_ok=True)

L = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
STORK = "Ciconia ciconia"
lines = []


def out(*a):
    s = " ".join(str(v) for v in a)
    print(s)
    lines.append(s)


def mismatches(read, ring):
    """Number of letters that were read but differ from the ring ('?' ignored)."""
    return sum(1 for p, q in zip(read, ring) if p != "?" and p != q)


def km(lat1, lon1, lat2, lon2):
    """Great-circle (haversine) distance in kilometres."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    a = (math.sin((p2 - p1) / 2) ** 2
         + math.cos(p1) * math.cos(p2) * math.sin(math.radians(lon2 - lon1) / 2) ** 2)
    return 6371 * 2 * math.asin(math.sqrt(a))


obs = pd.read_csv(os.path.join(HERE, "observations.csv"))
reg = pd.read_csv(os.path.join(HERE, "ring_register.csv"))
reads = obs[obs.ring_read.notna()].copy()
out(f"observations: {len(obs)} rows, with a ring reading: {len(reads)}")
out(f"register: {len(reg)} rings, of which White Stork: {(reg.species == STORK).sum()}")
out(f"all readings have 6 characters: {set(reads.ring_read.str.len()) == {6}}")

# ---------------------------------------------------------------- step 1
# tie each reading to the registered bird of the same species that matches best
# (fewest mismatches, then closest ringing site)
reg_by_sp = {sp: g.to_dict("records") for sp, g in reg.groupby("species")}
best_ring, best_km = [], []
for _, row in reads.iterrows():
    cands = sorted(
        (mismatches(row.ring_read, g["ring"]),
         km(row.latitude, row.longitude, g["ringed_latitude"], g["ringed_longitude"]),
         g["ring"])
        for g in reg_by_sp[row.species])
    best_ring.append(cands[0][2])
    best_km.append(cands[0][1])
reads["match"] = best_ring
reads["km"] = best_km
train = reads[reads.km < 5]  # seen near that bird's own nest/ringing site

counts = np.full((26, 26), 0.5)  # add-0.5 smoothing; rows = true, cols = read
n_letters = n_unk = n_wrong = 0
for read, ring in zip(train.ring_read, train.match):
    for p, q in zip(read, ring):
        n_letters += 1
        if p == "?":
            n_unk += 1
            continue
        counts[L.index(q), L.index(p)] += 1
        n_wrong += p != q
P = counts / counts.sum(axis=1, keepdims=True)  # P(read letter | true letter)
out(f"\n[1] error model learned from {len(train)} readings near their bird's ringing site")
out(f"    unreadable '?': {n_unk / n_letters:.1%} of letters; "
    f"wrong among read letters: {n_wrong / (n_letters - n_unk):.1%}")
conf = collections.Counter()
for read, ring in zip(train.ring_read, train.match):
    for p, q in zip(read, ring):
        if p != "?" and p != q:
            conf[(q, p)] += 1
out("    most common confusions (true -> read): "
    + ", ".join(f"{a}->{b} {n}" for (a, b), n in conf.most_common(14)))


def loglik(word, rs):
    return sum(np.log(P[L.index(t), L.index(c)])
               for s in rs for t, c in zip(word, s) if c != "?")


# ---------------------------------------------------------------- step 2
stork_rings = reg[reg.species == STORK].ring.tolist()
storks = reads[reads.species == STORK]
thr = np.log(0.03)  # per-letter log-likelihood below this = "not explained"
unexpl = []
for _, row in storks.iterrows():
    k = sum(c != "?" for c in row.ring_read)
    if k < 4:
        continue  # too little information to judge
    best = max(loglik(g, [row.ring_read]) for g in stork_rings) / k
    if best < thr:
        unexpl.append(row)
unexpl = pd.DataFrame(unexpl)
out(f"\n[2] stork readings (>=4 letters read) that no registered stork explains: {len(unexpl)}")
for (loc, dist), n in unexpl.groupby(["locality_id", "district"]).size().items():
    sub = obs[obs.locality_id == loc]
    out(f"    {loc} ({dist}, {sub.latitude.mean():.3f}N {sub.longitude.mean():.3f}E): {n}")
site = unexpl.locality_id.mode()[0]
nearest = min(km(obs[obs.locality_id == site].latitude.mean(),
                 obs[obs.locality_id == site].longitude.mean(),
                 g.ringed_latitude, g.ringed_longitude)
              for g in reg[reg.species == STORK].itertuples())
out(f"    nearest registered stork ringing site is {nearest:.0f} km away -> "
    "this is the Lower Prut, home of the unregistered stork")

cluster = storks[storks.locality_id == site]
cl = cluster.ring_read.tolist()
out(f"    all stork readings at {site}: {len(cl)} ({cluster.date.min()} .. {cluster.date.max()})")
for _, row in cluster.iterrows():
    out(f"      {row.date}  {row.ring_read}  observers={row.observers}")

# ---------------------------------------------------------------- step 3
out("\n[3] maximum-likelihood decoding, position by position")
word = ""
votes = []
for i in range(6):
    ll = np.zeros(26)
    for s in cl:
        if s[i] != "?":
            ll += np.log(P[:, L.index(s[i])])
    post = np.exp(ll - ll.max())
    post /= post.sum()
    top = np.argsort(-post)[:3]
    v = collections.Counter(s[i] for s in cl)
    votes.append(v)
    out(f"    pos {i + 1}: votes {dict(v.most_common())}  posterior "
        + ", ".join(f"{L[t]}={post[t]:.6f}" for t in top))
    word += L[top[0]]
out(f"    decoded ring: {word}")
out(f"    exact '{word}' readings: {cl.count(word)} of {len(cl)}")

# ---------------------------------------------------------------- step 4
out("\n[4] checks")
out(f"    in register: {word in set(reg.ring)}; min Hamming distance to any registered ring: "
    f"{min(sum(a != b for a, b in zip(word, g)) for g in reg.ring)}")
rng = np.random.default_rng(0)
boot = collections.Counter()
for _ in range(2000):
    sm = [cl[k] for k in rng.integers(0, len(cl), len(cl))]
    w = ""
    for i in range(6):
        ll = np.zeros(26)
        for s in sm:
            if s[i] != "?":
                ll += np.log(P[:, L.index(s[i])])
        w += L[ll.argmax()]
    boot[w] += 1
out("    bootstrap (2000 resamples): " + ", ".join(f"{w} {n}" for w, n in boot.most_common(4)))
scores = sorted([(loglik(g, cl), g) for g in reg.ring], reverse=True)[:3]
out(f"    log-likelihood of the cluster: {word} = {loglik(word, cl):.1f}; best registered rings: "
    + ", ".join(f"{g} = {s:.1f}" for s, g in scores))
out(f"    unexplained stork readings that {word} explains: "
    f"{sum(loglik(word, [s]) / sum(c != '?' for c in s) > thr for s in unexpl.ring_read)} of {len(unexpl)}")

flag = f"SIGMOID_{word}"
out(f"\nFLAG: {flag}")
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
        items = sorted(votes[i].items(), key=lambda t: -t[1])
        labels = [k for k, _ in items]
        ax.bar(labels, [n for _, n in items],
               color=["#2a6f97" if k == word[i] else "#bbbbbb" for k in labels])
        ax.set_title(f"pos {i + 1}: {word[i]}")
    axes[0].set_ylabel("readings")
    fig.suptitle(f"Lower Prut stork, {len(cl)} readings -> {word}")
    fig.tight_layout()
    fig.savefig(os.path.join(RES, "position_votes.png"), dpi=120)
except ImportError:
    pass
