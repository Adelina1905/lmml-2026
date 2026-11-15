#!/usr/bin/env python3
"""Training-side analysis for the Fez dispatch desk router (main.py).

Run from inside this folder:  python solve.py
Standard library only. Writes the report to results/.

1. checks main.py's groups and its own solver against the training answers
2. rebuilds the STATS table that main.py embeds and checks they are identical
3. scores main.py on the 400 training requests (quality, cost)
4. honest validation: repeated 5-fold cross-validation, stats re-fitted on the
   training folds only, routing the held-out fold with its own 20/request budget
5. stress test: re-weights the template mix at random to see how much the
   quality can move when the hidden test mix differs
6. runs main.py end to end on train-input.csv copied as test-input.csv
"""

import collections
import csv
import os
import random
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import main  # noqa: E402

OUT = os.path.join(HERE, "results")
os.makedirs(OUT, exist_ok=True)
LOG = []


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    LOG.append(s)


def read(name):
    with open(os.path.join(HERE, name), encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


inp = read("train-input.csv")
ans = {r["request_id"]: r for r in read("train-answers.csv")}
oracle = {r["request_id"]: r["model"] for r in read("train-output.csv")}

data = []
for r in inp:
    a = ans[r["request_id"]]
    ref = a["reference_answer"].strip().lower()
    data.append({
        "id": r["request_id"],
        "prompt": r["prompt"],
        "group": main.group(r["prompt"]),
        "c": a["cheap_answer"].strip().lower() == ref,
        "e": a["expensive_answer"].strip().lower() == ref,
    })


def fit(rows):
    st = collections.defaultdict(lambda: [0, 0, 0])
    for d in rows:
        s = st[d["group"]]
        s[0] += 1
        s[1] += d["c"]
        s[2] += d["e"]
    return {k: tuple(v) for k, v in st.items()}


def score(rows, models):
    q = sum((d["e"] if m == "expensive" else d["c"]) for d, m in zip(rows, models))
    cost = sum(main.COST_EXP if m == "expensive" else main.COST_CHEAP for m in models)
    return q / len(rows), cost


# 1-2. groups and stats table ------------------------------------------------
say("=== 1. per-group accuracy of each guide on the training set")
stats = fit(data)
say("{:<22} {:>4} {:>7} {:>7} {:>7}".format("group", "n", "cheap", "exp", "gain"))
for g in sorted(stats):
    n, c, e = stats[g]
    say("{:<22} {:>4} {:>7.2f} {:>7.2f} {:>+7.2f}".format(g, n, c / n, e / n, (e - c) / n))
same = stats == main.STATS
say("STATS embedded in main.py identical to the re-fitted table:", same)
assert same, "main.STATS is out of date"

# 3. in-sample score ----------------------------------------------------------
say("\n=== 2. main.py on the 400 training requests (budget 8000)")
models = main.route([d["prompt"] for d in data])
q, cost = score(data, models)
n_exp = models.count("expensive")
say("quality = {:.4f}  cost = {}  expensive = {}  cheap = {}".format(q, cost, n_exp, len(models) - n_exp))
say("gates (>= 0.80, <= 8000):", "PASS" if q >= 0.80 and cost <= 8000 else "FAIL")
for name, ms in [("all cheap", ["cheap"] * 400), ("all expensive", ["expensive"] * 400),
                 ("oracle train-output.csv", [oracle[d["id"]] for d in data])]:
    qq, cc = score(data, ms)
    say("  reference: {:<24} quality = {:.4f}  cost = {}".format(name, qq, cc))

# 4. cross-validation -------------------------------------------------------
say("\n=== 3. repeated 5-fold cross-validation (20 x 5 folds, stats fitted on 4/5)")
qs, costs = [], []
fails = 0
saved = main.STATS
for rep in range(20):
    rnd = random.Random(rep)
    idx = list(range(len(data)))
    rnd.shuffle(idx)
    for f in range(5):
        test_idx = set(idx[f::5])
        tr = [data[i] for i in idx if i not in test_idx]
        te = [data[i] for i in sorted(test_idx)]
        main.STATS = fit(tr)
        ms = main.route([d["prompt"] for d in te])
        qq, cc = score(te, ms)
        qs.append(qq)
        costs.append(cc / len(te))
        fails += not (qq >= 0.80 and cc <= 20 * len(te))
main.STATS = saved
qs.sort()
say("held-out quality: mean {:.4f}  min {:.4f}  5th pct {:.4f}  max {:.4f}".format(
    sum(qs) / len(qs), qs[0], qs[len(qs) // 20], qs[-1]))
say("held-out cost per request: max {:.2f} (budget 20)".format(max(costs)))
say("folds failing a gate: {} / {}".format(fails, len(qs)))

# 5. mix-shift stress test --------------------------------------------------
say("\n=== 4. mix-shift stress test (500 requests drawn with random template weights)")
by_kind = collections.defaultdict(list)
for d in data:
    by_kind[main.kind(d["prompt"])].append(d)
kinds = sorted(by_kind)
rnd = random.Random(7)
res = []
for t in range(2000):
    w = {k: len(by_kind[k]) * rnd.uniform(0.6, 1.4) for k in kinds}  # +-40 % per template
    tot = sum(w.values())
    sample = []
    for k in kinds:
        sample += rnd.choices(by_kind[k], k=round(500 * w[k] / tot))
    ms = main.route([d["prompt"] for d in sample])
    qq, cc = score(sample, ms)
    res.append((qq, cc, len(sample)))
qq = sorted(r[0] for r in res)
say("quality: mean {:.4f}  1st pct {:.4f}  min {:.4f}".format(sum(qq) / len(qq), qq[len(qq) // 100], qq[0]))
say("cost always within min(10000, 20*N):", all(c <= min(10000, 20 * n) for _, c, n in res))
say("runs below 0.80 quality: {} / {}".format(sum(q < 0.80 for q in qq), len(qq)))

# 6. end-to-end run of main.py -----------------------------------------------
say("\n=== 5. end-to-end: python main.py in a scratch folder (train copied as test-input.csv)")
tmp = tempfile.mkdtemp()
shutil.copy(os.path.join(HERE, "main.py"), tmp)
shutil.copy(os.path.join(HERE, "train-input.csv"), os.path.join(tmp, "test-input.csv"))
subprocess.run([sys.executable, "main.py"], cwd=tmp, check=True)
with open(os.path.join(tmp, "routing_log.csv"), encoding="utf-8", newline="") as fh:
    log = list(csv.DictReader(fh))
ok = ([r["request_id"] for r in log] == [d["id"] for d in data]
      and all(r["model"] in ("cheap", "expensive") for r in log))
q, cost = score(data, [r["model"] for r in log])
say("routing_log.csv rows = {}, ids in order and values valid: {}".format(len(log), ok))
say("quality = {:.4f}  cost = {}".format(q, cost))
shutil.copy(os.path.join(tmp, "routing_log.csv"), os.path.join(OUT, "routing_log_train.csv"))
shutil.rmtree(tmp)

with open(os.path.join(OUT, "report.txt"), "w", encoding="utf-8") as fh:
    fh.write("\n".join(LOG) + "\n")
print("\nreport written to results/report.txt")
