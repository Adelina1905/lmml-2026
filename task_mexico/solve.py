"""Reproduce and verify the task_mexico solution on the provided training data.

Run from inside the task folder:
    python solve.py

It runs the same linking code as main.py on data/train-input.csv, compares the result with
data/train-output.json, runs robustness tests, and saves everything to results/.
"""
import copy
import json
import os
import random
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import main as M  # noqa: E402

DATA_IN = os.path.join(HERE, "data", "train-input.csv")
DATA_OUT = os.path.join(HERE, "data", "train-output.json")
RES = os.path.join(HERE, "results")
os.makedirs(RES, exist_ok=True)

lines = []


def say(s=""):
    print(s)
    lines.append(s)


recs = M.load(DATA_IN)
truth = set(tuple(sorted(p)) for p in json.load(open(DATA_OUT, encoding="utf-8"))["matches"])
by = {r["id"]: r for r in recs}

say("=== task_mexico: Ruta de las Identidades ===")
say(f"records: {len(recs)}  per source table: {dict(sorted(Counter(r['table'] for r in recs).items()))}")
say(f"true matching pairs in train-output.json: {len(truth)}")

# 1. Main result
stats = {}
pred = M.link(recs, stats)
P = set(tuple(p) for p in pred)
tp, fp, fn = len(P & truth), len(P - truth), len(truth - P)
prec = tp / len(P) if P else 0.0
rec = tp / len(truth) if truth else 0.0
f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
say("")
say("--- 1. Result on training data ---")
say(f"predicted pairs: {len(P)}   TP: {tp}   FP (extra): {fp}   FN (missing): {fn}")
say(f"precision: {prec:.4f}   recall: {rec:.4f}   F1: {f1:.4f}")
say(f"exact match with train-output.json (PASS condition): {P == truth}")
say(f"linker stats: {stats}")
with open(os.path.join(RES, "train-output-predicted.json"), "w", encoding="utf-8") as fh:
    json.dump({"matches": pred}, fh)

# 2. Data facts that justify the method
say("")
say("--- 2. Facts about the duplicates (from the training labels) ---")
num_equal = sum(by[a]["nums"] == by[b]["nums"] for a, b in truth)
say(f"true pairs whose address digit groups are identical: {num_equal}/{len(truth)}")
blocks = Counter(r["nums"] for r in recs)
same_key_pairs = sum(n * (n - 1) // 2 for n in blocks.values())
say(f"all record pairs sharing the same digit key: {same_key_pairs} (= true pairs -> key is never shared by different people)")
pair_tables = Counter("".join(sorted((a[0], b[0]))) for a, b in truth)
say(f"true pairs by table combination: {dict(sorted(pair_tables.items()))} (never two records of the same table)")
say(f"group sizes (1 = reader appears once): {stats['group_sizes']}")
ns = Counter(M.name_score(by[a], by[b]) for a, b in truth)
say(f"name parts that agree on true pairs (0/1/2): {dict(sorted(ns.items()))}")

# 3. Robustness tests
say("")
say("--- 3. Robustness tests (do not affect the submission) ---")
rnd = random.Random(42)
victims_pool = sorted({b for a, b in truth})
rows = []
for trial in range(5):
    R = copy.deepcopy(recs)
    B = {r["id"]: r for r in R}
    for v in rnd.sample(victims_pool, 300):
        nums = list(B[v]["nums"])
        i = rnd.randrange(len(nums))
        if nums[i] == B[v]["zip"]:
            i = (i + 1) % len(nums)
        s = list(nums[i])
        j = rnd.randrange(len(s))
        s[j] = str((int(s[j]) + rnd.randint(1, 9)) % 10)
        nums[i] = "".join(s)
        B[v]["nums"] = tuple(sorted(nums))
    st = {}
    p2 = set(tuple(x) for x in M.link(R, st))
    rows.append((trial, len(truth - p2), len(p2 - truth), st["fuzzy_links"]))
    say(f"typo test {trial}: 300 records get a one-digit address typo -> missing {len(truth - p2)}, extra {len(p2 - truth)}, fallback links {st['fuzzy_links']}")
say("  (without the fallback all ~300-600 affected pairs would be missing)")
ids = list(by)
N = 20000
rej = sum(M.name_score(by[rnd.choice(ids)], by[rnd.choice(ids)]) == 0 for _ in range(N))
say(f"name guard: random different people rejected {rej}/{N} = {rej / N:.1%}; true pairs rejected {ns.get(0, 0)}/{len(truth)}")

metrics = {
    "records": len(recs), "true_pairs": len(truth), "predicted_pairs": len(P),
    "tp": tp, "fp": fp, "fn": fn, "precision": prec, "recall": rec, "f1": f1,
    "exact_match": P == truth, "linker_stats": stats,
    "typo_tests": [{"trial": t, "missing": m, "extra": e, "fallback_links": f} for t, m, e, f in rows],
    "name_guard_random_reject_rate": rej / N,
}
with open(os.path.join(RES, "metrics.json"), "w", encoding="utf-8") as fh:
    json.dump(metrics, fh, indent=2)
with open(os.path.join(RES, "report.txt"), "w", encoding="utf-8") as fh:
    fh.write("\n".join(lines) + "\n")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    gs = stats["group_sizes"]
    ax[0].bar([str(k) for k in gs], list(gs.values()), color="#4C78A8")
    ax[0].set_title("Readers by number of records")
    ax[0].set_xlabel("records per reader")
    ax[0].set_ylabel("readers")
    for k, v in gs.items():
        ax[0].text(str(k), v, str(v), ha="center", va="bottom")
    ax[1].bar(["true pairs", "predicted", "correct", "extra", "missing"], [len(truth), len(P), tp, fp, fn],
              color=["#9D9D9D", "#4C78A8", "#54A24B", "#E45756", "#F58518"])
    ax[1].set_title("Training data: prediction vs truth")
    for i, v in enumerate([len(truth), len(P), tp, fp, fn]):
        ax[1].text(i, v, str(v), ha="center", va="bottom")
    for a in ax:
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(RES, "summary.png"), dpi=130)
    print("saved results/summary.png")
except ImportError:
    print("matplotlib not installed - skipped the chart")

print("saved results/report.txt, results/metrics.json, results/train-output-predicted.json")
