"""
Stress test: run main.py against random black boxes of the same family
(main effects, one pairwise interaction, one hard step) and score each with
the official evaluation_script logic.

    python tests/stress_test.py main.py 0 100 standard
    python tests/stress_test.py main.py 0 100 hard     # tiny/huge regions, small jumps, curved main effects
    python tests/stress_test.py main.py 0 100 gates    # 1- and 3-variable gates (beyond the stated family)
"""
import sys, json, math, random, types, os, statistics
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
import harness, evaluation_script as ev

V = ["x1","x2","x3","x4","x5"]
pts = json.load(open(os.path.join(HERE, "challenge", "challenge_set.json")))["points"]

MODE = "standard"

def make(seed):
    R = random.Random(seed)
    hard = MODE in ("hard", "gates")
    lin = [R.choice([-1,1])*R.uniform(0.2,2.0) for _ in range(5)]
    gate = R.sample(range(5), R.choice([1, 3]) if MODE == "gates" else 2)
    inter = R.sample(range(5), 2)
    zero_cands = [k for k in range(5)]
    nz = R.choice([1,1,2])
    zeros = R.sample(range(5), nz)
    for k in zeros: lin[k] = 0.0
    a = [0.0]*5
    for k in gate:
        a[k] = (R.choice([R.uniform(0.1,0.3), R.uniform(0.4,3.0)]) * R.choice([1,1,-1])) if hard else R.uniform(0.4, 2.0) * R.choice([1,1,1,-1])
    # threshold: choose so active fraction in [3%, 50%]
    import numpy as np
    U = np.random.default_rng(seed).random((20000,5))
    s = U @ np.array(a)
    frac = R.choice([R.uniform(0.005,0.03), R.uniform(0.5,0.95), R.uniform(0.03,0.5)]) if hard else R.uniform(0.03, 0.5)
    thr = float(np.quantile(s, 1-frac))
    J = R.choice([R.uniform(0.1,0.3), R.uniform(0.3,3.0)]) if hard else R.uniform(0.3, 3.0)
    c = R.choice([-1,1])*R.uniform(0.5, 5.0)
    form = R.choice(["product","centered_product"])
    b0 = R.uniform(-2,2); sqk = R.randrange(5); sq = R.choice([0,0,R.uniform(-1,1)]) if hard else 0
    if sq: lin[sqk] = lin[sqk] or 0.5
    i, j = inter
    def f(x):
        y = b0 + sum(l*v for l,v in zip(lin,x)) + sq*(x[sqk]-0.3)**2
        y += c*((x[i]-0.5)*(x[j]-0.5) if form=="centered_product" else x[i]*x[j])
        if sum(ak*v for ak,v in zip(a,x)) >= thr: y += J
        return y
    relevant = sorted({k for k in range(5) if lin[k]!=0} | set(gate) | set(inter))
    HF = types.SimpleNamespace()
    HF.VAR_NAMES = V
    HF.TRUTH = {"interaction":{"variables":(V[i],V[j]),"coefficient":c},
                "discontinuity":{"gate_variables":tuple(V[k] for k in gate),
                    "boundary":{V[k]:a[k] for k in range(5)},"threshold":thr,"jump":J},
                "relevant":[V[k] for k in relevant],
                "irrelevant":[V[k] for k in range(5) if k not in relevant]}
    HF.gate_active = lambda x: sum(ak*v for ak,v in zip(a,x)) >= thr
    HF.evaluate = lambda x: round(f(x),6)
    HF.expression = lambda: "synthetic"
    return HF, dict(lin=lin,gate=gate,a=a,thr=thr,J=J,inter=inter,c=c,form=form,frac=frac)

if __name__ != "__main__":
    raise SystemExit
script = sys.argv[1]
n0, n1 = int(sys.argv[2]), int(sys.argv[3])
MODE = sys.argv[4] if len(sys.argv) > 4 else "standard"
scores=[]
for seed in range(n0, n1):
    HF, info = make(seed)
    ev.HF = HF
    out = harness.run(script, HF.evaluate, pts, 150, 50)
    r = ev.evaluate(out, "t", pts)
    scores.append(r["score"])
    if r["score"] < 95:
        print(seed, r["score"], {k:v["points"] for k,v in r["breakdown"].items()}, info, r["flags"])
        for k,v in r["breakdown"].items():
            if v["points"] < v["max"]: print("   ", k, v.get("detail"))
print(MODE, "n", len(scores), "min", min(scores), "mean", statistics.mean(scores), "below70", sum(s<70 for s in scores))
