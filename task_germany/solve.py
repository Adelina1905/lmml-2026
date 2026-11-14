"""
task_germany ("Black Box"): reproduce every result in results/.

    pip install -r requirements.txt
    python solve.py            # practice score + baseline + stress tests + plot
    python solve.py --quick    # practice score + baseline only

The deliverable is main.py (solve(oracle, points)). The organisers run it with
    python3 evaluation_script.py main.py --eval-type test
and a score >= 70 prints the flag (see FLAG.txt).
"""
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
os.makedirs(RES, exist_ok=True)


def run(cmd, out_file=None):
    print("$", " ".join(cmd))
    p = subprocess.run(cmd, cwd=HERE, capture_output=True, text=True)
    text = p.stdout + ("\n--- stderr (main.py log) ---\n" + p.stderr if p.stderr else "")
    if out_file:
        with open(os.path.join(RES, out_file), "w", encoding="utf-8") as fh:
            fh.write(text)
    print(p.stdout.splitlines()[2] if p.stdout.count("\n") > 2 else p.stdout)
    return p


def plot():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed, skipping plot")
        return
    sub = json.load(open(os.path.join(RES, "practice_run", "task_germany",
                                      "submission.json")))
    log = [json.loads(l) for l in open(os.path.join(
        RES, "practice_run", "task_germany", "query_log.jsonl"))]
    colors = {"design": "#888888", "bisect": "#d62728", "challenge": "#2ca02c"}
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.8))
    for q in log:
        ph = q["phase"].split(":")[0]
        c = colors.get(ph, "#1f77b4")
        ax[0].scatter(q["x"][0], q["x"][3], s=14, c=c)
    for ph, c in list(colors.items()) + [("gate test", "#1f77b4")]:
        ax[0].scatter([], [], c=c, label=ph)
    xs = [0.6, 1.0]
    ax[0].plot(xs, [1.7 - x for x in xs], "k--", lw=1, label="true x1+x4=1.7")
    b = sub["discontinuities"][0]["boundary"]
    a1, a4, t = b["coefficients"]["x1"], b["coefficients"]["x4"], b["threshold"]
    ax[0].plot(xs, [(t - a1 * x) / a4 for x in xs], "m:", lw=2,
               label="recovered boundary")
    ax[0].set_xlabel("x1"); ax[0].set_ylabel("x4")
    ax[0].set_title("Practice run: 149 queries projected on (x1, x4)")
    ax[0].legend(fontsize=7, loc="lower left")
    ax[0].set_xlim(-0.03, 1.03); ax[0].set_ylim(-0.03, 1.03)

    def parts(fname):
        txt = open(os.path.join(RES, fname), encoding="utf-8").read()
        out = {}
        for k in ["variables", "interaction", "discontinuity", "prediction"]:
            line = [l for l in txt.splitlines() if l.startswith(f"## {k}")][0]
            out[k] = float(line.split("—")[1].split("/")[0])
        return out
    m, bl = parts("practice_report_main.md"), parts("practice_report_baseline.md")
    keys = list(m)
    xi = range(len(keys))
    ax[1].bar([i - 0.2 for i in xi], [bl[k] for k in keys], 0.4,
              label="naive one-at-a-time baseline", color="#bbbbbb")
    ax[1].bar([i + 0.2 for i in xi], [m[k] for k in keys], 0.4,
              label="main.py", color="#1f77b4")
    ax[1].set_xticks(list(xi)); ax[1].set_xticklabels(keys)
    ax[1].set_ylabel("points"); ax[1].set_title("Practice score breakdown")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(RES, "practice_overview.png"), dpi=130)
    print("wrote results/practice_overview.png")


def main():
    py = sys.executable
    run([py, "evaluation_script.py", "main.py", "--eval-type", "train",
         "--save", os.path.join("results", "practice_run")],
        "practice_report_main.md")
    run([py, "evaluation_script.py", os.path.join("baseline", "naive_owat.py"),
         "--eval-type", "train"], "practice_report_baseline.md")
    if "--quick" not in sys.argv:
        lines = []
        for mode, n in [("standard", 150), ("hard", 200), ("gates", 100)]:
            p = subprocess.run([py, os.path.join("tests", "stress_test.py"),
                                "main.py", "0", str(n), mode],
                               cwd=HERE, capture_output=True, text=True)
            lines.append(p.stdout.strip())
            print(p.stdout.strip().splitlines()[-1])
        with open(os.path.join(RES, "stress_summary.txt"), "w") as fh:
            fh.write("Random black boxes of the same family, scored with the "
                     "official evaluation logic.\n"
                     "Columns: mode, number of functions, min score, mean "
                     "score, runs below 70.\n\n" + "\n".join(lines) + "\n")
    plot()


if __name__ == "__main__":
    main()
