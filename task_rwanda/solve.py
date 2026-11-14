"""
Task "Rwanda" - The Drying Floor at Karongi
-------------------------------------------
The LangGraph state machine in graph.py never left the measure/dry loop because
inspect_batch computed `ready` but never wrote `approved` back into the state,
so route_after_inspection always sent the lot back to dry_batch.

The fix (the only change in graph.py, one line in inspect_batch):

    return {"inspections": state["inspections"] + 1, "approved": ready}

This script:
  1. runs the official evaluation_script.py with the current Python and saves
     its output to run_log.txt (the log that gets submitted) and FLAG.txt
  2. traces every batch through the broken graph (graph_reference.py, which has
     the same inspect_batch as the handed-out file) and the fixed graph.py,
     node by node, and saves the proof to results/

Usage (from inside this folder):
    pip install -r requirements.txt
    python solve.py
"""

import csv
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
STEP_BUDGET = 18
sys.path.insert(0, HERE)

from langgraph.errors import GraphRecursionError  # noqa: E402

import graph as fixed  # noqa: E402
import graph_reference as broken  # noqa: E402


def run_official_evaluator():
    proc = subprocess.run(
        [sys.executable, os.path.join(HERE, "evaluation_script.py")],
        cwd=HERE, capture_output=True, text=True,
    )
    log = proc.stdout + proc.stderr
    with open(os.path.join(HERE, "run_log.txt"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write(log)
    match = re.search(r"Flag: (SIGMOID_[A-Z_]+)", proc.stdout)
    if proc.returncode == 0 and match:
        with open(os.path.join(HERE, "FLAG.txt"), "w", encoding="utf-8", newline="\n") as fh:
            fh.write(match.group(1) + "\n")
    return log, proc.returncode, match.group(1) if match else None


def trace(module, batch):
    """Run one batch through a graph module, recording every node visit."""
    app = module.build_graph()
    state = module.initial_state(batch["batch_id"], batch["moisture"])
    moisture = [state["moisture"]]
    path, final = [], dict(state)
    try:
        for update in app.stream(state, config={"recursion_limit": STEP_BUDGET},
                                 stream_mode="updates"):
            for node, delta in update.items():
                path.append(node)
                final.update(delta or {})
                if node == "dry_batch":
                    moisture.append(final["moisture"])
        status = "END"
    except GraphRecursionError:
        status = "GraphRecursionError"
    return {
        "batch_id": batch["batch_id"],
        "start_moisture": batch["moisture"],
        "status": status,
        "node_steps": len(path),
        "passes": final["passes"],
        "inspections": final["inspections"],
        "final_moisture": final["moisture"],
        "approved": final["approved"],
        "grade": final["grade"],
        "path": " > ".join(path),
    }, moisture


def plot(trajectories):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return "matplotlib not installed - plot skipped"
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.axhspan(fixed.MOISTURE_MIN, fixed.MOISTURE_MAX, color="#cfe8cf", label="accepted range 10.5-12.5 %")
    for batch_id, values in trajectories.items():
        ax.plot(range(len(values)), values, marker="o", label=batch_id)
    ax.set_xlabel("drying passes")
    ax.set_ylabel("moisture (%)")
    ax.set_title("Fixed graph: each lot is dried until it first enters the accepted range")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "moisture_trajectories.png"), dpi=120)
    return "saved results/moisture_trajectories.png"


def main():
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(HERE, "dataset", "batches.json"), encoding="utf-8") as fh:
        batches = json.load(fh)

    log, code, flag = run_official_evaluator()

    rows, trajectories = [], {}
    for label, module in (("broken", broken), ("fixed", fixed)):
        for batch in batches:
            row, moisture = trace(module, batch)
            rows.append({"graph": label, **row})
            if label == "fixed":
                trajectories[batch["batch_id"]] = moisture

    with open(os.path.join(RESULTS, "batch_traces.csv"), "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    plot_msg = plot(trajectories)

    lines = ["Task rwanda - The Drying Floor at Karongi", "=" * 42, "",
             "Official evaluation_script.py output (also in run_log.txt):", "", log.rstrip(), "",
             f"evaluator exit code: {code}", f"flag: {flag}", "",
             f"Per-batch traces (step budget = {STEP_BUDGET}):", ""]
    header = f"{'graph':7} {'batch':11} {'start':>5} {'status':20} {'steps':>5} {'passes':>6} {'final':>6} {'grade':>5}"
    lines += [header, "-" * len(header)]
    for r in rows:
        lines.append(f"{r['graph']:7} {r['batch_id']:11} {r['start_moisture']:>5} {r['status']:20} "
                     f"{r['node_steps']:>5} {r['passes']:>6} {r['final_moisture']:>6} {r['grade'] or '-':>5}")
    lines += ["", plot_msg, ""]
    report = "\n".join(lines)
    with open(os.path.join(RESULTS, "report.txt"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write(report)
    print(report)
    return code


if __name__ == "__main__":
    sys.exit(main())
