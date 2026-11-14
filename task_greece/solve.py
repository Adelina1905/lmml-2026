"""
Task "Greece" - Just a siblings' thing
--------------------------------------
Loads the three oracle models (Ares, Athena, Apollo), runs them on the test set
at a decision threshold of 0.50, computes the metrics that accuracy hides, and
picks the model that fits the business case:

    "missing a true catastrophic threat is far more expensive than a false alarm"
    -> minimise False Negatives -> maximise RECALL.

Usage (from inside this folder):
    pip install -r requirements.txt
    python solve.py

Outputs go to ./results/
"""

import io
import os
import pickle
import struct
import zipfile

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, fbeta_score,
    precision_score, recall_score, roc_auc_score,
)

HERE = os.path.dirname(os.path.abspath(__file__))

try:
    from model import FEATURE_COLUMNS  # needs torch, because model.py imports it
except ImportError:
    # No torch: read the FEATURE_COLUMNS list straight out of model.py's source
    import ast
    with open(os.path.join(HERE, "model.py"), encoding="utf-8") as _f:
        for _node in ast.parse(_f.read()).body:
            if isinstance(_node, ast.Assign) and _node.targets[0].id == "FEATURE_COLUMNS":
                FEATURE_COLUMNS = ast.literal_eval(_node.value)
MODELS = ["ares", "athena", "apollo"]
THRESHOLD = 0.50
RESULTS_DIR = os.path.join(HERE, "results")


# --------------------------------------------------------------------------- #
# 1. Model loading
# --------------------------------------------------------------------------- #
def load_with_torch(path):
    """Official way: rebuild the architecture from model.py and load the weights."""
    import torch
    from model import MythicThreatClassifier

    model = MythicThreatClassifier()
    state = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()  # inference mode
    weight = model.linear.weight.detach().numpy().reshape(-1)
    bias = float(model.linear.bias.detach().numpy().reshape(-1)[0])

    def predict_logits(x):
        with torch.no_grad():
            return model(torch.tensor(x, dtype=torch.float32)).numpy()

    return weight, bias, predict_logits


def load_without_torch(path):
    """
    Fallback when torch is not installed: a .pth file is a ZIP archive that holds
    a pickled state_dict plus raw little-endian float32 tensor storages.
    We read the storages directly. Gives exactly the same numbers as torch.load.
    """
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        root = names[0].split("/")[0]
        pkl = z.read(f"{root}/data.pkl")

        # Find which storage key ("0", "1", ...) belongs to which parameter
        storages = {}

        class _Unpickler(pickle.Unpickler):
            def find_class(self, module, name):
                if name == "_rebuild_tensor_v2":
                    return lambda storage, *a, **k: storage
                if module == "collections" and name == "OrderedDict":
                    from collections import OrderedDict
                    return OrderedDict
                return lambda *a, **k: None

            def persistent_load(self, pid):
                key = pid[2]
                if key not in storages:
                    raw = z.read(f"{root}/data/{key}")
                    storages[key] = np.array(
                        struct.unpack(f"<{len(raw) // 4}f", raw), dtype=np.float32
                    )
                return storages[key]

        state = _Unpickler(io.BytesIO(pkl)).load()

    weight = np.asarray(state["linear.weight"]).reshape(-1)
    bias = float(np.asarray(state["linear.bias"]).reshape(-1)[0])

    def predict_logits(x):
        return x @ weight + bias  # same as nn.Linear(9, 1)

    return weight, bias, predict_logits


def load_model(path):
    try:
        import torch  # noqa: F401
        return load_with_torch(path), "torch"
    except ImportError:
        return load_without_torch(path), "zip-fallback (torch not installed)"


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


# --------------------------------------------------------------------------- #
# 2. Data
# --------------------------------------------------------------------------- #
features = pd.read_csv(os.path.join(HERE, "test_dataset.csv"))
labels = pd.read_csv(os.path.join(HERE, "test_labels.csv"))

# Make sure rows line up 1:1 (join on the id, never trust row order blindly)
data = features.merge(labels, on="encounter_id", how="inner", validate="one_to_one")
assert len(data) == len(features) == len(labels), "id mismatch between files"

X = data[FEATURE_COLUMNS].to_numpy(dtype=np.float32)  # exact order from model.py
y = data["is_catastrophic_threat"].to_numpy(dtype=int)

os.makedirs(RESULTS_DIR, exist_ok=True)
lines = []


def log(s=""):
    print(s)
    lines.append(s)


log(f"Test set: {len(y)} encounters, {y.sum()} catastrophic (positive), "
    f"{len(y) - y.sum()} not catastrophic -> positive rate {y.mean():.1%}")
log(f"Decision threshold: {THRESHOLD}")
log()

# --------------------------------------------------------------------------- #
# 3. Evaluate every model
# --------------------------------------------------------------------------- #
rows, confusions = [], {}
for name in MODELS:
    (weight, bias, predict_logits), backend = load_model(
        os.path.join(HERE, "models", f"model_{name}.pth"))

    logits = predict_logits(X)
    proba = sigmoid(logits)                    # P(catastrophic)
    pred = (proba >= THRESHOLD).astype(int)    # 1 = raise the alarm

    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    confusions[name] = np.array([[tn, fp], [fn, tp]])
    used = [FEATURE_COLUMNS[i] for i in np.flatnonzero(weight)]

    rows.append({
        "model": name.upper(),
        "feature_used": ", ".join(used),
        "accuracy": accuracy_score(y, pred),
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred),
        "specificity": tn / (tn + fp),
        "f1": f1_score(y, pred),
        "f2": fbeta_score(y, pred, beta=2),
        "roc_auc": roc_auc_score(y, proba),
    })

    log(f"[{name.upper()}] loaded with {backend}")
    log(f"  weights = {np.round(weight, 4).tolist()}  bias = {bias}")
    log(f"  -> the model only looks at: {used}")

metrics = pd.DataFrame(rows).set_index("model")
metrics.to_csv(os.path.join(RESULTS_DIR, "metrics.csv"), float_format="%.4f")

log()
log("=== Metrics at threshold 0.50 ===")
log(metrics.drop(columns="feature_used").to_string(float_format=lambda v: f"{v:.4f}"))
log()

# --------------------------------------------------------------------------- #
# 4. Cost check: for which FN/FP cost ratios does each model win?
#    total_cost = FN * cost_FN + FP * cost_FP   (we set cost_FP = 1)
# --------------------------------------------------------------------------- #
log("=== Expected cost, cost_FP = 1, cost_FN = k ===")
for k in [1, 2, 5, 10, 50]:
    costs = {m: int(r.FN * k + r.FP) for m, r in metrics.iterrows()}
    low = min(costs.values())
    winners = [m for m, c in costs.items() if c == low]
    verdict = "tie (all equal)" if len(winners) > 1 else f"best {winners[0]}"
    log(f"  k={k:>3}: " + ", ".join(f"{m}={c}" for m, c in costs.items()) + f"  -> {verdict}")
log()

# --------------------------------------------------------------------------- #
# 5. Decision
# --------------------------------------------------------------------------- #
best = metrics["recall"].idxmax()        # fewest missed catastrophes
assert metrics["FN"].idxmin() == best
log(f"All three accuracies: {metrics['accuracy'].round(4).tolist()} (identical)")
log(f"Business rule: a missed threat (FN) costs far more than a false alarm (FP)")
log(f"-> choose the highest RECALL / fewest FN: {best} "
    f"(recall {metrics.loc[best, 'recall']:.2f}, FN {metrics.loc[best, 'FN']})")
flag = f"SIGMOID_{best}"
log(f"FLAG: {flag}")

with open(os.path.join(RESULTS_DIR, "report.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")
with open(os.path.join(HERE, "FLAG.txt"), "w", encoding="utf-8") as f:
    f.write(flag + "\n")

# --------------------------------------------------------------------------- #
# 6. Plots (optional; skipped if matplotlib is missing)
# --------------------------------------------------------------------------- #
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    for ax, name in zip(axes, MODELS):
        cm = confusions[name]
        ax.imshow(cm, cmap="Blues")
        for (i, j), v in np.ndenumerate(cm):
            ax.text(j, i, str(v), ha="center", va="center", fontsize=15,
                    color="white" if v > cm.max() / 2 else "black")
        ax.set_xticks([0, 1], ["pred 0\n(no alarm)", "pred 1\n(alarm)"])
        ax.set_yticks([0, 1], ["true 0", "true 1"])
        r = metrics.loc[name.upper()]
        ax.set_title(f"{name.upper()}\nacc {r.accuracy:.2f} | recall {r.recall:.2f} | "
                     f"precision {r.precision:.2f}", fontsize=10)
    fig.suptitle("Confusion matrices at threshold 0.50 (bottom-left = missed threats)")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "confusion_matrices.png"), dpi=130)

    fig, ax = plt.subplots(figsize=(8, 4.2))
    cols = ["accuracy", "precision", "recall", "f1", "f2"]
    xpos = np.arange(len(cols))
    for i, name in enumerate(metrics.index):
        ax.bar(xpos + (i - 1) * 0.27, metrics.loc[name, cols], width=0.27, label=name)
    ax.set_xticks(xpos, cols)
    ax.set_ylim(0, 1.05)
    ax.set_title("Same accuracy, very different recall / precision")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "metrics_comparison.png"), dpi=130)
    print("Plots saved to results/")
except ImportError:
    print("matplotlib not installed - plots skipped")
