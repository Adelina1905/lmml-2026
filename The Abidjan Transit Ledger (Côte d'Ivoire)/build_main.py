"""Generates the single-file submission main.py: embeds the replay + public training rows so main.py needs no other file."""
import json
from pathlib import Path
import model_utils as mu

HEADER = '''"""Self-contained submission for task_cote_divoire (The Abidjan Transit Ledger).

Same I/O contract as main_template.py: reads test-input.json, writes predictions.json ({"id", "label"} per row), both by
filename in the current directory, no arguments needed. The upload accepts a single .py file, so this file carries its
own training data (OLD-task replay sample + public set covering both tasks). On each run it trains the recovered model
jointly on all ten labels (full replay, the cure for catastrophic forgetting), saves it to
checkpoints/recovered_model.joblib, and predicts. Training takes about a second on CPU.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

DEFAULT_MODEL_PATH = Path("checkpoints") / "recovered_model.joblib"
'''

BODY = '''
def build_model() -> Pipeline:
    features = FeatureUnion([("word", TfidfVectorizer(analyzer="word", ngram_range=(1, 2), sublinear_tf=True, token_pattern=r"(?u)\\b\\w+\\b")), ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True))])
    return Pipeline([("features", features), ("clf", LinearSVC(C=1.0, random_state=0))])

def train() -> Pipeline:
    return build_model().fit([t for t, _ in TRAIN_ROWS], [LABELS[i] for _, i in TRAIN_ROWS])

def save_model(model: Pipeline, path: Path) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, path)
    except OSError as exc:
        print(f"could not save checkpoint to {path}: {exc}", file=sys.stderr)

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="test-input.json", help="filename only, no path")
    parser.add_argument("--output", default="predictions.json", help="filename only, no path")
    parser.add_argument("--model", default=str(DEFAULT_MODEL_PATH), help="where the recovered checkpoint is saved")
    args = parser.parse_args()
    input_path = Path(args.input)
    if(not input_path.exists()):
        print(f"input file not found: {input_path}", file=sys.stderr)
        return 2
    model = train()
    save_model(model, Path(args.model))
    with open(input_path, encoding="utf-8") as fh: records = json.load(fh)
    labels = model.predict([r["text"] for r in records])
    predictions = [{"id": r["id"], "label": str(label)} for r, label in zip(records, labels)]
    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump(predictions, fh, indent=2, ensure_ascii=False)
        fh.write("\\n")
    print(f"wrote {len(predictions)} predictions to {args.output}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
'''

def main() -> None:
    replay = mu.load_records(mu.HERE / "dataset" / "old_task_sample.json")
    gold = {r["id"]: r["label"] for r in mu.load_records(mu.HERE / "dataset" / "train-output.json")}
    public = mu.load_records(mu.HERE / "dataset" / "train-input.json")
    pairs = [(r["text"], r["label"]) for r in replay] + [(r["text"], gold[r["id"]]) for r in public]
    rows = sorted({(t, mu.ALL_LABELS.index(l)) for t, l in pairs})
    data = "LABELS = " + json.dumps(mu.ALL_LABELS) + "\n\nTRAIN_ROWS = [\n" + "".join(f"    ({json.dumps(t, ensure_ascii=False)}, {i}),\n" for t, i in rows) + "]\n"
    (mu.HERE / "main.py").write_text(HEADER + "\n" + data + BODY, encoding="utf-8")
    print(f"wrote main.py with {len(rows)} unique training rows")

if __name__ == "__main__":
    main()
