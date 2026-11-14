"""Builds checkpoints/vectorizer.joblib and checkpoints/recovered_model.joblib.

The shipped original/finetuned checkpoints were not part of the hand-out, so the recovered model is trained jointly
on all ten labels: the OLD-task replay sample plus the public set covering both tasks. Joint training (full replay)
is the standard cure for catastrophic forgetting. Run: python train_recovered.py [--cv]
"""
from __future__ import annotations
import argparse
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import FeatureUnion
from sklearn.svm import LinearSVC
import model_utils as mu

SEED = 0

def build_vectorizer() -> FeatureUnion:
    return FeatureUnion([("word", TfidfVectorizer(analyzer="word", ngram_range=(1, 2), sublinear_tf=True, token_pattern=r"(?u)\b\w+\b")), ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True))])

def build_model() -> LinearSVC:
    return LinearSVC(C=1.0, random_state=SEED)

def load_data() -> tuple[list[str], list[str], list[str]]:
    replay = mu.load_records(mu.HERE / "dataset" / "old_task_sample.json")
    gold = {r["id"]: r["label"] for r in mu.load_records(mu.HERE / "dataset" / "train-output.json")}
    public = mu.load_records(mu.HERE / "dataset" / "train-input.json")
    ids = [r["id"] for r in replay] + [r["id"] for r in public]
    texts = [r["text"] for r in replay] + [r["text"] for r in public]
    labels = [r["label"] for r in replay] + [gold[r["id"]] for r in public]
    return ids, texts, labels

def fit(texts: list[str], labels: list[str]):
    vectorizer = build_vectorizer().fit(texts)
    model = build_model().fit(mu.featurize(vectorizer, texts), labels)
    return vectorizer, model

def score(vectorizer, model, ids: list[str], texts: list[str], labels: list[str]) -> dict[str, float]:
    return mu.accuracy_by_task(dict(zip(ids, labels)), dict(zip(ids, mu.predict_labels(model, vectorizer, texts))))

def cross_validate(ids: list[str], texts: list[str], labels: list[str], folds: int = 5) -> None:
    """Held-out estimate. Folds split on unique text so duplicates never leak between train and test."""
    unique = sorted(set(zip(texts, labels)))
    utexts, ulabels = [t for t, _ in unique], [l for _, l in unique]
    results = []
    for fold, (tr, te) in enumerate(StratifiedKFold(n_splits=folds, shuffle=True, random_state=SEED).split(utexts, ulabels)):
        vectorizer, model = fit([utexts[i] for i in tr], [ulabels[i] for i in tr])
        acc = score(vectorizer, model, [str(i) for i in te], [utexts[i] for i in te], [ulabels[i] for i in te])
        results.append((acc["old"], acc["new"]))
        print(f"fold {fold}: old {acc['old']:.3f} new {acc['new']:.3f} ({len(te)} held-out texts)")
    mean = np.mean(results, axis=0)
    print(f"cv mean: old {mean[0]:.3f} new {mean[1]:.3f}")

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cv", action="store_true", help="print a held-out cross-validation estimate first")
    args = parser.parse_args()
    ids, texts, labels = load_data()
    if(args.cv): cross_validate(ids, texts, labels)
    vectorizer, model = fit(texts, labels)
    mu.save_model(vectorizer, mu.VECTORIZER_PATH)
    mu.save_model(model, mu.RECOVERED_MODEL_PATH)
    print(f"trained on {len(texts)} rows, classes {list(model.classes_)}")
    print("fit accuracy:", score(vectorizer, model, ids, texts, labels))
    print(f"saved {mu.VECTORIZER_PATH.name} and {mu.RECOVERED_MODEL_PATH.name}")

if __name__ == "__main__":
    main()
