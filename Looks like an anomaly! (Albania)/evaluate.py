"""Replay the supplied data chronologically; labels are used only for evaluation."""
import csv
import json
import math
import time
from pathlib import Path
from detector import Detector
ROOT = Path(__file__).resolve().parent
def load_data():
    with (ROOT / 'task_albania' / 'train-input.csv').open(newline='', encoding='utf-8-sig') as stream: readings = list(csv.DictReader(stream))
    with (ROOT / 'task_albania' / 'train-output.csv').open(newline='', encoding='utf-8-sig') as stream: labels = list(csv.DictReader(stream))
    assert len(readings) == len(labels) and all(a['timestamp'] == b['timestamp'] for a, b in zip(readings, labels))
    return readings, labels
def metrics_from_counts(tp, fp, fn, type_hits, type_totals):
    f1 = 2 * tp / max(1, 2 * tp + fp + fn)
    recall = {kind: type_hits[kind] / max(1, type_totals[kind]) for kind in ('point', 'drift', 'collective')}
    score = 100 * (0.8 * f1 + 0.05 * recall['point'] + 0.075 * recall['drift'] + 0.075 * recall['collective'])
    return {'score': max(1, score), 'f1': f1, 'precision': tp / max(1, tp + fp), 'recall': tp / max(1, tp + fn), 'type_recall': recall, 'tp': tp, 'fp': fp, 'fn': fn}
def metrics(scores, labels, cutoff=0.5):
    predicted = [s >= cutoff for s in scores]
    actual = [int(row['is_anomaly']) == 1 for row in labels]
    tp = sum(a and p for a, p in zip(actual, predicted))
    fp = sum(not a and p for a, p in zip(actual, predicted))
    fn = sum(a and not p for a, p in zip(actual, predicted))
    type_hits = {kind: sum(p and row['anomaly_type'] == kind for p, row in zip(predicted, labels)) for kind in ('point', 'drift', 'collective')}
    type_totals = {kind: sum(row['anomaly_type'] == kind for row in labels) for kind in type_hits}
    return metrics_from_counts(tp, fp, fn, type_hits, type_totals)
def replay(readings, detector=None):
    detector = detector or Detector()
    scores = [detector.update(row) for row in readings]
    assert all(isinstance(s, (int, float)) and math.isfinite(s) and 0 <= s <= 1 for s in scores)
    return scores
def main():
    readings, labels = load_data()
    started = time.perf_counter()
    scores = replay(readings)
    result = metrics(scores, labels)
    result['seconds'] = time.perf_counter() - started
    result['first_half'] = metrics(scores[:1800], labels[:1800])
    result['second_half'] = metrics(scores[1800:], labels[1800:])
    result['false_positive_indices'] = [i for i, (s, row) in enumerate(zip(scores, labels)) if(s >= 0.5 and row['is_anomaly'] == '0')]
    result['false_negative_indices'] = [i for i, (s, row) in enumerate(zip(scores, labels)) if(s < 0.5 and row['is_anomaly'] == '1')]
    (ROOT / 'evaluation.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, indent=2))
if(__name__ == '__main__'): main()
