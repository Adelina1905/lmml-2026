"""Exact O(n log n) decision-cutoff search for a fixed stream of anomaly scores.

Labels select a cutoff on the first half only. Equal scores move together.
This tunes the final decision cutoff, not the detector's internal parameters.
"""
import json
import math
from detector import DEFAULT_PARAMETERS, Detector
from evaluate import ROOT, load_data, metrics, metrics_from_counts, replay
KINDS = ('point', 'drift', 'collective')
def scan_thresholds(scores, labels, reference=0.5):
    if(not scores or len(scores) != len(labels)): raise ValueError('Expected nonempty, equally sized score and label lists.')
    if(not math.isfinite(reference) or not all(math.isfinite(score) and 0 <= score <= 1 for score in scores)): raise ValueError('Expected finite scores between zero and one and a finite reference.')
    for label in labels:
        if((label['is_anomaly'], label['anomaly_type']) not in (('0', 'normal'), ('1', 'point'), ('1', 'drift'), ('1', 'collective'))): raise ValueError('Inconsistent anomaly label.')
    ordered = sorted(zip(scores, labels), key=lambda pair: pair[0], reverse=True)
    totals = {kind: sum(label['anomaly_type'] == kind for label in labels) for kind in KINDS}
    hits = dict.fromkeys(KINDS, 0)
    tp, fp, fn = 0, 0, sum(totals.values())
    trials = []
    def record(cutoff):
        trials.append({'cutoff': cutoff, 'metrics': metrics_from_counts(tp, fp, fn, hits, totals)})
    # The first partition predicts no anomalies, including when max score is 1.
    record(reference if(reference > ordered[0][0]) else math.nextafter(ordered[0][0], math.inf))
    index = 0
    while(index < len(ordered)):
        upper = ordered[index][0]
        while(index < len(ordered) and ordered[index][0] == upper):
            kind = ordered[index][1]['anomaly_type']
            if(kind == 'normal'): fp += 1
            else:
                tp += 1
                fn -= 1
                hits[kind] += 1
            index += 1
        if(index == len(ordered)):
            cutoff = reference if(reference <= upper) else upper
        else:
            lower = ordered[index][0]
            cutoff = reference if(lower < reference <= upper) else lower + (upper - lower) / 2
            if(cutoff <= lower): cutoff = upper
        record(cutoff)
    # Prefer fewer mistakes, then the existing cutoff, when objective scores tie.
    best = max(trials, key=lambda trial: (trial['metrics']['score'], -trial['metrics']['fp'], -trial['metrics']['fn'], -abs(trial['cutoff'] - reference)))
    return {'cutoff': best['cutoff'], 'metrics': best['metrics'], 'boundaries_evaluated': len(trials), 'trials': trials}
def main():
    readings, labels = load_data()
    split = 1800
    if(len(readings) <= split): raise ValueError('Expected more than 1,800 rows for chronological validation.')
    # Scores are generated causally and remain fixed throughout the scan.
    raw_scores = replay(readings, Detector())
    selected = scan_thresholds(raw_scores[:split], labels[:split])
    cutoff = selected['cutoff']
    baseline = {'cutoff': 0.5, 'selection': metrics(raw_scores[:split], labels[:split]), 'validation': metrics(raw_scores[split:], labels[split:]), 'full': metrics(raw_scores, labels)}
    scanned = {'cutoff': cutoff, 'selection': metrics(raw_scores[:split], labels[:split], cutoff), 'validation': metrics(raw_scores[split:], labels[split:], cutoff), 'full': metrics(raw_scores, labels, cutoff)}
    report = {'method': 'Sort once, sweep every distinct score group, and maximize the competition score', 'scope': 'Exact optimum on fixed selection scores; does not optimize internal sensor thresholds or guarantee unseen performance', 'selection_rows': split, 'validation_note': 'Second half is reused validation, never used to choose the cutoff.', 'detector_parameters': dict(DEFAULT_PARAMETERS), 'boundaries_evaluated': selected['boundaries_evaluated'], 'baseline': baseline, 'scanned': scanned, 'trials': selected['trials']}
    output = ROOT / 'threshold_scan_results.json'
    output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({key: value for key, value in report.items() if(key != 'trials')}, indent=2))
    print(f'Results saved to {output}')
if(__name__ == '__main__'): main()
