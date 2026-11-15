"""Check the efficient sweep against exhaustive prediction evaluation."""
import math
import random
import unittest
from evaluate import metrics
from threshold_scan import scan_thresholds
def labels_for(kinds): return [{'is_anomaly': '0' if(kind == 'normal') else '1', 'anomaly_type': kind} for kind in kinds]
class ThresholdScanTests(unittest.TestCase):
    def test_matches_exhaustive_search_with_ties_and_all_anomaly_types(self):
        rng = random.Random(20261114)
        for _ in range(100):
            scores = [rng.choice([0.0, 0.2, 0.4, 0.5, 0.8, 1.0]) for _ in range(30)]
            labels = labels_for([rng.choice(['normal', 'point', 'drift', 'collective']) for _ in scores])
            result = scan_thresholds(scores, labels)
            cutoffs = [math.nextafter(max(scores), math.inf)] + sorted(set(scores))
            expected = max(metrics(scores, labels, cutoff)['score'] for cutoff in cutoffs)
            self.assertEqual(result['metrics']['score'], expected)
            self.assertEqual(result['metrics'], metrics(scores, labels, result['cutoff']))
            self.assertEqual(result['boundaries_evaluated'], len(set(scores)) + 1)
    def test_perfect_separation(self):
        result = scan_thresholds([0.1, 0.2, 0.3, 0.5, 0.7], labels_for(['normal', 'normal', 'normal', 'point', 'drift']))
        self.assertEqual((result['metrics']['tp'], result['metrics']['fp'], result['metrics']['fn']), (2, 0, 0))
        self.assertTrue(0.3 < result['cutoff'] <= 0.5)
    def test_equal_scores_cannot_be_split(self):
        result = scan_thresholds([0.4, 0.4], labels_for(['normal', 'point']))
        self.assertEqual(result['boundaries_evaluated'], 2)
        self.assertLess(result['metrics']['f1'], 1)
    def test_all_normal_including_score_one(self):
        result = scan_thresholds([0.0, 0.5, 1.0], labels_for(['normal'] * 3))
        self.assertGreater(result['cutoff'], 1)
        self.assertEqual(result['metrics']['fp'], 0)
    def test_all_anomalous_including_score_zero(self):
        result = scan_thresholds([0.0, 0.5, 1.0], labels_for(['point', 'drift', 'collective']))
        self.assertEqual(result['cutoff'], 0)
        self.assertEqual(result['metrics']['score'], 100)
    def test_rejects_invalid_inputs(self):
        for scores, labels in [([], []), ([0.5], []), ([float('nan')], labels_for(['point'])), ([0.5], [{'is_anomaly': '0', 'anomaly_type': 'point'}])]:
            with self.assertRaises(ValueError): scan_thresholds(scores, labels)
if(__name__ == '__main__'): unittest.main()
