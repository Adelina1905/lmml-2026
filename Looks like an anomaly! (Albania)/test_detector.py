"""Behavior checks using generated streams, independent of the competition labels."""
import math
import random
import unittest
from detector import Detector
from evaluate import load_data, replay
FIELDS = Detector.fields
CENTERS = (13.0, 72.0, 0.12, 32.0, 101.5, 510.0)
SCALES = (0.2, 0.6, 0.01, 0.35, 0.04, 10.0)
def synthetic(count, seed=123):
    rng = random.Random(seed)
    return [dict(zip(FIELDS, (center + scale * rng.gauss(0, 1) for center, scale in zip(CENTERS, SCALES))), bunker_zone='test', timestamp=str(i)) for i in range(count)]
class DetectorTests(unittest.TestCase):
    def test_normal_false_alarms(self):
        scores = replay(synthetic(5000))
        self.assertLess(sum(score >= 0.5 for score in scores) / len(scores), 0.005)
    def test_isolated_spikes_in_each_sensor(self):
        for index, field in enumerate(FIELDS):
            for direction in (-1, 1):
                readings = synthetic(601, seed=100 + index)
                readings[-1][field] += direction * 9 * SCALES[index]
                self.assertGreaterEqual(replay(readings)[-1], 0.5)
    def test_gradual_drifts_in_each_sensor(self):
        for index, field in enumerate(FIELDS):
            for direction in (-1, 1):
                readings = synthetic(720, seed=200 + index)
                for offset in range(120): readings[600 + offset][field] += direction * (offset + 1) / 120 * 8 * SCALES[index]
                scores = replay(readings)
                self.assertGreater(sum(score >= 0.5 for score in scores[-60:]) / 60, 0.8)
    def test_persistent_moderate_joint_shift(self):
        readings = synthetic(680, seed=321)
        for row in readings[600:]:
            for index in (2, 3, 5): row[FIELDS[index]] += 2.2 * SCALES[index]
        scores = replay(readings)
        self.assertGreater(sum(score >= 0.5 for score in scores[615:]) / 65, 0.7)
    def test_future_readings_do_not_change_prefix(self):
        readings, _ = load_data()
        complete = replay(readings)
        for end in (75, 591, 1012, 1915, 3075): self.assertEqual(replay(readings[:end]), complete[:end])
    def test_zone_state_is_independent(self):
        first, second = synthetic(700, seed=1), synthetic(700, seed=2)
        for row in second:
            row['bunker_zone'] = 'different'
            for field in FIELDS: row[field] *= 2
        expected_first, expected_second = replay(first), replay(second)
        combined = replay([row for pair in zip(first, second) for row in pair])
        self.assertEqual(combined[::2], expected_first)
        self.assertEqual(combined[1::2], expected_second)
    def test_change_of_units(self):
        readings = synthetic(800)
        for row in readings[600:660]: row['co2_ppm'] += 80
        converted = [{**row, **{field: row[field] * (i + 2) + (i + 1) * 10 for i, field in enumerate(FIELDS)}} for row in readings]
        self.assertTrue(all(abs(a - b) < 1e-9 for a, b in zip(replay(readings), replay(converted))))
    def test_invalid_readings_do_not_damage_state(self):
        readings = synthetic(200)
        detector = Detector()
        replay(readings[:100], detector)
        for invalid in ({}, {'temperature_c': None}, {**readings[0], 'co2_ppm': float('nan')}, {**readings[0], 'co2_ppm': float('inf')}):
            score = detector.update(invalid)
            self.assertTrue(math.isfinite(score) and 0 <= score <= 1)
        self.assertEqual(replay(readings[100:], detector), replay(readings)[100:])
if(__name__ == '__main__'): unittest.main()
