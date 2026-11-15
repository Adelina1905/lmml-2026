# Task Overview

## 📖 Storyline

Across Albania, thousands of concrete bunkers remain scattered through cities, mountains and the coastline. A preservation team has fitted one underground bunker with environmental and structural sensors to identify conditions that could threaten the site or the people working inside it.

The detector runs on a small monitoring device and receives **one sensor reading at a time**. It may remember what it has seen before, but it cannot inspect future measurements.

The stream contains several kinds of abnormal behavior. Some events appear as sudden isolated spikes, some emerge as slow drifts, and others are only suspicious when several moderate changes persist together.

## 📌 Task Title

Looks like an anomaly!

## 🔥 Difficulty

Medium

## 🎯 Objective

Implement a streaming anomaly detector in `detector.py`.

Your file must define:
```python
class Detector:
    def __init__(self):
        ...

    def update(self, reading: dict) -> float:
        ...

```

The organizer-owned harness invokes update() once per timestep. The detector may retain state between calls but must not access future readings.

Each call must return a finite anomaly score between 0.0 and 1.0.

## 🚩 FLAG

Format: `SIGMOID_<WORD>`.

The flag is released automatically by the platform when your final evaluation score reaches **70.00/100 or higher**. The flag value is not provided in this description.

## 📂 Task Input

[Data](https://drive.google.com/drive/folders/14AMHetQATgeB-wSm0-Da10NffyxOGQVo?usp=sharing) consists of a CSV, where `reading` contains:

- `timestamp`
- `bunker_zone`
- `temperature_c`
- `humidity_pct`
- `vibration_g`
- `acoustic_db`
- `air_pressure_kpa`
- `co2_ppm`

The evaluator runs Python 3.11 with `numpy==2.2.4`, `pandas==2.2.3` and `scikit-learn==1.6.1` (pinned in `requirements.txt`) plus the standard library, on CPU with a 60-second timeout. Internet access and installation of additional packages are not permitted.

## ✅ Expected Output

Submit:
```text
detector.py

```

The organizer harness imports your `Detector` class and calls `update()` once per timestep.

## 🏆 Scoring Criteria

Each submission receives a score from **1 to 100**.

Returned anomaly scores are thresholded at `0.50`. The evaluator then computes:

- overall F1
- recall on point anomalies
- recall on gradual drifts
- recall on collective anomalies

The final score is:
```text
100 × (
    0.80 × overall F1
  + 0.05 × point recall
  + 0.075 × drift recall
  + 0.075 × collective recall
)

```

The result is clamped to the range `1–100`.

**Acceptance threshold:** A final score of **70.00/100 or higher** releases the flag. Scores below 70.00 remain valid leaderboard scores but do not release the flag.

F1 is deliberately weighted most heavily so simply flagging everything as anomalous will not score well. The per-type recall terms reward detectors that handle all three anomaly families.

- ✅ Final score ≥ 70.00: 200 points (PASS), flag released
- ❌ Final score < 70.00: 0 points (FAIL)
