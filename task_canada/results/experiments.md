# Experiments log

Each row adds one idea on top of the row before it, unless noted otherwise.

- **Dev acc** is accuracy on `dev_reports.py`: 90 short fictional reports we wrote ourselves, EN+FR, 10 per hazard. It's only a rough sanity check, not the real score.
- **kNN agree** is how often a sealed report gets the same prediction as its nearest sealed neighbour. Real reports about the same kind of event tend to look alike, so a higher value means more consistent predictions on the real data. It uses no labels.
- **Vault** is pass/fail only. The vault never shows a score.

| # | Model | Dev acc | kNN agree | Vault |
|---|---|---|---|---|
| 0 | first_model.npz (baseline, raw corpus) | 0.678 | 0.461 | door 4: `SNOWBLIND`, the known decoy (the baseline's word) |
| 1 | boilerplate and content-free reports removed, random headers | 0.689 | – | nothing |
| 2 | + EN/FR function words masked out | 0.789 | 0.481 | nothing |
| 3 | + distractor sentences removed (first-sentence classifier) | 0.811 | – | nothing |
| 4 | + hand-written EN/FR template reports (`gen.py`) | 0.867 | – | nothing |
| 5 | + real dates kept, places randomised | 0.867 | 0.580 | nothing |
| 6 | + class-balanced self-training on the sealed fingerprints | 0.889 | 0.594 | nothing |
| 6b | row 6 + EM class-prior correction | – | – | nothing |
| 7 | + kNN label propagation (α=0.5, k=10, keep 80 %) | 0.878 | 0.629 | **door 3: `SNOWFLAKE`** |
| 7a | label propagation α=0.7, k=10, keep 80 % (setting used in `solve.py`) | 0.900 | 0.648 | door 3: `SNOWFLAKE` |
| 7b | label propagation α=0.5, k=10, keep 90 % | 0.889 | 0.654 | door 3: `SNOWFLAKE` |
| 7c | label propagation α=0.5, k=5, keep 80 %, weight 3 | 0.889 | 0.641 | door 3: `SNOWFLAKE` |
| 7d | label propagation α=0.8, k=15, keep 90 % | 0.856 | 0.653 | door 3: `SNOWFLAKE` |
| 7e | row 3 (no templates, no self-training) + label propagation | 0.811 | 0.648 | door 3: `SNOWFLAKE` |
| 7f | two rounds of label propagation | 0.856 | 0.683 | door 3: `SNOWFLAKE` |

## How to read this

- Every model with label propagation opens the same door, door 3, and no other door. That includes 7e, which uses neither our templates nor self-training.
- No improved model opens door 4, which belongs to the first model.
- So two different pipelines agree on one door that isn't the baseline's. That is the evidence that `SNOWFLAKE` is the real flag.
