# Evaluation — `naive_owat`

**Score: 22.1 / 100**

- queries used: **41 / 150**  (counted by the harness)
- recovered function: `2.6500 + 1.5000*(x1-0.5) + 0.4000*(x3-0.5) + -0.6000*(x5-0.5)`

| metric | value |
|---|---|
| rmse_overall | 0.455899 |
| rmse_unfiltered | 0.464872 |
| rmse_in_region | 1.557841 |
| region_points | 13 |
| boundary_points_excluded | 2 |
| max_abs_error | 2.138045 |

## variables  —  16.0 / 20
- x1: claimed relevant — correct
- x2: claimed inert — correct
- x3: claimed relevant — correct
- x4: claimed inert — MISSED (it matters)
- x5: claimed relevant — correct

## interaction  —  0.0 / 20
- no interaction reported between ['x3', 'x5'] — the single most important structural fact was missed

## discontinuity  —  0.0 / 30
- no discontinuity reported — the switching behaviour was missed entirely

## prediction  —  6.0 / 30
- source: predictions[]
- overall RMSE 0.45590 on 298 points (+5.6/20)
- RMSE inside the discontinuous region 1.55784 on 13 points (+0.5/10)
- 2 point(s) within 0.005 of the true switching surface excluded as a coin flip (unfiltered RMSE would be 0.46487)

## ground truth
```
f(x) = 2.0 + 1.5*x1 + 0.4*x3 - 0.6*x5 - 4.0*(x3-0.5)*(x5-0.5) + (1.5 if x1 + x4 >= 1.7 else 0.0)
```
- relevant: ['x1', 'x3', 'x4', 'x5'], inert: ['x2']

--- stderr (main.py log) ---
--- naive OWAT conclusion -------------------------------
  d f / d x1  =  +1.5000
  d f / d x2  =  +0.0000   <- 'irrelevant'
  d f / d x3  =  +0.4000
  d f / d x4  =  +0.0000   <- 'irrelevant'
  d f / d x5  =  -0.6000
  self-RMSE on its own 41 points: 1.40e-16  (looks perfect)
