# Evaluation — `task_germany`

**Score: 100.0 / 100**

- queries used: **149 / 150**  (counted by the harness)
- recovered function: `1.000000 + 1.500000*x1 + 2.400000*x3 + 1.400000*x5 - 4.000000*x3*x5 + (1.500000 if 1.000000*x1 + 1.000012*x4 >= 1.700071 else 0.0)`

| metric | value |
|---|---|
| rmse_overall | 0.0 |
| rmse_unfiltered | 0.0 |
| rmse_in_region | 0.0 |
| region_points | 13 |
| boundary_points_excluded | 2 |
| max_abs_error | 1e-06 |

## variables  —  20.0 / 20
- x1: claimed relevant — correct
- x2: claimed inert — correct
- x3: claimed relevant — correct
- x4: claimed relevant — correct
- x5: claimed relevant — correct

## interaction  —  20.0 / 20
- identified the ['x3', 'x5'] interaction (+12)
- coefficient -4.000 vs true -4.000 (0.0% off) (+8)

## discontinuity  —  30.0 / 30
- a discontinuity was reported (+8)
- gate variables ['x1', 'x4'] correct (+10)
- boundary orientation cos=1.0000 (true 1.0), offset err=0.0000 (+7)
- jump +1.500 vs true +1.500 (0.0% off) (+5)

## prediction  —  30.0 / 30
- source: predictions[]
- overall RMSE 0.00000 on 298 points (+20.0/20)
- RMSE inside the discontinuous region 0.00000 on 13 points (+10.0/10)
- 2 point(s) within 0.005 of the true switching surface excluded as a coin flip (unfiltered RMSE would be 0.00000)

## ground truth
```
f(x) = 2.0 + 1.5*x1 + 0.4*x3 - 0.6*x5 - 4.0*(x3-0.5)*(x5-0.5) + (1.5 if x1 + x4 >= 1.7 else 0.0)
```
- relevant: ['x1', 'x3', 'x4', 'x5'], inert: ['x2']

--- stderr (main.py log) ---
design: 32 corners + centre + 32 maximin-LHS = 65 queries
step search: gate candidates ['x1', 'x4'], 10/65 design points on, jump~+1.5000, max residual 4.93e-07
bisection: 5 boundary points, spent 124
gate test: boundary moves with ['x1', 'x4']
sparse model terms: ['1', 'x1', 'x3', 'x5', 'x3*x5']; jump 1.500000; max |residual| on 149 points 5.21e-07; label/half-space agreement 0.970
15 spare queries on challenge points nearest the boundary; model error there before refit: max 4.63e-07, median 2.58e-07
total queries 149/150
