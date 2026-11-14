"""
Black Box (task_germany): recover the structure of f(x1..x5) with <= 150 queries.

Model family that the procedure searches over (it is broad on purpose):

    f(x) = smooth(x) + J * [ a . x >= t ]
    smooth(x) = c0 + sum_i b_i x_i + sum_{i<j} c_ij x_i x_j + sum_i d_i x_i^2

Pipeline
  1. Space-filling design: all 32 corners of the cube, the centre, and
     interior points from a maximin Latin hypercube. Any half-space that
     cuts the cube contains a corner, so the step region is never missed,
     and the interior points stop the step from posing as an x_i*x_j term
     (on the corners alone, [x_i + x_j >= 1.7] equals x_i*x_j).
  2. Step search: for every pair of inputs, every direction (2 deg grid)
     and every threshold, add the step indicator to the 21-term quadratic
     least-squares fit and keep the candidate that removes most residual.
     The data is noise-free, so the right candidate leaves residuals ~1e-6.
     Labels are then re-derived from residuals (y - smooth is either ~0 or ~J).
  3. Bisection along segments that join an "off" point to an "on" point:
     every query halves the bracket, giving points on the switching
     surface to ~3e-4.
  4. Gate test: shift a tight bracket across the boundary along each input
     in turn. If neither label flips, that input does not move the
     boundary. This finds the gate variables without assuming there are two.
  5. Hyperplane fit (in the gate variables) through the boundary points.
  6. Sparse refit: drop terms whose coefficients are ~0, refit, read off
     relevant variables, interactions (x_i*x_j monomial coefficients),
     boundary, jump.
  7. Spare queries go on the challenge points closest to the boundary;
     for those the measured value is the prediction, and they also act as
     an out-of-sample check of the model.
"""
import itertools
import math

import numpy as np

VARS = ["x1", "x2", "x3", "x4", "x5"]
D = 5
PAIRS = list(itertools.combinations(range(D), 2))
TERMS = (["1"] + VARS + [f"{VARS[i]}*{VARS[j]}" for i, j in PAIRS]
         + [f"{v}**2" for v in VARS])
N_INTERIOR = 32
BISECT_TOL = 3e-4
N_SEGMENTS = 5
SHIFT = 0.3
LIN_EPS = 5e-3       # |coef| below this is treated as exactly zero
INT_EPS = 0.05       # interaction coefficients below this are not reported
SQ_EPS = 0.02
SEED = 20261114


def basis(X):
    X = np.atleast_2d(np.asarray(X, float))
    cols = [np.ones(len(X))] + [X[:, i] for i in range(D)]
    cols += [X[:, i] * X[:, j] for i, j in PAIRS]
    cols += [X[:, i] ** 2 for i in range(D)]
    return np.column_stack(cols)


def lhs_maximin(n, rng, tries=200):
    best, best_d = None, -1.0
    for _ in range(tries):
        cut = np.linspace(0, 1, n + 1)
        U = np.empty((n, D))
        for k in range(D):
            u = cut[:-1] + rng.random(n) / n
            U[:, k] = rng.permutation(u)
        U = 0.03 + 0.94 * U                      # keep off the faces
        d = np.min([np.min(np.sum((U[i + 1:] - U[i]) ** 2, 1))
                    for i in range(n - 1)])
        if d > best_d:
            best, best_d = U, d
    return best


class Data:
    """Every queried point and its answer, with budget-safe asking."""

    def __init__(self, oracle, reserve=0):
        self.o = oracle
        self.X, self.y = [], []
        self.seen = set()

    def ask(self, pts, phase):
        pts = [[min(1.0, max(0.0, float(v))) for v in p] for p in pts]
        new = []
        for p in pts:
            k = self.o.key(p)
            if k not in self.o.cache and k not in {self.o.key(q) for q in new}:
                new.append(p)
        if len(new) > self.o.remaining:
            raise RuntimeError("budget guard")
        ys = self.o.ask_many(pts, phase=phase)
        for p, yv in zip(pts, ys):
            k = self.o.key(p)
            if k not in self.seen:
                self.seen.add(k)
                self.X.append(p)
                self.y.append(float(yv))
        return ys

    def arrays(self):
        return np.array(self.X, float), np.array(self.y, float)


# ----------------------------------------------------------------------
def lstsq(A, y):
    return np.linalg.lstsq(A, y, rcond=None)[0]


def step_search(X, y):
    """Best half-space step in (pair, angle, threshold), quadratic smooth part."""
    n = len(X)
    Phi = basis(X)
    M = np.eye(n) - Phi @ np.linalg.pinv(Phi)
    r0 = M @ y
    thetas = np.deg2rad(np.arange(0.0, 360.0, 2.0))
    ct, st = np.cos(thetas), np.sin(thetas)
    best = (-1.0, None)
    for i, j in PAIRS:
        P = X[:, i:i + 1] * ct[None, :] + X[:, j:j + 1] * st[None, :]   # n x T
        # candidate thresholds = each point's own projection
        S = (P[:, None, :] >= P[None, :, :]).astype(float)               # n x n x T
        S = S.reshape(n, n * len(thetas))
        Sp = M @ S
        den = np.sum(Sp * Sp, 0)
        num = (Sp.T @ r0) ** 2
        score = np.where(den > 1e-8, num / np.maximum(den, 1e-12), -1.0)
        k = int(np.argmax(score))
        if score[k] > best[0]:
            m, ti = divmod(k, len(thetas))
            p = P[:, ti]
            thr = p[m]
            below = p[p < thr - 1e-12]
            t_mid = 0.5 * (thr + below.max()) if below.size else thr - 1e-3
            w = np.zeros(D)
            w[i], w[j] = ct[ti], st[ti]
            best = (score[k], dict(pair=(i, j), w=w, t=t_mid,
                                   labels=S[:, k].astype(bool)))
    return best[1]


def step_search_triples(X, y, n_dir=1500):
    """Fallback when no pair explains the step: 3-variable gates."""
    n = len(X)
    Phi = basis(X)
    M = np.eye(n) - Phi @ np.linalg.pinv(Phi)
    r0 = M @ y
    k = np.arange(n_dir) + 0.5                       # Fibonacci sphere
    phi = np.arccos(1 - 2 * k / n_dir)
    th = np.pi * (1 + 5 ** 0.5) * k
    U = np.column_stack([np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi),
                         np.cos(phi)])
    best = (-1.0, None)
    for tri in itertools.combinations(range(D), 3):
        P = X[:, list(tri)] @ U.T                                        # n x T
        for c0 in range(0, n_dir, 300):
            Pc = P[:, c0:c0 + 300]
            T = Pc.shape[1]
            S = (Pc[:, None, :] >= Pc[None, :, :]).astype(float).reshape(n, n * T)
            Sp = M @ S
            den = np.sum(Sp * Sp, 0)
            num = (Sp.T @ r0) ** 2
            score = np.where(den > 1e-8, num / np.maximum(den, 1e-12), -1.0)
            kk = int(np.argmax(score))
            if score[kk] > best[0]:
                m, ti = divmod(kk, T)
                p = Pc[:, ti]
                thr = p[m]
                below = p[p < thr - 1e-12]
                t_mid = 0.5 * (thr + below.max()) if below.size else thr - 1e-3
                w = np.zeros(D)
                w[list(tri)] = U[c0 + ti]
                best = (score[kk], dict(pair=tri, w=w, t=t_mid,
                                        labels=S[:, kk].astype(bool)))
    return best[1]


def fit_with_labels(X, y, L, cols=None):
    Phi = basis(X)
    if cols is not None:
        Phi = Phi[:, cols]
    A = np.column_stack([Phi, L.astype(float)])
    c = lstsq(A, y)
    return c[:-1], c[-1], y - A @ c


def relabel(X, y, cs, J, cols=None):
    Phi = basis(X)
    if cols is not None:
        Phi = Phi[:, cols]
    r = y - Phi @ cs
    return np.abs(r - J) < np.abs(r)


class Model:
    def __init__(self, cs, J, cols=None):
        self.cs, self.J, self.cols = cs, J, cols

    def smooth(self, X):
        Phi = basis(X)
        if self.cols is not None:
            Phi = Phi[:, self.cols]
        return Phi @ self.cs

    def is_on(self, x, yv):
        r = yv - self.smooth([x])[0]
        return abs(r - self.J) < abs(r)


# ----------------------------------------------------------------------
def solve(oracle, points):
    rng = np.random.default_rng(SEED)
    data = Data(oracle)
    log = []

    # 1. design ---------------------------------------------------------
    corners = [list(map(float, c)) for c in itertools.product([0, 1], repeat=D)]
    design = corners + [[0.5] * D] + lhs_maximin(N_INTERIOR, rng).tolist()
    data.ask(design, "design")
    log.append(f"design: 32 corners + centre + {N_INTERIOR} maximin-LHS = "
               f"{oracle.spent} queries")

    # 2. step search ----------------------------------------------------
    X, y = data.arrays()

    def settle(cand):
        L = cand["labels"]
        for _ in range(5):
            cs, J, res = fit_with_labels(X, y, L)
            L2 = relabel(X, y, cs, J)
            if np.array_equal(L2, L):
                break
            L = L2
        cs, J, res = fit_with_labels(X, y, L)
        return L, cs, J, res

    cand = step_search(X, y)
    L, cs, J, res = settle(cand)
    if np.max(np.abs(res)) > 1e-4:       # no pair explains it: try triples
        cand3 = step_search_triples(X, y)
        L3, cs3, J3, res3 = settle(cand3)
        if np.sum(res3 ** 2) < np.sum(res ** 2):
            cand, L, cs, J, res = cand3, L3, cs3, J3, res3
            log.append("pair search left residuals; a 3-variable gate fits better")
    model = Model(cs, J)
    log.append(f"step search: gate candidates {[VARS[k] for k in cand['pair']]}, "
               f"{int(L.sum())}/{len(L)} design points on, jump~{J:+.4f}, "
               f"max residual {np.max(np.abs(res)):.2e}")

    # 3. bisection ------------------------------------------------------
    on_idx = np.where(L)[0]
    off_idx = np.where(~L)[0]
    w, t = cand["w"], cand["t"]
    segs = []
    for a in off_idx:
        fa = X[a] @ w - t
        for b in on_idx:
            fb = X[b] @ w - t
            lam = 0.5 if fb - fa <= 1e-12 else min(1, max(0, -fa / (fb - fa)))
            c = X[a] + lam * (X[b] - X[a])
            segs.append((np.linalg.norm(X[b] - X[a]), a, b, c))
    chosen = []
    if segs:
        segs.sort(key=lambda s: s[0])
        short = segs[: max(40, len(segs) // 3)]
        chosen.append(short[0])
        while len(chosen) < N_SEGMENTS and len(chosen) < len(segs):
            best, bd = None, -1
            for s in segs:
                dmin = min(np.linalg.norm(s[3] - q[3]) for q in chosen)
                sc = dmin - 0.15 * s[0]
                if sc > bd and all(s is not q for q in chosen):
                    best, bd = s, sc
            chosen.append(best)

    brackets = []
    for _, a, b, _ in chosen:
        A, B = X[a].copy(), X[b].copy()
        steps = int(math.ceil(math.log2(max(np.linalg.norm(B - A), 1e-9)
                                        / BISECT_TOL)))
        if oracle.remaining < steps + 14:
            break
        for _ in range(steps):
            m = 0.5 * (A + B)
            ym = data.ask([m.tolist()], "bisect")[0]
            if model.is_on(m, ym):
                B = m
            else:
                A = m
        brackets.append((A, B))
    log.append(f"bisection: {len(brackets)} boundary points, "
               f"spent {oracle.spent}")

    # 4. gate test ------------------------------------------------------
    gate = []
    if brackets:
        A0, B0 = brackets[0]
        q0 = 0.5 * (A0 + B0)
        for k in range(D):
            if oracle.remaining < 4:
                break
            d = SHIFT if q0[k] + SHIFT <= 1.0 else -SHIFT
            As, Bs = A0.copy(), B0.copy()
            As[k] += d
            Bs[k] += d
            ya, yb = data.ask([As.tolist(), Bs.tolist()], f"gate:{VARS[k]}")
            la, lb = model.is_on(As, ya), model.is_on(Bs, yb)
            if la or not lb:
                gate.append(k)
    if not gate:
        gate = list(cand["pair"])
    log.append(f"gate test: boundary moves with {[VARS[k] for k in gate]}")

    # 5. hyperplane in the gate variables -----------------------------
    Q = np.array([0.5 * (A + B) for A, B in brackets]) if brackets else None
    if Q is not None and len(Q) >= len(gate):
        G = Q[:, gate]
        if len(gate) == 1:
            nrm = np.array([1.0])
            tt = float(G.mean())
        else:
            mu = G.mean(0)
            _, _, Vt = np.linalg.svd(G - mu)
            nrm = Vt[-1]
            tt = float(nrm @ mu)
        wv = np.zeros(D)
        wv[gate] = nrm
    else:
        wv, tt = w.copy(), t
    # orientation: "on" points on the >= side
    X, y = data.arrays()
    Lall = relabel(X, y, model.cs, model.J)
    side = X @ wv - tt
    if np.mean((side >= 0) == Lall) < 0.5:
        wv, tt = -wv, -tt
        side = -side
    agree = np.mean((side >= 0) == Lall)

    # 6. sparse refit ---------------------------------------------------
    cs_full, J_full, _ = fit_with_labels(X, y, Lall)
    keep = [0]
    for k in range(1, len(TERMS)):
        tol = LIN_EPS if k <= D else (INT_EPS if k <= D + len(PAIRS) else SQ_EPS)
        if abs(cs_full[k]) > tol:
            keep.append(k)
    cs, J, res = fit_with_labels(X, y, Lall, keep)
    model = Model(cs, J, keep)

    # 7. spare queries on challenge points nearest the boundary ---------
    P = np.array(points, float)
    nn = np.linalg.norm(wv)
    dist = np.abs(P @ wv - tt) / nn
    spare = max(0, oracle.remaining - 1)
    order = [int(i) for i in np.argsort(dist)][:spare]
    measured = {}
    val_err = []
    if order:
        ys = data.ask([P[i].tolist() for i in order], "challenge")
        for i, yv in zip(order, ys):
            measured[i] = float(yv)
            on = (P[i] @ wv - tt) >= 0
            pred = model.smooth([P[i]])[0] + (J if on else 0.0)
            val_err.append(abs(pred - yv))
        # final refit with everything we have
        X, y = data.arrays()
        Lall = relabel(X, y, model.cs, model.J, keep)
        cs, J, res = fit_with_labels(X, y, Lall, keep)
        model = Model(cs, J, keep)

    # orientation: jump reported positive, active side = higher side
    coef = dict(zip([TERMS[k] for k in keep], cs))
    if J < 0:
        coef["1"] = coef.get("1", 0.0) + J
        J = -J
        wv, tt = -wv, -tt
    # tidy scale: smallest non-zero gate coefficient = 1
    gate_abs = [abs(wv[k]) for k in gate if abs(wv[k]) > 1e-9]
    sc = 1.0 / min(gate_abs) if gate_abs else 1.0
    wv, tt = wv * sc, tt * sc

    # ------------------------------------------------------------------
    def smooth_pt(x):
        env = dict(zip(VARS, x))
        s = 0.0
        for name, c in coef.items():
            s += c * (1.0 if name == "1" else eval(name, {}, env))
        return s

    preds = []
    for i, p in enumerate(points):
        if i in measured:
            preds.append(round(measured[i], 6))
        else:
            on = float(np.dot(wv, p)) >= tt
            preds.append(round(smooth_pt(p) + (J if on else 0.0), 6))

    relevant = set(gate)
    for name, c in coef.items():
        for k, v in enumerate(VARS):
            if v in name.replace("**2", "").split("*"):
                relevant.add(k)
    rel = [VARS[k] for k in range(D) if k in relevant]
    irr = [VARS[k] for k in range(D) if k not in relevant]

    interactions = []
    for name, c in coef.items():
        if "*" in name and "**" not in name:
            a, b = name.split("*")
            interactions.append({
                "variables": [a, b], "type": "product",
                "coefficient": round(float(c), 6),
                "equivalent_centered_form":
                    f"{c:.6f}*({a}-0.5)*({b}-0.5) + linear/constant terms"})
    if len(interactions) > 1:      # keep the dominant ones only
        mx = max(abs(e["coefficient"]) for e in interactions)
        interactions = [e for e in interactions if abs(e["coefficient"]) >= 0.05 * mx]

    cond = " + ".join(f"{wv[k]:.6f}*{VARS[k]}" for k in range(D)
                      if abs(wv[k]) > 1e-12) + f" >= {tt:.6f}"
    smooth_str = " + ".join(
        (f"{c:.6f}" if name == "1" else f"{c:.6f}*{name}")
        for name, c in coef.items()).replace("+ -", "- ")
    expr = f"{smooth_str} + ({J:.6f} if {cond} else 0.0)"

    max_res = float(np.max(np.abs(res))) if len(res) else float("nan")
    log.append(f"sparse model terms: {list(coef)}; jump {J:.6f}; "
               f"max |residual| on {len(y)} points {max_res:.2e}; "
               f"label/half-space agreement {agree:.3f}")
    if val_err:
        log.append(f"{len(order)} spare queries on challenge points nearest "
                   f"the boundary; model error there before refit: max "
                   f"{max(val_err):.2e}, median {float(np.median(val_err)):.2e}")
    log.append(f"total queries {oracle.spent}/{oracle.budget}")
    for line in log:
        print(line)

    return {
        "method": "corner+LHS design, exhaustive step search, bisection of "
                  "the switching surface, gate test, sparse quadratic refit",
        "relevant_variables": rel,
        "irrelevant_variables": irr,
        "interactions": interactions,
        "discontinuities": [{
            "gate_variables": [VARS[k] for k in gate],
            "boundary": {
                "form": "linear",
                "coefficients": {v: round(float(wv[k]), 6)
                                 for k, v in enumerate(VARS)},
                "threshold": round(float(tt), 6),
                "active_when": "greater_equal",
            },
            "jump": round(float(J), 6),
            "condition": cond,
        }],
        "model_expression": expr,
        "predictions": preds,
        "notes": " | ".join(log),
    }
