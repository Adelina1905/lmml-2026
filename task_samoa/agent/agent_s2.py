"""s2: alpha-beta search with energy-aware leaf evaluation and fog-aware opponent model."""
import os, json
N = 15
NB = []
for i in range(N * N):
    r, c = divmod(i, N)
    l = []
    if r > 0: l.append(i - N)
    if r < N - 1: l.append(i + N)
    if c > 0: l.append(i - 1)
    if c < N - 1: l.append(i + 1)
    NB.append(tuple(l))
STARTS = (0, 224)

P = dict(depth=4, k_me=8, k_op=6, lam=1.2, lam_late=0.6, explore=1.6, access=0.55,
         risk=0.0, elsewhere_gain=3.5, elsewhere_cost=2, elsewhere_until=999)
if os.environ.get("AGENT_P"):
    P.update(json.loads(os.environ["AGENT_P"]))

MEM = {}


def reset_agent():
    MEM.clear()


class Ctx:
    pass


def choose_action(s):
    me = s["seat"]; op = 1 - me
    own0, w, c = s["owners"], s["weights"], s["costs"]
    if s["actions_remaining"] >= 255 or "W" not in MEM:
        MEM.clear()
        MEM["W"] = [0] * 225; MEM["C"] = [9] * 225; MEM["own"] = [-3] * 225
    W, C, mown = MEM["W"], MEM["C"], MEM["own"]
    for i in range(225):
        if w[i] is not None:
            W[i] = w[i]; C[i] = c[i]
        if own0[i] != -3:
            mown[i] = own0[i]
    legal = s["legal_actions"]
    if len(legal) == 1:
        return legal[0]
    # model ownership: visible truth + remembered enemy tiles in fog
    own = [own0[i] if own0[i] != -3 else (op if mown[i] == op else -3) for i in range(225)]
    hidden = [own0[i] == -3 for i in range(225)]
    rem = s["actions_remaining"]
    lm = (rem + 1) // 2; lo = rem // 2
    # does opponent still have unseen frontier (can expand elsewhere)?
    total_vis = sum(1 for i in range(225) if not hidden[i])
    ctx = Ctx()
    ctx.me = me; ctx.W = W; ctx.C = C; ctx.hidden = hidden
    late = max(0.0, min(1.0, lm / 30.0))
    ctx.late = late

    def bonus(own, t):
        unseen = 0; acc = 0.0
        for n in NB[t]:
            if hidden[n] and own[n] == -3:
                unseen += 1
            elif own[n] == -1:
                adj = False
                for m in NB[n]:
                    if own[m] == me:
                        adj = True; break
                if not adj:
                    a = W[n] / (3 + C[n])
                    if a > acc: acc = a
        return late * (P["explore"] * unseen + P["access"] * acc * 3)

    def moves(own, p, E):
        q = 1 - p; seen = set(); out = []
        for t in range(225):
            if own[t] == p:
                for n in NB[t]:
                    if n in seen: continue
                    o = own[n]
                    if o == -1:
                        seen.add(n)
                        if C[n] <= E: out.append((n, C[n], 0))
                    elif o == q and n not in STARTS:
                        seen.add(n)
                        if C[n] + 2 <= E: out.append((n, C[n] + 2, 1))
        return out

    def lam_for(left):
        if left <= 0: return 0.0
        f = min(1.0, left / 20.0)
        return P["lam_late"] + (P["lam"] - P["lam_late"]) * f if left > 1 else 0.3

    elsewhere = lm > 3 and P["elsewhere_gain"] > 0

    def search(own, E0, E1, sc, virt, p, depth, alpha, beta, lme, lop):
        # sc = margin (me - op); virt = my virtual bonus
        if depth == 0 or (lme == 0 and lop == 0):
            Em = E0 if me == 0 else E1; Eo = E1 if me == 0 else E0
            return sc + virt + lam_for(lme) * Em - lam_for(lop) * Eo, None
        left_p = lme if p == me else lop
        if left_p == 0:
            return search(own, E0, E1, sc, virt, 1 - p, depth, alpha, beta, lme, lop)
        Ep = E0 if p == 0 else E1
        mv = moves(own, p, Ep)
        if p == me:
            scored = []
            for (t, cost, cap) in mv:
                if cap:
                    g = 2 * W[t]; b = 0.0
                else:
                    b = bonus(own, t); g = W[t]
                scored.append(((g + b) / (3 + cost), t, cost, cap, g, b))
            scored.sort(reverse=True)
            scored = scored[:P["k_me"]]
        else:
            scored = []
            for (t, cost, cap) in mv:
                g = 2 * W[t] if cap else W[t]
                scored.append((g / (3 + cost), t, cost, cap, g, 0.0))
            scored.sort(reverse=True)
            scored = scored[:P["k_op"]]
            if elsewhere and Ep >= P["elsewhere_cost"]:
                scored.append((0, -2, P["elsewhere_cost"], 0, P["elsewhere_gain"], 0.0))
        opts = [(None, 0, 0, 0, 0.0)] + [(t, cost, cap, g, b) for (_, t, cost, cap, g, b) in scored]
        if left_p == 1 and len(opts) > 1:
            opts = opts[1:]
        nlme = lme - 1 if p == me else lme
        nlop = lop - 1 if p != me else lop
        best_mv = None
        maximizing = (p == me)
        best = -1e18 if maximizing else 1e18
        for (t, cost, cap, g, b) in opts:
            if t is None:
                nE = min(9, Ep + 3)
                nown = own; nsc = sc; nvirt = virt
            else:
                nE = Ep - cost
                if t == -2:
                    nown = own; nsc = sc - g; nvirt = virt
                else:
                    nown = list(own); nown[t] = p
                    if p == me:
                        nsc = sc + g; nvirt = virt + b
                    else:
                        nsc = sc - g; nvirt = virt
            if p == 0:
                v, _ = search(nown, nE, E1, nsc, nvirt, 1, depth - 1, alpha, beta, nlme, nlop)
            else:
                v, _ = search(nown, E0, nE, nsc, nvirt, 0, depth - 1, alpha, beta, nlme, nlop)
            if maximizing:
                if v > best: best = v; best_mv = t
                if best > alpha: alpha = best
            else:
                if v < best: best = v; best_mv = t
                if best < beta: beta = best
            if alpha >= beta: break
        return best, best_mv

    E = s["energy"]
    margin = s["scores"][me] - s["scores"][op]
    _, mv = search(own, E[0], E[1], margin, 0.0, me, P["depth"], -1e18, 1e18, lm, lo)
    if mv not in legal:
        mv = None if None in legal else legal[-1]
    return mv
