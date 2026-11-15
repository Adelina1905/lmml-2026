"""v4: efficient expansion + contact rules (reserve, recapture, intruder capture, freeze)."""
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

P = dict(explore=1.6, access=0.55, risk=0.5, rest_margin=1.15,
         reserve=5, recap=3.0, intruder=2.0, freeze_margin=0, freeze=1,
         contact_dist=2, dk=3.0, end_plan=1, end_n=4, lock=1, lock_margin=0, recap_behind=1.0, rush=0.0, rush_until=1, rush_ahead=4.0, rush_behind=-3.0, rush_margin=0)
if os.environ.get("AGENT_P"):
    P.update(json.loads(os.environ["AGENT_P"]))

MEM = {}


def reset_agent():
    MEM.clear()


def choose_action(s):
    me = s["seat"]; op = 1 - me
    own, w, c = s["owners"], s["weights"], s["costs"]
    E = s["energy"][me]; Eo = s["energy"][op]
    left = (s["actions_remaining"] + 1) // 2  # my decisions left incl. this one
    late = max(0.0, min(1.0, left / 30.0))
    margin = s["scores"][me] - s["scores"][op]
    prev = MEM.get("prev")
    if s["actions_remaining"] >= 255:
        prev = None; MEM.clear()
    stolen = set()
    if prev is not None:
        stolen = {i for i in range(225) if prev[i] == me and own[i] == op}
    MEM["seen_enemy"] = MEM.get("seen_enemy", False) or any(o == op for o in own)

    def mine_adj(t):
        for n in NB[t]:
            if own[n] == me: return True
        return False

    def enemy_adj(t):
        for n in NB[t]:
            if own[n] == op: return True
        return False

    cand = [a for a in range(225) if own[a] in (-1, op) and a not in (0, 224) and mine_adj(a)]
    MEM["prev"] = list(own)
    if not cand:
        return None
    cost = lambda a: c[a] + (2 if own[a] == op else 0)
    contact = MEM["seen_enemy"]

    def value(t):
        if own[t] == -1:
            v = w[t]
            unseen = 0; acc = 0.0
            for n in NB[t]:
                if own[n] == -3:
                    unseen += 1
                elif own[n] == -1 and not mine_adj(n):
                    acc = max(acc, w[n] / (3 + c[n]))
            v += late * (P["explore"] * unseen + P["access"] * acc * 3)
            if enemy_adj(t):
                v -= P["risk"] * w[t]
            if not (P["rush_until"] and contact):
                r_, c_ = divmod(t, N)
                prog = (r_ + c_) / 28.0 if me == 0 else (28 - r_ - c_) / 28.0
                v += (P["rush_ahead"] if margin > P["rush_margin"] else P["rush_behind"]) * prog
            return v / (P["dk"] + c[t])
        # capture
        recapturable = enemy_adj(t) and Eo >= c[t] + 2
        base = 2 * w[t] / (5 + c[t])
        if left <= 1:
            return base * 3
        if t in stolen:
            return base * (P["recap"] if margin > P["lock_margin"] else P["recap_behind"])
        if not recapturable:
            return base * P["intruder"]
        if P["freeze"] and margin > P["freeze_margin"]:
            return base * 0.5  # keeps the champion busy recapturing while we lead
        return 0.0

    scored = sorted(((value(a), a) for a in cand), reverse=True)
    if P["lock"] and margin > P["lock_margin"] and left > 1:
        locks = [a for a in cand if own[a] == op and enemy_adj(a)]
        if locks:
            tgt = min(locks, key=lambda a: (cost(a), -w[a]))
            if cost(tgt) <= E:
                return tgt
            # keep energy for the lock: only rest
            return None
    if left <= 1:
        aff = [a for a in cand if cost(a) <= E]
        if aff:
            return max(aff, key=lambda a: (2 if own[a] == op else 1) * w[a])
        return None
    reserve = P["reserve"] if contact else 0
    def ok(a):
        if own[a] == op:
            return cost(a) <= E
        return cost(a) <= E - (reserve if left > 3 else 0) or (E >= 9 and cost(a) <= E)
    afford = [(r, a) for r, a in scored if ok(a) and r > 0]
    if P["end_plan"] and 1 < left <= P["end_n"]:
        caps = [a for a in cand if own[a] == op]
        if caps:
            tgt = max(caps, key=lambda a: 2 * w[a] - 0.01 * cost(a))
            need = cost(tgt)
            # energy at my final decision if I rest from now on
            rests_left = left - 1
            def final_E(e_after):
                return min(9, e_after + 3 * (rests_left - 0))
            def final_E_after_spend(sp):
                e = E - sp
                return min(9, e + 3 * (rests_left - 1))
            afford = [(r, a) for r, a in afford if final_E_after_spend(cost(a)) >= need]
    if not afford:
        return None
    best_r, best_a = scored[0]
    ar, aa = afford[0]
    if E >= 7:
        return aa
    if best_r > ar * P["rest_margin"] and cost(best_a) <= E + 3:
        return None
    return aa
