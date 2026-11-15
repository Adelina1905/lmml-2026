#!/usr/bin/env python3
"""The Dispatch Desk of Fez: route each request to the cheap or the expensive guide.

Reads test-input.csv (request_id,prompt) from the working folder and writes
routing_log.csv (request_id,model) with model = cheap | expensive.
Standard library only, runs offline.

Idea
----
Every request is one of ten fixed templates (board price, walk, stall hours, ...).
main.py parses the request and solves it itself, which gives a few structural
features: is the board lookup direct or through "sells whatever X sells", is the
direct path in the walk already the shortest, are the crate numbers small, ...
For each (template, feature) group the training set tells us how often each
guide was right. The expected gain of paying for the master guide is

    gain = P(expensive right) - P(cheap right)

Requests are sorted by gain and the expensive guide gets the top ones while the
gain is positive and the total cost stays within the budget
(20 dirham per request, at most 10,000). Everything else goes to the apprentice.
"""

import csv
import heapq
import re

INPUT = "test-input.csv"
OUTPUT = "routing_log.csv"
COST_CHEAP, COST_EXP = 1, 25
BUDGET_PER_REQUEST = 20
BUDGET_CAP = 10000

# (template, feature) -> (n, cheap correct, expensive correct), counted on the
# 400 training requests (see solve.py, which regenerates this table).
STATS = {
    "board|direct":          (45, 45, 45),
    "board|indirect":        (22, 12, 22),
    "crates|easy":           (24, 22, 24),
    "crates|hard":           (19, 8, 19),
    "crates_sell":           (24, 0, 24),
    "stall":                 (37, 7, 35),
    "stall_pause":           (29, 2, 19),
    "takings":               (39, 6, 39),
    "takings_fee":           (28, 7, 15),
    "toll":                  (39, 6, 39),
    "toll_heavy":            (27, 0, 25),
    "walk|direct_shortest":  (37, 19, 33),
    "walk|detour_shorter":   (30, 1, 3),
}
# Unknown wording: fall back to the overall training rates (cheap 135/400,
# expensive 342/400), so an unrecognised request goes to the master guide.
DEFAULT = (400, 135, 342)


# ----------------------------------------------------------------- templates
def kind(p):
    if "gate charges" in p:
        return "toll_heavy" if "heavier" in p else "toll"
    if "stall opens" in p:
        return "stall_pause" if "pauses" in p else "stall"
    if "Walking times" in p:
        return "walk"
    if "takings" in p:
        return "takings_fee" if "fee" in p else "takings"
    if "board lists" in p:
        return "board"
    if "loads" in p and "crates" in p:
        return "crates_sell" if " sells " in p else "crates"
    return "other"


def shortest_walk(p):
    """(shortest distance, direct edge length) between the two named souks."""
    edges = re.findall(r"([A-Z][a-z]+)-([A-Z][a-z]+) (\d+) minutes", p)
    src, dst = re.search(r"from ([A-Z][a-z]+) to ([A-Z][a-z]+)", p).groups()
    graph = {}
    direct = None
    for a, b, w in edges:
        w = int(w)
        graph.setdefault(a, []).append((b, w))
        graph.setdefault(b, []).append((a, w))
        if {a, b} == {src, dst}:
            direct = w if direct is None else min(direct, w)
    dist = {src: 0}
    heap = [(0, src)]
    while heap:
        d, u = heapq.heappop(heap)
        if d > dist[u]:
            continue
        for v, w in graph.get(u, []):
            if d + w < dist.get(v, float("inf")):
                dist[v] = d + w
                heapq.heappush(heap, (d + w, v))
    return dist.get(dst), direct


def group(p):
    """The (template, feature) group a request belongs to."""
    k = kind(p)
    if k == "board":
        indirect = re.search(r"stall sells whatever", p) is not None
        return "board|indirect" if indirect else "board|direct"
    if k == "walk":
        best, direct = shortest_walk(p)
        if best is not None and direct is not None and direct == best:
            return "walk|direct_shortest"
        return "walk|detour_shorter"
    if k == "crates":
        n, per = map(int, re.search(r"loads (\d+) crates of [a-z ]+? with (\d+)", p).groups())
        easy = n <= 12 or n % 10 == 0 or per % 10 == 0
        return "crates|easy" if easy else "crates|hard"
    return k


def probabilities(p):
    """Laplace-smoothed P(cheap right), P(expensive right) for a request."""
    try:
        n, c, e = STATS.get(group(p), DEFAULT)
    except Exception:  # unexpected wording: treat as unknown
        n, c, e = DEFAULT
    return (c + 1) / (n + 2), (e + 1) / (n + 2)


# ------------------------------------------------------------------- routing
def route(prompts, budget=None):
    """Return a list of 'cheap' / 'expensive', one per prompt."""
    n = len(prompts)
    if budget is None:
        budget = min(BUDGET_CAP, BUDGET_PER_REQUEST * n)
    # largest k with COST_EXP*k + COST_CHEAP*(n-k) <= budget
    max_exp = max(0, (budget - COST_CHEAP * n) // (COST_EXP - COST_CHEAP))
    gains = []
    for i, p in enumerate(prompts):
        pc, pe = probabilities(p)
        gains.append((pe - pc, i))
    ranked = sorted((g for g in gains if g[0] > 0), key=lambda g: (-g[0], g[1]))
    chosen = {i for _, i in ranked[:max_exp]}
    return ["expensive" if i in chosen else "cheap" for i in range(n)]


def main():
    with open(INPUT, "r", encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    ids = [r["request_id"].strip() for r in rows]
    prompts = [r.get("prompt") or "" for r in rows]
    models = route(prompts)
    with open(OUTPUT, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["request_id", "model"])
        for rid, m in zip(ids, models):
            w.writerow([rid, m])


if __name__ == "__main__":
    main()
