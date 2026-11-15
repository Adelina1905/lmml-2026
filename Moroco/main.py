import csv
import re

INPUT_FILE = "test-input.csv"
OUTPUT_FILE = "routing_log.csv"

CHEAP_COST = 1
EXPENSIVE_COST = 25
TARGET_COST_PER_REQUEST = 19.0

FALLBACK_ORDER = ["takings-fee", "board-indirect", "stall-pause"]


def shortest_walk(prompt):
    """Return (shortest path length, direct edge length or None) for a walk question."""
    edges = re.findall(r"([A-Za-z]+)-([A-Za-z]+) (\d+) minutes", prompt)
    match = re.search(r"from ([A-Za-z]+) to ([A-Za-z]+)", prompt)
    if not edges or not match:
        return None, None
    start, goal = match.groups()
    graph = {}
    for a, b, w in edges:
        w = int(w)
        graph.setdefault(a, {})
        graph.setdefault(b, {})
        graph[a][b] = min(w, graph[a].get(b, w))
        graph[b][a] = min(w, graph[b].get(a, w))
    if start not in graph or goal not in graph:
        return None, None
    # Dijkstra on a handful of nodes: a simple scan is enough.
    dist = {node: float("inf") for node in graph}
    dist[start] = 0
    done = set()
    while len(done) < len(graph):
        node = min((n for n in graph if n not in done), key=lambda n: dist[n])
        done.add(node)
        for nxt, w in graph[node].items():
            if dist[node] + w < dist[nxt]:
                dist[nxt] = dist[node] + w
    return dist[goal], graph[start].get(goal)


def classify(prompt):

    if "board lists" in prompt:
        if "sells whatever" in prompt:
            return "board-indirect", "expensive"
        return "board-direct", "cheap"

    if "Walking times" in prompt:
        best, direct = shortest_walk(prompt)
        if best is not None and (direct is None or best < direct):
            return "walk-detour", "cheap"
        return "walk-direct", "expensive"
    
    crates = re.search(r"loads (\d+) crates of .*? with (\d+) ", prompt)
    if crates:
        if "in total" in prompt:
            count, per_crate = map(int, crates.groups())
            if count <= 12 or per_crate == 10:
                return "crates-total-easy", "cheap"
            return "crates-total-hard", "expensive"
        return "crates-remain", "expensive"


    if "gate charges" in prompt:
        if "heavier" in prompt:
            return "toll-heavy", "expensive"
        flat = re.search(r"charges (\d+) dirham for every cart", prompt)
        if flat and int(flat.group(1)) == 10:
            return "toll-x10", "cheap"
        return "toll-flat", "expensive"

    if "stall opens" in prompt:
        return ("stall-pause" if "pauses" in prompt else "stall-plain"), "expensive"

    if "takings" in prompt:
        return ("takings-fee" if "stall fee" in prompt else "takings-plain"), "expensive"


    return "unknown", "expensive"


def total_cost(routes):
    return sum(CHEAP_COST if guide == "cheap" else EXPENSIVE_COST for _, _, guide in routes)


def apply_budget_valve(routes):
    """Move expensive requests to cheap, in FALLBACK_ORDER, until under target."""
    budget = TARGET_COST_PER_REQUEST * len(routes)
    if total_cost(routes) <= budget:
        return routes
    routes = list(routes)
    for kind in FALLBACK_ORDER:
        for i, (rid, k, guide) in enumerate(routes):
            if total_cost(routes) <= budget:
                return routes
            if k == kind and guide == "expensive":
                routes[i] = (rid, k, "cheap")
    return routes


def main():
    with open(INPUT_FILE, "r", encoding="utf-8-sig", newline="") as handle:
        requests = list(csv.DictReader(handle))

    routes = []
    for row in requests:
        kind, guide = classify(row.get("prompt") or "")
        routes.append((row["request_id"], kind, guide))

    routes = apply_budget_valve(routes)

    with open(OUTPUT_FILE, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["request_id", "model"])
        for rid, _, guide in routes:
            writer.writerow([rid, guide])


if __name__ == "__main__":
    main()
