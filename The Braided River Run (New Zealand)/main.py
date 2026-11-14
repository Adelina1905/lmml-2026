"""The Braided River Run -- exact Dijkstra solver.

State = (row, col, bridges-down bitmask, minute mod tide_period). Move costs do
not depend on the clock, only the tide phase does, so Dijkstra over this state
space gives the fastest legal route. Reads test-input.txt, writes solution.txt.
"""

import heapq
import sys

from environment import ACTIONS, load_map, simulate


def solve(valley):
    bridges = sorted(valley.bridges)
    bit = {b: 1 << i for i, b in enumerate(bridges)}
    winch_mask = {w: sum(bit[b] for b in targets) for w, targets in valley.winches.items()}
    period = valley.tide_period

    start = (valley.start[0], valley.start[1], sum(bit[b] for b in valley.initial_open), 0)
    best = {start: 0}
    prev = {}
    queue = [(0, start)]

    while queue:
        t, state = heapq.heappop(queue)
        if best[state] < t:
            continue
        row, col, mask, _ = state
        if (row, col) == valley.goal:
            break
        for action, (d_row, d_col) in ACTIONS.items():
            new_row, new_col = row + d_row, col + d_col
            if not valley.in_bounds(new_row, new_col):
                continue
            symbol = valley.cell(new_row, new_col)
            if symbol in "#=":
                continue
            new_t = t + valley.duration(row, col, new_row, new_col)
            if symbol == "~" and not valley.tide_is_low(new_t):
                continue
            if symbol in bit and not mask & bit[symbol]:
                continue
            new_mask = mask ^ winch_mask[symbol] if action != "W" and symbol in winch_mask else mask
            new_state = (new_row, new_col, new_mask, new_t % period)
            if new_t < best.get(new_state, float("inf")):
                best[new_state] = new_t
                prev[new_state] = (state, action)
                heapq.heappush(queue, (new_t, new_state))

    route = []
    while state != start:
        state, action = prev[state]
        route.append(action)
    return "".join(reversed(route))


def main():
    map_path = sys.argv[1] if len(sys.argv) > 1 else "test-input.txt"
    out_path = sys.argv[2] if len(sys.argv) > 2 else "solution.txt"
    valley = load_map(map_path)
    route = solve(valley)
    minutes = simulate(valley, route)[-1][0]
    with open(out_path, "w", encoding="utf-8") as handle:
        handle.write("# {} minutes, {} actions\n{}\n".format(minutes, len(route), route))
    print("{}: {} minutes, {} actions -> {}".format(map_path, minutes, len(route), out_path))


if __name__ == "__main__":
    main()
