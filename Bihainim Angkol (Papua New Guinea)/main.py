"""Bihainim Angkol: a simple, readable fishing agent.

Reads test-input.csv (row,col,hold,steps_left) from the current directory and
writes test-output.csv (row,col,hold,steps_left,action) next to it.

The plan, in plain words:
  1. Paddle to the outer reef T (90% bites), not the seagrass (20% bites).
  2. Fish there until the hold is full (6 fish).
  3. Go home in time: once the steps left equal the trip home, leave.
  4. If there is no time left for a reef trip, fish the seagrass if it is
     on the way, otherwise just go home.
"""

import csv
from collections import deque

ACTIONS = ("NORTH", "SOUTH", "EAST", "WEST", "FISH", "WAIT")

MAP = (
    ". . . . T T .",
    "# # # # ~ # #",
    ". . . . s s .",
    ". . . . s s .",
    ". . . . . . .",
    ". . H . . . .",
)
GRID = [line.split() for line in MAP]
ROWS, COLS = len(GRID), len(GRID[0])
MOVES = {"NORTH": (-1, 0), "SOUTH": (1, 0), "EAST": (0, 1), "WEST": (0, -1)}
HOLD_CAPACITY = 6

BEACH = (5, 2)
REEF_SPOT = (0, 4)                      # the T cell right above the passage
SEAGRASS = [(2, 4), (2, 5), (3, 4), (3, 5)]


def is_water(r, c):
    return 0 <= r < ROWS and 0 <= c < COLS and GRID[r][c] != "#"


def distances_to(goal):
    """Number of moves from every water cell to `goal` (breadth-first search)."""
    dist = {goal: 0}
    queue = deque([goal])
    while queue:
        r, c = queue.popleft()
        for dr, dc in MOVES.values():
            nxt = (r + dr, c + dc)
            if is_water(*nxt) and nxt not in dist:
                dist[nxt] = dist[(r, c)] + 1
                queue.append(nxt)
    return dist


TO_HOME = distances_to(BEACH)
TO_REEF = distances_to(REEF_SPOT)
TO_SEAGRASS = {cell: distances_to(cell) for cell in SEAGRASS}


def move_toward(row, col, dist):
    """The move that brings the canoe one step closer, using a distance map."""
    return min(MOVES, key=lambda m: dist.get((row + MOVES[m][0], col + MOVES[m][1]), 99))


def choose_action(row, col, hold, steps_left):
    here = (row, col)
    trip_home = TO_HOME[here]

    # Hold full, or exactly enough time left to get home: go home now.
    if hold == HOLD_CAPACITY or steps_left <= trip_home:
        return "WAIT" if here == BEACH else move_toward(row, col, TO_HOME)

    # On the outer reef with room and time: fish.
    if GRID[row][col] == "T":
        return "FISH"

    # Enough time to reach the reef, fish at least once and come back: go there.
    if steps_left >= TO_REEF[here] + 1 + TO_HOME[REEF_SPOT]:
        return move_toward(row, col, TO_REEF)

    # Not enough time for the reef: use the seagrass if it still pays.
    if GRID[row][col] == "s":
        return "FISH"
    cell = min(SEAGRASS, key=lambda s: TO_SEAGRASS[s][here])
    if steps_left >= TO_SEAGRASS[cell][here] + 1 + TO_HOME[cell]:
        return move_toward(row, col, TO_SEAGRASS[cell])

    # Nothing worth trying: head home with what we have.
    return "WAIT" if here == BEACH else move_toward(row, col, TO_HOME)


def main():
    with open("test-input.csv", newline="") as src, open("test-output.csv", "w", newline="") as dst:
        out = csv.writer(dst)
        out.writerow(["row", "col", "hold", "steps_left", "action"])
        for rec in csv.DictReader(src):
            state = (int(rec["row"]), int(rec["col"]), int(rec["hold"]), int(rec["steps_left"]))
            out.writerow([*state, choose_action(*state)])


if __name__ == "__main__":
    main()
