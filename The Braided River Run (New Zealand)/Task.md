# Task Overview

## 📖 Storyline

A braided river runs through a gorge in New Zealand's South Island, cutting the valley into terraces. To get from the hut at the top to the hut at the bottom you cross five river channels — over swing bridges, most of them raised, or through fords that flood with the tide. The winches that lower the bridges are tucked away in side alleys, and some move more than one bridge. On the terraces, swamps fill the low ground and hills rise between them.

The park ranger's best time is **297 minutes**. Beat it.

## 📌 Task Title

The Braided River Run

## 🔥 Difficulty

Hard

## 🎯 Objective

Find the moves that take you from the start `S` to the goal `G` in **296 minutes or fewer** without breaking a rule. Moves take different times and the tide punishes bad timing, so build a search or learning agent on top of the simulator you are given. Nothing is random: the same moves always give the same result.

## 🚩 FLAG

`SIGMOID_<WORD>`, shown by the checker when your route is valid and takes fewer than 297 minutes. Enter it exactly as shown.

## 📂 Task Input

The provided [materials](https://drive.google.com/drive/folders/1oLxH-OAoWxqsFz9UeW6VEdg6Q-6dzGLj?usp=sharing) consist od the following:

| **File**                             | **What it is**                                                                                   |
| :----------------------------------- | :----------------------------------------------------------------------------------------------- |
| `test-input.txt`, `test-input.svg`   | the valley you must cross (47 × 61 cells), and a picture of it                                   |
| `train-input.txt`, `train-input.svg` | a smaller practice valley with the same rules, and its picture                                   |
| `train-output.txt`                   | the fastest route through the practice valley (114 minutes)                                      |
| `environment.py`                     | the simulator: replays and times a route, offers an agent interface (`ValleyEnv`) and draws maps |

Map symbols: `S` / `G` huts, `.` ground, `,` swamp, `#` rock, `=` deep water, `~` ford, `A` `B` … swing bridges, `a` `b` … winches.

1. Each action — `U`, `D`, `L`, `R` or `W` (wait) — takes time, starting from minute 0. A move takes 1 minute, 3 onto a swamp, plus 1 for every height level it climbs; a wait takes 1. Your time is the minute you reach `G`.
2. A bridge is passable only while it is down; bridges start up unless the map's `open` line lists them. Stepping onto a winch flips every bridge listed for it (waiting on it does not).
3. An action may end on a ford only at a minute that leaves a remainder of 0 to 4 when divided by 16 (minutes 1–4, 16–20, 32–36, …), whether it steps onto the ford or waits on it. Stepping off a ford is always allowed.
4. A move into rock, deep water, a raised bridge, a ford at high tide or off the map makes the route invalid, and so do actions after reaching `G`.

Each map file lists these settings, then the grid after the line `map` and one height digit (0–5) per cell after the line `heights`.

What you need installed: Python 3.11 (3.10–3.12 accepted), standard library only.

## ✅ Expected Output

Upload **`main.py`**. The platform runs it for up to 2 minutes in a folder that holds `test-input.txt` and `environment.py`, so open the map by its name and `import environment` directly. It must write **`solution.txt`** there and exit with code 0. The solution is your moves as the letters `U` `D` `L` `R` `W`; case, spaces, line breaks and anything after a `#` are ignored, and any other character makes the route invalid.

Check a route before you upload: `python environment.py --map test-input.txt --replay-file solution.txt` prints `valid run: N minutes` or why the route is invalid, and `--render route.svg` draws it on the map. A correct solver finds 114 minutes on the practice valley.

## 🏆 Scoring Criteria

- ✅ A valid route that reaches `G` in 296 minutes or fewer: **300 points (PASS)**, flag released.
- ❌ Anything else — an invalid route, 297 minutes or more, a crash or a timeout: **0 points (FAIL)**. You may resubmit.
