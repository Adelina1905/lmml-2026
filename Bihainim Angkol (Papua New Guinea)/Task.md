# Task Overview

## 📖 Storyline

On the shore of Kimbe Bay, in West New Britain, the village canoes go out at first light and are pulled back up the beach before dark. Your **angkol** (uncle) has fished these waters from his outrigger canoe for thirty years. He is steady and careful: he works the seagrass flats inside the reef and always has the canoe home with time to spare.

Last season your cousin rode along and kept a logbook of 400 days: where the canoe was, how many fish were aboard, what angkol did next, and what was landed on the beach.

Angkol is putting down his paddle. The family wants the canoe's next captain to be an agent that first learns to fish like him, and then brings home **more** than he ever did, because his catch has not grown in years.

## 📌 Task Title

Bihainim Angkol (Follow Your Uncle)

## 🔥 Difficulty

Medium

## 🎯 Objective

Learn from angkol's logbook, then improve on him: build an agent that lands **at least 1.5×** as many fish per day as angkol, on days you have not seen. Imitating him alone is not enough.

## 🚩 FLAG

`SIGMOID_<WORD>`. It is released automatically when your `main.py` passes the evaluation.

## 📂 Task Input

### The lagoon
```
    0 1 2 3 4 5 6
 0  . . . . T T .        H  village beach: fish are landed here
 1  # # # # ~ # #        s  seagrass flats      T  outer reef drop-off
 2  . . . . s s .        #  barrier reef (no canoe crosses it)
 3  . . . . s s .        ~  reef passage        .  open water
 4  . . . . . . .
 5  . . H . . . .

```

- A day is **24 steps**. It starts on the beach (`H`) with an empty hold.
- A **state** is `(row, col, hold, steps_left)`: row 0 is the open sea, row 5 the shore; `hold` is 0–6 fish; `steps_left` counts down from 24 to 1.
- **Actions:** `NORTH`, `SOUTH`, `EAST`, `WEST` move one cell; moving into the reef or off the map leaves the canoe where it is. `FISH` on a fishing ground (`s` or `T`) with fewer than 6 fish aboard adds one fish, with a chance that depends on the ground. Anywhere else it catches nothing. `WAIT` does nothing.
- **Tide:** a canoe that ends a step on the passage (`~`) may be pushed one cell south.
- **Landing:** whenever the canoe is on the beach, everything in the hold is landed. The day's score is the number of fish landed. Fish still aboard when the day ends are lost.
- Every chance event comes from the day's seed, so the same seed and the same decisions always give the same day.

`lagoon.py` holds every rule and constant above, and can fish as many days as you like.

### Files

Download: [task_papua_new_guinea-participant.zip](https://drive.google.com/drive/folders/1bwIZTzYD5i5H26CWYzCm3kgDBlQ-qtF2?usp=sharing)

| **File**              | **Content**                                                                                                                                                                |
| :-------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `lagoon.py`           | The simulator (`Lagoon(seed)`, `reset()`, `step(action)`) and helpers to play days, read and write policies, and export `main.py`                                          |
| `dataset/logbook.csv` | Angkol's 400 days, one row per decision: `day,row,col,hold,steps_left,action,reward` (`reward` = fish landed by that step)                                                 |
| `train-input.csv`     | Practice input: every state of the lagoon, `row,col,hold,steps_left`                                                                                                       |
| `train-output.json`   | Practice days (the logbook's days) and angkol's catch on each                                                                                                              |
| `main.py`             | Submission template: reads `test-input.csv` and writes a valid `test-output.csv`. Put your agent in `choose_action`, or replace the file with `lagoon.export_main(policy)` |

## ✅ Expected Output

Submit a single file, **`main.py`**:

- It reads **`test-input.csv`** (same columns as `train-input.csv`) from its working directory, by bare filename.
- It writes **`test-output.csv`** into the same directory: header `row,col,hold,steps_left,action`, one row for every row of `test-input.csv`, and `action` one of `NORTH`, `SOUTH`, `EAST`, `WEST`, `FISH`, `WAIT`.
- It runs alone, next to `test-input.csv` only, so your policy must be inside the file. `lagoon.export_main(policy)` writes such a `main.py` for you.
- Python 3.12, standard library and `numpy==2.2.6`, no internet, 60 seconds.

To check it on the practice days, run it on the practice input and compare with angkol:
```bash
cp train-input.csv test-input.csv
python main.py
python lagoon.py --policy test-output.csv --compare train-output.json

```

## 🏆 Scoring Criteria

- ✅ Average catch **≥ 1.5×** angkol's, on the same days and under the same tide and bites: 200 points (PASS), and the flag is released.
- ❌ Below 1.5×: 0 points (FAIL).
- ❌ `main.py` crashes, runs out of time, or `test-output.csv` misses a state, repeats one or uses an unknown action: 0 points (FAIL).
