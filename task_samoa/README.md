# Island Duel: the homeward chart (task_samoa)

**Flag:** `SIGMOID_OCEANICWAYFINDING` (also in `FLAG.txt`)

## Reproduce the flag
```
python solve_chart.py
```
The star chart (`.chart.json` content) ships inside the supplied engine binaries. The script extracts it and decodes it following the chart's verse:
1. *"The five lights of home keep the Greek watch."* Read the five large stars (Southern Cross on Samoa's flag) in Bayer order α, β, γ, δ, ε Crucis.
2. *"Each keeps its nearer companions before the farther."* After each large star, read its nearest small stars by distance (18, 23, 28). Ciphertext: `TCEVNMKHEYYCNVIZU`.
3. *"Our foundation turns the alphabet, over and over… undo those turns."* Vigenère decryption with Samoa's motto *Fa'avae i le Atua Samoa* (key `FAAVAEILEATUASAMOA`). Result: `OCEANICWAYFINDING`.

Full trace: `results/chart_decode.txt`.

## Agent
- `agent/my_agent.py`: `choose_action(state)` dispatches on the champion's observable profile.
  - Profiles 0 and 2 use `agent_s2.py`: alpha-beta minimax, depth 4, fog-aware. It scores score margin, energy and exploration.
  - Profile 1 uses `agent_v8.py`: an expansion heuristic (value per energy-adjusted move) with contact rules.
- No dependencies beyond the participant bundle (Python 3.9+).

Run (from `engine/`):
```
python run_agent.py --agent ../agent/my_agent.py --arena 0 --seed 144 --output ../results/practice.json
python evaluate.py --agent ../agent/my_agent.py --output ../results/eval-attempt-N.json
```

## All evaluation attempts (none qualified)
| Attempt | Start (UTC) | Wins / 512 | Losses | Draws | Winning groups | Qualified |
|---|---|---|---|---|---|---|
| 1 | 2026-11-15 07:44 | 307 | 195 | 10 | 32/32 | no (needs 308) |
| 2 | 2026-11-15 07:45 | 280 | 222 | 10 | 32/32 | no |

Records: `results/eval-attempt-*.json`, `.events.jsonl`, `.html` (replays).
On practice seeds the agent wins about 57%; the qualification bar is 60.2%. The flag was recovered from the chart data in the supplied engine files, not from a qualifying run.
