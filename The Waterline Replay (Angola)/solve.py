"""Find the trace item that caused the wrong water-status answer.

Usage: python solve.py test-input.json
Writes answer.txt next to the input file.
"""
import json, sys
from collections import defaultdict
from pathlib import Path

path = Path(sys.argv[1])
data = json.loads(path.read_text(encoding="utf-8"))
runs = defaultdict(list)
for e in data["events"]: runs[e["run_id"]].append(e)

# 1. Find the run whose answer contradicts the live tool result
bad_run = None
for rid, evs in runs.items():
    status = next(e["result"]["status"] for e in evs if e["event"] == "tool.result")
    answer = next(e["text"] for e in evs if e["event"] == "assistant.response")
    wrong = (status == "RESTORED") == answer.lower().startswith("no")
    print(f"{rid:18} tool={status:9} answer={answer!r} -> {'WRONG' if wrong else 'ok'}")
    if wrong: bad_run = rid

# 2. Find what the model received in the bad run that the good runs did not
def sources(rid):
    req = next(e for e in runs[rid] if e["event"] == "llm.request")
    return {m["source_id"]: m for m in req["messages"] if "source_id" in m}

good = set().union(*(sources(r) for r in runs if r != bad_run))
culprits = [sid for sid in sources(bad_run) if sid not in good]

answer = culprits[0]

# 3. Print proof: the culprit is an expired memory note
mem = next(e for e in runs[bad_run] if e["event"] == "memory.result" and e["item"]["source_id"] == answer)
print(f"\nCulprit {answer}: valid_until={mem['item']['valid_until']}, used at {mem['timestamp']}")
print(f"  text: {mem['item']['text']}")

# 4. Save the answer
(path.parent / "answer.txt").write_text(answer + "\n", encoding="utf-8")
print(f"\nFlag: {answer}  (saved to answer.txt)")
