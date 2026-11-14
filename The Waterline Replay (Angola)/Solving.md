# Solving: The Waterline Replay (Angola)

**Flag:** `SIGMOID_MEM_KILAMBA`

## Steps

1. **Find the wrong run.** In `incident-01` the live tool says `RESTORED`, but the bot answers "No".
2. **Compare it with its replay.** `replay-no-memory` repeats the same question with the same settings and the same tool result, but memory is off, and it answers "Yes".
3. **Spot the only difference.** `incident-01` received an extra memory note in its `llm.request`: `source_id: SIGMOID_MEM_KILAMBA`, saying "NOT restored".
4. **Confirm that it's bad.** The note expired at 11:00 but was used at 14:05, so it was outdated.
5. **Submit its `source_id`** in `answer.txt`.
