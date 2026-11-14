# Solving Bihainim Angkol (Papua New Guinea)

**Passed.** The grader scored 5.968 fish per day against angkol's 2.558, a ratio of 2.333× (target 1.5×). Flag: `SIGMOID_ELIGIBILITY`.

## How it was solved

1. **Read the rules in `lagoon.py`.** The map never changes; only the luck (bites and tide) changes from day to day. Seagrass `s` bites 20% of the time, the outer reef `T` 90%. The hold carries 6 fish. The passage `~` pushes the canoe south 40% of the time.
2. **Found angkol's weakness in the logbook.** He only fishes the seagrass, so he lands about 2.5 fish per day.
3. **Computed the exact optimum** by working backwards from the last step and picking the action with the best expected catch in every state. That gives 5.96 fish per day: go to `T`, fill the hold, come home on time. The 6-fish hold is the real ceiling.
4. **Wrote that strategy as four plain rules in `main.py`.** They score the same as the optimum (5.963 on the practice days):
   1. Go home if the hold is full or the steps left equal the trip home.
   2. Fish when on `T`.
   3. Head for `T` at (0,4) if there is time to get there, fish and return. If the tide pushes you back, try again.
   4. Otherwise fish the seagrass if there is still time, else go home.

   Moves come from breadth-first-search distances on the fixed map.
