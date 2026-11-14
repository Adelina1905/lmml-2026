# Solving — The Braided River Run (New Zealand)

The route is found with a shortest-path search (Dijkstra).

1. **Describe where you are** as: position (row, col), which bridges are down, and the minute mod 16 (the tide phase).
2. **Start** at `S` at minute 0, with only bridge `I` down.
3. **Always continue from the earliest minute found so far.** From there try every action `U D L R W`:
   - skip rock, deep water, leaving the map, and raised bridges;
   - skip a ford if you would arrive at high tide (minute mod 16 is 5 or more);
   - add the action's time (1 minute, 3 onto swamp, +1 per height level climbed, 1 for a wait);
   - stepping onto a winch flips its bridges.
4. **Remember only the fastest way** to reach each situation.
5. **Stop when `G` is reached.** Because the search always takes the earliest minute first, this is the fastest possible time.
6. **Walk back** from `G` to `S` to get the moves and write them to `solution.txt`.

Result: 293 minutes (target ≤ 296).
