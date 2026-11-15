# Task Overview

## 📖 Storyline

Dumont d'Urville sits on a scrap of rock off the coast of Adélie Land. The French have kept a station since 1956. It is a nice and warm (on Antarctica scale, of course) place in Antarctica. The coast is gentler than the plateau behind it: there are even mosses and lichens on the islands, snow petrels in the cliffs, and every autumn several thousand emperor penguins walk in over the new sea ice to lay their eggs within sight of the station's windows. That is why the biologists are there, and why they need the weather.

Every three hours, day and night, since the first wintering, someone at the station has read the instruments and sent the observation out by radio. This is one wintering's worth of those reports, from the end of March, when the penguins arrive, to early August, when the first chicks hatch on their fathers' feet. The biologist who asked for them wants to lay the colony's worst nights against the temperature curve.

The trouble is that the link was bad all winter, and some reports reached the other end changed. The radio operator kept a wall chart of the ones that could not be trusted. The chart did not survive the wintering. The reports did.

## 📌 Task Title

Look, a penguin! | Antarctica Level 1

## 🔥 Difficulty

Easy

## 🎯 Objective

Decode the messages, find every corrupted one, and redraw the operator's wall chart: days across, the eight observation hours (00, 03, ..., 21 UTC) down, one point for every corrupted message. Only then, the flag might be found.

## 🚩 FLAG

`SIGMOID_<WORD>`, where `<WORD>` can be retrieved only when the task is solved correctly and completely.

## 📂 Task Input

The provided [data](https://drive.google.com/drive/folders/1eFktkwt_Vg_JsepwNu6IhjxFaKA7zr8A?usp=sharing) contains a **`messages.txt`** file, with one line per observation, in chronological order:
```php-template
<id> <date> <hour>Z <message>=
0001 1987-03-07 00Z AAXX 07001 89624 41684 21102 11108 21167 39743 49798 58002 70211 81530 333 11098 21118=

```

- `id`: four-digit sequence number.
- `date`, `hour`: the observation time (UTC). The calendar is rebased and the year label is fictional, but the hour of day and the day-to-day sequence are the real ones.
- `message`: the FM-12 SYNOP text, ending with `=`.

**`synop_format.md`**: a description of every group this station uses, with units and code tables. The decoding itself is not the challenge, so everything needed to decode is in that file. Any SYNOP library may be used as well.

Some groups are reported only at certain hours, or only when there is something to report (clouds, present weather, gusts, 12-hour extremes). A group that is simply absent is **not** corruption.

## ✅ Expected Output

Submit on the platform:

The flag, `SIGMOID_<WORD>`. A URL to the script, proving the work done to retrieve the flag.

## 🏆 Scoring Criteria

- ✅ Correct flag and a write-up whose retrieves the flag: 150 points (PASS).
- ❌ Incorrect flag, or a correct flag without code and a chart that support it: 0 points (FAIL).
- ❌ No accepted submission within the time limit: 0 points (FAIL)
