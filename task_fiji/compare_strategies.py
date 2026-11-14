"""Local experiments only; this file is not part of the submission."""

import json
import math
import random
import statistics
from dataclasses import replace
from pathlib import Path

from reef import load_ocean, play_all
from starter_main import Strategy as Greedy


class RandomStrategy(Greedy):
    def __init__(self, n_spots, n_casts):
        super().__init__(n_spots, n_casts)
        self.rng = random.Random(0)

    def choose(self):
        return self.rng.randrange(self.n_spots)


class ExploreThenCommit(Greedy):
    def choose(self):
        for spot in range(self.n_spots):
            if self.casts[spot] < 15:
                return spot
        return super().choose()


class EpsilonGreedy(Greedy):
    def __init__(self, n_spots, n_casts):
        super().__init__(n_spots, n_casts)
        self.rng = random.Random(0)

    def choose(self):
        for spot in range(self.n_spots):
            if self.casts[spot] < 10:
                return spot
        if self.rng.random() < 0.1:
            return self.rng.randrange(self.n_spots)
        return super().choose()


class UCB(Greedy):
    coefficient = 5.0

    def choose(self):
        for spot in range(self.n_spots):
            if self.casts[spot] == 0:
                return spot
        log_t = math.log(sum(self.casts) + 1)
        return max(
            range(self.n_spots),
            key=lambda s: self.total[s] / self.casts[s]
            + self.coefficient * math.sqrt(log_t / self.casts[s]),
        )


class UCBConservative(UCB):
    coefficient = 8.0


def main():
    ocean = load_ocean(Path(__file__).with_name("train-output.json"))
    # Fresh seeds check sensitivity to the published days; the reef stays the same.
    fresh = replace(ocean, seeds=tuple(range(10001, 10301)))
    report = {}
    for strategy in (Greedy, RandomStrategy, ExploreThenCommit, EpsilonGreedy,
                     UCB, UCBConservative):
        row = {}
        for label, dataset in (("practice", ocean), ("fresh_seeds", fresh)):
            totals = play_all(strategy, dataset)
            row[label] = {
                "days": len(totals),
                "average": round(statistics.mean(totals), 2),
                "min": round(min(totals), 2),
                "max": round(max(totals), 2),
            }
        report[strategy.__name__] = row
        print(strategy.__name__, json.dumps(row), flush=True)
    Path(__file__).with_name("comparison_results.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
