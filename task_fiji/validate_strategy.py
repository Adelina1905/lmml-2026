"""Additional local validation. No hidden evaluation data is used."""

import ast
import json
import random
import statistics
from dataclasses import replace
from pathlib import Path

from main import Strategy
from reef import load_ocean, play_all


def without_strategy(source):
    tree = ast.parse(source)
    tree.body = [node for node in tree.body
                 if not (isinstance(node, ast.ClassDef) and node.name == "Strategy")]
    return ast.dump(tree)


def main():
    root = Path(__file__).resolve().parent
    assert without_strategy((root / "main.py").read_text()) == without_strategy(
        (root / "starter_main.py").read_text()
    ), "Only the Strategy class may change"
    agent = Strategy(10, 1000)
    for spot in range(10):
        assert agent.choose() == spot
        agent.update(spot, 0.0)
    assert 0 <= agent.choose() < 10
    assert agent.steps == 10
    assert Strategy(10, 1000).steps == 0
    # A single available spot and a budget shorter than initial exploration.
    for n_spots, n_casts in ((1, 20), (10, 3)):
        agent = Strategy(n_spots, n_casts)
        for _ in range(n_casts):
            spot = agent.choose()
            assert 0 <= spot < n_spots
            agent.update(spot, 0.0)

    ocean = load_ocean(root / "train-output.json")
    report = {}
    # Deliberately alter distributions, not only the day seeds.
    # These are synthetic robustness checks, not the hidden evaluation reef.
    for scenario in range(4):
        rng = random.Random(8000 + scenario)
        spots = []
        for spot in ocean.spots:
            if spot.kind == "steady":
                spots.append(replace(spot, mean=spot.mean * rng.uniform(0.8, 1.2),
                                     sd=spot.sd * rng.uniform(0.8, 1.2)))
            else:
                spots.append(replace(spot, p=spot.p * rng.uniform(0.8, 1.2),
                                     big=spot.big * rng.uniform(0.9, 1.1),
                                     big_sd=spot.big_sd * rng.uniform(0.8, 1.2)))
        variant = replace(ocean, spots=tuple(spots),
                          seeds=tuple(range(20001, 20101)))
        totals = play_all(Strategy, variant)
        row = {"days": len(totals), "average": round(statistics.mean(totals), 2),
               "min": round(min(totals), 2), "max": round(max(totals), 2)}
        report[f"synthetic_reef_{scenario + 1}"] = row
        print(f"synthetic_reef_{scenario + 1}: {row}", flush=True)
    (root / "robustness_results.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print("Protocol preservation and strategy state checks passed.")


if __name__ == "__main__":
    main()
