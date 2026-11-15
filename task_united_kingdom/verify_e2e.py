"""
Independent cross-check (NOT used by the metric in solve.py).

The venue cards and texts come from the public E2E NLG Challenge dataset
(Novikova, Dusek & Rieser, 2017: crowd-sourced restaurant descriptions).
Every reference in references.csv is a verbatim E2E description, so a panel
text that is NOT a verbatim E2E description must have been altered by its writer.

This script downloads the E2E files (or reads them from ./e2e/ if present) and
  1. counts, per deck, how many of the 40 texts are untouched E2E descriptions;
  2. compares the listener check (results/card_checks.csv, from solve.py) with
     the "altered" label card by card.

    python solve.py        # first
    python verify_e2e.py   # needs internet once (raw.githubusercontent.com)
"""
from __future__ import annotations

import re
import urllib.request
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
E2E_DIR = HERE / "e2e"
BASE = "https://raw.githubusercontent.com/tuetschek/e2e-dataset/master/"
FILES = ["trainset.csv", "devset.csv", "testset_w_refs.csv"]


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", str(s)).strip()


def load_e2e() -> set[str]:
    E2E_DIR.mkdir(exist_ok=True)
    texts = set()
    for f in FILES:
        path = E2E_DIR / f
        if not path.exists():
            print("downloading", BASE + f)
            urllib.request.urlretrieve(BASE + f, path)
        texts |= set(pd.read_csv(path).ref.map(norm))
    return texts


def main() -> None:
    e2e = load_e2e()
    refs = pd.read_csv(HERE / "dataset" / "references.csv")
    cards = pd.read_csv(HERE / "results" / "card_checks.csv")
    out = []
    out.append(f"references that are verbatim E2E descriptions: {refs.reference.map(norm).isin(e2e).mean():.1%}")
    cards["untouched"] = cards.text.map(norm).isin(e2e)
    tab = cards.pivot_table(index="round", columns="seat", values="untouched", aggfunc="sum")
    honest_by_e2e = tab.idxmax(axis=1)
    out.append("\nUntouched E2E descriptions per deck (out of 40):")
    out.append(tab.to_string())
    chosen = cards.groupby(["round", "seat"]).passes.mean().unstack().idxmax(axis=1)
    out.append("\nseat with 40/40 untouched texts : " + str(list(honest_by_e2e)))
    out.append("seat chosen by the listener check: " + str(list(chosen)))
    out.append("agree in every round: " + str(bool((honest_by_e2e == chosen).all())))

    altered = ~cards.untouched
    flagged = ~cards.passes
    tp = int((altered & flagged).sum())
    out.append(f"\nCard level, all 2,880 cards: {int(altered.sum())} altered texts, {int(flagged.sum())} flagged by the listener check")
    out.append(f"  precision (flagged cards that really were altered): {tp / flagged.sum():.1%}")
    out.append(f"  recall    (altered cards that got flagged)        : {tp / altered.sum():.1%}")
    out.append(f"  untouched texts flagged by mistake                : {int((~altered & flagged).sum())} of {int((~altered).sum())}")
    report = "\n".join(out)
    print(report)
    (HERE / "results" / "e2e_verification.txt").write_text(report + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
