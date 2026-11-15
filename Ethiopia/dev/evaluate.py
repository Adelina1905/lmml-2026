"""Score main.detect_splice on the training file + every simulated case.

A case passes like the real checker: |start error| <= 1.5 s and |end error| <= 1.5 s.

Run from the Ethiopia folder (after dev/make_sims.py):  python dev/evaluate.py
"""
import contextlib
import csv
import io
import sys
import time
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent))
import main as M  # noqa: E402

TOLERANCE = 1.5


def main():
    with open(HERE / "sims" / "truth.csv", newline="") as f:
        cases = list(csv.DictReader(f))

    per_kind = defaultdict(list)
    t0 = time.time()
    for case in cases:
        x, sr = M.load_audio(case["path"])
        log = io.StringIO()
        with contextlib.redirect_stderr(log):          # keep main.py's debug lines out of the table
            start, end = M.detect_splice(x, sr)
        winner = getattr(M, "LAST_WINNER", "-")
        s_err = abs(start - float(case["start_s"]))
        e_err = abs(end - float(case["end_s"]))
        ok = s_err <= TOLERANCE and e_err <= TOLERANCE
        per_kind[case["kind"]].append(ok)
        print(f"{'ok ' if ok else 'BAD'} {case['name']:18s} true {float(case['start_s']):6.2f}-{float(case['end_s']):6.2f}"
              f"  pred {start:6.2f}-{end:6.2f}  err {s_err:5.2f}/{e_err:5.2f}  view {winner}")

    print("\nper anomaly type:")
    for kind, oks in per_kind.items():
        print(f"  {kind:16s} {sum(oks)}/{len(oks)}")
    total = [ok for oks in per_kind.values() for ok in oks]
    print(f"\nTOTAL {sum(total)}/{len(total)} passed   ({(time.time() - t0) / len(total):.1f} s per file)")


if __name__ == "__main__":
    main()
