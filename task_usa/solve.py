#!/usr/bin/env python3
"""
Runs main.py on the training and test samples, checks the training answer,
runs an offline self-test, and writes everything to results/.

Usage (from inside task_usa/):   python solve.py
"""
import contextlib
import io
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import main  # noqa: E402

RES = HERE / "results"
RES.mkdir(exist_ok=True)
report = io.StringIO()


def out(*a):
    print(*a)
    print(*a, file=report)


def run(inp, outp):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        r = main.solve(HERE / inp, outp)
    out(buf.getvalue().rstrip())
    return r


out("=" * 70)
out("1) TRAINING SAMPLE (known answer)")
out("=" * 70)
pred = run("train-input.json", RES / "train-prediction.json")
gold = json.loads((HERE / "train-output.json").read_text(encoding="utf-8"))
tok_ok = pred["tokenizer_id"].lower() == gold["tokenizer_id"].lower()
msg_ok = main.norm(pred["decoded_message"]) == main.norm(gold["decoded_message"])
out(f"tokenizer_id correct: {tok_ok}   decoded_message correct: {msg_ok}   "
    f"score: {50 * tok_ok + 50 * msg_ok}/100")

out("\n" + "=" * 70)
out("2) TEST SAMPLE")
out("=" * 70)
test = run("test-input.json", HERE / "test-output.json")
flag = next((w for w in test["decoded_message"].replace("\n", " ").split()
             if w.startswith("SIGMOID_")), None)
out(f"flag found in decoded message: {flag}")
(HERE / "FLAG.txt").write_text((flag or "") + "\n", encoding="utf-8")

out("\n" + "=" * 70)
out("3) OFFLINE SELF-TEST: encode both calibration texts with every candidate,")
out("   feed the pair back in, and check the right tokenizer is identified")
out("=" * 70)
texts = {
    "train-text": json.loads((HERE / "train-input.json").read_text(encoding="utf-8"))["calibration"]["text"],
    "test-text": json.loads((HERE / "test-input.json").read_text(encoding="utf-8"))["calibration"]["text"],
}
secret = "SECRET_CHECK  with  double spaces\n\tünïcödé — ¡hola! 12345"
all_ok = True
for name, fname, pat, sp in main.CANDIDATES:
    if name in ("p50k_edit", "o200k_harmony"):
        continue  # same vocabulary/regex as p50k_base / o200k_base -> indistinguishable on plain text
    encode, _ = main.build(name, fname, pat, sp)
    if encode is None:
        out(f"  {name}: tiktoken not installed, skipped")
        continue
    for tname, t in texts.items():
        d = {"calibration": {"text": t, "token_ids": encode(t)},
             "challenge": {"token_ids": encode(secret)}}
        p = RES / "_selftest.json"
        p.write_text(json.dumps(d), encoding="utf-8")
        r = main.solve(p, RES / "_selftest_out.json", verbose=False)
        ok = r["tokenizer_id"] == name and r["decoded_message"] == secret
        all_ok &= ok
        out(f"  generated with {name:12s} on {tname:10s} -> identified {r['tokenizer_id']:12s} "
            f"{'OK' if ok else 'WRONG'}")
for f in ("_selftest.json", "_selftest_out.json"):
    (RES / f).unlink(missing_ok=True)
out(f"self-test: {'ALL OK' if all_ok else 'FAILURES'}")

(RES / "report.txt").write_text(report.getvalue(), encoding="utf-8")
print("\nreport written to results/report.txt")
