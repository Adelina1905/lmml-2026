"""Local stand-in for the organisers' check.py (which was not included in the task files).
Runs main.py exactly like the harness: in a fresh folder that contains only main.py and
the WAV, passing the bare filename. Then scores detection.csv against train-output.csv.
Usage:  python check_local.py main.py"""
import csv, os, shutil, subprocess, sys, tempfile
script = sys.argv[1] if len(sys.argv) > 1 else "main.py"
here = os.path.dirname(os.path.abspath(__file__))
with tempfile.TemporaryDirectory() as d:
    shutil.copy(os.path.join(here, script), os.path.join(d, "main.py"))
    shutil.copy(os.path.join(here, "train-input.wav"), os.path.join(d, "train-input.wav"))
    subprocess.run([sys.executable, "main.py", "train-input.wav"], cwd=d, check=True)
    files = sorted(os.listdir(d))
    rows = list(csv.reader(open(os.path.join(d, "detection.csv"))))
assert rows[0] == ["start_s", "end_s"], rows[0]
assert len(rows) == 2, rows
s, e = map(float, rows[1])
gs, ge = map(float, list(csv.reader(open(os.path.join(here, "train-output.csv"))))[1])
ok = abs(s - gs) <= 1.5 and abs(e - ge) <= 1.5
print("files after run:", files)
print(f"detected {s:.3f},{e:.3f}  truth {gs:.3f},{ge:.3f}  "
      f"errors {abs(s-gs):.3f} / {abs(e-ge):.3f} s  -> {'PASS' if ok else 'FAIL'}")
