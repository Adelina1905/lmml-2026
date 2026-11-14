from pathlib import Path
import re
import runpy
import shutil
import subprocess
import sys
import tempfile
import pandas as pd

def check_submission(script, working_directory, expected):
    result = subprocess.run([sys.executable, str(script)], cwd = working_directory, capture_output = True, text = True, timeout = 30)
    assert result.returncode == 0, result.stderr
    assert result.stdout == "", f"Unexpected stdout: {result.stdout}"
    assert result.stderr == "", f"Unexpected stderr: {result.stderr}"
    output = (working_directory / "test-output.txt").read_bytes()
    assert re.fullmatch(rb"SIGMOID_[A-Z]+\r?\n?", output), repr(output)
    assert output.decode("utf-8").strip() == expected, repr(output)

def main():
    folder = Path(__file__).resolve().parents[1]
    recover_word = runpy.run_path(str(folder / "main.py"))["recover_word"]
    data = pd.read_csv(folder / "data" / "train-input.csv")
    expected = (folder / "data" / "train-output.txt").read_text(encoding = "utf-8").strip()
    for seed in range(100):
        assert f"SIGMOID_{recover_word(data, random_state = seed)}" == expected, f"Failed seed {seed}"
    print("PASS: practice decoding for every seed from 0 through 99")
    with tempfile.TemporaryDirectory(prefix = "verification-", dir = folder) as temporary:
        temporary_folder = Path(temporary)
        script = temporary_folder / "main.py"
        shutil.copy2(folder / "main.py", script)
        practice_directory = temporary_folder / "practice"
        practice_directory.mkdir()
        shutil.copy2(folder / "data" / "train-input.csv", practice_directory / "test-input.csv")
        check_submission(script, practice_directory, expected)
        print(f"PASS: standalone submission produces {expected} from the practice table")
        synthetic = data.rename(columns = {"depth_hoar_pct": "layer_depth", "elevation_m": "altitude", "mean_grain_roundness": "visual_roundness", "overburden_load_kpa": "applied_load"})
        synthetic = synthetic.sample(frac = 1, random_state = 7)
        synthetic = synthetic.loc[:, list(reversed(synthetic.columns))]
        synthetic_directory = temporary_folder / "synthetic"
        synthetic_directory.mkdir()
        synthetic.to_csv(synthetic_directory / "test-input.csv", index = False)
        check_submission(script, synthetic_directory, "SIGMOID_LAVA")
        print("PASS: renamed features with shuffled rows and columns produce SIGMOID_LAVA, preserving repeated letters")
    print("PASS: bare input/output filenames, separate working directories, flag-only output, and no practice-file dependency")

if(__name__ == "__main__"):
    main()
