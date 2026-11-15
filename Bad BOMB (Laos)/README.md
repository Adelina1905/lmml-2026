# Bad BOMB (Laos)

**PASSED: 21/28 matches — 75%.** This exceeds the 70% threshold and meets the task's 200-point criterion.

- [Task.md](Task.md): original competition statement.
- [Solving.md](Solving.md): Obsidian-friendly solution notes, a workflow diagram, and training details.
- [test-output.txt](test-output.txt): the passing 28-image selection.
- [Candidate overview](results/local_detector_v1/top_28.jpg): the selected images and anomaly overlays.
- [Evaluator result](results/EVALUATOR_RESULT.json): recorded score and evidence reference.

The local route is **01 inspect → 02 prepare → 03 train**. Its scripts and launcher are in [scripts/](scripts/).

## Reproduce the local run

Use Python 3.12 on Windows and a compatible NVIDIA GPU. Download the dataset from the link in [Task.md](Task.md) and place the extracted PNG files in this task folder's `images/` directory. From this task folder, create an environment if needed and install the dependencies:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Then run the three stages:

```powershell
.\.venv\Scripts\python.exe .\scripts\step_01_inspect_dataset.py
.\.venv\Scripts\python.exe .\scripts\step_02_preprocess_images.py
.\.venv\Scripts\python.exe -u .\scripts\step_03_train_local_detector.py --run-name local_detector_rerun
```

The first local run downloads the pretrained encoder. A rerun saves its models, scores, and full HTML review in `results/local_detector_rerun/`; it leaves the passing `test-output.txt` unchanged. The HTML review is a local report that needs the dataset and generated heatmaps. It is excluded from Git together with the source images, processed views, model files, caches, and Python environment. The two figures embedded in the solution notes and the evaluator screenshot are included in Git.

The dataset has 1,225 unique images, despite the brief's 1,125. The evaluator did not disclose individual correct labels or show the flag. Old experiments and caches have been removed; the passing artifacts and working runtime remain.
