# The 300-Year Old Tiger Nest Painting (Bhutan)

Passed hidden evaluation with **SSIM 1.0000** and **100% correct strip order**. Flag: `SIGMOID_STITCHING`.

The solution compares strip edges using colour values and horizontal gradients, solves a minimum-cost neighbour assignment, and merges its cycles into one left-to-right sequence. It preserves the original strip pixels.

## Run

Use Python 3.13 and install the pinned dependencies:

```powershell
python -m pip install -r requirements.txt
python main.py
```

Place the PNG strips in `input/` beside `main.py` before running. The script writes `reconstructed_image.png` and `reconstructed_order.json` beside itself. Upload **`main.py`** to the competition evaluator.

The raw strips are excluded from Git. Download the [original dataset](https://drive.google.com/drive/folders/1L4L9B9ui35VR5XzGGiv2yBTyqltQCY4D?usp=sharing), extract it, and copy the PNG files from its `train-input/` folder into this task's `input/` folder to reproduce the demo. The committed reconstruction files are the demo results.

## Validation

With the original demo strips in `input/`, run:

```powershell
python validate.py
```

The validator compares against the supplied reference order and repeats reconstruction with four alternative strip widths and offsets. It updates the reconstruction files and `validation/results.json`. Local validation used Python 3.12.10; the task specifies Python 3.13 for evaluation.

## Files

- [Task.md](Task.md): original statement and scoring requirements.
- [Solving.md](Solving.md): Obsidian solution notes and formulas.
- [main.py](main.py): self-contained submission.
- [validate.py](validate.py) and `validation/`: reproducible checks, supplied reference order, and local results.
- `reconstructed_image.png` and `reconstructed_order.json`: demo reconstruction.
- `evidence/`: hidden evaluator feedback and screenshot.
- `flag.txt`: released flag.

`input/`, `_archive/`, Python caches, virtual environments, ZIP archives, and build directories are ignored.
