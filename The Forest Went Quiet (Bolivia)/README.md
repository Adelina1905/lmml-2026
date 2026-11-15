# The Forest Went Quiet (Bolivia)

Submission passed hidden evaluation. Flag: `SIGMOID_POISONING`.

- `main.py`: final, self-contained submission. Upload this file alone.
- `Task.md`: complete original task statement.
- `Solution.md`: Obsidian-friendly explanation of the investigation, implementation, and results.
- `flag.txt`: released flag.
- `materials/`: original audio, model, notebook, and example inputs, unchanged from the supplied archive.
- `scripts/validate_solution.py`: reproducible local validation using the supplied model.
- `results/`: local validation report, per-recording predictions, frequency-ablation measurements, and submission feedback screenshot.
- `results/figures/`: investigation images embedded in `Solution.md`.
- `scripts/plot_investigation.py`: recreates the tone close-up, frequency spectrum, zoomed spectrogram, and frequency-ablation chart; additionally requires Matplotlib.
- `_archive/`: temporary experiments, duplicate outputs, caches, and the previous Python environment; excluded from Git. These were archived because permanent deletion was blocked by automatic approval review.

To reproduce local validation, install `requirements.txt` in a Python environment and run `python scripts/validate_solution.py` from this folder. The validator updates `results/` and removes its temporary input/output files.

The submission requires only NumPy and SciPy. Running `python main.py` reads `test-input.npz` from the current directory and writes `test-output.npz` there.
