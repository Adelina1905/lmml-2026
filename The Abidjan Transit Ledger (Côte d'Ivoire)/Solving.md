# The Abidjan Transit Ledger

1. The hand-out had no checkpoints, so a new model was trained from scratch on all ten labels at once (replay). Training on old and new data together means nothing is forgotten.
2. Training data: `old_task_sample.json` (40 old-task rows) plus `train-input.json` matched by `id` to `train-output.json` (250 rows covering both tasks).
3. Model: TF-IDF on words, word pairs and 3–5 letter chunks, followed by a `LinearSVC` classifier.
4. Cross-validation on unseen sentences scored **old 0.986 / new 1.000**. Holding out whole sentence patterns scored about **0.93 / 0.97**. Both clear the 0.85 gates.
5. The upload accepts a single `.py` file, so `build_main.py` writes the training rows into `main.py`. Running `python main.py` trains the model, saves `checkpoints/recovered_model.joblib`, reads `test-input.json` and writes `predictions.json`.
