# Two Solitudes: solution

**Flag: `SIGMOID_SNOWFLAKE`**

## Reproduce

```
pip install -r requirements.txt      # numpy, scikit-learn (+ matplotlib for the chart, optional)
python solve.py                      # run inside this folder; writes model.npz, FLAG.txt, results/
python open_vault.py model.npz       # The vault releases: SIGMOID_SNOWFLAKE
python whichdoor.py first_model.npz model.npz
```

`solve.py` builds `model.npz` (W 9×4096, b 9) and runs the vault on it. The script is deterministic and takes about 10 minutes on a CPU. All provided task materials are copied into this folder, so it runs on its own.

| File | What it is |
|---|---|
| `solve.py` | the whole pipeline |
| `FLAG.txt` | the flag |
| `results/` | run log, prediction mix (csv + chart), experiments table, vault doors |
| `gen.py` | hand-written EN/FR sentence templates I wrote (synthetic, no real records) |
| `dev_reports.py` | 90 short fictional reports I wrote, used only as a sanity check |
| `whichdoor.py` | diagnostic: says **which** vault door a model opens (door index + word only) |
| `model.npz` | the final model |

## What was wrong with the first model

The first model scores 99 % on the corpus, but the corpus is full of planted shortcuts that real reports don't have. I found them by reading the corpus and checking which features drove the first model's decisions on the sealed fingerprints:

1. **Language is tied to the label (the "two solitudes").** Flood, winter and wildfire reports are about 70 % French, while storm, industrial and transport are about 70 % English. As a result, the first model's top features were function words like *a, un, the, des, la, de*, and its predictions on the real reports shifted with the language.
2. **Distractor sentences.** The label is the hazard of the *first* hazard sentence, and later sentences often describe a different hazard. For example, "An airliner crashed on approach…" appears more often in *winter* reports than in *transport* ones. This teaches the model that transport words are weak evidence, so real air crashes and train collisions were sent to ground or industrial.
3. **Boilerplate with skewed labels.** Sentences like "A state of emergency was declared" or "0 people were injured" have class mixes that depend on the language. About 1,400 reports contain nothing but boilerplate: their labels can't be inferred from the text, so they're just noise.
4. **Header confounds.** Place names repeat with particular labels (e.g. "Eastern Quebec, September" for ferry sinkings).

## Method (all inside `solve.py`)

1. **Parse each report** into a header (place, date) and sentences. Sentence templates whose label mix is close to uniform are treated as boilerplate. Reports with no hazard sentence are dropped.
2. **Remove distractors.** A sentence classifier trained on first sentences only (5-fold CV accuracy 93 %) flags later sentences that confidently belong to a different hazard. Those sentences are removed from the report and reused as training examples of their own hazard.
3. **Remove the remaining confounds.** Each report is rebuilt three times: hazard sentences in shuffled order, random same-language boilerplate added, and the place replaced with a random place while the real date is kept (seasonality is genuine).
4. **Broaden the vocabulary.** I wrote EN/FR templates (`gen.py`) for all nine hazards, covering words like *derailed, airliner, freighter, smallpox, canicule, verglas, éboulement* in both languages. They produce 5,400 extra reports.
5. **Mask language out.** The fingerprint buckets for English and French function words, the digit token, and frequent function-word bigrams get zero weight.
6. **Fit a logistic regression (C = 10)** on 32k rebuilt and generated reports.
7. **Adapt to the real reports without labels**, using only the sealed fingerprints:
   - five rounds of class-balanced self-training (the most confident 20 → 100 predictions per class, added with weight 5);
   - label propagation over a 10-nearest-neighbour cosine graph (TF-IDF, stopwords masked) of the real reports, with α = 0.7;
   - a final refit on the corpus plus the 80 % of real reports with the most confident propagated labels.

## Why SNOWFLAKE and not a decoy

The vault has 5 doors. The first model opens door 4 (`SNOWBLIND`), the baseline's decoy. Every model from steps 1–6 alone opened nothing. Every model with label propagation opened **door 3, `SNOWFLAKE`, and no other door**. That held across 7 configurations (α 0.5–0.8, k 5–15, kept share 80–90 %, one or two rounds), and also for a pipeline **without** my generator or self-training (cleaned corpus + label propagation only). Independent pipelines converging on one door, which isn't the baseline's, points to the door that sorts the real reports correctly.

## Rules check

- **No outside incident records.** The only extra text is my own templates and my own fictional dev reports.
- **The sealed reports are never labelled by any other means.** They're used only as unlabeled fingerprints for self-training, label propagation and building the neighbour graph.
- **The vault was used only as pass/fail.** I never read its internal agreement counts or decoded polynomial. `whichdoor.py` reports only which door index opened.
