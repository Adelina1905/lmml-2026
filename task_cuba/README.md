# task_cuba: Zemí, a light and fast retriever for the lector's slips

For each question (a slip from the box), return the 3 documents most likely to be the chapter it describes,
out of 1552 chapter-sized documents. To pass, all four limits must hold at once: Hit@3 ≥ 0.80, main.py + index/ ≤ 53 MB,
p95 latency ≤ 75 ms on one CPU thread, load time ≤ 10 s.

**Deliverable:** `submission.zip` (contains exactly `main.py` and `index/`).

## Result (`python local_check.py submission.zip`, public questions)

| measure | ours | limit | |
|---|---|---|---|
| index size | 51.75 MB | ≤ 53 MB | PASS |
| load time | 0.28 s | ≤ 10 s | PASS |
| p95 latency | 19.2 ms | ≤ 75 ms | PASS |
| Hit@3 | 0.922 | ≥ 0.80 | PASS |

Baseline for comparison: 80.25 MB and Hit@3 0.709, so it failed two limits.
On held-out halves of the public questions, 2-fold cross-validation of the weight choice gives a mean Hit@3 of 0.899 (5th percentile 0.866).

## Method (hybrid search)

1. **Query cleaning:** lower-case and strip accents; translate common Spanish words that slip into the questions (`guerra` → war, `gente` → people);
   drop words that describe the request rather than its content ("the chapter that tells the whole story of");
   fix typos: a word unknown to both the corpus and the model's vocabulary is replaced by its most frequent corpus neighbour at edit distance 1 (`Mnoroe` → monroe).
2. **Lexical signal (BM25):** stemmed BM25 over 300-word chunks (a document scores as its best chunk) plus BM25 over whole documents.
3. **Neural signal:** bge-small-en-v1.5 (int8 ONNX, 33.8 MB). The query is embedded at search time and compared with the precomputed embeddings
   of all 26,354 chunks of 200 words (a document scores as its best chunk) and with the embeddings of the 1552 titles.
4. **Fusion:** each signal is rescaled per query so that the median document scores 0 and the best scores 1, then the signals are combined as
   `1.0·lex_chunk + 0.6·lex_doc + 1.5·dense_chunk + 0.5·dense_title`. The top 3 documents are returned.

Size: the model is int8, the chunk vectors are int8 with one scale per row, and the BM25 weights are uint8 with gap-encoded ids, compressed.

## Files

| path | what |
|---|---|
| `solve.py` | full pipeline: model download (with SHA-256 check) → chunk embeddings → index → evaluation → `submission.zip` → `local_check.py` |
| `main.py` | the submission's retriever (the same file that is inside the zip) |
| `submission.zip` | **the deliverable** |
| `results/` | `report.md` (all metrics), `local_check.txt`, `fusion_grid.csv`, `misses.txt`, `hit_at_3_by_signal.png`, `solve_output.txt` |
| `data/`, `limits.json`, `local_check.py`, `template_main.py`, `baseline/` | the provided task materials (the baseline's index is left out because of its size) |

## Run

```
pip install -r requirements.txt
python solve.py          # rebuilds everything; computing the chunk embeddings takes about 50 min on 2 CPU cores (cached in cache/)
python solve.py --check  # only runs local_check.py on submission.zip
```

The model comes from the npm package `impro-plugin-semantic-mute-model@1.0.1`, which holds
BAAI/bge-small-en-v1.5 (MIT licence) quantised to int8, and its tokenizer. `solve.py` downloads it and checks both SHA-256 hashes.
