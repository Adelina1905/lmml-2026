# Task Overview

## 📖 Storyline

The Port of Abidjan, in Côte d'Ivoire, has run the same request-routing assistant for years: a small model that reads a clerk's plain-language request and tells the database layer which transactional query to run — look up a customer's orders, total revenue by category, insert a new order, update stock, close an account. It knows the port's old e-commerce ledger inside out.

This quarter, Abidjan became the logistics gateway for landlocked neighbors, and the assistant was fine-tuned to also route the new spatial and fleet-analytics requests: nearest warehouse, depots within a radius, delivery-delay trends, next-shipment lookups, fleet rankings. The fine-tuning worked — the new routes are flawless. The old ones are gone. Ask it to insert an order today and it confidently routes it as a spatial query. The ledger team needs one assistant that still does both jobs.

## 📌 Task Title

The Abidjan Transit Ledger

## 🔥 Difficulty

Medium

## 🎯 Objective

You are handed two checkpoints of the same routing model (a small shared text classifier, not a full LLM):

- `checkpoints/original_model.joblib` — trained only on the **old** task (five transactional-SQL template labels). Never saw the new task.
- `checkpoints/finetuned_model.joblib` — continued-trained on the **new** task only (five spatial/window-function template labels). This is the checkpoint you start from. Run both against a few old-task requests and see for yourself what happened.

You are also given `dataset/old_task_sample.json`, a sample of the original task's data (40 rows), and a public set covering **both** tasks (`dataset/train-input.json` / `train-output.json`) to self-check with locally.

Recover a single model that performs well on **both** tasks at once. Any approach that gets you there is acceptable — you don't need to justify which one you pick. What matters is the result: run your recovered model through `main_template.py`'s I/O contract and clear both accuracy gates below. Simply reverting to the original checkpoint or keeping the fine-tuned one as-is will each fail one gate — see the reference numbers below, neither endpoint is a shortcut.

## 🚩 FLAG

Format `SIGMOID_<WORD>`.

## 📂 Task Input

The files you need are stored [here](https://drive.google.com/drive/folders/1wgTAuF6sFH7mfCbmNuEeU4kN4XrW5_5M?usp=sharing). Records are `{"id", "text"[, "label"]}`.

| **File**                                 |  **Rows** | **Purpose**                                       |
| :--------------------------------------- | --------: | :------------------------------------------------ |
| `old_task_sample.json`                   |        40 | OLD-task sample, for replay/recovery              |
| `train-input.json` / `train-output.json` | 250 / 250 | Set covering **both** tasks, for local self-check |

Each task has 5 labels (`OLD_JOIN_CUSTOMER_ORDERS`, `OLD_GROUPBY_REVENUE_BY_CATEGORY`, `OLD_CRUD_INSERT_ORDER`, `OLD_CRUD_UPDATE_INVENTORY`, `OLD_CRUD_DELETE_ACCOUNT` for the old task; `NEW_SPATIAL_NEAREST_WAREHOUSE`, `NEW_SPATIAL_WITHIN_RADIUS`, `NEW_WINDOW_LAG_DELIVERY_DELAY`, `NEW_WINDOW_LEAD_NEXT_SHIPMENT`, `NEW_WINDOW_PARTITION_RANK_FLEET` for the new task). You are not asked to generate or execute real SQL — the model classifies a request into the query template that would generate it, which is what the evaluator scores.

Environment: Python 3.10–3.12 with `scikit-learn==1.8.0`, `numpy==2.5.1`, and `joblib==1.5.3`. No GPU, no API keys, no internet access — the model is a small CPU-only classifier; a full retrain-from-scratch takes well under a minute.

## ✅ Expected Output

Submit **(1)** `main.py` — your copy of `main_template.py`, I/O contract unchanged — and **(2)** your recovered checkpoint saved at `checkpoints/recovered_model.joblib`, the exact path `main.py` loads by default. Package both of these, plus the code you used to produce the checkpoint, behind a GitHub/Drive URL and submit that link.

For grading, the harness mounts the hidden `test-input.json` into your submission's working directory and runs `python main.py` with **no arguments**: it must read that file by filename and write `predictions.json` next to it, `{"id", "label"}` per row. You never see the hidden test split or run the comparator yourself. Self-check locally first against `dataset/train-input.json` / `train-output.json` (gold labels included) using `model_utils.accuracy_by_task` — the same function the organizer-side comparator uses — as demonstrated in `baseline_solution.ipynb`.

## 🏆 Scoring Criteria

Both gates below are checked **independently** by the organizer-side comparator against the hidden test split — passing one while failing the other does not release the flag:

- Old-task accuracy ≥ **0.85**
- New-task accuracy ≥ **0.85**

**Both pass: 200 points (PASS)**, flag released.✅

**Either fails: 0 points (FAIL).** ❌

Reference points, so you know what you're up against: the original checkpoint as-is scores old 1.00 / new 0.00; the fine-tuned checkpoint as-is scores old 0.00 / new 1.00. Both fail one gate completely.
