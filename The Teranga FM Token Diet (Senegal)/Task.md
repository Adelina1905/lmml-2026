## The Teranga FM Token Diet

# Task Overview

## 📖 Storyline

Teranga FM is a community radio station based in Kaolack, in the heart of Senegal's Peanut Basin. Alongside its broadcasts, it runs a text helpline that farmers and listeners use to get quick answers — about groundnut farming, the rainy season, or general facts about the country they can relay to buyers and visitors. Every answer goes out over a billed SMS gateway, and the station's LLM provider bills by the token, prompt and completion alike.

The station's intern wired up the helpline the easy way: every question, however simple, goes to the station's biggest model with a long, careful system prompt and a couple of worked examples attached, "just to be safe." It works — every answer is accurate — and it blows through the station's monthly token allowance before the month is half over. Your job is to answer the exact same battery of questions, just as accurately, for a lot fewer tokens.

## 📌 Task Title

The Teranga FM Token Diet

## 🔥 Difficulty

Medium

## 🎯 Objective

You are handed a battery of 20 fixed questions (`dataset/fact_bank.json`) and a small, fully transparent, offline simulator of an LLM gateway (`model_utils.py`) standing in for the real one — no API keys, no network calls, fully deterministic, and nothing about it hidden from you. For each question you choose four things: a **tier** (`cheap` or `large`), a **style** (`verbose` or `terse`), whether to attach a fixed **few-shot** example block, and whether to attach the question's **context** snippet (if it has one) to the prompt. Exactly what each choice costs in tokens and what it reliably produces is fully determined by `model_utils.py` and `dataset/fact_bank.json` — read them closely, not this document.

Every one of the 20 questions has a stated quality bar: your answer must include **every** fact the question requires, or that question fails — no partial credit. The total token count summed across all 20 answers (prompt tokens and completion tokens both count, exactly like a real LLM bill) must stay at or under **1200 tokens**. The intern's starting point — every question through the large tier, verbose, with the few-shot block attached (`dataset/example_naive_config.json`) — clears every quality bar but comes in at **3626 tokens**, well over double the cap. Your job is to find a routing config that still clears every quality bar and comes in under budget.

## 🚩 FLAG

Format `SIGMOID_<WORD>`.

## 📂 Task Input

The [materials](https://drive.google.com/drive/folders/1QuDKEu05swvKZiz_SDCzyLaEyKNT8iud?usp=sharing) you are provided with contain the following:

| **File**                    | **Purpose**                                                                                                    |
| :-------------------------- | :------------------------------------------------------------------------------------------------------------- |
| `fact_bank.json`            | The 20 questions, each with its required facts, and a context snippet for the questions that have one          |
| `example_naive_config.json` | The intern's starting point: every question routed to `large` / `verbose` / `fewshot: true` / `context: false` |
| `model_utils.py`            | The gateway simulator: prompt/completion templates, token counter and quality checker                          |
| `main_template.py`          | The template for your `main.py`: reads a routing config and writes `answers.json`                              |
| `baseline_solution.ipynb`   | A notebook that shows how to produce and self-check your answers locally                                       |

`model_utils.py` is the entire gateway simulator (prompt/completion templates, token counter, quality checker) and is exactly what both your own self-check and the organizer-side grader run. There is nothing about the cost or quality mechanism held back from you.

The scripts look for the two JSON files in a `dataset/` folder next to them (`dataset/fact_bank.json`, `dataset/example_naive_config.json`), so put them there after downloading.

Environment: Python 3.10–3.12, standard library only — no third-party packages, no GPU, no API keys, no internet access. Every run completes in well under a second.

## ✅ Expected Output

Submit:

1. `main.py` — your copy of `main_template.py`, I/O contract unchanged;
2. your `routing_config.json`: a JSON object with one entry per question id, each `{"tier": ..., "style": ..., "fewshot": ..., "context": ...}`. Package both, plus any script you used to produce your routing choices, behind a GitHub/Drive URL and submit that link.

Grading reconstructs every answer directly from your `routing_config.json` against the organizer's own `model_utils.py` and `fact_bank.json`. Running `python main.py` with **no arguments** (it reads `routing_config.json` and `dataset/fact_bank.json` by filename and writes `answers.json`) is how you produce and self-check your own answers locally, as demonstrated in `baseline_solution.ipynb`.

## 🏆 Scoring Criteria

Both gates below are checked **independently** by the organizer-side comparator — passing one while failing the other does not release the flag:

Quality gate: all 20 questions independently include every required fact. Budget gate: total tokens across all 20 answers ≤ the stated cap

- Both pass: 200 points (PASS), flag released.✅
- Either fails: 0 points (FAIL).❌

Reference point, so you know what you're up against: the intern's starting config (`dataset/example_naive_config.json`) clears the quality gate on all 20 questions, at 3626 tokens — well over the 1200 cap. Figuring out which changes close that gap, and by how much, is the task.

SenegalComplexity: Medium

- **Hint 1Unlock**
- **Hint 2Unlock**
- **Hint 3Unlock**

**Upload**

Accepted file types:**.json**

Max file size: 10 MB

**Submit**