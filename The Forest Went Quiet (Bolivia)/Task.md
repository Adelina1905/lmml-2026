# Task Overview

## 📖 Storyline

A biodiversity research station in the Bolivian Amazon uses an audio classifier to monitor birds, frogs, insects, and rain. After a model update, its dashboard reports unexpected changes in the recorded sound categories. Field researchers observe no matching change in the environment, and manual reviews disagree with several confident model predictions.

You receive confirmed failures, correctly classified reference recordings, and the deployed model. Some supplied failures may be ordinary model errors.

## 📌 Task Title

The Forest Went Quiet

## 🔥 Difficulty

Medium

## 🎯 Objective

Investigate what causes the observed failures. Produce modified recordings that reproduce those failures on new inputs while preserving the original audio.

Start here: open the provided baseline_solution.ipynb, edit `transform(audio)`, run the local checks, and submit the generated `main.py`.

## 🚩 FLAG

Format: **`SIGMOID_<WORD>`**. You receive the flag when your output passes all three checks listed in the Scoring Criteria on hidden recordings.

## 📂 Task Input

The provided [materials](https://drive.google.com/drive/folders/1w_JY-7QWHWxVQzYSh4J0sYOs4DgNOE-G?usp=sharing) contain the following:

| **Files**                                  | **Purpose**                                                                           |
| :----------------------------------------- | :------------------------------------------------------------------------------------ |
| `challenge-model.pt`, `challenge_model.py` | Four-class classifier; inspect it and run unlimited local predictions                 |
| `audio/`                                   | 31 recordings: 13 misclassified failures and 18 correctly classified references       |
| `recordings.csv`                           | Clip IDs, paths, correct labels, model classification status, and file hashes         |
| `train-input.npz`, `train-output.npz`      | Format example: the output is unchanged audio, not a solution                         |
| `baseline_solution.ipynb`                  | Editable baseline with input loading, transformation, output saving, and local checks |
| `investigation_tools.py`, `audio_io.py`    | Audio inspection and local checks                                                     |

All audio is mono, 16 kHz, and two seconds long. Class indices are **0 birds, 1 frogs, 2 insects, 3 rain**. Hidden evaluation uses source recordings absent from the public data and the model's task training split. In `recordings.csv`, `classification_status` is `misclassified` for failures and `correct` for references. The `label` column gives the correct class of each recording.

All assets are included and work offline once the dependencies pinned in `requirements.txt` are installed: Python 3.11 with NumPy 2.2.6 and SciPy 1.16.1, plus PyTorch 2.9.1 to run the supplied model locally.

## ✅ Expected Output

Upload a single Python file named **`main.py`**. The platform runs your script and grades the audio it produces.

**Your script must:**

1. Read **`test-input.npz`** from the current working directory.
2. Process **every recording** in that file. Each input contains a batch of **1–64 recordings**; your script must handle any batch size in that range.
3. Save all modified recordings in **one `test-output.npz` file** in the same directory, preserving their number and order.

The grader runs `python main.py` with no arguments. Use the filenames above directly, without absolute paths or interactive prompts.

**Input and output format**

Both files are NumPy `.npz` archives containing exactly the three arrays below. `B` is the number of recordings in the input. Each row of `audio` is one two-second recording containing 32,000 floating-point audio amplitude values between −1 and 1.

| **Array**     | **Required type and shape**                                            | **What to return**                                     |
| :------------ | :--------------------------------------------------------------------- | :----------------------------------------------------- |
| `audio`       | Finite `float32`, `(B, 32000)`                                         | Your modified audio, with the same shape and row order |
| `sample_rate` | Scalar `int32`, value `16000`                                          | Copy unchanged from the input                          |
| `clip_ids`    | Unique fixed-width Unicode strings, `(B,)`, at most 64 characters each | Copy unchanged from the input                          |

Load with `np.load(..., allow_pickle=False)` and save with `np.savez`, as shown in the notebook. Change only the audio values and follow the audio limits below.

**Execution limits**

| **Setting**          | **Limit**                                                       |
| :------------------- | :-------------------------------------------------------------- |
| Python and libraries | Python 3.11, standard library, NumPy 2.2.6, and SciPy 1.16.1    |
| Execution            | CPU only, offline, non-interactive                              |
| Time                 | 30 seconds for the entire batch                                 |
| Output file size     | 32 MiB maximum                                                  |
| Repeatability        | The same input must produce the same audio; seed any randomness |

Your submitted `main.py` must be self-contained; the supplied model and task helpers are available only for local investigation.

**Audio limits**

For every recording:

- Keep the edit's energy at most **1% of the original recording's energy**.
- Keep output samples strictly between **-1 and 1**, without clipping.
- Preserve the sample rate, clip IDs, order, and audio shape.

The notebook's checker calculates the edit energy percentage for you and explains any failed limit. Energy percentage is `100 * sum((output-input)**2) / sum(input**2)`.

## 🏆 Scoring Criteria

- ✅ Your output passes all three checks below on hidden recordings: 200 points (PASS), flag released
- ❌ Any check fails: 0 points (FAIL), no flag

| **Check**               | **Requirement**                                                                                                   |
| :---------------------- | :---------------------------------------------------------------------------------------------------------------- |
| Format and audio limits | Follow the Expected Output format and meet the audio limits for every recording                                   |
| Failure mechanism       | Reproduce the recurring cause of the supplied failures in every recording; approximate reconstruction is accepted |
| Success rate            | At least **90% overall** and **85% for each original class**                                                      |

A recording counts as a **success** when the model predicts the incorrect class associated with the recurring failure. Identifying that class is part of your investigation. Success rates are calculated within each original source recording, then averaged so that each source has equal weight.

The notebook checks your transformed audio and shows model predictions. These local checks do not confirm a pass: final grading also validates the output file, checks the failure mechanism, and measures success on hidden recordings.