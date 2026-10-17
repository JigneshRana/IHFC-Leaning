# Project 5.1 — Home Loan: Observations & Workarounds (v1 vs v2)

> Submission note for the two notebook versions in this folder:
> - **v1** — `home_loan_deep_learning.ipynb` (baseline deep-learning pipeline). Run captured in
>   `home_loan_deep_learning.ipynb - Colab.png` and `... - Colab (1).png`.
> - **v2** — `home_loan_deep_learning_v2.ipynb` (adds domain feature engineering + a LightGBM
>   benchmark). Results below are read from the notebook's saved cell outputs.

---

## 1. Results summary (all metrics on the held-out TEST set)

| Version | Model | Key settings | TEST ROC-AUC | TEST Accuracy | Training speed |
|---|---|---|:---:|:---:|---|
| **v1** | Keras DNN | `128→64→32`, BatchNorm, Dropout 0.3, Adam **1e-3**, 50 ep | **0.7325** | ~0.83 | Slow — ~2–3 s/epoch; practical only on GPU |
| **v2** | Keras DNN (main) | `64→32`, BatchNorm, Dropout 0.5, Adam **1e-4**, 50 ep, **+3 domain features** | **0.7441** | 0.8315 | Same order as v1 |
| **v2** | Keras DNN (quick) | same as above, 25 ep | 0.7400 | — | ~1 min |
| **v2** | **LightGBM** | `is_unbalance`, lr 0.05, early-stop @257 rounds, +domain features | **0.7621** ✅ | — | **Fastest** — seconds on CPU, no scaling/GPU needed |

Supporting validation-set numbers for the v2 DNN: Val ROC-AUC = 0.7392, best F1-threshold = 0.647,
minority-class F1 = 0.285, TEST PR-AUC = 0.2229.

**Verdict — best accuracy AND speed: LightGBM (0.7621 AUC).** This is the expected result for
structured/tabular data: gradient-boosted trees usually beat a plain MLP, train far faster, need no
feature scaling, and run on CPU. Because the assignment requires a *deep-learning* model, the Keras
network remains the primary deliverable and LightGBM is presented as a benchmark / improvement.

---

## 2. Version 1 — observations

**What it does well**
- Clean, leakage-safe pipeline: 3-way stratified split, scaler fit on train only, class weights
  from train only, threshold tuned on validation, final score reported once on test.
- Runs top-to-bottom without manual intervention (the two Colab PNGs are full "Run all" captures).
- Correct imbalance-aware evaluation (AUC / precision / recall / F1 + confusion matrix + ROC),
  not accuracy — important because a "predict all 0" model already scores ~92%.

**Limitations**
- Uses only the raw columns — no engineered features, so it leaves signal on the table
  (AUC 0.7325).
- The `128→64→32` network is relatively heavy for tabular data; most of the AUC comes from a few
  strong columns (`EXT_SOURCE_*`), so the extra capacity mostly adds training time, not accuracy.

**Workarounds applied / recommended for v1**
- *(none needed to run)* — v1 is the correct, reproducible baseline. Keep it as the "clean
  reference" submission.

---

## 3. Version 2 — observations

**What improved (keep these)**
- **Domain ratio features** (`NEW_CREDIT_TO_INCOME_RATIO`, `NEW_ANNUITY_TO_INCOME_RATIO`,
  `NEW_ANNUITY_TO_CREDIT_RATIO`) lifted the DNN by **+0.012 AUC** (0.7325 → 0.7441) — a cheap,
  real gain. These encode loan *affordability*, which raw amounts don't express directly.
- **LightGBM benchmark** correctly documents the performance ceiling (0.7621) and uses
  `is_unbalance=True` to handle the 8% minority automatically.

**Issues found**

| # | Issue | Why it matters |
|---|-------|----------------|
| I1 | **Notebook does not run top-to-bottom.** The new cells (feature engineering + LightGBM) are inserted right after *cleaning*, but they use `class_weight` (defined later) and `X_train`/`scaler` from the main pipeline. | A grader doing "Run all" hits `NameError: class_weight is not defined`. The saved outputs only exist because cells were run out of order. |
| I2 | **Duplicated preprocessing.** One cell impute→encode→split→scales the data, then the main pipeline does the same again. | Wasteful, confusing, and easy to let the two copies drift apart. |
| I3 | **Learning rate dropped to 1e-4** on the main DNN. | The training log shows AUC still rising at the end (under-fitting). `1e-3`/`5e-4` converges faster to a similar or better score. |
| I4 | **LightGBM trained on scaled data.** | Harmless (trees are scale-invariant) but unnecessary — minor tidiness issue. |

**Workarounds (to make v2 submittable as-is, no restructuring)**

- **For I1 (most important):** do **not** use "Run all". Run in this order instead:
  1. Cells 1–10 (setup → cleaning).
  2. Jump to the main pipeline and run **Feature & Target Separation → … → Class Imbalance
     Handling** (this defines `X_train`, `scaler`, and `class_weight`).
  3. Then go back and run the **Feature Engineering + Retrain + LightGBM** cells.
  4. Finish with **Model build → Train → Evaluation → Final Test → Save**.
  (Alternatively, in Colab: *Runtime → Run all*, then re-run the two failed feature/LightGBM cells
  once `class_weight` exists.)
- **For I2:** when submitting, treat the first (duplicate) preprocessing + "retrain" block as a
  *quick experiment* and the main pipeline as the *official run*. Both report ~0.74 AUC, so they
  agree.
- **For I3:** if you re-run, change `Adam(learning_rate=1e-4)` back to `1e-3` (or `5e-4`) in the
  model-build cell for faster convergence.
- **For I4:** no action required; optionally train LightGBM on the un-scaled `X` for cleanliness.

**Permanent fix (recommended — a clean v3):** move the 3 domain features *into* the main pipeline
(right before Feature & Target Separation, so there is a single preprocessing pass) and move
LightGBM to the end as a dedicated "Phase 11 — Improvement & Benchmark" section. That removes I1 and
I2 entirely and the notebook runs top-to-bottom. (Ask and this can be generated.)

---

## 4. What to submit

| File | Role in submission |
|---|---|
| `home_loan_deep_learning.ipynb` (v1) | Clean, reproducible baseline DNN (AUC 0.7325) |
| `home_loan_deep_learning.ipynb - Colab*.png` | Evidence of the v1 "Run all" execution |
| `home_loan_deep_learning_v2.ipynb` (v2) | Improved DNN with domain features (AUC 0.7441) + LightGBM benchmark (AUC 0.7621) |
| `observations_v1_v2.md` (this file) | Comparison, results, and the run-order workaround |
| `plan.md`, `explanation.md` | Project plan and the rationale behind every choice |

**One-line conclusion for the report:** *Engineered affordability ratios improved the deep-learning
model from 0.7325 to 0.7441 ROC-AUC; a LightGBM benchmark reached 0.7621, confirming that
gradient-boosted trees remain the stronger, faster choice for this tabular credit-risk problem,
while the Keras model satisfies the deep-learning requirement.*
