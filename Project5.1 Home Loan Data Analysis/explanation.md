# Project 5.1 — Home Loan: The "Why" Behind Every Choice

> Companion to `plan.md`. The plan says **what** to do; this file explains **why** — why this
> model, why these libraries, why the code is written this way. Cross-references point to your
> course notes (`5. Deep Learning Specialization/ClassnoteDeepLearning.md`) and instructor
> notebooks, so each decision ties back to something you've already learned.

---

## 1. Why frame this as *binary classification*?

The target `TARGET` has exactly two outcomes (repaid vs not). That is the textbook definition of
**binary classification** — Section 2 of your class notes ("Master Output Layer Decision Matrix")
maps this task directly to:

| Target | Output neurons | Final activation | Loss |
|--------|:---:|:---:|:---|
| scalar `y ∈ {0,1}` | **1** | **sigmoid** | **binary_crossentropy** |

So the *problem type dictates the last layer and the loss function* — we are not choosing them by
taste, we are reading them off the decision matrix. Everything else (hidden layers, optimizer) is
tuning on top of that fixed skeleton.

---

## 2. Why a plain feed-forward network (DNN), not a CNN or RNN?

| Model family | Designed for | Our data |
|---|---|---|
| **CNN** (L8) | grid data with spatial locality (images) | ❌ no spatial structure |
| **RNN/LSTM** (L11) | ordered sequences (text, time series) | ❌ rows are independent, no order |
| **Dense feed-forward (ANN/DNN)** (L3–L4) | fixed-length tabular feature vectors | ✅ each applicant = one vector of 120 features |

Each loan application is an **unordered, fixed-length vector of features**. There is no pixel
grid and no time axis, so the inductive biases of CNNs/RNNs buy us nothing. A stack of `Dense`
layers — the **Universal Approximation** architecture from class note §3.1 — is the correct and
simplest tool. *Match the architecture's assumptions to the data's structure.*

> Note: for 120 tabular features and heavy class imbalance, gradient-boosted trees (XGBoost /
> LightGBM) are often *the* industry winner. We still build a neural net because the assignment
> (and this course) is about **deep learning with Keras**. The plan suggests a tree baseline only
> as a sanity check.

---

## 3. Why each library?

| Library | Why it's here | Course tie-in |
|---|---|---|
| **pandas** | Load the 166 MB CSV, inspect, clean, encode — the standard tabular tool. | used throughout |
| **numpy** | Arrays/math that feed Keras; seeds for reproducibility. | §4 tensor math |
| **matplotlib / seaborn** | EDA plots (distributions, heatmaps) — you cannot clean what you cannot see. | EDA practice |
| **scikit-learn** | `train_test_split`, `StandardScaler`, `SimpleImputer`, `classification_report`, `roc_auc_score`, `compute_class_weight`. Keras does *modelling*; sklearn does *data prep + metrics*. | standard pairing |
| **tensorflow / keras** | The deep-learning framework the course teaches (L5). We use the **Sequential API** (L5.09). | §4, Lesson 5 |
| **imbalanced-learn (imblearn)** | Optional SMOTE for the 8%/92% imbalance. | §7 regularization/imbalance |

**Why Keras over raw TensorFlow or PyTorch?** The course covers all three (L5 TF, L6 PyTorch), but
Keras's `Sequential` is the highest-level, most readable API for a straightforward feed-forward
net — you describe layers as a list and `.compile()/.fit()` handle the training loop. Raw TF
(`GradientTape`, class note §4) or PyTorch (§5) are worth it when you need custom training loops;
we don't here.

---

## 4. Why the data-prep steps are in *this* order

The order is not cosmetic — it prevents **data leakage** (the single most common way to get a
fake-good score). Leakage = letting information from the test set influence training.

```
Clean → Encode → Select features → SPLIT → (fit scaler on train) → (resample train only) → model
```

- **Why drop `SK_ID_CURR` first?** It's a random ID — zero predictive signal. Feeding it in just
  adds noise and lets the model "memorise" rows.
- **Why impute missing values?** Neural nets cannot do arithmetic on `NaN` — a single `NaN`
  propagates to a `NaN` loss and training dies. We use **median** for numeric (robust to the heavy
  right-skew in income/credit amounts) and **mode/"Unknown"** for categoricals.
- **Why drop columns with >50–60% missing?** The building-info columns (`_AVG`/`_MODE`/`_MEDI`)
  are mostly empty; imputing a column that is 70% invented values adds more noise than signal.
- **Why treat `DAYS_EMPLOYED == 365243` as `NaN`?** That's a **sentinel value** (~1000 years) the
  data provider used for "not employed." Left as-is it would wreck scaling and the model would
  learn a nonsense spike. This is the kind of data-quality trap §14 (Debugging) warns about.
- **Why split *before* scaling and SMOTE?** If you fit the `StandardScaler` on the full dataset,
  the test set's mean/variance leak into training. If you SMOTE before splitting, synthetic copies
  of test-like rows end up in training. **Fit on train, apply to test.** This is the rule that most
  separates a trustworthy result from a misleading one.
- **Why `stratify=y`?** With only 8% positives, a random split could leave very few positives in
  the test set, making metrics unstable. Stratification keeps the 8%/92% ratio in every split.

---

## 5. Why encode categoricals the way we do (Assignment Task 1)

Neural nets do linear algebra — they need **numbers**, not strings. But *how* you turn a category
into numbers matters:

- **One-Hot Encoding** for nominal categories (`NAME_INCOME_TYPE`, `OCCUPATION_TYPE`, …). Why not
  just label-encode them `0,1,2,3`? Because label encoding invents a **false ordering** — the model
  would think `Working(3) > Pensioner(1)` numerically, which is meaningless. One-hot gives each
  category its own independent 0/1 column, no fake order.
- **Binary/label encoding** for true 2-level flags (`FLAG_OWN_CAR` Y/N). A single 0/1 column is
  enough; one-hot would just add a redundant column.
- **Why group rare levels into "Other" for high-cardinality columns?** `ORGANIZATION_TYPE` has
  dozens of values; one-hot would explode the feature count and create very sparse columns that the
  model can't learn from. Grouping rare levels keeps width manageable.

---

## 6. Why correlation-based feature dropping (Assignment Task 3)

- **Why drop one of each highly-correlated pair (|corr| > 0.85)?** Two near-duplicate features
  (e.g. `AMT_CREDIT` ↔ `AMT_GOODS_PRICE`, or the `_AVG`/`_MODE`/`_MEDI` triplets) carry the same
  information twice. Redundant inputs add parameters without adding signal, slow training, and can
  destabilise optimisation. Keeping one representative is cleaner.
- **Why engineer ratios** (`CREDIT_INCOME_RATIO`, `ANNUITY_INCOME_RATIO`)? Domain knowledge: a
  €200k loan is risky for a €20k income but fine for €500k. The *ratio* encodes affordability more
  directly than the two raw amounts, giving the network a stronger signal to latch onto.

---

## 7. Why this model architecture

```python
model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=(n_features,)),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(64, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(32, activation='relu'),
    layers.Dense(1, activation='sigmoid')   # binary → 1 neuron + sigmoid
])
```

Line-by-line reasoning (mirrors the class note §2 "Binary Classification" pattern):

- **`Dense` layers, decreasing 128 → 64 → 32** — the classic funnel. Early wide layers learn many
  low-level feature combinations; narrowing layers compress them into higher-level concepts
  (class note §3.1 hierarchical representation). Starting width scales with ~120 inputs.
- **`activation='relu'`** — ReLU is the default hidden activation (§2/§6): cheap, and it avoids the
  **vanishing-gradient** problem that sigmoid/tanh cause in deep stacks. Without a non-linear
  activation, stacked `Dense` layers collapse into a single linear model.
- **`Dropout(0.3)`** — regularisation (L7.18 / §7): randomly zeroes 30% of activations each step so
  the network can't over-rely on any one neuron. With 300k rows and 120+ features a plain MLP will
  overfit; dropout is the course's go-to fix.
- **`BatchNormalization`** — normalises each layer's inputs (§7). It stabilises and speeds up
  training and lets you use a higher learning rate. Especially helpful with many features of
  different scales.
- **`Dense(1, activation='sigmoid')`** — **sigmoid squashes the output to a probability in [0,1]**,
  exactly what binary classification needs. One neuron because there's one yes/no decision.

**Why not deeper/wider?** More capacity → more overfitting risk and slower training for little gain
on tabular data. Start small, grow only if the validation curve says you're underfitting (§7
bias-variance).

---

## 8. Why these compile choices

```python
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss='binary_crossentropy',
    metrics=['accuracy', keras.metrics.AUC(name='auc'),
             keras.metrics.Precision(), keras.metrics.Recall()]
)
```

- **`Adam` (L7.13)** — adaptive per-parameter learning rates = fast, robust convergence with little
  tuning. Your Lesson 7 notebooks implement SGD, Momentum, AdaGrad, RMSProp, Adadelta, Adam in
  sequence precisely to show *why* Adam is the sensible default: it combines momentum + RMSProp.
  `1e-3` is Adam's well-tested starting LR.
- **`binary_crossentropy`** — the loss mathematically matched to a sigmoid output (§2). It
  penalises confident-wrong predictions heavily, which is exactly the gradient signal we want.
- **Why `AUC`, `Precision`, `Recall` and not just `accuracy`?** See §9 — this is the single most
  important idea for this dataset.

---

## 9. Why accuracy is a *trap* here, and how we handle imbalance

Only **8% of loans are positive.** A model that blindly predicts "0 for everyone" scores **92%
accuracy** while catching **zero** of the cases we actually care about. So accuracy *looks* great
and is *useless*.

- **Why AUC / Precision / Recall / F1 instead?** They measure performance on the *minority* class.
  **Recall** = of all true defaults, how many did we catch (missing one = a bad loan approved).
  **Precision** = of those we flagged, how many were right. **AUC** summarises ranking quality
  across all thresholds. These are the metrics in the class note §2 for a reason.
- **Why use `class_weight` (recommended)?** `compute_class_weight('balanced', …)` tells `fit()` to
  penalise mistakes on the rare class ~11× more, so the optimiser stops ignoring it. It's free, has
  no leakage risk, and doesn't change the data.
- **Why SMOTE is the alternative, not the default?** SMOTE synthesises new minority rows to
  rebalance. It can help, but it (a) must run on **train only** (leakage otherwise), (b) invents
  data, and (c) enlarges the training set. Try it, but `class_weight` is the simpler first move.

---

## 10. Why these training callbacks

```python
callbacks = [
    keras.callbacks.EarlyStopping(monitor='val_auc', mode='max',
                                  patience=5, restore_best_weights=True),
    keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3),
]
```

- **`EarlyStopping`** — stop when validation AUC stops improving and roll back to the best epoch.
  Why? It prevents overfitting *and* wasted compute: beyond the sweet spot, training loss keeps
  dropping while validation gets worse (§7 bias–variance / overfitting curve).
- **`restore_best_weights=True`** — otherwise you'd keep the *last* (already-overfit) weights, not
  the best ones.
- **`ReduceLROnPlateau`** — when progress stalls, halve the learning rate so the optimiser can
  settle into a finer minimum instead of bouncing around it.
- **Why a 3-way train/val/test split?** Validation is used *during* tuning (it influences
  architecture, threshold, early stopping), so it is no longer unbiased. The **test set is touched
  once, at the very end**, to give an honest estimate of real-world performance.

---

## 11. Why tune the decision threshold

The sigmoid outputs a probability; converting it to a 0/1 label needs a cutoff. **0.5 is just a
default, not a law.** For an imbalanced lending problem, a bank usually prefers to **catch more
defaulters even at the cost of some false alarms** → lower the threshold to raise recall. We pick
the threshold on the **validation** set (to a business goal), then apply that fixed threshold once
to the test set. Choosing the threshold on the test set would be leakage.

---

## 12. One-paragraph summary you can say out loud

> "It's a binary classification on imbalanced tabular data, so I use a Keras `Sequential`
> feed-forward network with ReLU hidden layers, Dropout + BatchNorm for regularisation, and a
> single sigmoid output with binary-crossentropy loss and the Adam optimiser. I clean the
> Home-Credit quirks (the `365243` sentinel, high-missing building columns), one-hot the nominal
> categoricals, drop redundant correlated features, and — critically — split *before* scaling and
> resampling to avoid leakage. Because only 8% of loans are positive, I ignore accuracy and judge
> the model on AUC/recall/F1, handle the imbalance with class weights, tune the decision threshold
> on validation, and report the final score once on a held-out test set."

---

### Course cross-reference index
- Output-layer/loss choice → class note **§2** (Decision Matrix)
- Dense/DNN architecture & Universal Approximation → **§3**, Lessons 3–4
- TensorFlow/Keras Sequential API → **§4**, Lesson 5 (5.09)
- Optimizers (why Adam) → **§6**, Lesson 7 (7.03–7.13)
- Dropout / BatchNorm / regularization / hyperparameter tuning → **§7**, Lesson 7 (7.18)
- Debugging & data-quality traps → **§14**
