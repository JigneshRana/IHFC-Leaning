# Project 5.2 — Lending Club: The "Why" Behind Every Choice

> Companion to `plan.md`. The plan says **what** to do; this file explains **why** — why this
> model, why these libraries, why the code is written this way. Cross-references point to your
> course notes (`5. Deep Learning Specialization/ClassnoteDeepLearning.md`) and instructor
> notebooks, so each decision ties back to something you've already learned.

---

## 1. Why frame this as *binary classification*?

The target `not.fully.paid` is 0/1 (paid vs default). From the "Master Output Layer Decision
Matrix" in class note **§2**, a scalar `y ∈ {0,1}` target fixes three choices automatically:

| Target | Output neurons | Final activation | Loss |
|--------|:---:|:---:|:---|
| scalar `y ∈ {0,1}` | **1** | **sigmoid** | **binary_crossentropy** |

We don't *choose* the last layer or loss — the problem type **dictates** them. The rest is tuning
on that fixed skeleton.

---

## 2. Why a plain feed-forward network (DNN), not CNN/RNN?

Each loan is a **fixed-length vector of 13 features** with no spatial grid (so no CNN, L8) and no
time order (so no RNN/LSTM, L11). The right tool is a stack of `Dense` layers — the
feed-forward/Universal-Approximation architecture from class note §3.1 (Lessons 3–4). *Match the
architecture to the data's structure:* tabular → Dense MLP.

> With only 13 features and 9,578 rows, classic models (Logistic Regression, RandomForest) are
> strong and fast. We build a neural net because this is a **deep-learning** course project; the
> plan keeps a simple baseline only as a sanity check.

---

## 3. Why each library?

| Library | Why it's here | Course tie-in |
|---|---|---|
| **pandas** | Load `loan_data (2).csv`, inspect, one-hot `purpose`. | throughout |
| **numpy** | Arrays for Keras; reproducibility seeds. | §4 |
| **matplotlib / seaborn** | EDA: distributions, `purpose`-vs-default, correlation heatmap. | EDA practice |
| **scikit-learn** | `train_test_split`, `StandardScaler`, `classification_report`, `roc_auc_score`, `compute_class_weight`. | data prep + metrics |
| **tensorflow / keras** | DL framework the course teaches (L5); we use the **Sequential API** (L5.09). | §4, Lesson 5 |
| **imbalanced-learn** | Optional SMOTE for the 84%/16% imbalance. | §7 |

**Why Keras `Sequential` specifically?** It's the most readable high-level API for a straight
feed-forward net — layers as a list, `.compile()/.fit()` run the training loop. Raw TensorFlow
(`GradientTape`, §4) or PyTorch (§5) only pay off when you need a custom loop, which we don't.

---

## 4. Why the `input.csv` / `output.csv` shortcut exists — and why we mostly ignore it

The folder ships a **pre-processed** pair: `input.csv` (already numeric, `purpose` one-hot encoded,
18 columns) and `output.csv` (the target). You *could* load these straight into a model.

**Why we still start from the raw `loan_data (2).csv`:** Assignment **Task 1** is literally
"transform categorical values into numerical values" and **Task 2** is EDA — you can only
demonstrate those on the raw file. Doing the encoding yourself is the learning objective. Treat
`input/output.csv` as an answer key to check your encoded frame against.

---

## 5. Why encode `purpose` with One-Hot (Assignment Task 1)

`purpose` has 7 unordered categories (`credit_card`, `small_business`, …). Neural nets need numbers,
but *how* matters:

- **One-Hot (`pd.get_dummies(..., drop_first=True)`)** — each category becomes its own 0/1 column.
- **Why not label-encode to `0..6`?** That invents a **false order** — the model would treat
  `small_business(5) > credit_card(1)` as if 5 is "more" than 1, which is meaningless for a
  category. One-hot avoids any fake ranking.
- **Why `drop_first=True`?** With `k` categories you only need `k-1` columns — the dropped one is
  implied when all others are 0. Keeping all `k` creates perfect multicollinearity (the
  **dummy-variable trap**), redundant information that can destabilise training.
- **`credit.policy` is left alone** — it's already a clean 0/1 flag.

---

## 6. Why correlation-based feature selection (Assignment Task 3)

- **Why drop one of a highly-correlated pair?** Features like `fico` and `int.rate` move together
  (worse FICO → higher interest rate). Two features carrying the same information add parameters but
  no new signal, and can make optimisation less stable. Keep the stronger predictor, drop the
  redundant twin.
- **Why rank features by correlation with the target?** It tells the story for your write-up:
  `int.rate`, `fico`, `inq.last.6mths` and `credit.policy` are the strongest default signals —
  useful both for feature selection and for the business-insight conclusion.

---

## 7. Why the data-prep order prevents leakage

```
EDA → encode purpose → drop redundant features → SPLIT → fit scaler on train → resample train only → model
```

- **Why split before scaling?** Fitting `StandardScaler` on the whole dataset leaks the test set's
  statistics into training. **Fit on train, transform test** with those same train statistics.
- **Why scale at all?** Inputs like `revol.bal` (tens of thousands) and `int.rate` (~0.1) live on
  wildly different scales. Un-scaled, the large-magnitude features dominate the gradient and
  training is slow/unstable. `StandardScaler` puts everything on a comparable scale — §7 explains
  why normalisation matters for gradient descent.
- **Why `stratify=y`?** Only 16% are defaults; stratifying keeps that ratio in train/val/test so
  metrics stay meaningful.
- **Why SMOTE (if used) on train only?** Synthesising minority rows before the split would copy
  test-like rows into training = leakage.

---

## 8. Why this model architecture

```python
model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(n_features,)),
    layers.Dropout(0.3),
    layers.Dense(32, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(16, activation='relu'),
    layers.Dense(1, activation='sigmoid')   # binary → 1 neuron + sigmoid
])
```

This mirrors the class note §2 "Binary Classification" pattern, sized down for a small dataset:

- **Smaller layers (64 → 32 → 16)** — only ~13–19 input features and <10k rows. A big network would
  **overfit** instantly (memorise the training set). Capacity should match data size (§7 bias–
  variance). Start small, grow only if validation says you're underfitting.
- **`relu` hidden activations** — the standard non-linearity (§2/§6): cheap and avoids vanishing
  gradients. Without a non-linear activation, stacked Dense layers collapse to one linear layer.
- **`Dropout(0.3)`** (L7.18 / §7) — the main regulariser here. With so few rows, dropout is what
  keeps the model from memorising noise.
- **`Dense(1, sigmoid)`** — sigmoid outputs a probability in [0,1]; one neuron for one yes/no
  decision.
- **Why maybe skip BatchNorm here?** On a tiny tabular net it adds little; Dropout usually suffices.
  (It's optional in the plan — add it only if training is unstable.)

---

## 9. Why these compile choices

```python
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss='binary_crossentropy',
    metrics=['accuracy', keras.metrics.AUC(name='auc'),
             keras.metrics.Precision(), keras.metrics.Recall()]
)
```

- **`Adam` (L7.13)** — adaptive learning rates → fast, robust, little tuning. Lesson 7 builds up
  SGD → Momentum → AdaGrad → RMSProp → Adam precisely to show why Adam is the safe default.
- **`binary_crossentropy`** — the loss mathematically matched to a sigmoid output (§2).
- **AUC/Precision/Recall, not just accuracy** — see §10.

---

## 10. Why accuracy misleads here, and how we handle imbalance

Defaults are only **16%.** A model predicting "fully paid for everyone" scores **84% accuracy**
while catching **zero** defaults — high accuracy, zero business value.

- **Why AUC / Recall / Precision / F1?** They measure the *minority* class. **Recall** = how many
  real defaults we caught (a missed default = money lost). **Precision** = how many of our flags
  were right. **AUC** = ranking quality across thresholds. These are the class note §2 metrics.
- **Why `class_weight` (recommended)?** `compute_class_weight('balanced', …)` makes errors on the
  rare class cost ~5× more, so the optimiser stops ignoring defaults. No leakage, no invented data.
- **Why SMOTE is the backup, not the default?** It synthesises minority rows (train-only!), which
  can help but adds complexity and fabricated data. Try `class_weight` first.

---

## 11. Why these training callbacks

```python
callbacks = [
    keras.callbacks.EarlyStopping(monitor='val_auc', mode='max',
                                  patience=5, restore_best_weights=True),
    keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3),
]
```

- **`EarlyStopping` + `restore_best_weights`** — stop when validation AUC plateaus and keep the best
  epoch, not the last (overfit) one. Prevents overfitting and saves time (§7 overfitting curve).
- **`ReduceLROnPlateau`** — shrink the learning rate when progress stalls so the optimiser settles
  into a finer minimum.
- **Why a 3-way train/val/test split?** Validation guides tuning (so it's no longer unbiased); the
  **test set is used once at the end** for an honest performance estimate.

---

## 12. Why tune the decision threshold

The sigmoid gives a probability; a cutoff turns it into 0/1. **0.5 is only a default.** A lender
usually wants to **catch more defaulters even at the cost of a few false alarms**, so we lower the
threshold to raise recall — chosen on the **validation** set, then applied once to the test set
(choosing it on test would be leakage).

---

## 13. One-paragraph summary you can say out loud

> "It's binary classification on imbalanced tabular data, so I use a Keras `Sequential`
> feed-forward network with ReLU hidden layers, Dropout for regularisation, and a single sigmoid
> output with binary-crossentropy loss and Adam. I one-hot encode `purpose` (with `drop_first` to
> avoid the dummy trap), drop redundant correlated features, and split *before* scaling/resampling
> to avoid leakage. Since only 16% default, I ignore accuracy and judge on AUC/recall/F1, handle
> the imbalance with class weights, tune the threshold on validation, and report the final score
> once on a held-out test set."

---

### Course cross-reference index
- Output-layer/loss choice → class note **§2** (Decision Matrix)
- Dense/DNN architecture & Universal Approximation → **§3**, Lessons 3–4
- TensorFlow/Keras Sequential API → **§4**, Lesson 5 (5.09)
- Optimizers (why Adam) → **§6**, Lesson 7 (7.03–7.13)
- Dropout / regularization / hyperparameter tuning → **§7**, Lesson 7 (7.18)
- Debugging & data-quality traps → **§14**
