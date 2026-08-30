# Employee Turnover Analytics: Detailed Methodology & Explanations

This document provides a comprehensive, step-by-step educational guide explaining **why** specific methods, mathematical techniques, and library functions are chosen for this analysis to predict employee turnover at Portobello Tech.

---

## Table of Contents
1. [Phase 1: Setup and Exploratory Data Analysis (EDA)](#1-phase-1-setup-and-exploratory-data-analysis-eda)
2. [Phase 2: K-Means Clustering of Departures](#2-phase-2-k-means-clustering-of-departures)
3. [Phase 3: Preprocessing & Class Imbalance (SMOTE)](#3-phase-3-preprocessing--class-imbalance-smote)
4. [Phase 4: Model Training & 5-Fold Cross-Validation](#4-phase-4-model-training--5-fold-cross-validation)
5. [Phase 5: Model Evaluation & Metric Selection](#5-phase-5-model-evaluation--metric-selection)
6. [Phase 6: Retention Strategies & Risk Zones](#6-phase-6-retention-strategies--risk-zones)

---

## 1. Phase 1: Setup and Exploratory Data Analysis (EDA)

Before building predictive machine learning models, we must inspect the dataset's characteristics and perform quality checks.

### A. Missing Value Detection and Data Integrity
* **Missing Values (`isna()`)**: Missing values can prevent machine learning libraries like `scikit-learn` from executing correctly. If any null values exist, they must be resolved.
  * *Median Imputation*: If numerical features contain missing entries, filling them with the column's median is preferred over the mean. The median is robust to outliers, preventing extreme values from skewing the representative middle of the distribution.
* **Metadata Check**: Analyzing summary statistics (`df.describe()`) and dimensions (`df.shape`) helps confirm data range and boundaries. For example, `satisfaction_level` should lie within $[0.0, 1.0]$.

### B. Inferences from Project Counts and Turnover
Plotting the project count of employees grouped by their survival status (`left` as hue) helps reveal employee behaviors:
* **Overloaded Employees (Too Many Projects)**: Employees assigned to $6$ or $7$ projects show a nearly $100\%$ turnover rate. This points to extreme burnout.
* **Underutilized Employees (Too Few Projects)**: Employees assigned to only $2$ projects also show high turnover. This indicates underutilization, lack of engagement, or quiet quitting.
* **Optimal Workload**: Employees assigned to $3$ to $5$ projects show the lowest turnover, representing a healthy workload balance.

### C. Correlation Analysis
We compute Pearson's correlation coefficient ($r$) to identify linear associations:
$$r = \frac{\sum (x - \bar{x})(y - \bar{y})}{\sqrt{\sum (x - \bar{x})^2 \sum (y - \bar{y})^2}}$$
This helps us see which numeric factors (like satisfaction level vs. average working hours) show linear relationships with employee departure.

---

## 2. Phase 2: K-Means Clustering of Departures

We perform cluster analysis *exclusively* on employees who left the company (`left == 1`) using their job satisfaction (`satisfaction_level`) and performance rating (`last_evaluation`).

### A. Why Cluster Departed Employees?
Clustering allows us to segment the departed workforce into distinct personas. Understanding **why** groups of employees left tells us what retention interventions might have saved them.

### B. Why Use K-Means?
K-Means is a centroid-based clustering algorithm. It partitions the dataset into $K$ distinct, non-overlapping groups by minimizing the distance between each point and its assigned cluster centroid.
The mathematical objective is to minimize the Within-Cluster Sum of Squares (WCSS) or Inertia:
$$\text{WCSS} = \sum_{j=1}^{K} \sum_{i \in C_j} ||x_i - \mu_j||^2$$
where $\mu_j$ is the centroid of cluster $C_j$.

### C. Characterizing the 3 Employee Persona Clusters
Applying K-Means with $K=3$ on the departed subset reveals three classic workforce groups:
1. **Cluster 0: The Overworked & Highly Valued (High Evaluation, High Hours, Low Satisfaction)**
   * *Profile:* Employees with high performance ratings ($\ge 0.8$) who worked long monthly hours ($\ge 250$ hours) but had low job satisfaction ($\le 0.2$). These are top performers who left due to burnout.
2. **Cluster 1: The Disengaged / Bored (Moderate Evaluation, Low Hours, Moderate Satisfaction)**
   * *Profile:* Employees with average evaluations ($0.5$ to $0.6$), low working hours ($130$ to $160$ hours), and moderate satisfaction ($0.35$ to $0.45$). These employees were underutilized and likely left due to lack of growth or interest.
3. **Cluster 2: The High-Performing Stars (High Evaluation, Normal Hours, High Satisfaction)**
   * *Profile:* High-performing, highly satisfied employees ($\ge 0.7$ satisfaction, $\ge 0.8$ evaluation) who still left. This group represents competitive departures—star employees lured away by better offers, promotions, or compensation elsewhere.

---

## 3. Phase 3: Preprocessing & Class Imbalance (SMOTE)

### A. Categorical Encoding
Columns like `sales` (department) and `salary` are text-based. Machine learning models require numerical matrices.
* **One-Hot Encoding (`pd.get_dummies()`)**: Converts nominal categorical variables into separate binary columns ($0$ or $1$). For instance, `salary` ('low', 'medium', 'high') becomes three columns, preventing the model from assuming an incorrect numerical ordering.

### B. Class Imbalance Problem
In most organizations, the number of employees who stay is much larger than those who leave. In our dataset, only about $24\%$ of employees left.
If we train a model on this imbalanced data, it might achieve high accuracy simply by predicting that everyone stays. This makes it useless at identifying the actual departures.

### C. How SMOTE Solves Imbalance
SMOTE (Synthetic Minority Over-sampling Technique) creates synthetic data points for the minority class (`left == 1`) instead of simply duplicating existing rows.
* **The Mathematics of SMOTE**:
  1. For each minority class sample $x_i$, compute its $k$-nearest neighbors in the minority class (usually $k=5$).
  2. Randomly select one neighbor $x_{zi}$.
  3. Generate a synthetic sample $x_{new}$ along the line segment connecting $x_i$ and $x_{zi}$:
     $$x_{new} = x_i + \lambda \times (x_{zi} - x_i)$$
     where $\lambda$ is a random number drawn from a uniform distribution between $0$ and $1$.
This creates a continuous decision boundary rather than discrete, overfitted copies.

### D. Stratified Train-Test Split
To prevent data leakage, we perform a train-test split *before* upsampling. The test set must remain completely clean and representative of real-world proportions.
We use a **Stratified Split** (in an 80:20 ratio) to ensure the proportion of the target class (`left`) is identical in both the training and testing sets.

---

## 4. Phase 4: Model Training & 5-Fold Cross-Validation

We train three distinct classification algorithms:
1. **Logistic Regression**: A linear model that estimates probabilities using a logistic sigmoid function. It provides a baseline.
2. **Random Forest Classifier**: An ensemble of decision trees trained with bootstrap aggregating (bagging). It handles non-linear patterns and interactions well.
3. **Gradient Boosting Classifier**: An ensemble that builds decision trees sequentially, with each new tree correcting the residual errors of the previous ones.

### 5-Fold Cross-Validation (CV)
Cross-validation prevents overfitting and gives a robust estimate of model performance.
1. The training data is split into $5$ equal folds.
2. The model is trained on $4$ folds and validated on the remaining fold.
3. This process repeats $5$ times so every fold acts as the validation set once.
4. The final performance metrics are averaged across all $5$ runs.

---

## 5. Phase 5: Model Evaluation & Metric Selection

### A. Confusion Matrix Metrics
A confusion matrix maps predictions against reality:

| | Predicted Stay (0) | Predicted Left (1) |
| :--- | :---: | :---: |
| **Actual Stay (0)** | True Negative (TN) | False Positive (FP) |
| **Actual Left (1)** | False Negative (FN) | True Positive (TP) |

* **Precision**: Out of all employees predicted to leave, how many actually left?
  $$\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}$$
* **Recall (Sensitivity)**: Out of all employees who actually left, how many did we successfully identify?
  $$\text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}$$

### B. Justifying Recall as the Primary Metric
For Portobello Tech, **Recall is the critical metric**.
* **The Cost of a False Negative (FN)**: If the model predicts an employee will stay (0) but they actually leave (1), the company loses the employee. The cost of losing a key employee (recruitment, onboarding, lost productivity) is very high.
* **The Cost of a False Positive (FP)**: If the model predicts an employee will leave (1) but they intended to stay (0), the company might offer them a retention conversation or minor perk. This cost is minimal.
Since missing a flight-risk employee is much costlier than over-preparing for a loyal one, we prioritize maximizing **Recall**.

### C. ROC Curve and AUC Score
The Receiver Operating Characteristic (ROC) curve plots the True Positive Rate (Recall) against the False Positive Rate ($1 - \text{Specificity}$) at various classification thresholds.
The Area Under the Curve (AUC) measures overall model performance:
* $\text{AUC} = 1.0$: Perfect classification.
* $\text{AUC} = 0.5$: Equal to random guessing.

---

## 6. Phase 6: Retention Strategies & Risk Zones

Rather than treating all employees the same, we group them into four risk zones based on their predicted turnover probability ($p$):

1. **Safe Zone (Green) ($p < 20\%$)**
   * *Characteristics:* Highly satisfied, balanced workload, or recently promoted.
   * *Strategy:* Maintain current working conditions. Provide standard quarterly reviews and long-term career pathing.
2. **Low-Risk Zone (Yellow) ($20\% \le p < 60\%$)**
   * *Characteristics:* Minor satisfaction drop or slightly elevated working hours.
   * *Strategy:* Conduct proactive check-ins, offer soft skill development, and ensure workload distribution is fair.
3. **Medium-Risk Zone (Orange) ($60\% \le p < 90\%$)**
   * *Characteristics:* Overworked or under-promoted, showing clear signs of disengagement.
   * *Strategy:* Implement immediate workload adjustments, conduct structured 1-on-1 retention reviews, offer flexible working hours, and review compensation/salary level.
4. **High-Risk Zone (Red) ($p \ge 90\%$)**
   * *Characteristics:* High burnout (extreme project counts and hours) or severe dissatisfaction.
   * *Strategy:* Urgent executive/manager intervention. Offer immediate compensation reviews, mandatory project offloading, mental health days, or internal department transfers to reset their work style.

