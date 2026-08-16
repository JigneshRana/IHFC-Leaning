# 🎯 IHFC Machine Learning — Active Learning Progress & Memory Tracker

This document maintains real-time tracking of our learning journey, the exact file and topic in progress, past questions discussed, and upcoming topics so we can seamlessly resume anytime.

---

## 📍 Current State & Active File

* **Module**: [4.IHFC AIML: Machine Learning](file:///media/jignesh/Data/ihfc/IHFC-Leaning/4.IHFC%20AIML:%20Machine%20Learning)
* **Current Lesson**: [Lesson 04: Supervised Learning — Classification and its Application](file:///media/jignesh/Data/ihfc/IHFC-Leaning/4.IHFC%20AIML:%20Machine%20Learning/ReferanceMaterials/Instructor_Slides_and_Notebooks/Lesson_04_Supervised_Learning_%20Classification_and_its_Application)
* **Current Active Notebook**: [`4.1_Classification_and_Its_Applications_Part_1.ipynb`](file:///media/jignesh/Data/ihfc/IHFC-Leaning/4.IHFC%20AIML:%20Machine%20Learning/ReferanceMaterials/Instructor_Slides_and_Notebooks/Lesson_04_Supervised_Learning_%20Classification_and_its_Application/4.1_Classification_and_Its_Applications_Part_1.ipynb)
* **Active Dataset**: `Breast_cancer_dataset.csv` (Malignant=1 vs Benign=0)

---

## 📜 Questions & Topics Covered So Far

1. **Overall Notebook 4.1 Step-by-Step Architecture**
   - End-to-end classification workflow: Data Loading, Cleaning, Target Encoding (`M` $\to$ 1, `B` $\to$ 0), Train-Test Split with stratification, and `StandardScaler` pipeline.
2. **Logistic Regression & Sigmoid Activation**
   - Linear score to probability mapping $\sigma(z) = \frac{1}{1 + e^{-z}}$, log-loss optimization, and risk score thresholding.
3. **Comprehensive Classification Performance Metrics**
   - Confusion Matrix ($2 \times 2$ grid: TP, TN, FP, FN).
   - Precision vs Recall (Sensitivity) vs Specificity vs F1-Score.
   - Why **Recall (Sensitivity)** is top priority in medical diagnostics (minimizing False Negatives).
   - ROC Curve and **Youden's J Statistic** ($J = \text{TPR} - \text{FPR}$) for optimal threshold selection.
4. **Naive Bayes Classifier (`GaussianNB`)**
   - Bayes' Theorem: $P(y|X) = \frac{P(X|y)P(y)}{P(X)}$.
   - "Naive" conditional independence assumption $P(x_1, \dots, x_n|y) = \prod P(x_i|y)$.
   - GaussianNB vs MultinomialNB vs BernoulliNB.
5. **K-Nearest Neighbors (KNN)**
   - Distance-based "Lazy Learner" (Euclidean, Manhattan, Minkowski).
   - Odd $K$ selection, $K = \sqrt{N}$ thumb rule, and bias-variance tradeoff ($K=1$ overfits, large $K$ underfits).
   - Why Feature Scaling is strictly mandatory for KNN.
   - Hyperparameter tuning using `GridSearchCV`.

---

## ⏩ Next Topics in Line to Resume (ત્યારપછીના વિષયો)

1. **Decision Trees (CART)**:
   - How Decision Trees split nodes (Tree anatomy: Root, Decision, Leaf nodes).
   - Splitting criteria: **Gini Impurity** vs **Entropy / Information Gain**.
   - Overfitting in raw trees (Train Acc 100% vs Test Acc drop).
   - **Cost-Complexity Pruning (`ccp_alpha`)** and `max_depth` constraints.
2. **Support Vector Machines (SVM)**:
   - Maximum Margin Classifier, Hyperplane, and Support Vectors.
   - Linear SVM vs Non-linear SVM.
   - The **Kernel Trick** (Radial Basis Function / RBF, Polynomial).
   - Hyperparameters $C$ (Regularization / margin softness) and $\gamma$ (gamma / kernel bandwidth).
3. **Notebook 4.2 & 4.3**:
   - Multiclass Classification and imbalanced dataset handling (SMOTE resampling on `creditcard.csv`).

---

*(Updated automatically after every learning milestone.)*
