# Detailed Marketing Campaign Analysis Methodology & Explanations

This document provides a comprehensive, step-by-step educational guide explaining **why** specific methods, mathematical techniques, and library functions are chosen for the marketing campaign analysis.

---

## Table of Contents
1. [Introduction to the Marketing Mix (4 Ps)](#1-introduction-to-the-marketing-mix-4-ps)
2. [Phase 1: Verification of Data Importation](#2-phase-1-verification-of-data-importation)
3. [Phase 2: Sophisticated Data Cleaning & Group-By Imputation](#3-phase-2-sophisticated-data-cleaning--group-by-imputation)
4. [Phase 3: Feature Engineering](#4-phase-3-feature-engineered-variables)
5. [Phase 4: Categorical Encoding & Correlation Analysis](#5-phase-4-categorical-encoding--correlation-analysis)
6. [Phase 5: Hypothesis Testing & Statistical Formulas](#6-phase-5-hypothesis-testing--statistical-formulas)
7. [Phase 6: Visual Analytics & Dashboarding](#7-phase-6-visual-analytics--dashboarding)

---

## 1. Introduction to the Marketing Mix (4 Ps)
The Marketing Mix is a classic framework that categorizes a firm's tactical activities:
*   **Product**: What the customer buys (represented by spending on Wines, Fruits, Meat, Fish, Sweets, and Gold).
*   **Price**: The customer's financial capacity and sensitivity (represented by Income, purchases using discount deals, and household structure).
*   **Place**: Where and how purchases are made (represented by Web purchases, Store purchases, Catalog purchases, and Website visits).
*   **Promotion**: How campaigns are run and accepted (represented by campaigns 1-5, the final campaign response, customer complaints, and recency).

Data science allows us to analyze how these 4 Ps interact and predict which demographic groups are most likely to acquire products or respond to campaigns.

---

## 2. Phase 1: Verification of Data Importation
When we load raw CSV files, columns are often imported incorrectly due to formatting:
*   **Income**: Because incomes contain dollar signs and commas (e.g. `"$84,835.00"`), Pandas reads them as text (`object` type) instead of numbers. We cannot compute averages, sums, or correlations on text. Verification shows that we must remove `$`, `,`, and strip spaces before casting it to a float.
*   **Dt_Customer**: Stored as text. We must convert it to a datetime object so we can calculate a customer's loyalty duration or group them chronologically.

---

## 3. Phase 2: Sophisticated Data Cleaning & Group-By Imputation

### A. Categorical Group Scrutiny
In our raw dataset, `Marital_Status` contains irregular categories:
*   `Alone` (3 occurrences), `YOLO` (2 occurrences), and `Absurd` (2 occurrences).
*   **Why clean them?** Leaving these categories as-is creates tiny sample sizes that make grouping statistics highly unstable. We map `Alone`, `YOLO`, and `Absurd` to `Single` since they describe un-partnered individuals, increasing statistical power.

### B. Group-By Imputation of Income
We have missing income values. A simple imputation (like using the global average income of all customers) would bias our data because a customer with a PhD generally has a different average income than a customer with Basic education.
*   **Why use conditional imputation?** The prompt states that customers with similar `Education` and `Marital_Status` have comparable yearly incomes.
*   **Method**: We group the dataset by `['Education', 'Marital_Status']`, calculate the median income for each group, and fill the missing income values using that specific group's median. The median is chosen over the mean because it is robust to extreme outliers (like the `$666,666` entry).

### C. Outlier Detection using the IQR Method
Outliers (extreme values) can heavily skew our averages and hypothesis tests.
*   **The IQR (Interquartile Range) Rule**:
    *   $IQR = Q3 - Q1$ (where $Q3$ is the 75th percentile and $Q1$ is the 25th percentile).
    *   Lower Bound = $Q1 - 1.5 \times IQR$
    *   Upper Bound = $Q3 + 1.5 \times IQR$
    *   Any values outside these boundaries are flagged as outliers.
*   **Birth Year Outliers**: Customers born in `1893`, `1899`, and `1900` are flagged since they correspond to ages above 110, which are highly likely data entry errors.
*   **Income Outliers**: The max income of `$666,666` is far beyond the Upper Bound. We cap or remove these extreme outliers to stabilize our statistical models.

---

## 4. Phase 3: Feature Engineered Variables
To analyze behavior, we combine raw variables to create more descriptive features:
1.  **Age**: $\text{Age} = 2014 - \text{Year\_Birth}$ (representing their age during the campaign year).
2.  **Total\_Children**: $\text{Kidhome} + \text{Teenhome}$ (representing household size and time constraints).
3.  **Total\_Spending**: $\text{MntWines} + \text{MntFruits} + \text{MntMeat} + \text{MntFish} + \text{MntSweets} + \text{MntGold}$ (representing customer lifetime value).
4.  **Total\_Purchases**: $\text{NumWebPurchases} + \text{NumCatalogPurchases} + \text{NumStorePurchases}$ (representing purchasing volume).

---

## 5. Phase 4: Categorical Encoding & Correlation Analysis

### A. One-Hot vs. Ordinal Encoding
Machine learning algorithms and correlation matrices can only process numbers, so categorical columns must be encoded:
*   **Ordinal Encoding**: Used when categories have a natural ranking. In our case, `Education` is ordinal:
    $$\text{Basic} \rightarrow 0,\quad \text{2n Cycle} \rightarrow 1,\quad \text{Graduation} \rightarrow 2,\quad \text{Master} \rightarrow 3,\quad \text{PhD} \rightarrow 4$$
*   **One-Hot Encoding**: Used when categories have no natural order (nominal). In our case, `Marital_Status` and `Country` are nominal. We convert them into binary dummy columns ($0$ or $1$) to avoid implying any mathematical order.

### B. Correlation Heatmap
We use the **Pearson Correlation Coefficient ($r$)** to check relationships between pairs of variables:
$$r = \frac{\sum (x - \bar{x})(y - \bar{y})}{\sqrt{\sum (x - \bar{x})^2 \sum (y - \bar{y})^2}}$$
*   $r = 1$: Perfect positive correlation (e.g., as spending rises, purchases rise).
*   $r = 0$: No correlation.
*   $r = -1$: Perfect negative correlation (e.g., as children at home rise, spending falls).
*   A Seaborn heatmap visualizes this matrix using colors, helping us spot key patterns instantly.

---

## 6. Phase 5: Hypothesis Testing & Statistical Formulas

Hypothesis testing allows us to mathematically prove or disprove our business assumptions. We evaluate them at a significance level of $\alpha = 0.05$. If $p\text{-value} < 0.05$, we reject the null hypothesis ($H_0$) and accept the alternative ($H_a$).

### Hypothesis 1: Older individuals lean toward traditional in-store shopping.
*   **Null Hypothesis ($H_0$)**: There is no correlation (or a negative correlation) between `Age` and `NumStorePurchases`.
*   **Alternative Hypothesis ($H_a$)**: There is a significant positive correlation between `Age` and `NumStorePurchases`.
*   **Method**: Calculate Pearson's correlation coefficient ($r$) and its corresponding p-value. If $p < 0.05$ and $r > 0$, the hypothesis is supported.

### Hypothesis 2: Customers with children prefer online shopping.
*   **Null Hypothesis ($H_0$)**: Customers with children make the same or fewer web purchases than those without children.
*   **Alternative Hypothesis ($H_a$)**: Customers with children make significantly more web purchases.
*   **Method**: Independent Two-Sample T-Test:
    $$t = \frac{\bar{x}_1 - \bar{x}_2}{\sqrt{\frac{s_1^2}{n_1} + \frac{s_2^2}{n_2}}}$$
    Where Group 1 is Customers with Children, and Group 2 is Customers without Children.

### Hypothesis 3: Physical store sales face cannibalization by Web/Catalog sales.
*   **Null Hypothesis ($H_0$)**: There is no negative correlation between store purchases and web/catalog purchases.
*   **Alternative Hypothesis ($H_a$)**: There is a significant negative correlation between store purchases and web/catalog purchases (indicating that as online shopping increases, store shopping decreases).
*   **Method**: Examine Pearson's correlation coefficient. Note: If $r$ is positive, it means they are *complements* (active shoppers shop more across all channels), disproving cannibalization.

### Hypothesis 4: The US outperforms the rest of the world in purchase volumes.
*   **Null Hypothesis ($H_0$)**: The average purchase volume of US customers is less than or equal to the average purchase volume of the rest of the world.
*   **Alternative Hypothesis ($H_a$)**: US customers have a significantly higher average purchase volume.
*   **Method**: Independent Two-Sample T-test comparing `Total_Purchases` between the US group and the combined Non-US group.

---

## 7. Phase 6: Visual Analytics & Dashboarding
To communicate findings to business stakeholders, we choose specific chart types:
*   **Bar Plots (Product Performance)**: We aggregate and rank the total spending across Wine, Fruits, Meat, Fish, Sweets, and Gold to highlight cash cows and low-revenue products.
*   **Box Plots (Age vs. Response & Children vs. Spending)**: Box plots are excellent for showing the distribution (median, interquartile range, spread) of numerical spending across discrete categories.
*   **Complain Analysis (Education Breakdown)**: Segmented count plots or percentage bars showing the distribution of complaining customers across their educational backgrounds, helping customer service teams target resolution efforts.
