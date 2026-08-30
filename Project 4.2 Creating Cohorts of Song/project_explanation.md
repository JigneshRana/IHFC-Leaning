# Creating Cohorts of Songs: Detailed Methodology & Explanations

This document provides a comprehensive, step-by-step educational guide explaining **why** specific methods, mathematical techniques, and library functions are chosen for this analysis to create cohorts (groups) of songs from the Rolling Stones Spotify dataset.

---

## Table of Contents
1. [Phase 1: Project Setup & Exploratory Data Analysis (EDA)](#1-phase-1-project-setup--exploratory-data-analysis-eda)
2. [Phase 2: Data Cleaning & Wrangling](#2-phase-2-data-cleaning--wrangling)
3. [Phase 3: Exploratory Analysis & Feature Engineering](#3-phase-3-exploratory-analysis--feature-engineering)
4. [Phase 4: Dimensionality Reduction](#4-phase-4-dimensionality-reduction)
5. [Phase 5: Cluster Analysis (Creating Cohorts)](#5-phase-5-cluster-analysis-creating-cohorts)
6. [Phase 6: Cluster Profiling & Recommendations](#6-phase-6-cluster-profiling--recommendations)

---

## 1. Phase 1: Project Setup & Exploratory Data Analysis (EDA)
Before writing clustering models, we must understand the structure and properties of the dataset.

### Why do we inspect metadata using `df.info()`, `df.head()`, and `df.describe()`?
* **`df.shape`**: Tells us the size of the dataset (number of songs and features). For this dataset, we can see how many Rolling Stones songs are available on Spotify.
* **`df.info()`**: Shows us column names, data types, and non-null counts. For instance, `release_date` might be loaded as a string (object) and needs to be parsed into a `datetime` object to extract the release year.
* **`df.head()`**: Gives us a quick visual confirmation of the first few rows. This helps us see if there are row index columns, formatting issues, or obvious placeholder values.
* **`df.describe()`**: Generates summary statistics (mean, standard deviation, minimum, maximum, and percentiles) for numerical columns. This is our first clue about features that are on different scales (e.g., `loudness` ranges from $-60$ to $0$ dB, `tempo` is in BPM from $60$ to $200+$, while `acousticness` is strictly between $0.0$ and $1.0$).

---

## 2. Phase 2: Data Cleaning & Wrangling
Raw data collected from APIs like Spotify often contains duplicates, missing values, or inconsistent entries.

### A. Missing Value Detection and Treatment
* **Missing Values (`isna()`)**: If any values are missing, we must decide whether to drop them or fill them.
  * *Dropping (`dropna()`)*: Safe only if the missing data is extremely minimal (< 2% of the dataset) and random.
  * *Imputation (`fillna()`)*: If essential features (like `danceability` or `energy`) are missing, we can fill them with the **median** value of that column (the median is preferred over the mean because it is not affected by extreme outliers).
* **Duplicate Detection (`duplicated()`)**: Spotify albums sometimes contain duplicate tracks (e.g., a studio version and a remaster or live version). For clustering cohorts of unique songs, we should identify and remove duplicates by matching song names and album names, or using the unique `id`.

### B. Feature Scaling (Why standardizing is crucial for Clustering)
Clustering algorithms like K-Means calculate the distance (usually Euclidean distance) between data points to group them.
If one feature is `duration_ms` (values in hundreds of thousands, e.g., $250,000$ ms) and another is `danceability` (values between $0$ and $1$), the distance formula will be completely dominated by the duration. A difference of $10,000$ ms will dwarf a $0.5$ difference in danceability.

To solve this, we must scale our features so they contribute equally to the distance calculation.

#### 1. Standardization (Z-Score Normalization)
Standardization centers the data so that the mean ($\mu$) is $0$ and the standard deviation ($\sigma$) is $1$.
$$x_{\text{standardized}} = \frac{x - \mu}{\sigma}$$
* **When to use**: Standardization is robust to outliers and is the standard choice for K-Means clustering and Principal Component Analysis (PCA) because it preserves the shape of the original distribution while making all features comparable.

#### 2. Min-Max Normalization (Scaling to $[0, 1]$)
Transforms the data to fit strictly within a bounded range, usually $[0, 1]$.
$$x_{\text{normalized}} = \frac{x - x_{\text{min}}}{x_{\text{max}} - x_{\text{min}}}$$
* **When to use**: Useful when the distribution is not normal (Gaussian) and we want strict boundaries. However, extreme outliers will squash the normal values into a very tight cluster, which is why Z-Score Standardization is preferred for clustering.

---

## 3. Phase 3: Exploratory Analysis & Feature Engineering
Exploratory Data Analysis (EDA) helps us discover patterns, trends, and relationships.

### A. Recommending Albums based on Popular Songs
To recommend the two best albums based on the number of popular songs:
1. Define a threshold for "popular songs" (e.g., songs with a `popularity` score in the top 25% or above a fixed score like $40$ or $50$).
2. Count the number of popular songs in each album using a group-by operation: `df[df['popularity'] >= threshold].groupby('album')['name'].count()`.
3. Sort the results in descending order and identify the top 2 albums.

### B. Analyzing Correlation and Evolution of Popularity
* **Correlation**: We use Pearson's correlation coefficient ($r$) to measure the strength of linear relationship between `popularity` and song characteristics (e.g., does high `energy` or high `danceability` correlate with higher popularity?).
  $$r = \frac{\sum (x - \bar{x})(y - \bar{y})}{\sqrt{\sum (x - \bar{x})^2 \sum (y - \bar{y})^2}}$$
* **Temporal Evolution**: By extracting the year from the `release_date`, we can plot how popularity and audio features have evolved over the decades. The Rolling Stones have been active since the 1960s, allowing us to see if newer or older songs are more popular today.

---

## 4. Dimensionality Reduction
When clustering, we analyze many features (danceability, energy, acousticness, valence, tempo, liveness, loudness, etc.). This is high-dimensional space.

### The Curse of Dimensionality
As the number of dimensions (features) increases:
1. The volume of the space increases exponentially, making the data points very sparse.
2. The distance between any two points converges, meaning that all points start to look equally far apart. This makes distance-based clustering (like K-Means) perform poorly.
3. Visualization is impossible because humans can only perceive 2D or 3D plots.

### Principal Component Analysis (PCA)
PCA is a mathematical technique that reduces the dimensions of the data while retaining as much of the variance (information) as possible.
* **How it works**: It finds new, perpendicular axes called **Principal Components (PCs)**.
  * **PC1** is the axis along which the data varies the most.
  * **PC2** is perpendicular to PC1 and captures the second highest amount of variance.
* **Why it's significant**:
  * **Noise Reduction**: By keeping only the top principal components, we discard low-variance components which often represent noise.
  * **Visualization**: We can project our multi-dimensional songs onto a 2D scatter plot (using PC1 and PC2) to visually check if natural clusters exist.
  * **Computational Efficiency**: Running clustering algorithms on 2 or 3 principal components instead of 10+ features is much faster.

---

## 5. Cluster Analysis (Creating Cohorts)
The primary objective is to group songs into cohorts based on their audio features.

### A. K-Means Clustering Algorithm
K-Means is a centroid-based clustering algorithm.
1. **Initialization**: Choose $K$ initial points at random to act as cluster centers (centroids).
2. **Assignment**: Assign each song to the nearest centroid based on Euclidean distance:
   $$d(p, q) = \sqrt{\sum_{i=1}^{n} (p_i - q_i)^2}$$
3. **Update**: Recalculate the centroid of each cluster as the mean of all songs assigned to it.
4. **Repeat**: Repeat steps 2 and 3 until the centroids no longer move or the maximum iterations are reached.

### B. Finding the Optimal Number of Clusters ($K$)
Choosing the right $K$ is critical. We use two complementary techniques:

#### 1. The Elbow Method (Inertia)
* **Inertia** is the sum of squared distances of samples to their closest cluster center (Within-Cluster Sum of Squares - WCSS):
  $$\text{WCSS} = \sum_{j=1}^{K} \sum_{i \in C_j} ||x_i - \mu_j||^2$$
* As $K$ increases, WCSS will naturally decrease (because with more centroids, points are closer to their centers).
* **The Elbow**: We plot WCSS against $K$. The point where the rate of decrease changes sharply (forming an "elbow") represents the optimal balance between cluster compactness and simplicity.

#### 2. Silhouette Analysis
* The **Silhouette Coefficient** measures how similar a song is to its own cluster compared to other clusters.
* For a song $i$:
  $$s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}$$
  where:
  * $a(i)$ is the average distance between song $i$ and all other songs in the same cluster.
  * $b(i)$ is the average distance between song $i$ and the songs in the closest neighboring cluster.
* The average silhouette score ranges from $-1$ to $+1$:
  * **Close to +1**: The song is far away from neighboring clusters (well clustered).
  * **Close to 0**: The song is on the boundary between two clusters.
  * **Close to -1**: The song is placed in the wrong cluster.
* We select the $K$ that maximizes the average silhouette score.

---

## 6. Cluster Profiling & Recommendations
Once the cohorts are created, we must translate the math back into musical descriptions.

### Defining Cohorts by Feature Averages
By calculating the mean of each audio feature within each cluster, we can label the cohorts:
* **Cohort A (e.g., "Acoustic Ballads")**: High acousticness, low energy, low loudness, slow tempo.
* **Cohort B (e.g., "High-Energy Rock Anthems")**: High energy, high loudness, high valence (happy/cheerful), low acousticness.
* **Cohort C (e.g., "Live Concert Hits")**: High liveness (crowd noise), high energy, high loudness.
* **Cohort D (e.g., "Danceable Grooves")**: High danceability, strong rhythm, moderate tempo.

These cohorts can be fed directly into a recommendation engine. If a listener enjoys a song in "High-Energy Rock Anthems," the system should recommend other songs from that specific cohort.
