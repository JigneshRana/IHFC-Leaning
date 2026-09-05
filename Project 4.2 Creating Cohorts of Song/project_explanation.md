# Creating Cohorts of Songs: Detailed Methodology & Explanations

This document provides a comprehensive, step-by-step educational guide explaining **why** specific methods, mathematical techniques, and library functions are chosen for this analysis to create cohorts (groups) of songs from the Rolling Stones Spotify dataset, and how these cohorts power a personalized **Song Recommendation Engine** as taught in **Lesson 07: Recommendation Systems**.

---

## Table of Contents
1. [Phase 1: Project Setup & Exploratory Data Analysis (EDA)](#1-phase-1-project-setup--exploratory-data-analysis-eda)
2. [Phase 2: Data Cleaning, Wrangling & Scaling](#2-phase-2-data-cleaning-wrangling--scaling)
3. [Phase 3: Exploratory Analysis & Feature Engineering](#3-phase-3-exploratory-analysis--feature-engineering)
4. [Phase 4: Dimensionality Reduction (PCA)](#4-phase-4-dimensionality-reduction-pca)
5. [Phase 5: Cluster Analysis (Creating Cohorts)](#5-phase-5-cluster-analysis-creating-cohorts)
6. [Phase 6: Cohort Profiling & Sonic Definitions](#6-phase-6-cohort-profiling--sonic-definitions)
7. [Phase 7: Cohort-Based Recommendation Engine (Lesson 07 Integration)](#7-phase-7-cohort-based-recommendation-engine-lesson-07-integration)

---

## 1. Phase 1: Project Setup & Exploratory Data Analysis (EDA)
Before writing clustering models or recommendation algorithms, we must understand the structure, data types, and properties of the dataset.

### A. Inspecting Metadata (`df.info()`, `df.describe()`)
* **`df.shape`**: Reveals the dataset dimensions ($1,610$ rows and $17$ columns).
* **`df.info()`**: Shows column names, memory usage, and non-null counts. Confirms that text fields (`name`, `album`, `release_date`, `id`, `uri`) are stored as strings (`object`), integer fields (`track_number`, `popularity`, `duration_ms`) as integers, and audio metrics as floating-point numbers (`float64`).
* **`df.describe()`**: Computes summary statistics (mean, standard deviation, minimum, quartiles, and maximum). This provides immediate insight into scale disparities:
  * Audio metrics (`acousticness`, `danceability`, `energy`, `liveness`, `valence`, `speechiness`) are normalized within $[0.0, 1.0]$.
  * `loudness` ranges between $-24.4$ dB and $-2.1$ dB.
  * `tempo` ranges between $65.0$ and $216.0$ BPM.
  * `duration_ms` ranges between $21,000$ ms (21 seconds) and $981,866$ ms (16.3 minutes).

### B. Feature Categorization
To build a reliable machine learning model, we must separate columns into two distinct categories:
1. **Metadata & Identifier Columns:** `name`, `album`, `release_date`, `id`, `uri`, `track_number`. These identify tracks for human interpretation and recommendation lookups, but must **never** be passed into distance-based clustering algorithms.
2. **Audio Acoustic Features:** `acousticness`, `danceability`, `energy`, `instrumentalness`, `liveness`, `loudness`, `speechiness`, `tempo`, `valence`. These 9 numeric dimensions represent the perceptual and sonic properties of the sound recording.

---

## 2. Phase 2: Data Cleaning, Wrangling & Scaling

### A. Missing Value Detection & Handling
* **Missing Values (`df.isna().sum()`):** All $1,610$ entries are complete across all 17 columns. No imputation (`fillna`) or record deletion (`dropna`) is required.

### B. Duplicate Detection: Database IDs vs. Catalog Titles
In music streaming services like Spotify, duplicates exist at two distinct levels:
1. **Database-Level Duplicates (Unique Spotify `id`):**
   * Testing `df.duplicated(subset=['id']).sum()` yields **0 duplicate IDs**. Every single row represents a unique track entity in Spotify's catalog.
2. **Catalog-Level Song Title Duplicates (`name`):**
   * Testing `df.duplicated(subset=['name']).sum()` reveals **656 duplicate song titles**.
   * *Why do these exist?* The Rolling Stones have released original studio albums, 2009/2010 remasters, 50th Anniversary editions, Super Deluxe boxsets, and live concert recordings (e.g., *Licked Live In NYC*, *Live At The El Mocambo*).
   * *How should they be handled?*
     * **For Clustering:** We retain all 1,610 recordings. A live performance of *"Start Me Up"* has high `liveness` ($> 0.85$) and raw crowd energy, whereas the studio remastered track has high loudness and clean production. They represent distinct sonic profiles that belong to different cohorts.
     * **For Recommendation:** We implement **title deduplication** in our recommendation engine so a user querying *"Angie"* receives recommendations for different acoustic songs (e.g., *"Wild Horses"*, *"Fool To Cry"*), rather than 5 alternative masterings of *"Angie"*.

### C. Feature Scaling: Why Standardization is Crucial for Clustering
Clustering algorithms like K-Means calculate the distance (typically Euclidean distance) between data points to assign clusters:
$$d(\mathbf{p}, \mathbf{q}) = \sqrt{\sum_{i=1}^{n} (p_i - q_i)^2}$$

If features are unscaled, variables with large absolute ranges (e.g., `tempo` in BPM from 60 to 200, or `duration_ms` in hundreds of thousands) will completely dominate the distance metric. A difference of 20 BPM between two songs would create a squared difference of $400$, completely overwhelming a $0.2$ difference in `danceability` (squared difference of $0.04$).

#### 1. Standardization (Z-Score Normalization)
Standardization transforms each feature to have a mean ($\\mu$) of $0$ and a standard deviation ($\\sigma$) of $1$:
$$z = \frac{x - \mu}{\sigma}$$
* **Why chosen:** Z-score normalization preserves the shape of the underlying distribution, does not bound data artificially, and is the standard prerequisite for Principal Component Analysis (PCA) and K-Means clustering.

#### 2. Min-Max Normalization (Scaling to $[0, 1]$)
$$x_{\\text{norm}} = \frac{x - x_{\\text{min}}}{x_{\\text{max}} - x_{\\text{min}}}$$
* **Limitation:** Min-max scaling is sensitive to outliers. A single extreme outlier (e.g., an unusually long track or very quiet track) compresses all standard tracks into a tiny sub-interval, distorting distance relationships. Therefore, Z-Score standardization (`StandardScaler`) is preferred.

---

## 3. Phase 3: Exploratory Analysis & Feature Engineering

### A. Recommending Albums Based on Popular Songs
To recommend the best starting albums for a general audience:
1. We compute the **75th percentile** of song popularity across the entire catalog ($27.0$). Songs with a popularity score $\\ge 27.0$ represent the top quartile of the catalog.
2. We filter the dataset for tracks meeting this threshold and aggregate by album: `popular_tracks.groupby('album')['name'].count()`.
3. **Findings:**
   * **Honk (Deluxe):** 18 popular tracks. A comprehensive career-spanning greatest hits compilation featuring the band's most celebrated anthems across six decades.
   * **Exile On Main Street (2010 Re-Mastered):** 18 popular tracks. The definitive remastered studio release of the band's critically acclaimed double album.
   * *(Note: Exile On Main Street Deluxe Version also has 18 popular tracks, representing another release of the same core album).*
4. **Recommendation:** Recommend **Honk (Deluxe)** for greatest-hits appeal and **Exile On Main Street (2010 Re-Mastered)** for the essential studio album experience.

### B. Audio Feature Distributions
Histograms and Kernel Density Estimation (KDE) curves reveal the band's musical signature:
* `energy` is heavily left-skewed with a peak near $0.85$, demonstrating their hard-driving rock foundation.
* `liveness` exhibits a distinctive **bimodal distribution**: a cluster of studio recordings below $0.30$, and a secondary cluster of live concert tracks above $0.70$.
* `valence` averages $0.58$, confirming an upbeat, groove-heavy musical character.
* `speechiness` is uniformly low ($< 0.15$), confirming purely musical content without spoken word or rap segments.

### C. Overall Correlation with Popularity
Using Pearson's correlation coefficient ($r$):
$$r = \frac{\sum (x - \bar{x})(y - \bar{y})}{\sqrt{\sum (x - \bar{x})^2 \sum (y - \bar{y})^2}}$$
* **Positive Drivers:** `loudness` ($+0.156$) and `danceability` ($+0.141$). Louder, rhythmically stable, and punchier tracks receive higher streaming counts.
* **Negative Drivers:** `liveness` ($-0.206$) and `speechiness` ($-0.137$). Casual Spotify users stream studio tracks significantly more than live concert recordings.

### D. Temporal Evolution of Feature Correlations Across Decades
The problem statement explicitly prompts us to: *"Examine the relationship between a song's popularity and various factors, exploring how this correlation has evolved."*

By grouping tracks by release decade (1960s to 2020s) and computing the correlation of each audio feature with popularity, we discover major historical shifts:
1. **The Acoustic Era (1960s–1970s):**
   * In the 1960s ($+0.150$) and 1970s ($+0.184$), `acousticness` was **positively correlated** with popularity. Early folk-blues and melodic arrangements were massive hits.
   * From the 1980s onward, this correlation dropped to negative/neutral values ($-0.155$ to $-0.008$).
2. **The Danceability & Funk-Rock Era (1970s–2000s):**
   * `danceability` became strongly positive in the 1970s ($+0.214$), 1980s ($+0.239$), and 2000s ($+0.314$), corresponding to the band's groove-heavy dance-rock era (e.g., *"Miss You"*).
3. **The Loudness Wars & Modern Remastering (1980s–2010s):**
   * `loudness` had negligible correlation in the 1960s ($+0.007$), but surged to $+0.272$ in the 1980s, $+0.346$ in the 1990s, and $+0.354$ in the 2010s. Modern streaming listeners strongly favor dynamic, loudly mastered remasters.
4. **Persistent Live Track Penalty:**
   * In **every single decade**, `liveness` remained negatively correlated with popularity ($-0.127$ to $-0.407$), proving an enduring listener preference for clean studio recordings.

---

## 4. Phase 4: Dimensionality Reduction (PCA)

### A. The Curse of Dimensionality
When analyzing 9 continuous audio features:
1. **Space Sparsity:** The volume of the 9-dimensional space grows exponentially, causing data points to become sparse.
2. **Distance Convergence:** In high-dimensional spaces, the ratio between the distance to the nearest neighbor and the distance to the farthest neighbor approaches 1:
   $$\lim_{d \to \infty} \frac{d_{\\text{max}} - d_{\\text{min}}}{d_{\\text{min}}} = 0$$
   This makes distance-based algorithms like K-Means struggle because all songs start to appear equally distant.
3. **Multicollinearity:** Features like `energy` and `loudness` are strongly correlated ($r = +0.76$). Retaining both gives redundant weight to loudness/energy in Euclidean distance.

### B. Principal Component Analysis (PCA)
PCA transforms correlated variables into a set of linearly uncorrelated orthogonal variables called **Principal Components (PCs)**.
* Mathematically, PCA computes the eigenvectors $\mathbf{v}_i$ and eigenvalues $\lambda_i$ of the feature covariance matrix $\\mathbf{\\Sigma}$:
  $$\mathbf{\\Sigma} \mathbf{v}_i = \lambda_i \mathbf{v}_i$$
* The eigenvalues $\lambda_i$ quantify the variance captured by each principal component.

### C. Explained Variance Analysis
* **PC1:** Explains **$32.45\%$** of total variance.
* **PC2:** Explains **$17.94\%$** of total variance.
* **Cumulative PC1 + PC2:** Explains **$50.38\%$** of total variance, providing a high-fidelity 2D visual projection.
* **80% Threshold:** Capturing over $80\%$ of variance requires 5 components ($81.38\%$).

### D. Understanding Principal Components via Feature Loadings
By examining the eigenvector weights (loadings) for each feature on PC1 and PC2, we decode their musical meaning:
* **PC1 (Raw Live Concert Energy vs. Studio Polish):**
  * Strong positive loadings: `energy` ($+0.452$), `liveness` ($+0.444$), and `loudness` ($+0.370$).
  * Strong negative loading: `danceability` ($-0.418$).
  * *Interpretation:* Positive PC1 represents intense live concert jams with crowd noise; negative PC1 represents rhythmic, clean studio takes.
* **PC2 (Valence & Positivity vs. Mellow Acoustic):**
  * Strong positive loadings: `valence` ($+0.598$), `energy` ($+0.386$), and `loudness` ($+0.357$).
  * Strong negative loading: `acousticness` ($-0.448$).
  * *Interpretation:* Positive PC2 represents bright, cheerful, electric rock anthems; negative PC2 represents quiet, melancholic acoustic ballads.

---

## 5. Phase 5: Cluster Analysis (Creating Cohorts)

### A. Clustering Algorithm Selection
* **K-Means Clustering:** Chosen for this project because it partitions standardized audio spaces into spherical, evenly-sized, and highly interpretable clusters with clearly defined centroids.
* **Why not Hierarchical Clustering?** Hierarchical clustering requires computing an $\\mathcal{O}(N^2)$ distance matrix and is computationally expensive for large streaming catalogs, whereas K-Means scales linearly ($\\mathcal{O}(K \\cdot N \\cdot d)$).
* **Why not DBSCAN?** DBSCAN groups points by density and flags low-density points as noise (-1). In a music recommendation system, every song must belong to an actionable cohort; labeling 20% of the catalog as "unclusterable noise" is unacceptable for recommendation playlists.

### B. Finding the Optimal Number of Clusters ($K$)

#### 1. The Elbow Method (Inertia / WCSS)
* **Within-Cluster Sum of Squares (WCSS):**
  $$\text{WCSS} = \sum_{j=1}^{K} \sum_{x_i \in C_j} \|x_i - \mu_j\|^2$$
* Plotting WCSS against $K=2..9$ shows a distinct reduction in the rate of descent around **$K=3$**.

#### 2. Silhouette Analysis
* The Silhouette Coefficient for a song $i$ measures how close it is to its own cluster relative to the nearest neighboring cluster:
  $$s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}$$
  where $a(i)$ is the mean intra-cluster distance, and $b(i)$ is the mean nearest-cluster distance.
* **Metric Values:**
  * $K=2$: Silhouette Score = $0.2149$
  * $K=3$: Silhouette Score = **$0.1896$**
  * $K=4$: Silhouette Score = $0.1854$
  * $K=5$: Silhouette Score = $0.1599$
* **Selection Rationale:** While $K=2$ has a slightly higher silhouette score, it merely splits songs into "Live vs. Studio." Selecting **$K=3$** maintains a strong silhouette coefficient ($0.1896$) while providing the necessary granularity to separate studio tracks into **Upbeat Danceable Rock** and **Acoustic Melodic Ballads**.

### C. Cluster Centroids
The cluster centroids represent the mathematical prototypes of each cohort:
$$\mu_j = \frac{1}{|C_j|} \sum_{x_i \in C_j} x_i$$
Transforming the centroids using PCA allows us to plot them as prominent anchors in 2D space, illustrating the central sonic identity of each cohort.

---

## 6. Phase 6: Cohort Profiling & Sonic Definitions

Based on unscaled feature averages and standardized Z-score deviations, we define our three cohorts:

### 1. Cohort 0: High-Energy Live & Concert Recordings (596 tracks / 37.0%)
* **Sonic Profile:** Very high `liveness` ($0.821$, $+1.61\\sigma$), maximum `energy` ($0.924$, $+0.78\\sigma$), high `loudness` ($-5.38$ dB), and rapid `tempo` ($137.9$ BPM).
* **Musical Characteristics:** Live concert performances, stadium bootlegs, and raw stage recordings filled with crowd noise.
* **Representative Tracks:** *Loving Cup (Live)*, *The Last Time (Mono Version)*, *Sway (Live / 2009 Mix)*.
* **Target Audience:** Listeners seeking authentic live concert atmosphere and raw stadium energy.

### 2. Cohort 1: Upbeat & Danceable Studio Rock Hits (584 tracks / 36.3%)
* **Sonic Profile:** High `danceability` ($0.564$, $+0.59\\sigma$), high `energy` ($0.821$), exceptionally high `valence` ($0.789$, $+0.98\\sigma$ — happy, cheerful mood), and low `acousticness` ($0.186$).
* **Musical Characteristics:** Rhythmic studio rock anthems, funk-rock grooves, and cheerful, celebratory radio hits.
* **Representative Tracks:** *Paint It, Black*, *(I Can't Get No) Satisfaction*, *Start Me Up*.
* **Target Audience:** Perfect for workout playlists, party mixes, and road trips.

### 3. Cohort 2: Acoustic, Melodic & Slow Ballads (430 tracks / 26.7%)
* **Sonic Profile:** High `acousticness` ($0.430$, $+1.18\\sigma$), lower `loudness` ($-9.72$ dB), moderate `energy` ($0.571$), and relaxed `tempo` ($115.1$ BPM).
* **Musical Characteristics:** Softer, acoustic guitar-driven ballads, melodic blues songs, and introspective vocal-focused tracks.
* **Representative Tracks:** *Angie*, *Wild Horses*, *Gimme Shelter*, *Sympathy For The Devil*.
* **Target Audience:** Ideal for relaxation, acoustic listening, and quiet evening playlists.

---

## 7. Phase 7: Cohort-Based Recommendation Engine (Lesson 07 Integration)

### A. The Business Problem
As stated in the project scenario:
> *"Spotify aims to create cohorts of different songs to enhance song recommendations. These cohorts will be based on various relevant features, ensuring that each group contains similar types of songs."*

In **Lesson 07: Recommendation Systems**, we learned that industrial recommender systems use a **Two-Stage Architecture**:

```mermaid
graph LR
    A["Catalog (1,610+ Songs)"] --> B["Stage 1: Cohort Retrieval (K-Means)"]
    B --> C["Cohort Candidates (~500 Songs)"]
    C --> D["Stage 2: Similarity Ranking (Cosine Similarity)"]
    D --> E["Top N Recommendations"]
```

### B. Stage 1: Candidate Generation (Cohort Retrieval)
Comparing a user's song against every single track in a 100-million song catalog is computationally prohibitive. By assigning every song to a cohort during offline batch processing, the system retrieves a candidate pool of musically compatible tracks in $\\mathcal{O}(1)$ time using the nearest cluster centroid:
$$\text{Cohort}(x) = \arg\min_{j} \|x - \mu_j\|^2$$

### C. Stage 2: Similarity Ranking via Cosine Similarity
Within the retrieved cohort, the system ranks candidate songs using **Cosine Similarity** on standardized audio feature vectors:
$$\text{Cosine Similarity}(\mathbf{u}, \mathbf{v}) = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\|_2 \|\mathbf{v}\|_2} = \frac{\sum_{i=1}^{n} u_i v_i}{\sqrt{\sum_{i=1}^{n} u_i^2} \sqrt{\sum_{i=1}^{n} v_i^2}}$$

#### Why Cosine Similarity over Euclidean Distance?
1. **Directional Alignment:** Cosine similarity measures the angle between two feature vectors rather than their absolute magnitude. Two songs that share the exact same proportion of energy, valence, and acousticness will have a cosine similarity of $1.0$, even if one is slightly louder than the other.
2. **Normalized Scale:** Cosine similarity produces values bounded strictly within $[-1.0, +1.0]$, making similarity thresholds intuitive to interpret.

### D. Track Title Deduplication
A major real-world challenge in music catalogs is duplicate versions of the same composition across original albums, remasters, and compilation discs.
Our recommendation engine extracts the base title (stripping suffixes like `" - Live"`, `" - 2009 Mix"`, `" - Remaster"`) and drops duplicate base titles:
```python
candidates['base_title'] = candidates['name'].str.split(' - ').str[0].str.strip()
candidates = candidates.drop_duplicates(subset=['base_title'])
```
This guarantees that when a user requests recommendations for *"Angie"*, the system suggests different songs (*"Till The Next Goodbye"*, *"Fool To Cry"*, *"Wild Horses"*) rather than alternate releases of *"Angie"*.

### E. Addressing the Cold-Start Problem
Collaborative Filtering algorithms (like Matrix Factorization or User-User CF) suffer from the **Item Cold-Start Problem**: when a brand-new song is released, it has zero user listening history, zero ratings, and cannot be recommended.
Because our cohort recommender relies strictly on **audio content features**, as soon as Spotify's audio analysis pipeline computes the 9 acoustic metrics for a new track:
1. The track is scaled using `StandardScaler`.
2. The model determines its cohort via `kmeans.predict()`.
3. The track can be immediately recommended to users listening to that cohort with zero cold-start delay!

### F. Hybrid Recommender Architecture
In production streaming platforms, this content-based cohort engine is combined with Collaborative Filtering to form a **Hybrid Recommender**:
1. **Cohort Filter (Content):** Ensures candidate songs match the listener's current acoustic mood (e.g. acoustic ballads).
2. **Collaborative Filter (User Preferences):** Ranks the cohort candidates based on which songs the user or similar users have liked in the past.
