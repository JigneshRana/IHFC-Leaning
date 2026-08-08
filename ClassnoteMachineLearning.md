# Machine Learning — Complete Class Notes & Concepts

Welcome to **`ClassnoteMachineLearning.md`**! This document serves as a comprehensive, structured repository for Machine Learning notes, theoretical foundations, mathematical formulations, code implementations, and practical workflows.

---

## 📂 Table of Contents
0. [Key Industry Terminology](#0-key-industry-terminology)
1. [Overview & Machine Learning Fundamentals](#1-overview--machine-learning-fundamentals)
   - [1.0 Historical Evolution & Journey](#10-the-historical-evolution--journey-of-machine-learning)
   - [1.1 Hierarchy (AI, ML, DL)](#11-hierarchy-of-artificial-intelligence-machine-learning--deep-learning)
   - [1.2 Classical Programming vs. Machine Learning Paradigm](#12-classical-programming-vs-machine-learning-paradigm)
   - [1.3 Modern AI Architectural Spectrum & Learning Path](#13-modern-ai-industry-architectural-spectrum--learning-path)
   - [1.4 Core ML & Deep Learning Libraries Ecosystem](#14-core-ml--deep-learning-libraries-ecosystem)
2. [Supervised Learning](#2-supervised-learning)
   - [Regression Models](#regression-models)
   - [Classification Models](#classification-models)
3. [Unsupervised Learning](#3-unsupervised-learning)
   - [Clustering](#clustering)
   - [Dimensionality Reduction](#dimensionality-reduction)
4. [Feature Engineering & Data Preprocessing](#4-feature-engineering--data-preprocessing)
5. [Model Evaluation & Validation](#5-model-evaluation--validation)
6. [Bias-Variance Tradeoff & Regularization](#6-bias-variance-tradeoff--regularization)
7. [Ensemble Techniques](#7-ensemble-techniques)
8. [Practical Machine Learning Workflow](#8-practical-machine-learning-workflow)
9. [Code Snippets & Cheatsheets](#9-code-snippets--cheatsheets)
10. [Reference Links & External Learning Resources](#10-reference-links--external-learning-resources)

---

## 0. Key Industry Terminology

| Term | 1 to 2 Line Definition / Description |
| :--- | :--- |
| **ML (Machine Learning)** | A branch of Artificial Intelligence where algorithms learn statistical patterns from data to make predictions or decisions automatically without being explicitly programmed. |
| **MLOps (Machine Learning Operations)** | The engineering practice of automating and managing the entire ML lifecycle—including data pipelines, CI/CD deployment, continuous monitoring, and model governance in production. |
| **Generative AI (GenAI)** | A category of AI models trained on vast datasets to create brand new content (such as text, images, code, audio, and video) rather than just categorizing existing data. |
| **LLM (Large Language Model)** | A deep neural network (typically Transformer-based) trained on massive text corpora with billions of parameters to understand, summarize, translate, and generate human language. |
| **Base Model (Foundation Model)** | A large-scale neural network pre-trained on broad data that serves as a general foundation, designed to be fine-tuned or adapted for specific downstream tasks. |
| **Base LLM** | A raw Large Language Model trained strictly to complete text by predicting the next token, before any instruction-tuning (IT) or Reinforcement Learning from Human Feedback (RLHF) is applied. |
| **RAG (Retrieval-Augmented Generation)** | An architecture framework that enhances LLM responses by retrieving relevant facts/documents from external databases or vector stores before generating answers, preventing hallucinations. |
| **MCP (Model Context Protocol)** | An open standard protocol that enables AI models and agents to securely connect to external tools, databases, local file systems, and API services through a standardized interface. |
| **MultiAgent (Multi-Agent System)** | An architectural framework where multiple specialized AI agents collaborate, divide complex workflows, share context, and coordinate tool execution to accomplish tasks beyond single-agent capabilities. |
| **Transformer** | A breakthrough neural network architecture relying on self-attention mechanisms to process sequential data in parallel, serving as the core foundation for modern LLMs and Generative AI. |

---

## 1. Overview & Machine Learning Fundamentals

### 1.0 The Historical Evolution & Journey of Machine Learning

#### 🏛️ The Coining of "Machine Learning" (Arthur Samuel, 1959)
* **Arthur Samuel** (an IBM computer scientist and pioneer in computer gaming) coined the term **"Machine Learning"** in **1959**.
* He demonstrated this by creating a self-learning **Checkers program** on the IBM 704 computer. The program played thousands of games against itself, calculated board positions, and gradually learned to beat human players.
* **Arthur Samuel's Classic Definition**:
  > *"Machine Learning is the field of study that gives computers the ability to learn without being explicitly programmed."*

---

#### 📈 Andrew Ng's Scale Paradigm: Traditional ML vs. Deep Learning

Pioneering AI educator **Andrew Ng** illustrated a fundamental law of AI scalability that explains why the industry shifted from traditional ML to Deep Learning and LLMs:

```
  Model Performance
         ▲
         │                                       /  Large Neural Nets / Transformers (LLMs)
         │                                      /
         │                                     /   Medium Neural Nets
         │                                    /
         │  ---------------------------------/     Small Neural Nets
         │  --------------------------------/      Traditional ML (SVMs, Random Forests, Logistic Reg)
         │  ───────────────────────────────/       (Performance plateaus as data volume increases)
         │
         └─────────────────────────────────────────────────────────────────► Amount of Data / Compute
```

* **Traditional ML Bottleneck**: Algorithms like Support Vector Machines (SVMs), Logistic Regression, and Decision Trees hit a **performance plateau**. Adding millions of extra data points yields diminishing returns.
* **Deep Learning & LLM Scaling Law**: Large Neural Networks (Deep Learning and Transformers) scale continuously—the more data and compute you feed them, the better their performance becomes ($P \propto \text{Data} \times \text{Parameters} \times \text{Compute}$).

---

#### ⏳ Complete Chronological Timeline of Machine Learning & AI

```mermaid
timeline
    title Journey & Milestones of Machine Learning
    1950 : Turing Test (Alan Turing)
    1957 : Perceptron (Frank Rosenblatt)
    1959 : "Machine Learning" Coined (Arthur Samuel - IBM Checkers)
    1969 : XOR Limitation & 1st AI Winter (Minsky & Papert)
    1986 : Backpropagation Breakthrough (Rumelhart, Hinton, Williams)
    1990s : Statistical ML Era (SVMs, Random Forests, Decision Trees)
    1997 : Deep Blue beats Kasparov | LSTM Invented (Schmidhuber)
    2012 : AlexNet & Deep Learning Revolution (Hinton et al. ImageNet)
    2014 : GANs (Goodfellow) & Adam Optimizer
    2017 : Transformer Architecture ("Attention Is All You Need" - Google)
    2018 : BERT (Google) & GPT-1 (OpenAI)
    2020 : GPT-3 Emergence (175B Parameters)
    2022 : ChatGPT (GPT-3.5) & RLHF Alignment Mainstream Wave
    2023-2026 : GPT-4, Gemini (Google), Llama 3 (Meta), RAG & Autonomous AI Agents
```

| Era / Year | Milestone / Breakthrough | Significance & Impact |
| :--- | :--- | :--- |
| **1950** | **Turing Test (Alan Turing)** | Introduced the concept of machine intelligence testing ("Can machines think?"). |
| **1957** | **Perceptron (Frank Rosenblatt)** | First hardware implementation of an artificial single-layer neural network. |
| **1959** | **Arthur Samuel Coined "Machine Learning"** | Demonstrated self-learning software using an IBM Checkers program. |
| **1969** | **Minsky & Papert's "Perceptrons" Book** | Proved single-layer perceptrons couldn't solve XOR; triggered the 1st "AI Winter". |
| **1986** | **Backpropagation Algorithm** | Popularized by Geoffrey Hinton et al., enabling efficient training of multi-layer neural networks. |
| **1990s** | **Statistical Machine Learning Era** | Vapnik's Support Vector Machines (SVM) and Breiman's Random Forests dominated industry applications. |
| **1997** | **Deep Blue & LSTM Networks** | IBM's Deep Blue beat chess champion Kasparov; Hochreiter & Schmidhuber introduced LSTMs. |
| **2012** | **AlexNet (ImageNet Victory)** | GPU-accelerated deep CNN smashed vision benchmarks; kicked off the **Deep Learning Revolution**. |
| **2017** | **Transformer Architecture (Google)** | Paper *"Attention Is All You Need"* replaced RNNs/LSTMs with parallel self-attention mechanisms. |
| **2018 - 2020** | **BERT & GPT Series (OpenAI/Google)** | Self-supervised pre-training at scale; GPT-3 demonstrated zero-shot and few-shot language capabilities. |
| **2022** | **ChatGPT & RLHF Alignment** | Reinforcement Learning from Human Feedback (RLHF) made conversational GenAI accessible globally. |
| **2023 - 2026+** | **Multimodal LLMs & AI Agents** | GPT-4, Gemini, Claude, Llama 3, RAG systems, and autonomous multi-agent tool execution. |

---

### 1.1 Hierarchy of Artificial Intelligence, Machine Learning & Deep Learning

![Artificial Intelligence, Machine Learning, and Deep Learning Hierarchy](images/ai_ml_dl_hierarchy.png)

```
┌──────────────────────────────────────────────────────────────────────────┐
│ ARTIFICIAL INTELLIGENCE (AI)                                             │
│  Systems or machines mimicking human intelligence to perform tasks       │
│  ┌────────────────────────────────────────────────────────────────────┐  │
│  │ MACHINE LEARNING (ML)                                              │  │
│  │  Algorithms learning statistical patterns from data automatically │  │
│  │  ┌──────────────────────────────────────────────────────────────┐  │  │
│  │  │ DEEP LEARNING (DL)                                           │  │  │
│  │  │  Multi-layer neural networks (Transformers, CNNs, RNNs)     │  │  │
│  │  └──────────────────────────────────────────────────────────────┘  │  │
│  └────────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────────┘
```

* **Artificial Intelligence (AI)**: The broader field of creating intelligent systems capable of reasoning, problem-solving, and decision-making.
* **Machine Learning (ML)**: A sub-field of AI focused on learning mapping functions $f(X) \to y$ from data without explicit hardcoded rules.
* **Deep Learning (DL)**: A sub-field of ML based on deep artificial neural networks capable of learning hierarchical feature representations directly from raw unstructured data.

---

### 1.2 Classical Programming vs. Machine Learning Paradigm

![Classical Programming vs Machine Learning Paradigm](images/classical_vs_machine_learning.png)

```
Classical Programming Paradigm (Software 1.0):
┌───────────┐
│   Rules   │──┐
└───────────┘  │    ┌──────────────────────┐
               ├────► Classical Programming ├────► Answers
┌───────────┐  │    └──────────────────────┘
│   Data    │──┘
└───────────┘

Machine Learning Paradigm (Software 2.0):
┌───────────┐
│   Data    │──┐
└───────────┘  │    ┌──────────────────────┐
               ├────►   Machine Learning   ├────► Rules (Trained Model)
┌───────────┐  │    └──────────────────────┘
│  Answers  │──┘
└───────────┘
```

#### Detailed Breakdown & Comparison

| Dimension | Classical Programming (Software 1.0) | Machine Learning Paradigm (Software 2.0) |
| :--- | :--- | :--- |
| **Input Ingredients** | Human-written **Rules** (if-else logic, explicit math formulas) $+$ Input **Data**. | Historical **Data** (features $X$) $+$ Ground-truth **Answers** (labels $y$). |
| **Output Result** | Computed **Answers** (calculated output values). | Learned **Rules** (trained model function $f(X) \to y$, parameters $\theta$). |
| **Who Writes the Rules?** | Human programmers explicitly code every single condition and edge case. | The ML algorithm automatically extracts and learns rules by analyzing patterns. |
| **Handling Complexity** | Becomes brittle and unmaintainable for complex tasks (e.g. face recognition, sentiment analysis). | Excels at high-dimensional, unstructured, complex problems (images, audio, text, sensor data). |
| **Analogy** | Following a rigid, hardcoded baking recipe to make a cake. | Feeding a chef 1,000 cakes and raw ingredients so the chef deduces the recipes. |

* **Technical Definition (Tom Mitchell, 1997)**: A computer program is said to learn from experience **E** with respect to some class of tasks **T** and performance measure **P**, if its performance at tasks in **T**, as measured by **P**, improves with experience **E**.

---

#### 💡 How Machine Learning Works: Representation Learning & Coordinate Transformations

![Data Representation Learning & Coordinate Change](images/coordinate_change_representation.png)

```
Step 1: Raw Data                    Step 2: Coordinate Change              Step 3: Better Representation
(Entangled in original space)       (Mathematical transformation)          (Linearly separable at x = 0)
        y │                                y │                                    y │
          │  ●   ●                           │ \  ●   ●                             │  │
       ○  │    ●   ●                      ○  │  \   ●   ●                        ○   │  │ ●   ●
     ○  ○ │  ●                              ○  ○ │   \  ●                        ○  ○  │  │   ●   ●
       ○  │    ●                              ○  │ \  \   ●                        ○   │  │ ●
          │  ●                                   │  \  \                         ───┼──┼──────────► x
   ───────┼──────────► x                  ───────┼───\──\───► x                     │  │ ●
          │                                      │    \  \                          │  │   ●
```

##### The Core Mechanism of Machine Learning:
* **Machine Learning is Searching for Useful Representations**: Machine Learning algorithms do not just "memorize" data; they apply **geometric transformations** (rotations, scaling, projections, activations) to project raw input data into a new coordinate space where the underlying pattern or decision boundary becomes simple and linearly separable.
* **Classical ML vs. Deep Learning Representation**:
  * **Classical ML**: Relies on engineers to manually engineer coordinate transformations (e.g. PCA, SVM kernel trick, polynomial projections).
  * **Deep Learning**: Automatically learns complex multi-layered coordinate transformations sequentially across neural network layers.

---

### 1.3 Modern AI Industry Architectural Spectrum & Learning Path

![Modern AI & MLOps Architectural Spectrum](images/ai_mlops_agent_architecture.png)

```mermaid
flowchart TD
    subgraph Traditional_ML["Path 1: Traditional Custom ML Pipeline (↑ Time, ↓ Compute/Resource)"]
        Data[Data Input] --> ManualML[Manual ML Pipeline]
        ManualML --> MLOps[MLOps System]
        MLOps --> S1[Server Deployment]
        MLOps --> S2[Local Device]
        MLOps --> S3[API Gateways]
    end

    subgraph GenAI_Architecture["Path 2: Modern GenAI & Foundation Model Spectrum (↓ Time, ↑ Compute/Resource)"]
        TF[Transformer Architecture] --> BaseLLM[Base LLM / Foundation Model]
        
        %% Fine-tuning path
        BaseLLM --> FTData[Data + Fine-tuning]
        FTData --> FTModel[Fine-Tuned Model]
        FTModel --> Cloud1[Cloud Server]

        %% RAG path
        BaseLLM --> RAG[RAG Application]
        RAG --> Cloud2[Cloud Server]

        %% Multi-Agent path
        MultiLLM[Multiple LLMs] --> Decider[Brain / Decider System]
        RAG <--> Decider
        Decider --> Agent[Autonomous Agent]
        Agent --> Cloud3[Cloud Deployment]
        Agent --> EdgeMachine[Local Machine Execution]
    end
```

#### Comparison of Architectural Trade-offs
| Spectrum / Approach | Time to Value ($\text{Time}$) | Resource & Compute Requirements ($\text{Resources}$) | Infrastructure & Deployment | Primary Use Cases |
| :--- | :--- | :--- | :--- | :--- |
| **Traditional Custom ML** | **Higher Time ($\uparrow$)**: Requires manual feature engineering, custom training & tuning. | **Lower Resources ($\downarrow$)**: Runs efficiently on standard CPUs / lightweight servers. | Deployed via MLOps pipelines to Local, Server, or API Gateways. | Structured tabular data, Fraud scoring, Churn prediction, Sensor metrics. |
| **Foundation Models / GenAI** | **Lower Time ($\downarrow$)**: Pre-trained Base LLMs provide immediate out-of-the-box capabilities. | **Higher Resources ($\uparrow$)**: Requires GPU clusters, high memory, and token budget. | Deployed on Cloud Servers, Vector DBs (RAG), and Autonomous Agent Runtimes. | Text generation, Document Q&A (RAG), Code synthesis, Multi-Agent decision-making. |

---

### Primary Types of Machine Learning

```
                               ┌───────────────────────────┐
                               │ Machine Learning Types    │
                               └─────────────┬─────────────┘
                                             │
       ┌──────────────────────┬──────────────┴───────────────┬──────────────────────┐
       ▼                      ▼                              ▼                      ▼
┌──────────────┐       ┌──────────────┐              ┌──────────────┐       ┌──────────────┐
│ 1.Supervised │       │2.Unsupervised│              │3.Semi-Superv.│       │4.Reinforcement│
│   Learning   │       │   Learning   │              │   Learning   │       │   Learning   │
└──────┬───────┘       └──────────────┘              └──────────────┘       └──────┬───────┘
       │ (Labeled Data)  (Unlabeled Data)             (Small Labeled +             │ (Environment &
       ├───────────────┐                              Large Unlabeled)             │  Feedback Loop)
       ▼               ▼                                                           ▼
┌─────────────┐ ┌──────────────┐                                            ┌──────────────┐
│ Regression  │ │Classification│                                            │  Recurrent   │
│ (Continuous)│ │ (Categories) │                                            │Feedback Loop │
└─────────────┘ └──────────────┘                                            └──────────────┘
       ▲               ▲
       └───────┬───────┘
               │ (Meta-Technique)
        ┌──────────────┐
        │   Ensemble   │
        │   Learning   │
        └──────────────┘
```

| # | Type / Paradigm | Data & Signal Source | Learning Mechanism & Objective | Key Examples & Use Cases |
| :--- | :--- | :--- | :--- | :--- |
| **1** | **Supervised Learning** | Fully Labeled dataset $(X, y)$ | Learns an explicit mapping function $f(X) \to y$ from known input-output pairs. | Spam detection, House pricing, Credit scoring |
| **2** | **Unsupervised Learning** | Unlabeled dataset $(X)$ | Discovers intrinsic patterns, clusters, hidden structures, or reduced representations without targets. | Customer segmentation, Anomaly detection, PCA |
| **3** | **Semi-Supervised Learning** | Small labeled dataset $+$ large volume of unlabeled data | Leverages unlabeled data distribution to improve learning accuracy when labeling is expensive. | Medical image diagnosis, Protein annotation, Web categorization |
| **4** | **Reinforcement Learning** | Environment, Agent, States ($S$), Actions ($A$), and Reward signals ($R$) | Learns an optimal policy $\pi(s)$ via **feedback loops** (trial-and-error) to maximize long-term cumulative reward. | Robotics, Autonomous vehicles, Game AI (Chess/Go), RLHF (LLM alignment) |

---

### Core Sub-Types & Model Combinations

1. **Regression (Sub-type)**: Output is a continuous numerical value (e.g., salary, price, temperature).
2. **Classification (Sub-type)**: Maps continuous inputs/probabilities into discrete groups/categories (e.g., $0$ vs $1$, Cat vs Dog).
3. **Ensemble Learning (Meta-type / Multi-model Synergy)**: A powerful approach where **multiple models work together** to produce a single, superior prediction:
   * **Bagging (Parallel)**: Trains multiple models independently on different data subsets and averages their votes (e.g., Random Forest).
   * **Boosting (Sequential)**: Trains models in series, where each new model focuses on correcting the errors/feedback of previous models (e.g., XGBoost, LightGBM).
   * **Stacking (Heterogeneous)**: Combines predictions of completely different model types using a meta-model.

---

### 1.4 Core ML & Deep Learning Libraries Ecosystem

| Library | Backed / Created By | Primary Domain & Focus | Key Capabilities & Features | Typical Import & Usage |
| :--- | :--- | :--- | :--- | :--- |
| **SciPy** | SciPy / NumFOCUS Community | Scientific Computing & Advanced Math | Advanced numerical integration, optimization (`scipy.optimize`), hypothesis testing & probability (`scipy.stats`), linear algebra, and signal processing. | `import scipy as sp`<br>`from scipy import stats, optimize` |
| **Scikit-learn (`scikit`)** | INRIA / Open-Source Community | Traditional Machine Learning | Industry-standard toolkit for classical ML: classification, regression, clustering, feature scaling (`StandardScaler`), model selection, and pipelines. | `import sklearn`<br>`from sklearn.ensemble import RandomForestClassifier` |
| **TensorFlow** | **Google** (Google Brain / DeepMind) | Industrial & Enterprise Deep Learning | End-to-end production DL framework with Keras API, static graph compilation, mobile/edge deployment (TF Lite), and TF Serving for high-throughput production. | `import tensorflow as tf`<br>`from tensorflow import keras` |
| **PyTorch** | **Meta** (FAIR / PyTorch Foundation) | Research & Generative AI Deep Learning | Dynamic computational graphs (`autograd`), highly pythonic syntax, dominant ecosystem for Transformers, LLMs (Hugging Face), and modern GenAI research. | `import torch`<br>`import torch.nn as nn` |
| **Keras** | François Chollet / **Google** & Community | High-Level Deep Learning API | User-friendly, high-level API designed for fast experimentation. Keras 3 functions as a multi-backend API supporting **TensorFlow**, **PyTorch**, and **JAX** seamlessly. | `import keras`<br>`from keras import layers, models` |

---

## 2. Supervised Learning

![Supervised Learning Method & Workflow](images/supervised_learning_diagram.png)

> **Key Rule**: A supervised learning method uses labeled data to predict outcomes guided by specific input-output pairs. Here, both inputs and outputs are known.

Supervised Learning is divided into two primary categories based on the nature of the target variable $y$: **Regression** and **Classification**.

### 💡 Core Distinction: Regression vs. Classification

```
                          ┌──────────────────────────┐
                          │   Supervised Learning    │
                          └────────────┬─────────────┘
                                       │
            ┌──────────────────────────┴──────────────────────────┐
            ▼                                                     ▼
  ┌───────────────────┐                                 ┌───────────────────┐
  │    REGRESSION     │                                 │  CLASSIFICATION   │
  │ Continuous Output │                                 │ Categorical/Group │
  │  y ∈ (-∞, +∞)     │                                 │     y ∈ {0, 1}    │
  └───────────────────┘                                 └───────────────────┘
   Predicts a quantity                                   Predicts a category
   (e.g., $450,000, 36.6°C)                             (e.g., 0: Safe, 1: Spam)
```

#### Detailed Comparison Table
| Feature / Dimension | Regression Models | Classification Models |
| :--- | :--- | :--- |
| **Output Type** | Continuous numerical value ($y \in \mathbb{R}$) | Discrete label / category ($y \in \{0, 1\}$ or $\{C_1, C_2, \dots, C_K\}$) |
| **Primary Goal** | Estimate a quantity or value on a continuous scale. | Group data points into predefined categories. |
| **Decision Mechanism** | Direct numerical output based on features. | Probability estimation + **Thresholding** (e.g., if $P(y=1) \ge 0.5 \implies 1$). |
| **Common Loss Functions** | Mean Squared Error (MSE), Mean Absolute Error (MAE), RMSE | Binary Cross-Entropy (Log Loss), Categorical Cross-Entropy |
| **Example Questions** | *"What will be the price of this house?"* <br> *"What will tomorrow's temperature be?"* | *"Is this email spam (1) or inbox (0)?"* <br> *"Is this transaction fraudulent (1) or legitimate (0)?"* |

---

### How Classification Converts Numbers into Groups (0 and 1)

Classification often works internally by predicting a **continuous probability score** between $0$ and $1$, and then applying a **decision threshold** to assign a discrete group label:

1. **Raw Numerical Output ($z$)**: The model computes a linear combination of input features:
   $$z = \theta_0 + \theta_1 x_1 + \theta_2 x_2 + \dots + \theta_n x_n = \theta^T X$$
2. **Probability Mapping (Sigmoid Activation)**: The raw score $z$ (which can be any real number from $-\infty$ to $+\infty$) is squeezed into a probability range $[0, 1]$:
   $$\hat{p} = \sigma(z) = \frac{1}{1 + e^{-z}}$$
3. **Thresholding (Converting into Groups 0 or 1)**:
   $$\hat{y} = \begin{cases} 1 & \text{if } \hat{p} \ge \text{Threshold } (e.g., 0.5) \quad \Rightarrow \text{Group 1 (Positive Class)} \\ 0 & \text{if } \hat{p} < \text{Threshold } (e.g., 0.5) \quad \Rightarrow \text{Group 0 (Negative Class)} \end{cases}$$

---

### Regression Models
Predicting a continuous target variable $y \in \mathbb{R}$.

#### Linear Regression
* **Hypothesis Function**: 
  $$\hat{y} = \theta_0 + \theta_1 x_1 + \theta_2 x_2 + \dots + \theta_n x_n = \theta^T X$$
* **Cost Function (Mean Squared Error)**:
  $$J(\theta) = \frac{1}{2m} \sum_{i=1}^{m} (\hat{y}^{(i)} - y^{(i)})^2$$
* **Normal Equation**:
  $$\theta = (X^T X)^{-1} X^T y$$

---

### Classification Models
Predicting discrete class labels $y \in \{0, 1, \dots, K-1\}$.

#### Logistic Regression
* **Sigmoid Function**:
  $$\sigma(z) = \frac{1}{1 + e^{-z}}$$
* **Decision Boundary**: $\hat{p} \ge 0.5 \implies \hat{y} = 1$
* **Binary Cross-Entropy Loss**:
  $$J(\theta) = -\frac{1}{m} \sum_{i=1}^{m} \left[ y^{(i)} \log(\hat{y}^{(i)}) + (1 - y^{(i)}) \log(1 - \hat{y}^{(i)}) \right]$$

---

## 3. Unsupervised Learning

![Unsupervised Learning Method & Workflow](images/unsupervised_learning_diagram.png)

> **Key Concept**: Unsupervised Learning automatically groups raw unlabeled data into distinct categories or clusters based on inherent similarities (such as age group, demographic traits, or feature density) without any prior target labels.

---

### Clustering
#### K-Means Clustering
1. Initialize $K$ cluster centroids randomly.
2. **Assignment Step**: Assign each data point to the nearest centroid.
3. **Update Step**: Recompute the centroid of each cluster as the mean of all points assigned to it.
4. Repeat until centroids converge.

---

### Semi-Supervised Learning

![Semi-Supervised Learning Method & Workflow](images/semi_supervised_learning_diagram.png)

> **Key Concept**: Semi-Supervised Learning combines a **small amount of labeled data** with a **large volume of unlabeled raw data**. The machine learning model uses the labeled points as anchors and leverages the structural distribution of the unlabeled data to infer labels across fine-grained target categories (e.g. Babies, Teens, Tweens, Adults, Youth, Senior citizens).

* **When to Use**: When acquiring target labels is extremely expensive, time-consuming, or requires domain expertise (e.g. medical image annotation, genome sequencing), but large amounts of raw unlabeled data are readily available.
* **Common Techniques**: Pseudo-Labeling, Label Propagation, Self-Training, Semi-Supervised VAEs / GANs.

---

## 4. Feature Engineering & Data Preprocessing

* **Scaling Techniques**:
  * **Standardization (Z-Score)**: $x' = \frac{x - \mu}{\sigma}$ (Mean = 0, Std = 1)
  * **Min-Max Scaling**: $x' = \frac{x - x_{min}}{x_{max} - x_{min}}$ (Bounds $[0, 1]$)
* **Categorical Encoding**:
  * **One-Hot Encoding**: For nominal categories without ordinal relationship.
  * **Ordinal Encoding**: For categories with inherent order (e.g., Low, Medium, High).

---

## 5. Model Evaluation & Validation

### Classification Metrics
| Metric | Formula | Interpretation |
| :--- | :--- | :--- |
| **Accuracy** | $\frac{TP + TN}{TP + TN + FP + FN}$ | Overall proportion of correct predictions. |
| **Precision** | $\frac{TP}{TP + FP}$ | Out of all predicted positives, how many were truly positive? (Low FP focus) |
| **Recall (Sensitivity)** | $\frac{TP}{TP + FN}$ | Out of all actual positives, how many did we catch? (Low FN focus) |
| **F1-Score** | $2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$ | Harmonic mean of Precision and Recall. |

---

## 6. Bias-Variance Tradeoff & Regularization

* **High Bias (Underfitting)**: Model is too simple, fails to capture underlying patterns.
* **High Variance (Overfitting)**: Model is too complex, fits noise in training data.
* **Regularization Techniques**:
  * **L1 Regularization (Lasso)**: Adds $\lambda \sum |\theta_j|$ penalty. Promotes sparsity (feature selection).
  * **L2 Regularization (Ridge)**: Adds $\lambda \sum \theta_j^2$ penalty. Shrinks coefficients towards zero.

---

## 7. Ensemble Techniques

* **Bagging (Bootstrap Aggregating)**: Trains parallel models on random bootstrap samples (e.g., Random Forest). Reduces variance.
* **Boosting**: Trains sequential models where each model corrects errors of previous models (e.g., XGBoost, LightGBM, AdaBoost). Reduces bias.
* **Stacking**: Combines predictions of diverse base models using a meta-learner.

---

## 8. Practical Machine Learning Workflow

```mermaid
flowchart TD
    A[Business Problem Definition] --> B[Data Collection & Exploration]
    B --> C[Data Cleaning & Preprocessing]
    C --> D[Feature Engineering & Selection]
    D --> E[Model Selection & Training]
    E --> F[Hyperparameter Tuning]
    F --> G[Model Evaluation & Validation]
    G --> H{Performance Satisfactory?}
    H -- No --> D
    H -- Yes --> I[Model Deployment & Monitoring]
```

---

## 9. Code Snippets & Cheatsheets

### Python Baseline Model Template
```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

# 1. Load Data
# df = pd.read_csv("data.csv")

# 2. Split Features & Target
# X = df.drop(columns=["target"])
# y = df["target"]

# 3. Train-Test Split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 4. Preprocessing & Modeling Pipeline
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)

# model = RandomForestClassifier(n_estimators=100, random_state=42)
# model.fit(X_train_scaled, y_train)

# 5. Evaluation
# y_pred = model.predict(X_test_scaled)
# print(classification_report(y_test, y_pred))
```

---

## 10. Reference Links & External Learning Resources

| Resource Name | Website URL | Description & Key Focus |
| :--- | :--- | :--- |
| **Deep Learning with Python** | [deeplearningwithpython.io](https://deeplearningwithpython.io) | Official website for François Chollet's foundational book on deep learning concepts, representation transformations, and Keras practices. |
| **Scikit-learn** | [scikit-learn.org](https://scikit-learn.org) | Official documentation, tutorials, user guides, and API specifications for classical Machine Learning algorithms in Python. |
| **TensorFlow** | [tensorflow.org](https://www.tensorflow.org) | Google's official platform, tutorials, API docs, and deployment ecosystem (TF Lite, TF Serving, Keras) for deep learning. |
| **PyTorch** | [pytorch.org](https://pytorch.org) | Meta / PyTorch Foundation official documentation, tutorials, PyTorch Hub, dynamic graph autograd reference, and deep learning framework ecosystem. |
| **Keras API** | [keras.io](https://keras.io) | Official portal for Keras 3 multi-backend deep learning API supporting TensorFlow, PyTorch, and JAX. |
| **SciPy Scientific Computing** | [scipy.org](https://scipy.org) | Official portal for scientific computing, numerical optimization, integration, and statistics in Python. |
| **DeepLearning.AI** | [deeplearning.ai](https://www.deeplearning.ai) | Andrew Ng's education platform featuring comprehensive courses on ML fundamentals, Deep Learning specialization, and LLMs. |
| **Hugging Face Hub & Docs** | [huggingface.co](https://huggingface.co) | Open-source platform for pre-trained Transformers, LLMs, datasets, and model evaluations. |

---

*(This document is set up as your central Machine Learning reference notebook. Future notes, algorithms, math proofs, and code examples will be continuously updated here!)*
