# Viva Preparation Guide: Student Grade Level Classification

This guide covers conceptual, mathematical, algorithmic, and implementation questions based on `student_grade_classification.py`.

---

## 0. Glossary of Acronyms and Technical Terms

| Acronym | Full Form | Plain-English Definition |
|---|---|---|
| **PCA** | Principal Component Analysis | An unsupervised technique that compresses data by finding new perpendicular axes (called principal components) that capture the maximum variance (spread) in the data without looking at target labels. |
| **LDA** | Linear Discriminant Analysis | A supervised technique that compresses data by finding axes (called linear discriminants) that maximize the distance between different classes while keeping samples of the same class tightly packed together. |
| **ML** | Machine Learning | A branch of artificial intelligence where computer algorithms learn patterns directly from sample data to make predictions or decisions without being explicitly rule-programmed. |
| **SES** | Socioeconomic Status | An economic and sociological measure of an individual or family's standing based on variables like income, residence, education, and family structure. |
| **SVM** | Support Vector Machine | A classification algorithm that searches for the optimal decision boundary (a line or hyperplane) with the maximum margin (distance) between different classes. |
| **SVC** | Support Vector Classifier | The specific class/implementation of Support Vector Machines designed for classification tasks (predicting categories rather than continuous numbers). |
| **RBF** | Radial Basis Function | A mathematical function used as a kernel trick in SVM to project complex, curved, or non-linearly separable data into a higher-dimensional space where it can be separated linearly. |
| **RF** | Random Forest | An ensemble machine learning algorithm that builds a "forest" of multiple decision trees on random subsets of data and takes their majority vote for the final prediction. |
| **ANOVA** | Analysis of Variance | A statistical hypothesis test that checks whether the mean values of a feature are significantly different across three or more distinct class groups. |
| **MI** | Mutual Information | A measure from information theory that calculates how much information knowing one variable gives about another; detects both linear and non-linear dependencies. |
| **CV** | Cross-Validation | A model validation technique (such as 5-Fold CV) that splits data into multiple subsets, training and testing multiple times so that every data point gets evaluated, avoiding lucky or unlucky splits. |
| **PC** | Principal Component | A single new synthetic feature created by PCA as a linear combination of original features, ordered by how much variance it explains (PC1 explains the most, followed by PC2, etc.). |
| **LD** | Linear Discriminant | A single new synthetic axis created by LDA, ordered by how much between-class separation it produces (LD1 provides the highest class separation). |
| **F1 Score** | Balanced F-Measure | The harmonic mean of Precision and Recall. It provides a balanced assessment of a classifier's performance, especially when classes are uneven. |
| **TP / TN / FP / FN** | True Positive / True Negative / False Positive / False Negative | The four possible classification outcomes: correctly predicted positive, correctly predicted negative, incorrectly predicted positive, and incorrectly predicted negative. |
| **CSV** | Comma-Separated Values | A plain-text file format where tabular data is stored using commas to separate column values for each row. |

---

## 1. Project Overview and Dataset

### Question: What is the objective of this project?
**Answer:** The objective is to build and evaluate a multi-class classification pipeline that predicts a student's `grade_level` (from Grade 6 to Grade 12) using demographic and socioeconomic features. The project implements data preprocessing, statistical feature selection, dimensionality reduction (PCA and LDA), and model benchmarking using Random Forest and Support Vector Machines (SVM).

### Question: What is the dataset size and composition?
**Answer:**
- Total records: 12,025 rows
- Raw columns: 10
- Non-predictive dropped columns: `student_id` (arbitrary identifier), `academic_year`, `survey_date` (administrative metadata)
- Remaining features: 6 input features (`age`, `gender`, `school_type`, `residence_type`, `family_size`, `num_siblings`)
- Target column: `grade_level` (7 distinct classes: 6, 7, 8, 9, 10, 11, 12)

### Question: Why were `student_id`, `academic_year`, and `survey_date` dropped?
**Answer:**
- `student_id` is an arbitrary unique key with near 12,000 distinct values; keeping it would lead to extreme overfitting or spurious correlations.
- `academic_year` and `survey_date` represent collection timestamps rather than student-level socioeconomic or academic characteristics. Including them could cause temporal data leakage.

### Question: How were categorical variables handled?
**Answer:** Label encoding was applied to nominal categorical features (`gender`, `school_type`, `residence_type`) converting string categories into integers (0, 1, 2). For tree-based models like Random Forest, label encoding is effective. For distance-based models like SVM, features were subsequently standardized.

### Question: Why is feature standardization (StandardScaler) necessary?
**Answer:** StandardScaler transforms features to have zero mean and unit variance:
$$z = \frac{x - \mu}{\sigma}$$
Distance-dependent algorithms (like SVM with RBF kernel) and variance-maximizing methods (like PCA) are sensitive to feature scales. If `age` ranges from 10 to 18 while `gender` ranges from 0 to 2, unscaled distance metrics would be disproportionately dominated by features with numerically larger magnitudes.

---

## 2. Task (i): Target Column and Class Distribution

### Question: What is the target column and how many classes are there?
**Answer:** The target column is `grade_level`. It contains 7 discrete classes: grades 6, 7, 8, 9, 10, 11, and 12.

### Question: Is the target class balanced or imbalanced?
**Answer:** The classes are approximately balanced across the 7 grades, with each grade containing between 1,600 and 1,800 students.

### Question: What is the random baseline accuracy for this problem?
**Answer:** For a balanced 7-class problem, random guessing yields an accuracy of:
$$\text{Baseline Accuracy} = \frac{1}{7} \approx 14.28\%$$
Any model achieving significantly above 14.28% is capturing meaningful statistical patterns.

---

## 3. Task (ii): Feature Selection

### Question: Which feature selection methods did you use and why?
**Answer:** Three complementary filter methods were employed:
1. **Chi-Square Test ($\chi^2$):** Evaluates whether the observed frequency distribution of categorical/discrete features differs significantly from the expected distribution under the null hypothesis of independence with the target classes.
2. **Mutual Information (`mutual_info_classif`):** Measures the reduction in uncertainty of the target variable given knowledge of a feature. Unlike linear correlation, mutual information detects non-linear dependencies.
3. **ANOVA F-Test (`f_classif`):** Calculates the ratio of between-group variance to within-group variance ($F = \frac{MS_{between}}{MS_{within}}$) to assess whether mean feature values differ significantly across target classes.

### Question: Why did you normalize scores before combining them?
**Answer:** The three methods output scores on completely different scales:
- $\chi^2$ scores reached over 11,000 for `age`.
- ANOVA F-scores reached approximately 11,900 for `age`.
- Mutual Information scores were in the range of 0.0 to 1.02.

Min-Max normalization scales each method's score distribution to the interval $[0, 1]$:
$$S_{norm} = \frac{S - S_{min}}{S_{max} - S_{min}}$$
Averaging the normalized scores prevents methods with large numeric scales from dominating the combined ranking.

### Question: What were the results of feature selection?
**Answer:**
- `age` achieved a normalized score of 1.000 across all three tests.
- All other features (`school_type`, `gender`, `family_size`, `residence_type`, `num_siblings`) scored below 0.001.
- Final selected features: `age`, `school_type`, `gender` (keeping the top three to retain multi-feature input).

### Question: Why does `age` have such a disproportionately high score?
**Answer:** In academic systems, student grade level is directly structured by chronological age (e.g., 11-year-olds attend Grade 6, 17-year-olds attend Grade 12). Socioeconomic factors such as residence type or family size do not determine school entry age or grade progression in this dataset.

---

## 4. Task (iii): Dimensionality Reduction (PCA vs. LDA)

### Question: What is the core difference between PCA and LDA?
**Answer:**
| Dimension | Principal Component Analysis (PCA) | Linear Discriminant Analysis (LDA) |
|---|---|---|
| **Supervision** | Unsupervised (ignores target labels $y$) | Supervised (uses target labels $y$) |
| **Objective** | Maximizes variance of data projections | Maximizes separation between classes while minimizing within-class spread |
| **Mathematical Criterion** | Eigenvectors of covariance matrix $\Sigma$ | Eigenvectors of $S_W^{-1} S_B$ |
| **Component Limit** | Up to total feature count $p$ ($p=6$) | At most $\min(p, C - 1) = \min(6, 6) = 6$ |

### Question: Explain the mathematical criterion of LDA.
**Answer:** LDA seeks projection vectors $w$ that maximize Fisher's criterion:
$$J(w) = \frac{w^T S_B w}{w^T S_W w}$$
Where:
- $S_B$ is the **Between-class scatter matrix**:
  $$S_B = \sum_{c=1}^C N_c (\mu_c - \mu)(\mu_c - \mu)^T$$
- $S_W$ is the **Within-class scatter matrix**:
  $$S_W = \sum_{c=1}^C \sum_{x \in D_c} (x - \mu_c)(x - \mu_c)^T$$

### Question: What happened when PCA and LDA were applied to this dataset?
**Answer:**
- **PCA:** The top 2 principal components explained only 49.38% of total variance (PC1: 32.30%, PC2: 17.08%). 6 components were required to retain 100% variance. The 2D PCA scatter plot showed severe class overlap because variance in socioeconomic features does not correspond to class boundaries.
- **LDA:** The first linear discriminant (LD1) accounted for **99.99%** of between-class variance. The 2D scatter plot exhibited clear separation between lower and higher grade levels.

### Question: Was PCA or LDA more useful for this dataset, and why?
**Answer:** **LDA was significantly more useful.** Because grade levels are primarily separated along one underlying axis (`age`), supervised projection via LDA aligns directly with that axis. PCA, being unsupervised, captured irrelevant variance from socioeconomic noise features.

---

## 5. Task (iv): Machine Learning Models and Comparison

### Question: Which two models did you choose and why?
**Answer:**
1. **Random Forest Classifier:** A non-parametric ensemble method based on bagging decision trees. Chosen because it handles non-linear relationships, is invariant to monotonic feature scaling, and resists overfitting.
2. **Support Vector Classifier (SVC with RBF Kernel):** A geometric margin-based classifier. Chosen as a contrast to tree ensembles because it projects inputs into an infinite-dimensional feature space using the Radial Basis Function kernel to identify optimal separating hyperplanes.

### Question: How does Random Forest work?
**Answer:**
- **Bootstrap Aggregating (Bagging):** Draws $B$ bootstrap samples from training data with replacement.
- **Random Feature Subspace:** At each split node, considers only a random subset of features (typically $\sqrt{p}$) to decorrelate individual trees.
- **Aggregation:** Collects predictions from all trees and outputs the majority class vote:
  $$\hat{y} = \text{mode}\{T_1(x), T_2(x), \dots, T_B(x)\}$$

### Question: How does SVM with RBF kernel work?
**Answer:**
SVM finds the maximum-margin hyperplane separating classes. For non-linearly separable data, it uses the kernel trick:
$$K(x, x') = \exp(-\gamma ||x - x'||^2)$$
This computes inner products in a transformed higher-dimensional space without explicitly calculating coordinates in that space.
- Hyperparameter $C=10$: Regularization controlling penalty for misclassifications.
- Hyperparameter $\gamma = \text{'scale'}$: Defines the radius of influence of individual support vectors ($\gamma = \frac{1}{p \cdot \text{Var}(X)}$).

### Question: Compare the performance of Random Forest and SVM on this dataset.
**Answer:**
| Metric | Random Forest (All Features) | SVM RBF (All Features) |
|---|:---:|:---:|
| **Test Accuracy** | **0.4383 (43.83%)** | 0.4270 (42.70%) |
| **Weighted Precision** | **0.4381** | 0.4237 |
| **Weighted Recall** | **0.4383** | 0.4270 |
| **Weighted F1 Score** | **0.4379** | 0.4230 |
| **5-Fold CV Accuracy** | 0.4226 +/- 0.0067 | **0.4269 +/- 0.0059** |

Both models achieve ~43% to 44% accuracy, which is approximately 3 times better than random baseline (14.28%).

### Question: Why is the accuracy capped around 44%?
**Answer:**
1. Adjacent grades share identical ages (e.g., age 14 can correspond to Grade 8, Grade 9, or Grade 10 depending on birth month and enrollment age).
2. The remaining features (`school_type`, `residence_type`, `family_size`) provide negligible discriminatory signal for grade assignment.
3. As observed in the confusion matrices, classification errors occur almost exclusively between adjacent grades (e.g., Grade 8 predicted as Grade 7 or 9), not distant grades.

---

## 6. Evaluation Metrics and Validation Strategy

### Question: Why use Weighted F1 Score instead of Macro F1 Score?
**Answer:**
- **Macro F1:** Computes the arithmetic mean of F1 scores across all classes without weighting by support:
  $$\text{Macro F1} = \frac{1}{C}\sum_{c=1}^C F1_c$$
- **Weighted F1:** Weights each class F1 score by its true class support $N_c$:
  $$\text{Weighted F1} = \sum_{c=1}^C \frac{N_c}{N} F1_c$$
Weighted averaging accounts for slight class support variations in test splits.

### Question: What is 5-Fold Cross-Validation and why is it preferred over a single train-test split?
**Answer:** In 5-fold cross-validation, the dataset is partitioned into 5 equal folds. The model is trained on 4 folds and tested on the remaining fold, repeating 5 times so every sample is evaluated once. This prevents evaluation bias caused by an unusually favorable or unfavorable single train-test split and provides a measure of performance variance ($\mu \pm \sigma$).

### Question: What did the confusion matrices reveal?
**Answer:**
- Strongest diagonal values were at the boundary grades: Grade 6 and Grade 12 (students aged 10-11 and 17-18 have minimal age overlap).
- Middle grades (Grades 8, 9, 10) had lower precision and recall due to age overlap among 13, 14, and 15-year-olds.
- Off-diagonal errors were concentrated strictly along the first sub-diagonals, proving the models learned the correct ordinal progression.

---

## 7. Fast-Fire Conceptual Definitions

- **Curse of Dimensionality:** As feature dimensions increase, data space volume grows exponentially, making data sparse and causing distance metrics to lose discriminative power.
- **Eigenvalue in PCA:** Represents the variance of the data along the corresponding eigenvector (principal component direction).
- **Fisher's Linear Discriminant:** The axis that maximizes between-class variance relative to within-class variance.
- **Overfitting:** When a model learns noise and idiosyncrasies of training data, failing to generalize to unseen test data.
- **Gini Impurity (Random Forest):** A measure of node heterogeneity used for splitting:
  $$I_G(p) = 1 - \sum_{i=1}^C p_i^2$$

