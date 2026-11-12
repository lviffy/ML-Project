# Group Work Division & Video Presentation Plan

This document outlines the breakdown of the project among **5 group members**, ensuring each person owns a distinct technical component, speaks for **3 to 4 minutes** (total duration: 15–20 minutes), and fulfills the video presentation requirements:
- Presenter clearly visible on camera.
- Walkthrough of actual code from [`student_grade_classification.py`](student_grade_classification.py).
- Step-by-step mathematical derivations or explanations on whiteboard / digital screen.
- Interactive visualizations and graphs from the [`plots/`](plots) directory.

---

## Team Overview & Task Allocation

| Member | Module / Role | Primary Task | Code Lines | Speaking Time |
|---|---|---|:---:|:---:|
| **Member 1** | Data Pipeline, Preprocessing & Problem Formulation | Task (i) | Lines 30–125 | 3 – 4 mins |
| **Member 2** | Statistical Feature Selection & Multi-Method Ranking | Task (ii) | Lines 126–248 | 3 – 4 mins |
| **Member 3** | Dimensionality Reduction Analysis (PCA vs. LDA) | Task (iii) | Lines 249–380 | 3 – 4 mins |
| **Member 4** | ML Model 1: Random Forest Classifier & Feature Importances | Task (iv) Part 1 | Lines 381–485, 550–585 | 3 – 4 mins |
| **Member 5** | ML Model 2: SVM, Confusion Matrices & Comparative Synthesis | Task (iv) Part 2 | Lines 486–549, 586–645 | 3 – 4 mins |

---

## Detailed Member Breakdown

### Member 1: Data Pipeline, Preprocessing & Problem Formulation

- **Project Tasks Covered:** Task (i) and Data Preprocessing
- **Code Reference:** `student_grade_classification.py` (Lines 30 to 125)
- **Visualizations to Show:**
  - [`01_target_distribution.png`](plots/01_target_distribution.png) (Target class distribution across Grade 6–12)

#### What to Show and Explain:
1. **Dataset Loading and Shape:** Explain the structure of `student_grade_ses_dataset-selected-columns.csv` (12,025 rows, 10 columns).
2. **Target Identification (Task i):**
   - Explain why `grade_level` is the target column.
   - State the number of classes (7 discrete classes: grades 6 to 12).
   - Point out that random guessing yields $\frac{1}{7} \approx 14.28\%$ baseline accuracy.
3. **Data Cleaning:**
   - Walk through the code dropping non-predictive columns (`student_id`, `academic_year`, `survey_date`).
   - Explain why `student_id` is an arbitrary key that would cause severe overfitting, and why dates are administrative metadata.
4. **Encoding & Standardization:**
   - Explain `LabelEncoder` for categorical variables (`gender`, `school_type`, `residence_type`).
   - Explain `StandardScaler` transforming features to zero mean and unit variance.

#### Whiteboard / Screen Calculation:
Write down and explain the Standardization formula:
$$z = \frac{x - \mu}{\sigma}$$
Explain why distance-based algorithms (SVM) and variance-based methods (PCA) require scaling so that features with larger ranges (like `age` $\in [10, 18]$) do not arbitrarily dominate features with smaller ranges (like `gender` $\in [0, 2]$).

---

### Member 2: Statistical Feature Selection & Multi-Method Ranking

- **Project Tasks Covered:** Task (ii) Feature Selection
- **Code Reference:** `student_grade_classification.py` (Lines 126 to 248)
- **Visualizations to Show:**
  - [`02_feature_selection_comparison.png`](plots/02_feature_selection_comparison.png) (Side-by-side bar plots of Chi2, MI, and ANOVA F-test)
  - [`03_combined_feature_ranking.png`](plots/03_combined_feature_ranking.png) (Final normalized combined ranking)

#### What to Show and Explain:
1. **The Three Selection Techniques:**
   - **Chi-Square Test ($\chi^2$):** Tests independence between categorical/discrete inputs and target classes. Point out why values are shifted to be non-negative (`X - X.min()`).
   - **Mutual Information (`mutual_info_classif`):** Measures reduction in entropy of target given a feature; captures non-linear relationships.
   - **ANOVA F-Test (`f_classif`):** Tests if the mean of a continuous/ordinal feature is significantly different across classes.
2. **Normalization Strategy:**
   - Explain that raw scores are on completely different scales ($\chi^2 > 11,000$ vs $\text{MI} \approx 1.0$).
   - Show the code using `MinMaxScaler` to scale scores to $[0, 1]$ before averaging.
3. **Findings & Final Feature Set:**
   - Walk through the result: `age` achieved 1.000, while all other socioeconomic features scored $< 0.001$.
   - State the final selected features (`age`, `school_type`, `gender`) and explain why `age` dominates the academic grade level.

#### Whiteboard / Screen Calculation:
Write down the Chi-Square formula and ANOVA F-statistic:
$$\chi^2 = \sum \frac{(O - E)^2}{E}, \quad F = \frac{\text{Between-Group Variance}}{\text{Within-Group Variance}} = \frac{MS_B}{MS_W}$$
Explain that when between-group variance is significantly larger than within-group variance, the F-statistic becomes very large and the p-value approaches 0.

---

### Member 3: Dimensionality Reduction Analysis (PCA vs. LDA)

- **Project Tasks Covered:** Task (iii) PCA and LDA
- **Code Reference:** `student_grade_classification.py` (Lines 249 to 380)
- **Visualizations to Show:**
  - [`04_pca_explained_variance.png`](plots/04_pca_explained_variance.png) (Scree plot with individual and cumulative variance)
  - [`05_pca_2d_scatter.png`](plots/05_pca_2d_scatter.png) (PCA 2D projection colored by grade)
  - [`06_lda_2d_scatter.png`](plots/06_lda_2d_scatter.png) (LDA 2D projection colored by grade)
  - [`07_pca_vs_lda.png`](plots/07_pca_vs_lda.png) (Side-by-side comparison)

#### What to Show and Explain:
1. **PCA Mechanism (Unsupervised):**
   - Explain that PCA finds axes of maximum variance without looking at labels $y$.
   - Show that PC1 explains 32.3% and PC2 explains 17.1% (cumulative: 49.4%).
   - Show in `05_pca_2d_scatter.png` that data points from different grades overlap heavily.
2. **LDA Mechanism (Supervised):**
   - Explain that LDA seeks directions that maximize the distance between class means while minimizing spread within each class.
   - Show that LD1 accounts for **99.99%** of between-class variance.
   - Show in `06_lda_2d_scatter.png` the separation achieved by LD1.
3. **Synthesis (Why Useful or Not Useful):**
   - **PCA was not useful** for classification because variance in socioeconomic features did not correspond to class boundaries.
   - **LDA was highly useful** because it aligned projection directly with the class-discriminating feature (`age`).

#### Whiteboard / Screen Calculation:
Write down and compare the objective functions:
$$\text{PCA: } \max_w w^T \Sigma w \quad \text{subject to } w^T w = 1$$
$$\text{LDA (Fisher's Criterion): } \max_w J(w) = \frac{w^T S_B w}{w^T S_W w}$$
Explain the terms: $S_B$ (Between-class scatter) and $S_W$ (Within-class scatter).

---

### Member 4: Machine Learning Model 1 — Random Forest Classifier

- **Project Tasks Covered:** Task (iv) Part 1 — Model 1 Architecture & Evaluation
- **Code Reference:** `student_grade_classification.py` (Lines 381 to 485, 550 to 585)
- **Visualizations to Show:**
  - [`11_rf_feature_importance.png`](plots/11_rf_feature_importance.png) (Random Forest Gini importances)
  - [`12_correlation_heatmap.png`](plots/12_correlation_heatmap.png) (Feature correlation matrix)

#### What to Show and Explain:
1. **Train-Test Stratification:**
   - Explain `train_test_split(..., test_size=0.2, stratify=y, random_state=42)`.
   - Explain why stratification is essential to preserve class proportions in training (9,620 samples) and testing (2,405 samples).
2. **Random Forest Architecture:**
   - Explain Bagging (Bootstrap Aggregation) with 200 decision trees (`n_estimators=200`).
   - Explain tree depth limitation (`max_depth=15`) to control variance and prevent overfitting.
   - Explain majority voting among trees for multi-class prediction.
3. **Performance Metrics Obtained:**
   - Accuracy: **0.4383 (43.83%)**
   - Weighted Precision: **0.4381**
   - Weighted Recall: **0.4383**
   - Weighted F1 Score: **0.4379**
4. **Pros and Cons of Random Forest:**
   - *Pros:* Captures non-linear relationships, robust against outliers, provides feature importances.
   - *Cons:* Computationally heavier, memory-intensive with large tree ensembles, less interpretable than single trees.

#### Whiteboard / Screen Calculation:
Write down the Gini Impurity formula used for decision tree splits:
$$I_G(p) = 1 - \sum_{i=1}^C p_i^2$$
Explain with an example: if a node has pure class samples ($p_1 = 1$), $I_G = 0$; if classes are uniformly mixed, $I_G$ is maximized.

---

### Member 5: Machine Learning Model 2 (SVM) & Comparative Performance Evaluation

- **Project Tasks Covered:** Task (iv) Part 2 — Model 2, Confusion Matrices & Final Comparison
- **Code Reference:** `student_grade_classification.py` (Lines 486 to 549, 586 to 645)
- **Visualizations to Show:**
  - [`08_model_comparison.png`](plots/08_model_comparison.png) (RF vs SVM bar charts across metrics)
  - [`09_confusion_matrices.png`](plots/09_confusion_matrices.png) (Side-by-side confusion matrices)
  - [`10_cv_comparison.png`](plots/10_cv_comparison.png) (5-fold cross validation accuracy box plot)

#### What to Show and Explain:
1. **Support Vector Machine (SVC) Architecture:**
   - Explain maximum-margin hyperplane optimization.
   - Explain the Radial Basis Function (RBF) kernel mapping non-linear data into higher dimensional space.
   - Explain hyperparameters: $C=10$ (misclassification penalty) and $\gamma=\text{'scale'}$.
   - Performance: Accuracy = **0.4270**, Precision = **0.4237**, Recall = **0.4270**, F1 = **0.4230**.
2. **Model Comparison & Confusion Matrix Analysis:**
   - Show `08_model_comparison.png`: Random Forest slightly outperforms SVM on all features, while SVM performs better on LDA features.
   - Show `09_confusion_matrices.png`: Explain that both models perform best on boundary grades (Grade 6 and Grade 12) and that errors are concentrated strictly on adjacent grades (e.g., Grade 8 predicted as Grade 7 or 9).
3. **5-Fold Cross-Validation:**
   - Random Forest: $0.4226 \pm 0.0067$
   - SVM (RBF): $0.4269 \pm 0.0059$
   - Explain that low standard deviation ($\approx 0.006$) demonstrates model stability across folds.
4. **Pros and Cons of SVM:**
   - *Pros:* Strong theoretical foundation, effective in high dimensions, memory-efficient (stores only support vectors).
   - *Cons:* Training scales quadratically $O(N^2)$ with sample size, sensitive to feature scaling and hyperparameter choice.

#### Whiteboard / Screen Calculation:
Write down the RBF Kernel function and F1-score formula:
$$K(x, x') = \exp(-\gamma ||x - x'||^2), \quad F_1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$
Explain how the RBF kernel acts as a similarity measure decaying with Euclidean distance.

---

## Video Recording Checklist

- [ ] Ensure all 5 members are present with cameras turned on.
- [ ] Each member speaks for at least 3 minutes (total time between 15 and 20 minutes).
- [ ] Share screen to display code in `student_grade_classification.py`.
- [ ] Display visual plots from `plots/` when explaining results.
- [ ] Write mathematical formulas on a whiteboard or drawing screen during each member's turn.
- [ ] Confirm audio and video quality before submitting.

