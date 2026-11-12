# Student Grade Level Classification

A multi-class machine learning project predicting student academic grade level (Grades 6 through 12) from demographic and socioeconomic attributes.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset Summary](#dataset-summary)
- [Methodology and Tasks](#methodology-and-tasks)
  - [Task (i): Target Column and Class Breakdown](#task-i-target-column-and-class-breakdown)
  - [Task (ii): Feature Selection](#task-ii-feature-selection)
  - [Task (iii): Dimensionality Reduction (PCA vs. LDA)](#task-iii-dimensionality-reduction-pca-vs-lda)
  - [Task (iv): ML Model Comparison](#task-iv-ml-model-comparison)
- [Project Structure](#project-structure)
- [Installation and Execution](#installation-and-execution)
- [List of Generated Visualizations](#list-of-generated-visualizations)
- [Team Documentation](#team-documentation)

---

## Project Overview

The objective of this project is to build an end-to-end classification pipeline that evaluates whether demographic and socioeconomic features can predict a student's enrolled grade level. The workflow encompasses:

1. Data exploration, cleaning, and standardization.
2. Multi-method statistical feature selection (Chi-Square, Mutual Information, ANOVA F-Test).
3. Supervised versus unsupervised dimensionality reduction (PCA vs. LDA).
4. Model benchmarking and comparison between Random Forest and Support Vector Machine (SVM) using 5-fold cross-validation.

---

## Dataset Summary

- **Source File:** `student_grade_ses_dataset-selected-columns.csv`
- **Total Records:** 12,025 rows
- **Total Columns:** 10 columns
- **Input Features Kept:** 6 (`age`, `gender`, `school_type`, `residence_type`, `family_size`, `num_siblings`)
- **Dropped Columns:**
  - `student_id`: Arbitrary unique identifier (near 12,000 distinct values).
  - `academic_year`, `survey_date`: Administrative collection metadata with risk of temporal leakage.
- **Preprocessing:**
  - Missing Values: None found (0 missing entries).
  - Categorical Encoding: Scikit-learn `LabelEncoder` applied to `gender`, `school_type`, and `residence_type`.
  - Standardization: Scikit-learn `StandardScaler` transforming features to zero mean and unit variance ($z = \frac{x - \mu}{\sigma}$).

---

## Methodology and Tasks

### Task (i): Target Column and Class Breakdown

- **Target Column:** `grade_level`
- **Total Classes:** 7 distinct classes (`6`, `7`, `8`, `9`, `10`, `11`, `12`)
- **Distribution:** Approximately balanced (~1,600 to 1,800 students per grade level).
- **Random Baseline Accuracy:** $\frac{1}{7} \approx 14.28\%$

---

### Task (ii): Feature Selection

Three complementary statistical filter methods were evaluated and combined:

1. **Chi-Square Test ($\chi^2$):** Assesses independence between categorical/discrete predictors and grade levels.
2. **Mutual Information:** Measures non-linear dependency and information gain.
3. **ANOVA F-Test:** Compares the ratio of between-group variance to within-group variance.

Because raw test statistics operate on vastly different scales, scores were normalized to $[0, 1]$ using Min-Max scaling before averaging:

| Feature | Chi-Square (norm) | Mutual Info (norm) | ANOVA F-Test (norm) | Combined Score | Rank |
|---|:---:|:---:|:---:|:---:|:---:|
| **age** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | 1 |
| **school_type** | 0.0001 | 0.0011 | 0.0000 | 0.0004 | 2 |
| **gender** | 0.0001 | 0.0000 | 0.0001 | 0.0001 | 3 |
| **family_size** | 0.0001 | 0.0000 | 0.0000 | 0.0000 | 4 |
| **residence_type** | 0.0001 | 0.0000 | 0.0000 | 0.0000 | 5 |
| **num_siblings** | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 6 |

**Selected Features:** `age`, `school_type`, `gender`

**Key Finding:** `age` is overwhelmingly the primary predictor of `grade_level`. Socioeconomic factors (family size, residence type, siblings) carry near-zero discriminative capacity for grade assignment.

---

### Task (iii): Dimensionality Reduction (PCA vs. LDA)

| Criterion | Principal Component Analysis (PCA) | Linear Discriminant Analysis (LDA) |
|---|---|---|
| **Learning Paradigm** | Unsupervised (ignores target labels $y$) | Supervised (uses target labels $y$) |
| **Optimization Target** | Maximizes total data variance | Maximizes between-class variance relative to within-class variance |
| **Top 2 Components Variance** | 49.38% of total variance | **100.00%** of between-class variance (LD1 alone explains 99.99%) |
| **Separation Quality** | Heavy class overlap | Clear class progression along LD1 |
| **Conclusion for Dataset** | **Not effective** for classification | **Highly effective** for linear class discrimination |

---

### Task (iv): ML Model Comparison

Two distinct model families were benchmarked:
1. **Random Forest Classifier:** Ensemble bagging model (200 estimators, max depth 15).
2. **Support Vector Machine (SVC):** Non-linear margin classifier with Radial Basis Function (RBF) kernel ($C=10, \gamma=\text{'scale'}$).

#### Test Set Performance (All Features, 80/20 Stratified Split)

| Model | Accuracy | Precision (Weighted) | Recall (Weighted) | F1-Score (Weighted) |
|---|:---:|:---:|:---:|:---:|
| **Random Forest** | **0.4383** | **0.4381** | **0.4383** | **0.4379** |
| **SVM (RBF Kernel)** | 0.4270 | 0.4237 | 0.4270 | 0.4230 |

#### 5-Fold Stratified Cross-Validation

| Model | Mean CV Accuracy | Standard Deviation |
|---|:---:|:---:|
| **Random Forest** | 0.4226 | +/- 0.0067 |
| **SVM (RBF Kernel)** | **0.4269** | +/- 0.0059 |

#### Performance Across Data Subsets

| Data Representation | Random Forest Accuracy | SVM (RBF) Accuracy |
|---|:---:|:---:|
| All Features (Baseline) | 0.4383 | 0.4270 |
| Selected Features (`age`, `school_type`, `gender`) | **0.4453** | 0.4274 |
| PCA Transformed Data | 0.4383 | 0.3580 |
| LDA Transformed Data | 0.4349 | **0.4432** |

**Interpretation:**
- Both models perform at ~43% to 44% accuracy, which is approximately 3 times better than random guessing (14.28%).
- As shown in the confusion matrices, classification errors occur almost exclusively between adjacent grades (e.g., Grade 8 predicted as Grade 7 or 9) due to natural age overlap among students.

---

## Project Structure

```
PROJECT/
├── student_grade_classification.py          # Complete, self-contained project code
├── student_grade_ses_dataset-selected-columns.csv # Dataset file
├── PROJECT_REQUIREMENTS.md                  # Official assignment requirements and checklist
├── VIVA_PREPARATION.md                      # Comprehensive viva questions, answers, and glossary
├── GROUP_WORK_DIVISION.md                   # 5-person video presentation plan and code split
├── README.md                                # Project documentation
└── plots/                                   # Generated high-resolution visualization plots
    ├── 01_target_distribution.png
    ├── 02_feature_selection_comparison.png
    ├── 03_combined_feature_ranking.png
    ├── 04_pca_explained_variance.png
    ├── 05_pca_2d_scatter.png
    ├── 06_lda_2d_scatter.png
    ├── 07_pca_vs_lda.png
    ├── 08_model_comparison.png
    ├── 09_confusion_matrices.png
    ├── 10_cv_comparison.png
    ├── 11_rf_feature_importance.png
    └── 12_correlation_heatmap.png
```

---

## Installation and Execution

### Prerequisites

- Python 3.10 or higher
- Recommended virtual environment setup

### Setup Instructions

1. Clone or navigate to the project directory:
   ```bash
   cd /path/to/PROJECT
   ```

2. Create and activate a virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. Install the required dependencies:
   ```bash
   pip install scikit-learn pandas numpy matplotlib seaborn
   ```

4. Run the classification pipeline:
   ```bash
   python student_grade_classification.py
   ```

All numerical metrics, ranking tables, classification reports, and cross-validation summaries will print to the console, and all 12 plots will be automatically saved to the `plots/` directory.

---

## List of Generated Visualizations

All plots are stored in the `plots/` folder:

1. `01_target_distribution.png`: Frequency distribution of students across Grade 6 through 12.
2. `02_feature_selection_comparison.png`: Side-by-side comparison of Chi-Square, Mutual Information, and ANOVA F-scores.
3. `03_combined_feature_ranking.png`: Combined normalized feature ranking bar chart.
4. `04_pca_explained_variance.png`: Scree plot showing individual and cumulative variance explained by principal components.
5. `05_pca_2d_scatter.png`: Two-dimensional PCA projection colored by grade level.
6. `06_lda_2d_scatter.png`: Two-dimensional LDA projection showing class separation along LD1.
7. `07_pca_vs_lda.png`: Side-by-side visual comparison between PCA and LDA projections.
8. `08_model_comparison.png`: Grouped bar charts comparing Accuracy, Precision, Recall, and F1-score for Random Forest vs. SVM.
9. `09_confusion_matrices.png`: Confusion matrices for Random Forest and SVM highlighting adjacent-grade error patterns.
10. `10_cv_comparison.png`: Box plot showing the spread of 5-fold cross-validation accuracy.
11. `11_rf_feature_importance.png`: Feature importance ranking based on Random Forest Gini impurity reduction.
12. `12_correlation_heatmap.png`: Pearson correlation matrix among all encoded features.

---

## Team Documentation

- **Requirements & Rubric:** Refer to `PROJECT_REQUIREMENTS.md` for guidelines and evaluation criteria.
- **Viva Preparation:** Refer to `VIVA_PREPARATION.md` for detailed conceptual, mathematical, and implementation Q&A including an acronym glossary.
- **Video Presentation Guide:** Refer to `GROUP_WORK_DIVISION.md` for individual member assignments, timing (3-4 minutes each), code line references, and whiteboard calculations.

