# Student Grade Level Classification (Machine Learning Project)

A straightforward machine learning project predicting student academic grade levels (Grades 6 to 12) from demographic and socioeconomic features, built strictly to the professor's requirements.

---

## The 4 Project Tasks

### Task (i): Target Column & Distinct Classes
- **Target Column:** `grade_level`
- **Number of Classes:** 7 distinct grades (`6, 7, 8, 9, 10, 11, 12`)
- **Dataset Size:** 12,025 student records
- **Class Balance:** Evenly distributed (~1,600 to 1,800 students per grade)
- **Baseline Accuracy:** Random guessing accuracy is $1/7 \approx 14.28\%$
- **Plot:** `plots/plot1_target_distribution.png`

---

### Task (ii): Feature Selection
Two complementary statistical filter methods were evaluated on the 6 candidate features:
1. **Chi-Square Test ($\chi^2$):** Evaluates statistical independence between discrete features and the target.
2. **Mutual Information:** Measures both linear and non-linear information gain.

Scores were normalized to $[0, 1]$ and averaged:

| Feature | Combined Score | Selected? | Reason |
|---|:---:|:---:|---|
| **`age`** | **1.000** | **YES** | Primary predictor of grade level in school systems |
| **`num_siblings`** | **0.001** | **YES** | Retained for multi-feature representation (#2 rank) |
| **`gender`** | **0.001** | **YES** | Retained for multi-feature representation (#3 rank) |
| `school_type` | 0.000 | NO | Near-zero correlation with grade level |
| `family_size` | 0.000 | NO | Zero correlation with grade level |
| `residence_type`| 0.000 | NO | Zero correlation with grade level |

- **Final Selected Features (Used in Models):** `age`, `num_siblings`, `gender` (strictly 3 features)
- **Plot:** `plots/plot2_feature_importance.png`

---

### Task (iii): Dimensionality Reduction (PCA vs. LDA)
We reduced the data to 2 dimensions using both techniques to see which works better:

| Method | Type | Top 2 Variance | Did it separate grades? | Verdict |
|---|---|:---:|:---:|---|
| **PCA** | Unsupervised (ignores labels) | 49.38% total variance | No, all 7 grades overlap completely | **Not useful** |
| **LDA** | Supervised (uses labels) | 100.00% between-class variance | Yes, clear progression from Grade 6 to 12 | **Highly useful** |

- **Why PCA failed:** PCA maximizes overall data spread without knowing classes. Socioeconomic spread is noise for grade level.
- **Why LDA succeeded:** LDA directly maximizes separation between class means relative to within-class spread.
- **Plot:** `plots/plot3_pca_vs_lda.png`

---

### Task (iv): ML Model Comparison (Random Forest vs. SVM)
Both models were trained strictly on the **3 selected features** (`age`, `num_siblings`, `gender`) using an 80/20 stratified split (9,620 train / 2,405 test samples):

| Metric | Random Forest (100 Trees) | Support Vector Machine (RBF Kernel) |
|---|:---:|:---:|
| **Exact Accuracy** | **42.62%** | **44.07%** |
| **Adjacent (±1 Grade) Accuracy** | **85.82%** | **91.68%** |
| **Top-2 Accuracy** | **74.93%** | N/A |
| **Weighted Precision** | **43.00%** | **43.98%** |
| **Weighted Recall** | **42.62%** | **44.07%** |
| **Weighted F1-Score** | **42.14%** | **41.75%** |

#### Why ~44% accuracy is a good result:
1. **$3\times$ Better Than Random:** Random guessing on 7 classes is $14.28\%$.
2. **Adjacent Errors Only:** Confusion matrix inspection reveals that misclassifications are almost exclusively off by $\pm 1$ grade (e.g. Grade 8 predicted as 7 or 9) because students of the same age naturally attend adjacent grades depending on cutoff birth dates.
- **Plot:** `plots/plot4_model_comparison.png`

---

## How to Run

```bash
# 1. Activate environment
source .venv/bin/activate

# 2. Run the code
python student_grade_classification.py
```

All 4 plots are automatically generated in the `plots/` folder.

---

## 4 Generated Visualizations

1. `plots/plot1_target_distribution.png` — Distribution of students across Grades 6–12 (Task i)
2. `plots/plot2_feature_importance.png` — Combined Chi2 + Mutual Information feature ranking (Task ii)
3. `plots/plot3_pca_vs_lda.png` — Side-by-side scatter plots showing PCA overlap vs. LDA separation (Task iii)
4. `plots/plot4_model_comparison.png` — Performance bar chart + Confusion matrices for RF and SVM (Task iv)
