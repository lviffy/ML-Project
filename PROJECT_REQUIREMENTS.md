# ML Project Requirements

## Dataset

- **File:** `student_grade_ses_dataset-selected-columns.csv`
- **Rows:** 12,025 | **Columns:** 10

| # | Column | Type | Unique Values |
|---|--------|------|---------------|
| 1 | `student_id` | object | 12,000 |
| 2 | `academic_year` | object | 4 — `2021-2022`, `2022-2023`, `2023-2024`, `2024-2025` |
| 3 | `survey_date` | object | 75 |
| 4 | `age` | int | 9 — `10–18` |
| 5 | `gender` | object | 3 — `Female`, `Male`, `Other` |
| 6 | `grade_level` | int | 7 — `6, 7, 8, 9, 10, 11, 12` |
| 7 | `school_type` | object | 3 — `Government`, `Government-Aided`, `Private-Unaided` |
| 8 | `residence_type` | object | 3 — `Rural`, `Semi-Urban`, `Urban` |
| 9 | `family_size` | int | 9 — `2–10` |
| 10 | `num_siblings` | int | 9 — `0–8` |

---

## Group Rules

- **Max members per group:** 5
- **Max groups per dataset:** 2 (code & explanations must differ between groups)

---

## Tasks

### (i) Identify Target & Classes
- Determine the **target column** for classification.
- Report **how many distinct classes** exist in the target.

### (ii) Feature Selection
- Apply **any combination** of feature selection methods (e.g., filter, wrapper, embedded) to identify the most important features.
- Justify **why** those specific methods were chosen.
- List the **final selected features**.

### (iii) Dimensionality Reduction — PCA or LDA
- Apply **PCA** (Principal Component Analysis) **or LDA** (Linear Discriminant Analysis).
- Analyze whether it was **useful or not** for this dataset and explain why.

### (iv) ML Model Comparison
- Train **any two ML models** of your choice (e.g., Random Forest & SVM, Logistic Regression & KNN, etc.).
- **Compare their performances** using appropriate evaluation metrics (accuracy, precision, recall, F1-score, confusion matrix, etc.).

### (v) Video Presentation (Max 20 minutes)
Every group member must present **at least 3 minutes** each. The presentation must include:

| Requirement | Details |
|-------------|---------|
| **Visibility** | The presenter must be **visible on camera** |
| **No slide-reading** | Do NOT simply read from slides |
| **Code walkthrough** | Explain from the **actual lines of code** |
| **Math / Whiteboard** | Write mathematical calculations on a whiteboard or screen |
| **Visualizations** | Show **interactive graphs and visualizations** |

#### Topics to cover in the video:
1. **Feature Selection** — Why these specific methods? What are the final features?
2. **PCA / LDA** — What happens when applied? Why was it useful or not for your dataset?
3. **ML Models** — Which models were used? How did they perform?
4. **Mechanism & Pros/Cons** — Explain the inner workings, advantages, and disadvantages of each method used.
5. **Evaluation Metrics** — Present and explain the metrics obtained for each model.

---

## Suggested Workflow

```
1. Load & explore the dataset
2. Preprocessing (handle missing values, encode categoricals, scale numerics)
3. Identify target column → grade_level (7 classes)
4. Feature selection (e.g., Chi-Square + Mutual Information + Recursive Feature Elimination)
5. Apply PCA or LDA and analyze impact
6. Train Model 1 (e.g., Random Forest)
7. Train Model 2 (e.g., SVM / KNN / Logistic Regression)
8. Evaluate & compare both models
9. Record video presentation
```

---

## Deliverables Checklist

- [ ] Jupyter Notebook / Python script with all code
- [ ] Feature selection analysis with justification
- [ ] PCA/LDA analysis with explanation of usefulness
- [ ] Two trained ML models with performance comparison
- [ ] Evaluation metrics (accuracy, precision, recall, F1, confusion matrix)
- [ ] Video (≤ 20 min, every member ≥ 3 min, on camera, code + math + visualizations)

