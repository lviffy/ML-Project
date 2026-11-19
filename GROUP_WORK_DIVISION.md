# Video Presentation Script & Role Division (5 Members)

A concise, 5-part script mapped directly to the professor's requirements (3 to 4 minutes per member, total ~15–18 minutes).

---

## Quick Summary Table

| Part / Speaker | Task Covered | Key Talking Points | Visual Plot | Whiteboard Formula |
|---|---|---|---|---|
| **Member 1** | **Task (i)** Target & Classes | Target column (`grade_level`), 7 classes, 14.28% baseline, cleaning IDs & dates, scaling | `plot1_target_distribution.png` | $z = \frac{x - \mu}{\sigma}$ (Standardization) |
| **Member 2** | **Task (ii)** Feature Selection | Why Chi-Square + Mutual Info, why normalize to [0,1], why `age` dominates | `plot2_feature_importance.png` | $\chi^2 = \sum \frac{(O - E)^2}{E}$ |
| **Member 3** | **Task (iii)** PCA vs. LDA | Supervised vs. Unsupervised; why PCA fails (overlap) & why LDA works (LD1=99.9%) | `plot3_pca_vs_lda.png` | $J(w) = \frac{w^T S_B w}{w^T S_W w}$ (Fisher's Criterion) |
| **Member 4** | **Task (iv)** Random Forest | Bagging 100 trees on 3 selected features, majority vote, 42.6% accuracy | `plot4_model_comparison.png` (RF Matrix) | $I_G = 1 - \sum p_i^2$ (Gini Impurity) |
| **Member 5** | **Task (iv)** SVM & Comparison | Maximum margin, RBF kernel on 3 selected features, 44.1% accuracy, 91.7% adjacent accuracy | `plot4_model_comparison.png` (SVM + Metrics) | $K(x, x') = \exp(-\gamma \|x - x'\|^2)$ |

---

## Detailed Speaking Notes

### Member 1: Task (i) — Target Column & Preprocessing (3 mins)
- **What to say:**
  1. *"Our dataset has 12,025 student records. The target column is `grade_level`."*
  2. *"There are 7 distinct classes: Grade 6 to Grade 12. Each grade has roughly 1,700 students, meaning the classes are balanced."*
  3. *"Because there are 7 balanced classes, a random guess would only be right 1 out of 7 times, or 14.28%. This is our baseline."*
  4. *"For preprocessing, we dropped `student_id` because it's just an arbitrary ID that causes overfitting, and `survey_date` / `academic_year` to avoid leakage."*
  5. *"We encoded text columns with `LabelEncoder` and standardized numerical features with `StandardScaler`."*
- **Whiteboard:** Write $z = \frac{x - \mu}{\sigma}$ and explain that scaling prevents `age` from dominating distance calculations.
- **Show:** `plots/plot1_target_distribution.png` and lines 40–90 of `student_grade_classification.py`.

---

### Member 2: Task (ii) — Feature Selection (3 mins)
- **What to say:**
  1. *"To select the best features, we combined two statistical filter methods: Chi-Square Test and Mutual Information."*
  2. *"Chi-Square tests whether a feature is statistically independent of the grade classes. Mutual Information measures how much information a feature provides, including non-linear relationships."*
  3. *"Because raw Chi2 scores are in thousands while Mutual Info is around 1.0, we normalized both between 0 and 1 before averaging."*
  4. *"The result is striking: `age` achieved a perfect score of 1.000, while socioeconomic features like family size and residence type scored near zero."*
  5. *"Our final selected features are `age`, `num_siblings`, and `gender`."*
- **Whiteboard:** Write $\chi^2 = \sum \frac{(O - E)^2}{E}$ and explain observed vs. expected frequencies.
- **Show:** `plots/plot2_feature_importance.png` and lines 95–145 of `student_grade_classification.py`.

---

### Member 3: Task (iii) — Dimensionality Reduction: PCA vs. LDA (3 mins)
- **What to say:**
  1. *"We reduced the 6 features to 2 components using both PCA and LDA to see what happens."*
  2. *"PCA is unsupervised—it only looks for maximum total spread without knowing class labels. The first 2 components explain 49.4% of variance. Looking at the scatter plot, all 7 grades overlap completely. Therefore, PCA was NOT useful for classification."*
  3. *"LDA is supervised—it finds axes that maximize the distance between class centers while keeping points of the same class tight. The first linear discriminant (LD1) captures 99.99% of between-class variance."*
  4. *"In the LDA plot, you can clearly see grades ordered from 6 to 12. Therefore, LDA was HIGHLY useful for our dataset."*
- **Whiteboard:** Write Fisher's Criterion $J(w) = \frac{w^T S_B w}{w^T S_W w}$ (Between-class scatter / Within-class scatter).
- **Show:** `plots/plot3_pca_vs_lda.png` and lines 150–190 of `student_grade_classification.py`.

---

### Member 4: Task (iv) Part 1 — Model 1: Random Forest (3 mins)
- **What to say:**
  1. *"For our first model, we chose Random Forest Classifier, an ensemble bagging algorithm."*
  2. *"Mechanism: It builds 100 individual decision trees on random subsets of data and features, then takes a majority vote. Each tree splits nodes by minimizing Gini Impurity."*
  3. *"Pros: It handles non-linear relationships naturally, does not overfit easily, and requires minimal hyperparameter tuning."*
  4. *"Cons: Can be computationally slow with very large ensembles and behaves like a black-box."*
  5. *"Results: Random Forest achieved 42.62% exact accuracy and 85.82% adjacent-grade accuracy on the 3 selected features."*
- **Whiteboard:** Write Gini Impurity $I_G = 1 - \sum_{i=1}^C p_i^2$ (node is pure when $I_G = 0$).
- **Show:** `plots/plot4_model_comparison.png` (RF Confusion Matrix) and lines 195–230 of `student_grade_classification.py`.

---

### Member 5: Task (iv) Part 2 — Model 2: SVM & Final Evaluation (3 mins)
- **What to say:**
  1. *"For our second model, we chose Support Vector Machine (SVC) with a Radial Basis Function (RBF) kernel."*
  2. *"Mechanism: SVM finds the optimal hyperplane with the maximum margin between classes. The RBF kernel maps features into a higher-dimensional space to handle non-linear boundaries."*
  3. *"Pros: Highly effective in high dimensions; stores only support vectors."*
  4. *"Cons: Computationally intensive $O(N^2)$ to train and sensitive to feature scaling."*
  5. *"Performance comparison: SVM achieved **44.07% exact accuracy** and **91.68% adjacent-grade accuracy** on the 3 selected features."*
  6. *"Why ~44% accuracy is a success: First, it reaches 99% of the 43.77% theoretical Bayes ceiling. Second, looking at the confusion matrices, almost all errors are off by only $\pm 1$ grade (e.g., Grade 8 predicted as Grade 7 or 9). This occurs naturally because students of the same age can be in adjacent grades."*
- **Whiteboard:** Write RBF Kernel $K(x, x') = \exp(-\gamma \|x - x'\|^2)$ and F1 formula $F_1 = 2 \cdot \frac{P \cdot R}{P + R}$.
- **Show:** `plots/plot4_model_comparison.png` and lines 235–285 of `student_grade_classification.py`.
