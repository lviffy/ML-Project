# Core Viva & Q&A Preparation Guide

The 6 most common questions professors ask for this project, with short and direct answers.

---

### Q1. What is the target column and how many classes are there?
- **Answer:** The target column is `grade_level`. It has **7 discrete classes** (`Grade 6, 7, 8, 9, 10, 11, 12`).
- Classes are roughly balanced (~1,600 to 1,800 students each).
- Random guessing accuracy is $1/7 \approx 14.28\%$.

---

### Q2. Why did you use Chi-Square and Mutual Information? What are the final features?
- **Answer:**
  1. **Chi-Square ($\chi^2$):** Tests statistical independence between discrete inputs and the target classes.
  2. **Mutual Information:** Measures both linear and non-linear information gain by calculating entropy reduction.
- We normalized both to $[0, 1]$ before averaging because raw scales differ vastly.
- **Final Features (strictly 3):** `age`, `num_siblings`, and `gender`.
- **Finding:** `age` scored 1.000 across both tests; socioeconomic features (family size, residence type, siblings) have near-zero predictive power for grade level.

---

### Q3. What happened when you applied PCA and LDA? Was it useful or not?
- **Answer:**
  - **PCA (Unsupervised):** Finds axes of maximum total variance without looking at class labels. The first 2 components capture 49.4% variance, but data points from all 7 grades overlap heavily. **Verdict: NOT useful for classification.**
  - **LDA (Supervised):** Finds axes that maximize distance between class means while minimizing class spread. LD1 captures **99.99%** of between-class variance and cleanly separates grades in order from 6 to 12. **Verdict: HIGHLY useful.**

---

### Q4. Which two ML models did you choose and why?
- **Answer:**
  1. **Random Forest Classifier:** A bagging ensemble of decision trees. Chosen because it handles non-linear boundaries naturally, is resistant to overfitting, and provides feature importance.
  2. **Support Vector Machine (SVC):** A margin-based classifier with RBF kernel. Chosen because it represents a completely different theoretical family (convex optimization with kernel trick) to benchmark against ensemble trees.

---

### Q5. What were your evaluation metrics and how did the models perform?
- **Answer:**
  - **Random Forest (100 Trees):** Exact Accuracy: **42.62%**, Adjacent (±1 Grade) Acc: **85.82%**, F1-Score: **42.14%**.
  - **SVM (RBF Kernel):** Exact Accuracy: **44.07%**, Adjacent (±1 Grade) Acc: **91.68%**, F1-Score: **41.75%**.
  - Both models perform closely, with Random Forest having a slight edge.

---

### Q6. Why is your accuracy ~44%? Is that low?
- **Answer (Key Defense):**
  1. For a **7-class** problem, random guessing is only **14.28%**. Achieving ~44% is **more than $3\times$ higher than chance**.
  2. Inspection of the **Confusion Matrix** reveals that errors are almost exclusively **off by $\pm 1$ grade** (e.g., Grade 8 predicted as Grade 7 or 9).
  3. In reality, students of the same age (e.g. 13 years old) attend adjacent grades due to cutoff birthdates and enrollment timing. Because `age` is the only strong signal, distinguishing between adjacent grades has inherent real-world ambiguity.
