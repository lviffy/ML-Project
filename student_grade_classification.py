# =============================================================================
# ML Project: Student Grade Level Classification
# Dataset: Student Grade & Socioeconomic Status Dataset
# =============================================================================
#
# What this script does (step by step):
#   1. Loads the dataset and explores it
#   2. Finds the target column and counts classes
#   3. Cleans and prepares the data for ML
#   4. Selects the best features using 3 methods
#   5. Applies PCA and LDA to reduce dimensions
#   6. Trains two ML models (Random Forest and SVM)
#   7. Compares how well each model performs
# =============================================================================


# --------------------------------------------------
# Step 0: Import all the libraries we need
# --------------------------------------------------

import pandas as pd                  # for working with tables/dataframes
import numpy as np                   # for math operations
import matplotlib                    
matplotlib.use('Agg')                # lets us save plots without a display
import matplotlib.pyplot as plt      # for making graphs
import seaborn as sns                # for nicer-looking graphs
import os                            # for creating folders
import warnings
warnings.filterwarnings('ignore')    # hide unnecessary warnings

# scikit-learn imports (our ML toolkit)
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.feature_selection import chi2, mutual_info_classif, f_classif
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, classification_report, confusion_matrix,
                             ConfusionMatrixDisplay)

# Create a folder to save all our plots
os.makedirs("plots", exist_ok=True)

# Make plots look clean
sns.set_theme(style="whitegrid")
plt.rcParams.update({'figure.dpi': 120})

print("All libraries loaded.\n")


# =============================================================================
# TASK (i): Load the dataset, find the target column, count classes
# =============================================================================

# Load the CSV file into a dataframe (like a spreadsheet)
df = pd.read_csv("student_grade_ses_dataset-selected-columns.csv")

# Basic info about the dataset
print("DATASET OVERVIEW")
print("-" * 40)
print(f"Number of rows    : {df.shape[0]}")
print(f"Number of columns : {df.shape[1]}")
print(f"\nColumn names and their data types:")
print(df.dtypes)
print(f"\nFirst 5 rows of the dataset:")
print(df.head())

# -- Finding the Target Column --
# The dataset is called "student_grade_ses_dataset".
# Looking at the columns, "grade_level" is what we want to predict.
# It tells us which grade (class 6 to 12) a student is in.

target = "grade_level"
num_classes = df[target].nunique()        # how many unique grades
class_list = sorted(df[target].unique())  # list of all grades

print(f"\n{'=' * 40}")
print(f"TARGET COLUMN     : {target}")
print(f"NUMBER OF CLASSES : {num_classes}")
print(f"CLASS VALUES      : {class_list}")
print(f"{'=' * 40}")

# Plot: How many students are in each grade?
fig, ax = plt.subplots(figsize=(8, 5))
grade_counts = df[target].value_counts().sort_index()
ax.bar(grade_counts.index.astype(str), grade_counts.values,
       color=sns.color_palette("viridis", num_classes))
ax.set_xlabel("Grade Level")
ax.set_ylabel("Number of Students")
ax.set_title("How Many Students in Each Grade?")
# Add count labels on top of each bar
for i, val in enumerate(grade_counts.values):
    ax.text(i, val + 30, str(val), ha='center')
plt.tight_layout()
plt.savefig("plots/01_target_distribution.png")
plt.close()
print("[OK] Saved: plots/01_target_distribution.png")


# =============================================================================
# DATA PREPROCESSING (Cleaning and Preparing the Data)
# =============================================================================

# -- Step 1: Remove columns that won't help with prediction --
# "student_id" is just a unique ID, not useful for prediction
# "survey_date" and "academic_year" are just dates, not real features

columns_to_drop = ["student_id", "survey_date", "academic_year"]
df_clean = df.drop(columns=columns_to_drop)
print(f"\nDropped columns: {columns_to_drop}")
print(f"Columns kept: {list(df_clean.columns)}")

# -- Step 2: Check for missing values --
print(f"\nMissing values per column:")
print(df_clean.isnull().sum())
# Great -- no missing values in this dataset!

# -- Step 3: Convert text columns to numbers --
# ML models can only work with numbers, so we convert text like
# "Male"/"Female" into numbers like 0/1/2.

text_columns = ["gender", "school_type", "residence_type"]

df_encoded = df_clean.copy()
encoders = {}  # save encoders so we know what number means what

print("\nConverting text to numbers:")
for col in text_columns:
    le = LabelEncoder()
    df_encoded[col] = le.fit_transform(df_encoded[col])
    encoders[col] = le
    mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    print(f"  {col}: {mapping}")

# -- Step 4: Split into features (X) and target (y) --
# X = the input columns (what we use to predict)
# y = the output column (what we want to predict)

X = df_encoded.drop(columns=[target])
y = df_encoded[target]

feature_names = list(X.columns)
print(f"\nFeatures ({len(feature_names)}): {feature_names}")
print(f"Target: {target}")
print(f"Total samples: {len(y)}")

# -- Step 5: Scale the features --
# Scaling makes all features have mean=0 and std=1.
# This is important because some features have big numbers (like age=18)
# and some have small numbers (like num_siblings=2).
# Without scaling, the bigger numbers would unfairly dominate.

scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=feature_names)
print("[OK] Features scaled (standardized).")


# =============================================================================
# TASK (ii): Feature Selection
# =============================================================================
# We want to find which features are actually useful for predicting grade_level.
# We use 3 different methods and combine their results.

print(f"\n{'=' * 50}")
print("FEATURE SELECTION")
print(f"{'=' * 50}")

# --- Method 1: Chi-Square Test ---
# This tests if there is a statistical relationship between each feature
# and the target. Higher score = stronger relationship.
# Note: Chi-Square needs positive numbers, so we shift values up.

X_positive = X - X.min()  # make all values non-negative
chi2_scores, chi2_pvalues = chi2(X_positive, y)

print("\nMethod 1: Chi-Square Test")
print("  (Tests statistical relationship between feature and target)")
for name, score, pval in zip(feature_names, chi2_scores, chi2_pvalues):
    print(f"  {name:20s}  Score: {score:10.2f}   p-value: {pval:.6f}")

# --- Method 2: Mutual Information ---
# This measures how much information a feature gives us about the target.
# It can detect non-linear relationships too.
# Higher score = feature is more informative.

mi_scores = mutual_info_classif(X_scaled, y, random_state=42)

print("\nMethod 2: Mutual Information")
print("  (Measures how much info each feature gives about the target)")
for name, score in zip(feature_names, mi_scores):
    print(f"  {name:20s}  Score: {score:.4f}")

# --- Method 3: ANOVA F-Test ---
# This compares the variance of each feature across different classes.
# If a feature varies a lot between classes, it's probably useful.
# Higher score = feature separates classes better.

f_scores, f_pvalues = f_classif(X_scaled, y)

print("\nMethod 3: ANOVA F-Test")
print("  (Checks if feature values differ significantly across classes)")
for name, score, pval in zip(feature_names, f_scores, f_pvalues):
    print(f"  {name:20s}  Score: {score:10.2f}   p-value: {pval:.6f}")

# --- Combine all 3 methods ---
# Normalize each method's scores to 0-1 range, then average them.

chi2_norm = MinMaxScaler().fit_transform(chi2_scores.reshape(-1, 1)).flatten()
mi_norm = MinMaxScaler().fit_transform(mi_scores.reshape(-1, 1)).flatten()
f_norm = MinMaxScaler().fit_transform(f_scores.reshape(-1, 1)).flatten()
avg_scores = (chi2_norm + mi_norm + f_norm) / 3

print("\nCombined Ranking (average of all 3 methods, normalized 0-1):")
ranking = sorted(zip(feature_names, avg_scores), key=lambda x: x[1], reverse=True)
for name, score in ranking:
    print(f"  {name:20s}  Combined Score: {score:.4f}")

# Pick the top features (at least 3)
selected_features = [name for name, score in ranking if score >= 0.1]
if len(selected_features) < 3:
    selected_features = [name for name, _ in ranking[:3]]

print(f"\nFinal selected features: {selected_features}")

# Plot: Compare all 3 methods side by side
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].barh(feature_names, chi2_scores, color="#2196F3")
axes[0].set_title("Chi-Square Scores")
axes[0].set_xlabel("Score")

axes[1].barh(feature_names, mi_scores, color="#4CAF50")
axes[1].set_title("Mutual Information Scores")
axes[1].set_xlabel("Score")

axes[2].barh(feature_names, f_scores, color="#FF9800")
axes[2].set_title("ANOVA F-Test Scores")
axes[2].set_xlabel("Score")

plt.suptitle("Feature Importance: Three Methods Compared", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("plots/02_feature_selection_comparison.png", bbox_inches="tight")
plt.close()
print("[OK] Saved: plots/02_feature_selection_comparison.png")

# Plot: Combined ranking
fig, ax = plt.subplots(figsize=(9, 5))
names_sorted = [n for n, _ in sorted(zip(feature_names, avg_scores), key=lambda x: x[1])]
scores_sorted = sorted(avg_scores)
ax.barh(names_sorted, scores_sorted, color=sns.color_palette("viridis", len(names_sorted)))
ax.set_xlabel("Combined Score (0 to 1)")
ax.set_title("Combined Feature Importance Ranking")
for i, val in enumerate(scores_sorted):
    ax.text(val + 0.01, i, f"{val:.3f}", va="center")
plt.tight_layout()
plt.savefig("plots/03_combined_feature_ranking.png")
plt.close()
print("[OK] Saved: plots/03_combined_feature_ranking.png")


# =============================================================================
# TASK (iii): PCA and LDA
# =============================================================================
# Both PCA and LDA reduce the number of dimensions (features).
# PCA: Unsupervised -- finds directions with the most variance (spread).
# LDA: Supervised   -- finds directions that best separate the classes.

print(f"\n{'=' * 50}")
print("DIMENSIONALITY REDUCTION: PCA and LDA")
print(f"{'=' * 50}")

# --- PCA (Principal Component Analysis) ---
# PCA does NOT look at the target (y). It just finds directions where
# the data spreads out the most. It's "unsupervised".

pca = PCA()  # keep all components to see how much each explains
X_pca = pca.fit_transform(X_scaled)

print("\nPCA Results:")
print("  Each principal component explains this much of the total variance:")
cumulative = 0
for i, var in enumerate(pca.explained_variance_ratio_):
    cumulative += var
    print(f"  PC{i+1}: {var*100:.2f}%   (cumulative: {cumulative*100:.2f}%)")

# Plot: How much variance each PC explains
fig, ax = plt.subplots(figsize=(9, 5))
pcs = range(1, len(pca.explained_variance_ratio_) + 1)
ax.bar(pcs, pca.explained_variance_ratio_, alpha=0.7, color="#2196F3", label="Individual")
ax.plot(pcs, np.cumsum(pca.explained_variance_ratio_), "ro-", label="Cumulative")
ax.axhline(y=0.95, color="gray", linestyle="--", alpha=0.7, label="95% line")
ax.set_xlabel("Principal Component")
ax.set_ylabel("Variance Explained")
ax.set_title("PCA: How Much Variance Each Component Explains")
ax.legend()
plt.tight_layout()
plt.savefig("plots/04_pca_explained_variance.png")
plt.close()
print("[OK] Saved: plots/04_pca_explained_variance.png")

# PCA with just 2 components (so we can plot it in 2D)
pca_2d = PCA(n_components=2)
X_pca_2d = pca_2d.fit_transform(X_scaled)

fig, ax = plt.subplots(figsize=(9, 7))
scatter = ax.scatter(X_pca_2d[:, 0], X_pca_2d[:, 1],
                     c=y, cmap="viridis", alpha=0.4, s=10)
plt.colorbar(scatter, ax=ax, label="Grade Level")
ax.set_xlabel(f"PC1 ({pca_2d.explained_variance_ratio_[0]*100:.1f}%)")
ax.set_ylabel(f"PC2 ({pca_2d.explained_variance_ratio_[1]*100:.1f}%)")
ax.set_title("PCA: Data Projected onto 2 Components")
plt.tight_layout()
plt.savefig("plots/05_pca_2d_scatter.png")
plt.close()
print("[OK] Saved: plots/05_pca_2d_scatter.png")


# --- LDA (Linear Discriminant Analysis) ---
# LDA DOES look at the target (y). It finds directions that best
# separate the different classes. It's "supervised".
# Max LDA components = min(num_features, num_classes - 1) = min(6, 6) = 6

n_lda = min(len(feature_names), num_classes - 1)
lda = LDA(n_components=n_lda)
X_lda = lda.fit_transform(X_scaled, y)

print(f"\nLDA Results:")
print(f"  Number of LDA components: {n_lda}")
print(f"  Each component explains this much of the between-class variance:")
for i, var in enumerate(lda.explained_variance_ratio_):
    print(f"  LD{i+1}: {var*100:.2f}%")

# LDA 2D scatter plot
fig, ax = plt.subplots(figsize=(9, 7))
scatter = ax.scatter(X_lda[:, 0], X_lda[:, 1],
                     c=y, cmap="viridis", alpha=0.4, s=10)
plt.colorbar(scatter, ax=ax, label="Grade Level")
ax.set_xlabel(f"LD1 ({lda.explained_variance_ratio_[0]*100:.1f}%)")
ax.set_ylabel(f"LD2 ({lda.explained_variance_ratio_[1]*100:.1f}%)")
ax.set_title("LDA: Data Projected onto 2 Components")
plt.tight_layout()
plt.savefig("plots/06_lda_2d_scatter.png")
plt.close()
print("[OK] Saved: plots/06_lda_2d_scatter.png")

# Plot: PCA vs LDA side by side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

ax1.scatter(X_pca_2d[:, 0], X_pca_2d[:, 1], c=y, cmap="viridis", alpha=0.4, s=8)
ax1.set_title("PCA (Unsupervised)")
ax1.set_xlabel(f"PC1 ({pca_2d.explained_variance_ratio_[0]*100:.1f}%)")
ax1.set_ylabel(f"PC2 ({pca_2d.explained_variance_ratio_[1]*100:.1f}%)")

sc = ax2.scatter(X_lda[:, 0], X_lda[:, 1], c=y, cmap="viridis", alpha=0.4, s=8)
ax2.set_title("LDA (Supervised)")
ax2.set_xlabel(f"LD1 ({lda.explained_variance_ratio_[0]*100:.1f}%)")
ax2.set_ylabel(f"LD2 ({lda.explained_variance_ratio_[1]*100:.1f}%)")
plt.colorbar(sc, ax=ax2, label="Grade Level")

plt.suptitle("PCA vs LDA: Which Separates Classes Better?", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("plots/07_pca_vs_lda.png", bbox_inches="tight")
plt.close()
print("[OK] Saved: plots/07_pca_vs_lda.png")

# Summary of PCA vs LDA
pca_var = np.sum(pca_2d.explained_variance_ratio_) * 100
lda_var = np.sum(lda.explained_variance_ratio_[:2]) * 100
print(f"\nPCA vs LDA Summary:")
print(f"  PCA top 2 components explain {pca_var:.1f}% of total variance")
print(f"  LDA top 2 components explain {lda_var:.1f}% of between-class variance")
print(f"  LDA is better for classification because it directly tries")
print(f"  to separate the classes, while PCA just looks at spread.")


# =============================================================================
# TASK (iv): Train and Compare Two ML Models
# =============================================================================
# Model 1: Random Forest -- builds many decision trees and takes a vote
# Model 2: SVM (Support Vector Machine) -- finds the best boundary between classes

print(f"\n{'=' * 50}")
print("MODEL TRAINING AND COMPARISON")
print(f"{'=' * 50}")

# -- Split data into training (80%) and testing (20%) --
# We train on 80% and test on 20% to see how well the model generalizes.
# stratify=y makes sure each grade has the same proportion in train and test.

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTraining samples: {len(X_train)}")
print(f"Testing samples : {len(X_test)}")


# --- Model 1: Random Forest ---
# How it works:
#   - Creates many decision trees (like flowcharts)
#   - Each tree learns from a random subset of the data
#   - Final prediction = majority vote of all trees
# Pros: Handles complex patterns, hard to overfit, works well out of the box
# Cons: Slow with very large datasets, hard to interpret

print("\nTraining Model 1: Random Forest...")
rf_model = RandomForestClassifier(
    n_estimators=200,   # number of trees
    max_depth=15,       # max depth of each tree
    random_state=42     # for reproducibility
)
rf_model.fit(X_train, y_train)          # train the model
rf_predictions = rf_model.predict(X_test)  # make predictions on test data

# Calculate metrics for Random Forest
rf_accuracy  = accuracy_score(y_test, rf_predictions)
rf_precision = precision_score(y_test, rf_predictions, average="weighted", zero_division=0)
rf_recall    = recall_score(y_test, rf_predictions, average="weighted", zero_division=0)
rf_f1        = f1_score(y_test, rf_predictions, average="weighted", zero_division=0)

print(f"  Accuracy  : {rf_accuracy:.4f}  (what % of predictions were correct)")
print(f"  Precision : {rf_precision:.4f}  (of predicted grade X, how many were actually X)")
print(f"  Recall    : {rf_recall:.4f}  (of actual grade X, how many did we find)")
print(f"  F1 Score  : {rf_f1:.4f}  (balance between precision and recall)")


# --- Model 2: SVM (Support Vector Machine) ---
# How it works:
#   - Finds the best boundary (hyperplane) that separates classes
#   - RBF kernel allows it to handle non-linear boundaries
# Pros: Works well in high dimensions, effective with clear margins
# Cons: Slow with large datasets, sensitive to feature scaling

print("\nTraining Model 2: SVM (Support Vector Machine)...")
svm_model = SVC(
    kernel="rbf",       # radial basis function kernel (handles curves)
    C=10,               # how strict the boundary is
    gamma="scale",      # kernel coefficient
    random_state=42
)
svm_model.fit(X_train, y_train)
svm_predictions = svm_model.predict(X_test)

# Calculate metrics for SVM
svm_accuracy  = accuracy_score(y_test, svm_predictions)
svm_precision = precision_score(y_test, svm_predictions, average="weighted", zero_division=0)
svm_recall    = recall_score(y_test, svm_predictions, average="weighted", zero_division=0)
svm_f1        = f1_score(y_test, svm_predictions, average="weighted", zero_division=0)

print(f"  Accuracy  : {svm_accuracy:.4f}")
print(f"  Precision : {svm_precision:.4f}")
print(f"  Recall    : {svm_recall:.4f}")
print(f"  F1 Score  : {svm_f1:.4f}")


# --- Side-by-side comparison ---
print(f"\n{'=' * 60}")
print("COMPARISON TABLE")
print(f"{'=' * 60}")
print(f"{'Metric':<15} {'Random Forest':>15} {'SVM (RBF)':>15}")
print(f"{'-'*15} {'-'*15} {'-'*15}")
print(f"{'Accuracy':<15} {rf_accuracy:>15.4f} {svm_accuracy:>15.4f}")
print(f"{'Precision':<15} {rf_precision:>15.4f} {svm_precision:>15.4f}")
print(f"{'Recall':<15} {rf_recall:>15.4f} {svm_recall:>15.4f}")
print(f"{'F1 Score':<15} {rf_f1:>15.4f} {svm_f1:>15.4f}")

# Plot: Bar chart comparing both models
fig, ax = plt.subplots(figsize=(10, 6))
metrics_names = ["Accuracy", "Precision", "Recall", "F1 Score"]
rf_scores = [rf_accuracy, rf_precision, rf_recall, rf_f1]
svm_scores = [svm_accuracy, svm_precision, svm_recall, svm_f1]

x = np.arange(len(metrics_names))
width = 0.35

bars1 = ax.bar(x - width/2, rf_scores, width, label="Random Forest", color="#2196F3")
bars2 = ax.bar(x + width/2, svm_scores, width, label="SVM (RBF)", color="#FF9800")

ax.set_ylabel("Score")
ax.set_title("Random Forest vs SVM: Performance Comparison")
ax.set_xticks(x)
ax.set_xticklabels(metrics_names)
ax.set_ylim(0, 1.0)
ax.legend()
ax.bar_label(bars1, fmt="%.3f", fontsize=9, padding=3)
ax.bar_label(bars2, fmt="%.3f", fontsize=9, padding=3)
plt.tight_layout()
plt.savefig("plots/08_model_comparison.png")
plt.close()
print("\n[OK] Saved: plots/08_model_comparison.png")


# --- Confusion Matrices ---
# A confusion matrix shows exactly where the model gets confused.
# Rows = actual grade, Columns = predicted grade.
# Diagonal = correct predictions, Off-diagonal = mistakes.

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

cm_rf = confusion_matrix(y_test, rf_predictions, labels=class_list)
ConfusionMatrixDisplay(cm_rf, display_labels=class_list).plot(ax=ax1, cmap="Blues")
ax1.set_title("Random Forest: Confusion Matrix")

cm_svm = confusion_matrix(y_test, svm_predictions, labels=class_list)
ConfusionMatrixDisplay(cm_svm, display_labels=class_list).plot(ax=ax2, cmap="Oranges")
ax2.set_title("SVM: Confusion Matrix")

plt.suptitle("Where Do the Models Make Mistakes?", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("plots/09_confusion_matrices.png", bbox_inches="tight")
plt.close()
print("[OK] Saved: plots/09_confusion_matrices.png")


# --- Detailed Classification Reports ---
# Shows precision, recall, and F1 for EACH grade separately.

print(f"\n{'=' * 60}")
print("DETAILED REPORT: Random Forest")
print(f"{'=' * 60}")
print(classification_report(y_test, rf_predictions,
                            target_names=[f"Grade {g}" for g in class_list]))

print(f"{'=' * 60}")
print("DETAILED REPORT: SVM (RBF)")
print(f"{'=' * 60}")
print(classification_report(y_test, svm_predictions,
                            target_names=[f"Grade {g}" for g in class_list]))


# --- 5-Fold Cross Validation ---
# Instead of one train/test split, we split the data 5 different ways
# and average the results. This gives a more reliable estimate.

print(f"{'=' * 60}")
print("5-FOLD CROSS VALIDATION")
print(f"{'=' * 60}")

rf_cv_scores = cross_val_score(
    RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42),
    X_scaled, y, cv=5, scoring="accuracy"
)
svm_cv_scores = cross_val_score(
    SVC(kernel="rbf", C=10, gamma="scale", random_state=42),
    X_scaled, y, cv=5, scoring="accuracy"
)

print(f"\nRandom Forest : {rf_cv_scores.mean():.4f} (+/- {rf_cv_scores.std():.4f})")
print(f"  Each fold   : {np.round(rf_cv_scores, 4)}")
print(f"SVM (RBF)     : {svm_cv_scores.mean():.4f} (+/- {svm_cv_scores.std():.4f})")
print(f"  Each fold   : {np.round(svm_cv_scores, 4)}")

# Plot: Cross validation box plot
fig, ax = plt.subplots(figsize=(8, 5))
bp = ax.boxplot([rf_cv_scores, svm_cv_scores],
                tick_labels=["Random Forest", "SVM (RBF)"],
                patch_artist=True, widths=0.4)
bp["boxes"][0].set_facecolor("#2196F3")
bp["boxes"][1].set_facecolor("#FF9800")
for box in bp["boxes"]:
    box.set_alpha(0.7)
ax.set_ylabel("Accuracy")
ax.set_title("5-Fold Cross Validation: Accuracy Spread")
plt.tight_layout()
plt.savefig("plots/10_cv_comparison.png")
plt.close()
print("[OK] Saved: plots/10_cv_comparison.png")


# --- Random Forest Feature Importance ---
# Random Forest can tell us which features it found most useful.

rf_full = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42)
rf_full.fit(X_scaled, y)

fig, ax = plt.subplots(figsize=(9, 5))
importances = rf_full.feature_importances_
sorted_idx = np.argsort(importances)
ax.barh([feature_names[i] for i in sorted_idx],
        importances[sorted_idx],
        color=sns.color_palette("viridis", len(feature_names)))
ax.set_xlabel("Importance Score")
ax.set_title("Random Forest: Which Features Matter Most?")
for i, idx in enumerate(sorted_idx):
    ax.text(importances[idx] + 0.002, i, f"{importances[idx]:.4f}", va="center")
plt.tight_layout()
plt.savefig("plots/11_rf_feature_importance.png")
plt.close()
print("[OK] Saved: plots/11_rf_feature_importance.png")


# --- Correlation Heatmap ---
# Shows how features relate to each other.
# Values close to +1 or -1 mean strong relationship.
# Values close to 0 mean no relationship.

fig, ax = plt.subplots(figsize=(9, 7))
corr = df_encoded.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))  # hide upper triangle (it's a mirror)
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
            center=0, ax=ax, square=True)
ax.set_title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig("plots/12_correlation_heatmap.png")
plt.close()
print("[OK] Saved: plots/12_correlation_heatmap.png")


# =============================================================================
# FINAL SUMMARY
# =============================================================================

print(f"\n{'=' * 60}")
print("FINAL PROJECT SUMMARY")
print(f"{'=' * 60}")

print(f"""
DATASET:
  Student Grade & Socioeconomic Status Dataset
  {df.shape[0]} samples, {df.shape[1]} columns

TASK (i) - TARGET:
  Column  : grade_level
  Classes : {num_classes} grades -> {class_list}

TASK (ii) - FEATURE SELECTION:
  Methods used: Chi-Square, Mutual Information, ANOVA F-Test
  Selected features: {selected_features}
  Key finding: 'age' is by far the most important feature.

TASK (iii) - PCA vs LDA:
  PCA top 2 components explain {pca_var:.1f}% of total variance
  LDA top 2 components explain {lda_var:.1f}% of between-class variance
  Conclusion: LDA is more useful here because it directly
  optimizes for separating the grade level classes.

TASK (iv) - MODEL COMPARISON:
  Random Forest : Accuracy={rf_accuracy:.4f}  F1={rf_f1:.4f}
  SVM (RBF)     : Accuracy={svm_accuracy:.4f}  F1={svm_f1:.4f}
  Cross-Val RF  : {rf_cv_scores.mean():.4f} +/- {rf_cv_scores.std():.4f}
  Cross-Val SVM : {svm_cv_scores.mean():.4f} +/- {svm_cv_scores.std():.4f}

All plots saved in: plots/
""")

print(f"{'=' * 60}")
print("PROJECT COMPLETE")
print(f"{'=' * 60}")
