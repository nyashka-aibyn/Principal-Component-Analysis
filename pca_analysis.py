
# =====================================================
# PCA METHOD - WATER POTABILITY DATASET
# Principal Component Analysis
# =====================================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# -----------------------------------------------------
# 1. DATA LOADING
# -----------------------------------------------------

os.makedirs("results", exist_ok=True)

data = pd.read_csv("data/water_potability.csv")

print("DATASET INFORMATION")
print(data.head())

print("\nDataset shape:")
print(data.shape)

print("\nDataset information:")
print(data.info())

# -----------------------------------------------------
# 2. EXPLORATORY DATA ANALYSIS (EDA)
# -----------------------------------------------------

print("\nDESCRIPTIVE STATISTICS")
print(data.describe())

print("\nMissing values:")
print(data.isnull().sum())

print("\nTarget distribution:")
print(data["Potability"].value_counts())

# Target distribution graph
plt.figure(figsize=(7, 5))
sns.countplot(data=data, x="Potability")
plt.title("Water Potability Distribution")
plt.xlabel("Potability (0 = Not Drinkable, 1 = Drinkable)")
plt.ylabel("Number of Samples")
plt.savefig("results/target_distribution.png")
plt.close()

# Feature distributions
data.hist(figsize=(14, 10), bins=30)
plt.suptitle("Feature Distributions")
plt.tight_layout()
plt.savefig("results/feature_distributions.png")
plt.close()

# Correlation heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(data.corr(numeric_only=True),
            annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig("results/correlation_heatmap.png")
plt.close()

# -----------------------------------------------------
# 3. DATA CLEANING
# -----------------------------------------------------

print("\nDATA CLEANING")

# Remove duplicate rows
data = data.drop_duplicates()

# Remove rows without target values
data = data.dropna(subset=["Potability"])

# Separate features and target
X = data.drop("Potability", axis=1)
y = data["Potability"].astype(int)

print("Dataset after cleaning:", data.shape)

print("\nMissing values after removing target:")
print(X.isnull().sum())

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# -----------------------------------------------------
# 4. BASELINE MODEL - ORIGINAL FEATURES
# -----------------------------------------------------

print("\nBASELINE MODEL")

baseline_model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("classifier", RandomForestClassifier(
        n_estimators=100,
        random_state=42
    ))
])

baseline_model.fit(X_train, y_train)

baseline_predictions = baseline_model.predict(X_test)

baseline_accuracy = accuracy_score(
    y_test, baseline_predictions
)

baseline_precision = precision_score(
    y_test, baseline_predictions, zero_division=0
)

baseline_recall = recall_score(
    y_test, baseline_predictions, zero_division=0
)

baseline_f1 = f1_score(
    y_test, baseline_predictions, zero_division=0
)

print("Original Features Results:")
print("Accuracy:", baseline_accuracy)
print("Precision:", baseline_precision)
print("Recall:", baseline_recall)
print("F1 Score:", baseline_f1)

# -----------------------------------------------------
# 5. PCA - DIMENSIONALITY REDUCTION
# -----------------------------------------------------

print("\nPCA ANALYSIS")

pca_model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("pca", PCA(n_components=0.95)),
    ("classifier", RandomForestClassifier(
        n_estimators=100,
        random_state=42
    ))
])

pca_model.fit(X_train, y_train)

pca_predictions = pca_model.predict(X_test)

# Get PCA information
pca = pca_model.named_steps["pca"]

print("\nOriginal number of features:", X.shape[1])
print("Number of PCA components:", pca.n_components_)

print("\nExplained variance ratio:")
print(pca.explained_variance_ratio_)

print("Total retained information:",
      sum(pca.explained_variance_ratio_))

# PCA explained variance graph
plt.figure(figsize=(8, 5))

plt.plot(
    range(1, len(pca.explained_variance_ratio_) + 1),
    np.cumsum(pca.explained_variance_ratio_),
    marker="o"
)

plt.axhline(y=0.95, color="red", linestyle="--",
            label="95% Information")

plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("PCA Explained Variance")
plt.legend()
plt.grid()
plt.savefig("results/pca_explained_variance.png")
plt.close()

# -----------------------------------------------------
# 6. PCA MODEL EVALUATION
# -----------------------------------------------------

pca_accuracy = accuracy_score(y_test, pca_predictions)

pca_precision = precision_score(
    y_test, pca_predictions, zero_division=0
)

pca_recall = recall_score(
    y_test, pca_predictions, zero_division=0
)

pca_f1 = f1_score(
    y_test, pca_predictions, zero_division=0
)

print("\nPCA MODEL RESULTS")
print("Accuracy:", pca_accuracy)
print("Precision:", pca_precision)
print("Recall:", pca_recall)
print("F1 Score:", pca_f1)

print("\nClassification Report:")
print(classification_report(
    y_test, pca_predictions, zero_division=0
))

# Confusion matrix
cm = confusion_matrix(y_test, pca_predictions)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")

plt.title("PCA Model Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("results/confusion_matrix.png")
plt.close()

# -----------------------------------------------------
# 7. COMPARISON OF RESULTS
# -----------------------------------------------------

comparison = pd.DataFrame({
    "Metric": ["Accuracy", "Precision", "Recall", "F1 Score"],
    "Original Features": [
        baseline_accuracy,
        baseline_precision,
        baseline_recall,
        baseline_f1
    ],
    "PCA Features": [
        pca_accuracy,
        pca_precision,
        pca_recall,
        pca_f1
    ]
})

print("\nMODEL COMPARISON")
print(comparison)

comparison.to_csv(
    "results/model_comparison.csv",
    index=False
)

# Comparison graph
comparison.set_index("Metric").plot(
    kind="bar",
    figsize=(10, 6)
)

plt.title("Original Features vs PCA Features")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("results/model_comparison.png")
plt.close()

# -----------------------------------------------------
# 8. CONCLUSION
# -----------------------------------------------------

print("\nFINAL CONCLUSION")

print("Original features:", X.shape[1])
print("PCA components:", pca.n_components_)
print("Retained information:",
      round(sum(pca.explained_variance_ratio_) * 100, 2), "%")

if pca_accuracy > baseline_accuracy:
    print("PCA model achieved better Accuracy.")
elif pca_accuracy < baseline_accuracy:
    print("Original features achieved better Accuracy.")
else:
    print("Both models achieved the same Accuracy.")

print("\nAll analysis completed successfully!")
print("Results are saved in the results folder.")
