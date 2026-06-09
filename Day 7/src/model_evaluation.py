import pandas as pd

df = pd.read_csv("data/credit_card_fraud_10k.csv")

print(df.columns.tolist())
print(df.head())



import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score
)

from sklearn.preprocessing import LabelEncoder

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    classification_report,
    auc
)

from sklearn.calibration import calibration_curve


# ==========================================
# CREATE GRAPHS FOLDER
# ==========================================

os.makedirs("graphs", exist_ok=True)


# ==========================================
# LOAD DATASET
# ==========================================

df = pd.read_csv("data/credit_card_fraud_10k.csv")

print("=" * 50)
print("DATASET INFORMATION")
print("=" * 50)

print("\nShape:")
print(df.shape)

print("\nClass Distribution:")
print(df["is_fraud"].value_counts())

print("\nFraud Percentage:")
print(df["is_fraud"].value_counts(normalize=True) * 100)


# ==========================================
# ENCODE CATEGORICAL COLUMN
# ==========================================

encoder = LabelEncoder()

df["merchant_category"] = encoder.fit_transform(
    df["merchant_category"]
)

print("\nCategorical column encoded successfully.")


# ==========================================
# FEATURES AND TARGET
# ==========================================

X = df.drop("is_fraud", axis=1)
y = df["is_fraud"]

print("\nFeature Shape:", X.shape)
print("Target Shape:", y.shape)


# ==========================================
# TRAIN TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Shape:", X_train.shape)
print("Testing Shape :", X_test.shape)


# ==========================================
# STRATIFIED K FOLD
# ==========================================

skf = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


# ==========================================
# MODELS
# ==========================================

models = {
    "Logistic Regression":
        LogisticRegression(max_iter=1000),

    "Random Forest":
        RandomForestClassifier(
            n_estimators=100,
            random_state=42
        ),

    "Gradient Boosting":
        GradientBoostingClassifier(
            random_state=42
        )
}


# ==========================================
# 5-FOLD CROSS VALIDATION
# ==========================================

print("\n")
print("=" * 50)
print("5-FOLD CROSS VALIDATION RESULTS")
print("=" * 50)

cv_results = {}

for name, model in models.items():

    scores = cross_val_score(
        model,
        X,
        y,
        cv=skf,
        scoring="roc_auc"
    )

    cv_results[name] = scores

    print(f"\n{name}")
    print("AUC Scores:", scores)
    print(
        f"Mean AUC = {scores.mean():.4f} ± {scores.std():.4f}"
    )


# ==========================================
# ROC CURVES
# ==========================================

plt.figure(figsize=(8, 6))

for name, model in models.items():

    model.fit(X_train, y_train)

    probabilities = model.predict_proba(X_test)[:, 1]

    fpr, tpr, _ = roc_curve(
        y_test,
        probabilities
    )

    auc_score = roc_auc_score(
        y_test,
        probabilities
    )

    plt.plot(
        fpr,
        tpr,
        label=f"{name} (AUC={auc_score:.3f})"
    )

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves Comparison")
plt.legend()

plt.savefig(
    "graphs/roc_curves.png",
    bbox_inches="tight"
)

plt.show()


# ==========================================
# PRECISION-RECALL CURVE
# ==========================================

plt.figure(figsize=(8, 6))

for name, model in models.items():

    probabilities = model.predict_proba(X_test)[:, 1]

    precision, recall, _ = precision_recall_curve(
        y_test,
        probabilities
    )

    pr_auc = auc(recall, precision)

    plt.plot(
        recall,
        precision,
        label=f"{name} (AUC={pr_auc:.3f})"
    )

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend()

plt.savefig(
    "graphs/pr_curve.png",
    bbox_inches="tight"
)

plt.show()


# ==========================================
# CLASSIFICATION REPORT
# ==========================================

print("\n")
print("=" * 50)
print("CLASSIFICATION REPORTS")
print("=" * 50)

for name, model in models.items():

    predictions = model.predict(X_test)

    print("\n")
    print("-" * 50)
    print(name)
    print("-" * 50)

    print(
        classification_report(
            y_test,
            predictions
        )
    )


# ==========================================
# CALIBRATION CURVE
# ==========================================

plt.figure(figsize=(8, 6))

for name, model in models.items():

    probabilities = model.predict_proba(X_test)[:, 1]

    prob_true, prob_pred = calibration_curve(
        y_test,
        probabilities,
        n_bins=10
    )

    plt.plot(
        prob_pred,
        prob_true,
        marker="o",
        label=name
    )

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)

plt.xlabel("Mean Predicted Probability")
plt.ylabel("Observed Frequency")
plt.title("Calibration Curve")
plt.legend()

plt.savefig(
    "graphs/calibration_curve.png",
    bbox_inches="tight"
)

plt.show()


# ==========================================
# BEST MODEL
# ==========================================

print("\n")
print("=" * 50)
print("SUMMARY")
print("=" * 50)

for name, scores in cv_results.items():

    print(
        f"{name}: "
        f"{scores.mean():.4f} ± {scores.std():.4f}"
    )

print("\nAll graphs saved in graphs folder.")
print("Day 7 Task Completed Successfully!")