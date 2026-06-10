
# Import Libraries
# ---------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import RFECV

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score



# STEP 1: Load Dataset

df = pd.read_csv("data/titanic.csv")

print("\nDataset Shape:")
print(df.shape)

print("\nFirst 5 Rows:")
print(df.head())



# STEP 2: BASELINE MODEL
#
# Using original features only


baseline_df = df.copy()

# Select original features
baseline_features = [
    "Pclass",
    "Sex",
    "Age",
    "Fare",
    "Embarked"
]

X_baseline = baseline_df[baseline_features]
y = baseline_df["Survived"]



# Handle Missing Values


# Fill Age with median
X_baseline["Age"] = X_baseline["Age"].fillna(
    X_baseline["Age"].median()
)

# Fill Embarked with mode
X_baseline["Embarked"] = X_baseline["Embarked"].fillna(
    X_baseline["Embarked"].mode()[0]
)



# Encode Categorical Variables

le_sex = LabelEncoder()
le_embarked = LabelEncoder()

X_baseline["Sex"] = le_sex.fit_transform(
    X_baseline["Sex"]
)

X_baseline["Embarked"] = le_embarked.fit_transform(
    X_baseline["Embarked"]
)



# Train/Test Split

X_train, X_test, y_train, y_test = train_test_split(
    X_baseline,
    y,
    test_size=0.2,
    random_state=42
)


# ============================================================
# Train Baseline Model

baseline_model = RandomForestClassifier(
    random_state=42
)

baseline_model.fit(X_train, y_train)

baseline_predictions = baseline_model.predict(X_test)

baseline_accuracy = accuracy_score(
    y_test,
    baseline_predictions
)

print("\nBaseline Accuracy:")
print(round(baseline_accuracy, 4))


# ============================================================
# STEP 3: FEATURE ENGINEERING

engineered_df = df.copy()


# ------------------------------------------------------------
# Feature 1: Title
# Extract Mr, Mrs, Miss etc from Name
# ------------------------------------------------------------
engineered_df["Title"] = engineered_df["Name"].str.extract(
    r",\s*([^\.]+)\."
)

# Reduce rare titles
engineered_df["Title"] = engineered_df["Title"].replace(
    [
        "Lady",
        "Countess",
        "Capt",
        "Col",
        "Don",
        "Dr",
        "Major",
        "Rev",
        "Sir",
        "Jonkheer",
        "Dona"
    ],
    "Rare"
)

engineered_df["Title"] = engineered_df["Title"].replace(
    {
        "Mlle": "Miss",
        "Ms": "Miss",
        "Mme": "Mrs"
    }
)


# ------------------------------------------------------------
# Feature 2: FamilySize
# ------------------------------------------------------------
engineered_df["FamilySize"] = (
    engineered_df["SibSp"] +
    engineered_df["Parch"] +
    1
)


# ------------------------------------------------------------
# Feature 3: IsAlone
# ------------------------------------------------------------
engineered_df["IsAlone"] = np.where(
    engineered_df["FamilySize"] == 1,
    1,
    0
)


# ------------------------------------------------------------
# Feature 4: FarePerPerson
# ------------------------------------------------------------
engineered_df["FarePerPerson"] = (
    engineered_df["Fare"] /
    engineered_df["FamilySize"]
)


# ------------------------------------------------------------
# Fill Age Missing Before AgeBand
# ------------------------------------------------------------
engineered_df["Age"] = engineered_df["Age"].fillna(
    engineered_df["Age"].median()
)


# ------------------------------------------------------------
# Feature 5: AgeBand
# ------------------------------------------------------------
engineered_df["AgeBand"] = pd.cut(
    engineered_df["Age"],
    bins=[0, 16, 32, 48, 64, 100],
    labels=[0, 1, 2, 3, 4]
)

engineered_df["AgeBand"] = (
    engineered_df["AgeBand"].astype(int)
)


# ============================================================
# Select Features
# ============================================================

feature_columns = [
    "Pclass",
    "Sex",
    "Age",
    "Fare",
    "Embarked",
    "Title",
    "FamilySize",
    "IsAlone",
    "FarePerPerson",
    "AgeBand"
]

X = engineered_df[feature_columns]
y = engineered_df["Survived"]


# ============================================================
# Handle Missing Values
# ============================================================

X["Embarked"] = X["Embarked"].fillna(
    X["Embarked"].mode()[0]
)

X["Fare"] = X["Fare"].fillna(
    X["Fare"].median()
)

X["FarePerPerson"] = X["FarePerPerson"].fillna(
    X["FarePerPerson"].median()
)


# ============================================================
# Encode Categorical Columns
# ============================================================

label_columns = [
    "Sex",
    "Embarked",
    "Title"
]

for col in label_columns:
    encoder = LabelEncoder()
    X[col] = encoder.fit_transform(X[col])


# ============================================================
# STEP 4: Train Engineered Model
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

engineered_model = RandomForestClassifier(
    random_state=42
)

engineered_model.fit(
    X_train,
    y_train
)

engineered_predictions = engineered_model.predict(
    X_test
)

engineered_accuracy = accuracy_score(
    y_test,
    engineered_predictions
)

print("\nEngineered Accuracy:")
print(round(engineered_accuracy, 4))


# ============================================================
# STEP 5: SelectKBest
# ============================================================

selector = SelectKBest(
    score_func=f_classif,
    k=5
)

selector.fit(X, y)

scores = pd.DataFrame({
    "Feature": X.columns,
    "Score": selector.scores_
})

scores = scores.sort_values(
    by="Score",
    ascending=False
)

print("\nTop Features (SelectKBest)")
print(scores)


# ============================================================
# Graph 1 - SelectKBest Scores
# ============================================================

plt.figure(figsize=(10, 6))

plt.bar(
    scores["Feature"],
    scores["Score"]
)

plt.xticks(rotation=45)

plt.title(
    "SelectKBest Feature Scores"
)

plt.tight_layout()

plt.savefig(
    "graphs/selectkbest_scores.png"
)

plt.close()


# ============================================================
# STEP 6: RFECV
# ============================================================

rf_model = RandomForestClassifier(
    random_state=42
)

rfecv = RFECV(
    estimator=rf_model,
    step=1,
    cv=5,
    scoring="accuracy"
)

rfecv.fit(X, y)

print("\nOptimal Features Selected:")
print(rfecv.n_features_)

selected_features = X.columns[
    rfecv.support_
]

print("\nRFECV Selected Features:")
print(list(selected_features))


# ============================================================
# Graph 2 - RFECV
# ============================================================

plt.figure(figsize=(10, 6))

plt.plot(
    range(
        1,
        len(rfecv.cv_results_["mean_test_score"]) + 1
    ),
    rfecv.cv_results_["mean_test_score"]
)

plt.xlabel(
    "Number of Features"
)

plt.ylabel(
    "Cross Validation Accuracy"
)

plt.title(
    "RFECV Feature Selection"
)

plt.grid()

plt.savefig(
    "graphs/rfecv_features.png"
)

plt.close()


# ============================================================
# Graph 3 - Accuracy Comparison
# ============================================================

plt.figure(figsize=(6, 5))

plt.bar(
    ["Baseline", "Engineered"],
    [
        baseline_accuracy,
        engineered_accuracy
    ]
)

plt.ylabel("Accuracy")

plt.title(
    "Baseline vs Engineered Model"
)

plt.savefig(
    "graphs/baseline_vs_engineered.png"
)

plt.close()


# ============================================================
# Final Summary
# ============================================================

print("\n" + "=" * 60)

print("FINAL RESULTS")

print("=" * 60)

print(
    f"Baseline Accuracy: {baseline_accuracy:.4f}"
)

print(
    f"Engineered Accuracy: {engineered_accuracy:.4f}"
)

print(
    f"Improvement: "
    f"{(engineered_accuracy - baseline_accuracy):.4f}"
)

print("\nBest Features from RFECV:")

for feature in selected_features:
    print(feature)

print("\nGraphs Saved Successfully!")

print("=" * 60)