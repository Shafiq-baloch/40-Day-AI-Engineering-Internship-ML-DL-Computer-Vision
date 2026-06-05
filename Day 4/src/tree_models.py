import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score


#load the dataset
df = pd.read_csv("data/titanic.csv")

print(df.head())
print(df.shape)

#select features and target variable
X = df.drop("Survived", axis=1)
y = df["Survived"]

print('X shape:', X.shape)
print('y shape:', y.shape)

#split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

#identify numerical and categorical columns
numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns

categorical_cols = X.select_dtypes(include=["object"]).columns

print("Numeric Columns:")
print(numeric_cols)

print("\nCategorical Columns:")
print(categorical_cols)

#preprocessing for numerical data          
numeric_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])             

#preprocessing for categorical data
categorical_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

#column transformer to apply the appropriate transformations to each column
preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_pipeline, numeric_cols),
    ("cat", categorical_pipeline, categorical_cols)
])

#Decision Tree Depth = 3
dt_model_3 = DecisionTreeClassifier(
    max_depth=3,
    random_state=42
)

#full pipeline for Decision Tree
model_dt3 = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", dt_model_3)
])


#train the Decision Tree model
model_dt3.fit(X_train, y_train)

#predict and evaluate the Decision Tree model
y_pred_dt3 = model_dt3.predict(X_test)

#Accuracy for Decision Tree Depth = 3

#train accuracy
train_acc_dt3 = model_dt3.score(X_train, y_train)

#test accuracy
test_acc_dt3 = accuracy_score(y_test, y_pred_dt3)

print("Decision Tree (max_depth=3)")
print("Train Accuracy:", train_acc_dt3)
print("Test Accuracy:", test_acc_dt3)


#Decision Tree max_Depth = 10

dt_model_10 = DecisionTreeClassifier(
    max_depth=10,
    random_state=42
)

model_dt10 = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", dt_model_10)
])

model_dt10.fit(X_train, y_train)

y_pred_dt10 = model_dt10.predict(X_test)

train_acc_dt10 = model_dt10.score(X_train, y_train)
test_acc_dt10 = accuracy_score(y_test, y_pred_dt10)

print("\nDecision Tree (max_depth=10)")
print("Train Accuracy:", train_acc_dt10)
print("Test Accuracy:", test_acc_dt10)



#Training Random Forest Classifier
rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model_rf = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", rf_model)
])

model_rf.fit(X_train, y_train)

y_pred_rf = model_rf.predict(X_test)

train_acc_rf = model_rf.score(X_train, y_train)
test_acc_rf = accuracy_score(y_test, y_pred_rf)

print("\nRandom Forest")
print("Train Accuracy:", train_acc_rf)
print("Test Accuracy:", test_acc_rf)

#feature importance from Random Forest what features are most important for predicting survival

import numpy as np

importances = model_rf.named_steps["model"].feature_importances_

feature_names = model_rf.named_steps["preprocessor"].get_feature_names_out()

feat_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
})

feat_df = feat_df.sort_values(by="Importance", ascending=False)

print("\nTop 10 Features:")
print(feat_df.head(10))


#plot feature importance
top_features = feat_df.head(10)

plt.figure(figsize=(10,6))
plt.barh(top_features["Feature"], top_features["Importance"])
plt.gca().invert_yaxis()
plt.title("Top 10 Feature Importances (Random Forest)")
plt.show()

#tree visualization
from sklearn.tree import plot_tree

single_tree = model_rf.named_steps["model"].estimators_[0]

plt.figure(figsize=(15,8))

plot_tree(single_tree,
          max_depth=2,
          filled=True,
          feature_names=feature_names)

plt.show()