import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler


#load dataset
df = pd.read_csv('data/housing.csv')

#check dataset
print(df.head())

#check shape
print('Dataset Shape:', df.shape)


# Target column (what we want to predict)
y = df["median_house_value"]

# Features (everything except target)
X = df.drop("median_house_value", axis=1)

print("X shape:", X.shape)
print("y shape:", y.shape)

# Numeric columns (numbers only)
numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns

# Categorical columns (text/data like labels)
categorical_cols = X.select_dtypes(include=["object"]).columns

# Print results
print("\nNumeric Columns:")
print(numeric_cols)

print("\nCategorical Columns:")
print(categorical_cols)

# Optional: counts
print("\nNumber of Numeric Columns:", len(numeric_cols))
print("Number of Categorical Columns:", len(categorical_cols))


numeric_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_pipeline, numeric_cols),
    ("cat", categorical_pipeline, categorical_cols)
])


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

preprocessor.fit(X_train)

X_train_processed = preprocessor.transform(X_train)
X_test_processed = preprocessor.transform(X_test)

print("Train shape:", X_train_processed.shape)
print("Test shape:", X_test_processed.shape)


joblib.dump(preprocessor, "preprocessor.pkl")

from sklearn.pipeline import Pipeline

full_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor)
])