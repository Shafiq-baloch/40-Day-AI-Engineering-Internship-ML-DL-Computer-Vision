import pandas as pd

df = pd.read_csv("data/titanic.csv")

print(df.head())
print(df.shape)

#choose taget column
y = df["Survived"]
X = df.drop("Survived", axis=1)

#identify categorical and numerical columns
numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns
categorical_cols = X.select_dtypes(include=["object"]).columns

#numeric pipeline
from sklearn.pipeline import Pipeline   
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

numeric_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

#categorical pipeline
from sklearn.preprocessing import OneHotEncoder
categorical_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])  

#column transformer
from sklearn.compose import ColumnTransformer   

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_pipeline, numeric_cols),
    ("cat", categorical_pipeline, categorical_cols)
])

#train and test split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

#logistic regression model
from sklearn.linear_model import LogisticRegression 
model  = LogisticRegression(max_iter=1000)

#full pipeline
from sklearn.pipeline import Pipeline

clf = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", model)
])  


#train the model
clf.fit(X_train, y_train)

#predict on the test set
y_pred = clf.predict(X_test)

#accuracy score
from sklearn.metrics import accuracy_score

acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc:.4f}")

#confusion matrix
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)
print(cm)

#classification report
from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))



#roc curve and auc
from sklearn.metrics import roc_auc_score

y_prob = clf.predict_proba(X_test)[:, 1]

auc = roc_auc_score(y_test, y_prob)
print("AUC Score:", auc)

#final evaluation summary
print("Accuracy:", acc)
print("AUC:", auc)
print("\nConfusion Matrix:\n", cm)