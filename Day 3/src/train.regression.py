import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv('data/housing.csv')

print(df.head())
print(df.shape)

#Separate x and y
X = df.drop('median_house_value', axis=1)
y = df['median_house_value']


#quick check
print(X.columns)
print(y.head())


#Split data into train and test
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

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

#train the regression model
from sklearn.linear_model import LinearRegression

model = LinearRegression()

#full pipeline
from sklearn.pipeline import Pipeline

clf = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", model)
])

#train the model
clf.fit(X_train, y_train)

#predict on test set
y_pred = clf.predict(X_test)

#root mean squared error
from sklearn.metrics import mean_squared_error
import numpy as np

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print("RMSE:", rmse)


#r2 score
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_pred)
print("R2 Score:", r2)


# Get feature names after preprocessing
feature_names = clf.named_steps["preprocessor"].get_feature_names_out()

# Get model coefficients
model_coefficients = clf.named_steps["model"].coef_

# Create a DataFrame to display feature importance
import pandas as pd

coef_df = pd.DataFrame({
    "Feature": feature_names,
    "Coefficient": model_coefficients
})

#sort important features
coef_df = coef_df.sort_values(by="Coefficient", ascending=False)

#show top 10 features
print("Top 10 Positive Features:")
print(coef_df.head(10))

print("\nTop 10 Negative Features:")
print(coef_df.tail(10))

#plot feature importance
import matplotlib.pyplot as plt

top_features = coef_df.head(10)

plt.figure(figsize=(10,6))
plt.barh(top_features["Feature"], top_features["Coefficient"])
plt.title("Top Positive Features Affecting House Price")
plt.gca().invert_yaxis()
plt.show()

#ridge regression
from sklearn.linear_model import Ridge

ridge_model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", Ridge(alpha=1.0))
])

ridge_model.fit(X_train, y_train)

y_pred_ridge = ridge_model.predict(X_test)

#evaluate ridge model
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

rmse_ridge = np.sqrt(mean_squared_error(y_test, y_pred_ridge))
r2_ridge = r2_score(y_test, y_pred_ridge)

print("Ridge RMSE:", rmse_ridge)
print("Ridge R2:", r2_ridge)

#lasso regression
from sklearn.linear_model import Lasso

lasso_model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", Lasso(alpha=0.1))
])

lasso_model.fit(X_train, y_train)

y_pred_lasso = lasso_model.predict(X_test)

#evaluate lasso model
rmse_lasso = np.sqrt(mean_squared_error(y_test, y_pred_lasso))
r2_lasso = r2_score(y_test, y_pred_lasso)

print("Lasso RMSE:", rmse_lasso)
print("Lasso R2:", r2_lasso)

#compare models
print("\nModel Comparison:")
print("Linear Regression R2:", r2)
print("Ridge R2:", r2_ridge)
print("Lasso R2:", r2_lasso)