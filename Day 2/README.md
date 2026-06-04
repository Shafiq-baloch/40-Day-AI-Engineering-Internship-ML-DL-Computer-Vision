Day 2 Internship Report — Preprocessing Pipeline
🧾 Title

Day 2: Building a Data Preprocessing Pipeline with Zero Data Leakage

🎯 Objective

The objective of this task was to build a complete machine learning preprocessing pipeline using Scikit-learn that:

Handles numeric and categorical data separately
Fills missing values properly
Applies feature scaling
Encodes categorical variables
Prevents data leakage
Saves the final pipeline for reuse

📊 Dataset Used
Dataset: California Housing Dataset
Target Variable: median_house_value
Features include:
Numeric: longitude, latitude, income, population, etc.
Categorical: ocean_proximity

🧠 Concepts Learned
1. Feature Separation

The dataset was split into:

X (Features) → input variables
y (Target) → value to predict (house price)
2. Train-Test Split

Data was split into training and testing sets using:

80% training data
20% testing data
Ensures unbiased model evaluation
3. Column Types Identification
Numeric columns → handled using scaling
Categorical columns → handled using encoding
4. Pipelines

Two separate pipelines were created:

Numeric Pipeline:
SimpleImputer (median)
StandardScaler
Categorical Pipeline:
SimpleImputer (most frequent)
OneHotEncoder
5. ColumnTransformer

Both pipelines were combined using ColumnTransformer to apply correct transformations to each column type automatically.

6. Data Leakage Prevention

All transformations were:

Fit ONLY on training data
Applied to test data separately

This ensured no information from test data leaked into training.

7. Pipeline Serialization

The final preprocessing pipeline was saved using joblib for reuse in model training and deployment.

⚙️ Final Workflow
Raw Data
   ↓
Train/Test Split
   ↓
Numeric Pipeline ───┐
                    ├── ColumnTransformer → Preprocessed Data
Categorical Pipeline ─┘
   ↓
Saved as .pkl file
💻 Tools Used
Python
Pandas
NumPy
Scikit-learn (Pipeline, ColumnTransformer, SimpleImputer, OneHotEncoder, StandardScaler)
Joblib
📦 Output
Clean preprocessing pipeline created
No data leakage
Ready for model training
Saved as: preprocessor.pkl
🏁 Conclusion

This task helped in understanding how real-world machine learning pipelines are built. The focus was on proper preprocessing, separating data types, and ensuring no leakage between training and testing data. The final pipeline is reusable and production-ready.