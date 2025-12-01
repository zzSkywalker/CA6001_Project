import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Read raw data
df = pd.read_csv("./old.csv")

# ============================================================
# 0. Remove non-informative or constant columns
# ============================================================
cols_to_drop = ['EmployeeNumber', 'EmployeeCount', 'StandardHours', 'Over18']
df = df.drop(columns=cols_to_drop)

# ============================================================
# 1. Define feature groups
# ============================================================

# Categorical features for One-Hot Encoding
# StockOptionLevel is added here for OneHotEncoder
onehot_cols = [
    "BusinessTravel",
    "Department",
    "EducationField",
    "Gender",
    "JobRole",
    "MaritalStatus",
    "OverTime",
    "StockOptionLevel"     # Newly added for One-Hot Encoding
]

# Ordinal categorical features for Label Encoding
label_cols = [
    "Education",
    "EnvironmentSatisfaction",
    "JobInvolvement",
    "JobLevel",
    "JobSatisfaction",
    "PerformanceRating",
    "RelationshipSatisfaction",
    "WorkLifeBalance"
]

# Numerical features for StandardScaler
numeric_cols = [
    "Age",
    "DailyRate",
    "DistanceFromHome",
    "HourlyRate",
    "MonthlyIncome",
    "MonthlyRate",
    "NumCompaniesWorked",
    "PercentSalaryHike",
    "TotalWorkingYears",
    "TrainingTimesLastYear",
    "YearsAtCompany",
    "YearsInCurrentRole",
    "YearsSinceLastPromotion",
    "YearsWithCurrManager"
]

# Target column
target_col = "Attrition"

# ============================================================
# 2. Apply Label Encoding to ordinal categorical features
# ============================================================
le_dict = {}
for col in label_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    le_dict[col] = le   # Save encoders for future inference

# ============================================================
# 3. Build preprocessing pipeline
# ============================================================
preprocessor = ColumnTransformer(
    transformers=[
        ("onehot", OneHotEncoder(drop="first"), onehot_cols),
        ("scaler", StandardScaler(), numeric_cols)
    ],
    remainder="passthrough"   # Keep LabelEncoded columns
)

# ============================================================
# 4. Split feature matrix and target vector
# ============================================================
X = df.drop(columns=[target_col])
y = df[target_col]

# ============================================================
# 5. Train-test split
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ============================================================
# 6. Fit transformer on training set and transform both sets
# ============================================================
X_train_transformed = preprocessor.fit_transform(X_train)
X_test_transformed = preprocessor.transform(X_test)

# ============================================================
# 7. Save processed outputs
# ============================================================
pd.DataFrame(X_train_transformed).to_csv("X_train_processed.csv", index=False)
pd.DataFrame(X_test_transformed).to_csv("X_test_processed.csv", index=False)

y_train.to_csv("y_train.csv", index=False)
y_test.to_csv("y_test.csv", index=False)

print("Preprocessing completed.")
