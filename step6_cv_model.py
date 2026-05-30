import pandas as pd
import numpy as np

from sklearn.model_selection import cross_val_score
from xgboost import XGBRegressor

# 🔹 Load dataset
df = pd.read_excel("step4_cleaned_dataset.xlsx", engine="openpyxl")

# 🔹 Best features (no PI)
X = df[['LL','OMC','MDD']]
y = df['CBR']

# 🔹 Model (simple, stable)
xgb = XGBRegressor(
    n_estimators=100,
    learning_rate=0.1,
    random_state=42
)

# 🔹 5-Fold Cross Validation
scores = cross_val_score(xgb, X, y, cv=5, scoring='r2')

print("\n===== CROSS VALIDATION RESULTS =====")
print("R2 scores:", scores)
print("Mean R2:", np.mean(scores))
print("Std Dev:", np.std(scores))