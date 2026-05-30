import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold
from xgboost import XGBRegressor
from sklearn.metrics import r2_score

# 🔹 Load data
df = pd.read_excel("step4_cleaned_dataset.xlsx", engine="openpyxl")

X = df[['LL','OMC','MDD']]
y = df['CBR']

# =====================================================
# 🔥 CREATE BINS (IMPORTANT)
# =====================================================

y_binned = pd.qcut(y, q=5, labels=False)

# =====================================================
# 🔹 STRATIFIED K-FOLD
# =====================================================

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

r2_scores = []

for train_idx, test_idx in skf.split(X, y_binned):
    
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)

    r2_scores.append(r2)

# =====================================================
# 🔹 RESULTS
# =====================================================

print("\n===== STRATIFIED CV RESULTS =====")
print("R2 scores:", r2_scores)
print("Mean R2:", np.mean(r2_scores))
print("Std Dev:", np.std(r2_scores))