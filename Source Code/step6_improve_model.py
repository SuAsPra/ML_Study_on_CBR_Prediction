import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor

# =====================================================
# 🔹 LOAD CLEAN DATA
# =====================================================

df = pd.read_excel("step4_cleaned_dataset.xlsx", engine="openpyxl")

print("\nDataset shape:", df.shape)

# =====================================================
# 🔥 USE BEST FEATURES (NO PI)
# =====================================================

X = df[['LL','OMC','MDD']]
y = df['CBR']

# =====================================================
# 🔹 TRAIN-TEST SPLIT
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTrain shape:", X_train.shape)
print("Test shape:", X_test.shape)

# =====================================================
# 🔥 IMPROVED XGBOOST MODEL (TUNED)
# =====================================================

xgb = XGBRegressor(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

xgb.fit(X_train, y_train)

# =====================================================
# 🔹 PREDICTION
# =====================================================

y_pred = xgb.predict(X_test)

# =====================================================
# 🔹 EVALUATION
# =====================================================

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\n===== IMPROVED XGBOOST =====")
print("RMSE:", rmse)
print("R2:", r2)

# =====================================================
# 🔹 FEATURE IMPORTANCE
# =====================================================

plt.figure()
importances = xgb.feature_importances_
plt.bar(X.columns, importances)
plt.title("Feature Importance (Improved XGBoost)")
plt.xticks(rotation=45)
plt.show()

# =====================================================
# 🔹 SAVE MODEL (OPTIONAL)
# =====================================================

import joblib
joblib.dump(xgb, "xgb_cbr_model.pkl")

print("\n✅ Model saved as xgb_cbr_model.pkl")