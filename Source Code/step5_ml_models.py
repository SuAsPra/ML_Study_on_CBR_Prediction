import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score

# 🔹 Load cleaned dataset
df = pd.read_excel("step4_cleaned_dataset.xlsx", engine="openpyxl")

# 🔹 Split features & target
X = df.drop('CBR', axis=1)
y = df['CBR']

# 🔹 Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTrain shape:", X_train.shape)
print("Test shape:", X_test.shape)

# =====================================================
# 🔹 MODEL 1: LINEAR REGRESSION
# =====================================================

lr = LinearRegression()
lr.fit(X_train, y_train)

y_pred_lr = lr.predict(X_test)

print("\n===== LINEAR REGRESSION =====")
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_lr)))
print("R2:", r2_score(y_test, y_pred_lr))

# =====================================================
# 🔹 MODEL 2: RANDOM FOREST
# =====================================================

rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)

print("\n===== RANDOM FOREST =====")
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_rf)))
print("R2:", r2_score(y_test, y_pred_rf))

# =====================================================
# 🔹 MODEL 3: XGBOOST
# =====================================================

xgb = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
xgb.fit(X_train, y_train)

y_pred_xgb = xgb.predict(X_test)

print("\n===== XGBOOST =====")
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_xgb)))
print("R2:", r2_score(y_test, y_pred_xgb))

# =====================================================
# 🔹 FEATURE IMPORTANCE (Random Forest)
# =====================================================

import matplotlib.pyplot as plt

importances = rf.feature_importances_
plt.bar(X.columns, importances)
plt.title("Feature Importance (Random Forest)")
plt.xticks(rotation=45)
plt.show()