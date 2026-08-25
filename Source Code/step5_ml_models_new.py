import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score

# =====================================================
# 🔹 LOAD DATA
# =====================================================

df = pd.read_excel("step4_cleaned_dataset.xlsx", engine="openpyxl")

print("\nDataset shape:", df.shape)

# =====================================================
# 🔹 DEFINE TARGET
# =====================================================

y = df['CBR']

# =====================================================
# 🔥 MODEL SET 1 — WITH PI
# =====================================================

print("\n==============================")
print("MODEL WITH PI")
print("==============================")

X1 = df[['LL','PI','OMC','MDD']]

X_train, X_test, y_train, y_test = train_test_split(
    X1, y, test_size=0.2, random_state=42
)

# Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

print("\nLinear Regression (WITH PI)")
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_lr)))
print("R2:", r2_score(y_test, y_pred_lr))

# Random Forest
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

print("\nRandom Forest (WITH PI)")
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_rf)))
print("R2:", r2_score(y_test, y_pred_rf))

# XGBoost
xgb = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
xgb.fit(X_train, y_train)
y_pred_xgb = xgb.predict(X_test)

print("\nXGBoost (WITH PI)")
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_xgb)))
print("R2:", r2_score(y_test, y_pred_xgb))

# Feature Importance (RF)
plt.figure()
importances = rf.feature_importances_
plt.bar(X1.columns, importances)
plt.title("Feature Importance (WITH PI)")
plt.xticks(rotation=45)
plt.show()

# =====================================================
# 🔥 MODEL SET 2 — WITHOUT PI
# =====================================================

print("\n==============================")
print("MODEL WITHOUT PI")
print("==============================")

X2 = df[['LL','OMC','MDD']]

X_train, X_test, y_train, y_test = train_test_split(
    X2, y, test_size=0.2, random_state=42
)

# Linear Regression
lr2 = LinearRegression()
lr2.fit(X_train, y_train)
y_pred_lr2 = lr2.predict(X_test)

print("\nLinear Regression (WITHOUT PI)")
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_lr2)))
print("R2:", r2_score(y_test, y_pred_lr2))

# Random Forest
rf2 = RandomForestRegressor(n_estimators=100, random_state=42)
rf2.fit(X_train, y_train)
y_pred_rf2 = rf2.predict(X_test)

print("\nRandom Forest (WITHOUT PI)")
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_rf2)))
print("R2:", r2_score(y_test, y_pred_rf2))

# XGBoost
xgb2 = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
xgb2.fit(X_train, y_train)
y_pred_xgb2 = xgb2.predict(X_test)

print("\nXGBoost (WITHOUT PI)")
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_xgb2)))
print("R2:", r2_score(y_test, y_pred_xgb2))

# =====================================================
# 🔹 FINAL COMPARISON SUMMARY
# =====================================================

print("\n==============================")
print("FINAL COMPARISON")
print("==============================")

print("\nXGBoost WITH PI R2:", r2_score(y_test, y_pred_xgb))
print("XGBoost WITHOUT PI R2:", r2_score(y_test, y_pred_xgb2))