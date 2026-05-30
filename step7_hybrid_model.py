import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score
from xgboost import XGBRegressor

# =====================================================
# 🔹 LOAD DATA
# =====================================================

df = pd.read_excel("step2_structured.xlsx", engine="openpyxl")

# =====================================================
# 🔹 CLEAN NUMERIC COLUMNS (same as before)
# =====================================================

for col in df.columns:
    if col not in ['UCSCS', 'AASHTO', 'doi']:
        df[col] = df[col].astype(str)
        df[col] = df[col].str.extract(r'([-+]?\d*\.?\d+)')
        df[col] = pd.to_numeric(df[col], errors='coerce')

# =====================================================
# 🔹 SELECT FEATURES (INCLUDING NEW ONES)
# =====================================================

features = ['LL_', 'OMC_', 'MDDg/cm3', 'Fines', 'Sand_', 'Gravel_']
target = 'CBR_'

df_model = df[features + [target]]

# Rename for simplicity
df_model = df_model.rename(columns={
    'LL_': 'LL',
    'OMC_': 'OMC',
    'MDDg/cm3': 'MDD',
    'Sand_': 'Sand',
    'Gravel_': 'Gravel',
    'CBR_': 'CBR'
})

# =====================================================
# 🔥 PHASE 1 — NO IMPUTATION MODEL
# =====================================================

df_no_imp = df_model.dropna()

X1 = df_no_imp.drop('CBR', axis=1)
y1 = df_no_imp['CBR']

print("\nNO IMPUTATION DATA SHAPE:", X1.shape)

# Stratified bins
y_binned = pd.qcut(y1, q=5, labels=False)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

scores_no_imp = []

for train_idx, test_idx in skf.split(X1, y_binned):
    X_train, X_test = X1.iloc[train_idx], X1.iloc[test_idx]
    y_train, y_test = y1.iloc[train_idx], y1.iloc[test_idx]

    model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    scores_no_imp.append(r2_score(y_test, y_pred))

print("\nNO IMPUTATION R2:", scores_no_imp)
print("Mean:", np.mean(scores_no_imp))

# =====================================================
# 🔥 PHASE 2 — IMPUTATION MODEL
# =====================================================

imputer = SimpleImputer(strategy='median')

X_full = df_model.drop('CBR', axis=1)
y_full = df_model['CBR']

X_imputed = pd.DataFrame(imputer.fit_transform(X_full), columns=X_full.columns)

print("\nIMPUTED DATA SHAPE:", X_imputed.shape)

# Stratified bins
y_binned_full = pd.qcut(y_full, q=5, labels=False, duplicates='drop')

scores_imp = []

for train_idx, test_idx in skf.split(X_imputed, y_binned_full):
    X_train, X_test = X_imputed.iloc[train_idx], X_imputed.iloc[test_idx]
    y_train, y_test = y_full.iloc[train_idx], y_full.iloc[test_idx]

    model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    scores_imp.append(r2_score(y_test, y_pred))

print("\nIMPUTATION R2:", scores_imp)
print("Mean:", np.mean(scores_imp))

# =====================================================
# 🔹 FINAL COMPARISON
# =====================================================

print("\n==============================")
print("FINAL COMPARISON")
print("==============================")

print("No Imputation Mean R2:", np.mean(scores_no_imp))
print("Imputation Mean R2:", np.mean(scores_imp))