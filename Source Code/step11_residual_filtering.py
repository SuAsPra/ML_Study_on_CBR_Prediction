import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score
from xgboost import XGBRegressor

# =====================================================
# 🔹 LOAD DATA
# =====================================================

df = pd.read_excel("step2_structured.xlsx", engine="openpyxl")

# =====================================================
# 🔹 CLEAN NUMERIC
# =====================================================

for col in df.columns:
    if col not in ['UCSCS', 'AASHTO', 'doi']:
        df[col] = df[col].astype(str)
        df[col] = df[col].str.extract(r'([-+]?\d*\.?\d+)')
        df[col] = pd.to_numeric(df[col], errors='coerce')

# =====================================================
# 🔹 RENAME
# =====================================================

df = df.rename(columns={
    'LL_': 'LL',
    'OMC_': 'OMC',
    'MDDg/cm3': 'MDD',
    'Sand_': 'Sand',
    'Gravel_': 'Gravel',
    'CBR_': 'CBR'
})

# =====================================================
# 🔥 COHESIVE ONLY
# =====================================================

def classify_soil(x):
    x = str(x).lower()
    if 'clay' in x or 'silt' in x:
        return 'Cohesive'
    else:
        return 'Other'

df['Soil_Type'] = df['UCSCS'].apply(classify_soil)
df = df[df['Soil_Type'] == 'Cohesive']

# =====================================================
# 🔥 FEATURE ENGINEERING
# =====================================================

df['OMC_MDD'] = df['OMC'] / df['MDD']
df['LL_MDD'] = df['LL'] / df['MDD']
df['LL_OMC'] = df['LL'] * df['OMC']

features = ['LL', 'OMC', 'MDD', 'Fines', 'Sand', 'Gravel',
            'OMC_MDD', 'LL_MDD', 'LL_OMC']

target = 'CBR'

df_model = df[features + [target]]

# =====================================================
# 🔹 IMPUTATION
# =====================================================

imputer = SimpleImputer(strategy='median')

X = df_model.drop('CBR', axis=1)
y = df_model['CBR']

X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# =====================================================
# 🔥 STEP 1: TRAIN INITIAL MODEL
# =====================================================

model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
model.fit(X_imputed, y)

y_pred = model.predict(X_imputed)

# =====================================================
# 🔥 STEP 2: COMPUTE RESIDUALS
# =====================================================

residuals = abs(y - y_pred)

# Remove worst 10%
threshold = residuals.quantile(0.90)

df_filtered = df_model[residuals < threshold]

print("\nOriginal size:", len(df_model))
print("Filtered size:", len(df_filtered))

# =====================================================
# 🔥 STEP 3: RETRAIN WITH CLEAN DATA
# =====================================================

X = df_filtered.drop('CBR', axis=1)
y = df_filtered['CBR']

X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# Stratified bins
y_binned = pd.qcut(y, q=5, labels=False, duplicates='drop')

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

scores = []

for train_idx, test_idx in skf.split(X_imputed, y_binned):

    X_train, X_test = X_imputed.iloc[train_idx], X_imputed.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    model = XGBRegressor(
        n_estimators=100,
        learning_rate=0.1,
        random_state=42
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    scores.append(r2_score(y_test, y_pred))

# =====================================================
# 🔹 RESULTS
# =====================================================

print("\n===== STEP 11 RESULTS =====")
print("R2 scores:", scores)
print("Mean R2:", np.mean(scores))
print("Std Dev:", np.std(scores))