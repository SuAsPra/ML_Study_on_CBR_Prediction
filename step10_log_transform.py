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

# =====================================================
# 🔹 SELECT
# =====================================================

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
# 🔥 LOG TRANSFORM
# =====================================================

y_log = np.log1p(y)

# =====================================================
# 🔹 STRATIFIED CV
# =====================================================

y_binned = pd.qcut(y_log, q=5, labels=False, duplicates='drop')

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

scores = []

for train_idx, test_idx in skf.split(X_imputed, y_binned):

    X_train, X_test = X_imputed.iloc[train_idx], X_imputed.iloc[test_idx]
    y_train, y_test = y_log.iloc[train_idx], y_log.iloc[test_idx]

    model = XGBRegressor(
        n_estimators=100,
        learning_rate=0.1,
        random_state=42
    )

    model.fit(X_train, y_train)

    y_pred_log = model.predict(X_test)

    # 🔥 convert back
    y_pred = np.expm1(y_pred_log)
    y_true = np.expm1(y_test)

    scores.append(r2_score(y_true, y_pred))

# =====================================================
# 🔹 RESULTS
# =====================================================

print("\n===== STEP 10 RESULTS =====")
print("R2 scores:", scores)
print("Mean R2:", np.mean(scores))
print("Std Dev:", np.std(scores))