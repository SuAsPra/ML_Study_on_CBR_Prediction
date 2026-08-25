import pandas as pd
import numpy as np
import joblib

from sklearn.impute import SimpleImputer
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
# 🔥 RESIDUAL FILTERING
# =====================================================

model_temp = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
model_temp.fit(X_imputed, y)

y_pred = model_temp.predict(X_imputed)
residuals = abs(y - y_pred)

threshold = residuals.quantile(0.90)

df_filtered = df_model[residuals < threshold]

# =====================================================
# 🔹 FINAL TRAINING
# =====================================================

X_final = df_filtered.drop('CBR', axis=1)
y_final = df_filtered['CBR']

X_final_imputed = pd.DataFrame(imputer.fit_transform(X_final), columns=X_final.columns)

final_model = XGBRegressor(
    n_estimators=100,
    learning_rate=0.1,
    random_state=42
)

final_model.fit(X_final_imputed, y_final)

# =====================================================
# 🔥 SAVE EVERYTHING
# =====================================================

joblib.dump(final_model, "final_cbr_model.pkl")
joblib.dump(imputer, "imputer.pkl")
joblib.dump(features, "features.pkl")

print("\n✅ FINAL MODEL SAVED!")
print("Files created:")
print("- final_cbr_model.pkl")
print("- imputer.pkl")
print("- features.pkl")