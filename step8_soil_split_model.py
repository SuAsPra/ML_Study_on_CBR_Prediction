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
# 🔹 CLEAN NUMERIC COLUMNS
# =====================================================

for col in df.columns:
    if col not in ['UCSCS', 'AASHTO', 'doi']:
        df[col] = df[col].astype(str)
        df[col] = df[col].str.extract(r'([-+]?\d*\.?\d+)')
        df[col] = pd.to_numeric(df[col], errors='coerce')

# =====================================================
# 🔹 RENAME COLUMNS
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
# 🔥 IMPROVED SOIL CLASSIFICATION
# =====================================================

def classify_soil(x):
    x = str(x).lower()
    
    if 'clay' in x or 'silt' in x:
        return 'Cohesive'
    
    elif 'sand' in x or 'gravel' in x:
        return 'Granular'
    
    else:
        return 'Mixed'

df['Soil_Type'] = df['UCSCS'].apply(classify_soil)

print("\n===== SOIL TYPE DISTRIBUTION =====")
print(df['Soil_Type'].value_counts())

# =====================================================
# 🔹 FEATURES
# =====================================================

features = ['LL', 'OMC', 'MDD', 'Fines', 'Sand', 'Gravel']
target = 'CBR'

# =====================================================
# 🔥 MODEL FOR EACH SOIL TYPE
# =====================================================

soil_types = ['Cohesive', 'Granular']

for soil in soil_types:
    
    print("\n==============================")
    print(f"SOIL TYPE: {soil}")
    print("==============================")
    
    df_soil = df[df['Soil_Type'] == soil]
    
    if len(df_soil) < 100:
        print("❌ Not enough data, skipping...")
        continue
    
    df_model = df_soil[features + [target]]
    
    # =====================================================
    # 🔹 IMPUTATION
    # =====================================================
    
    imputer = SimpleImputer(strategy='median')
    
    X = df_model.drop('CBR', axis=1)
    y = df_model['CBR']
    
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
    
    # =====================================================
    # 🔹 STRATIFIED BINNING
    # =====================================================
    
    y_binned = pd.qcut(y, q=5, labels=False, duplicates='drop')
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    scores = []
    
    # =====================================================
    # 🔹 TRAIN MODEL
    # =====================================================
    
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
        
        r2 = r2_score(y_test, y_pred)
        scores.append(r2)
    
    # =====================================================
    # 🔹 RESULTS
    # =====================================================
    
    print("R2 scores:", scores)
    print("Mean R2:", np.mean(scores))
    print("Std Dev:", np.std(scores))