import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor


# =====================================================
# LOAD FINAL MODEL ARTIFACTS
# =====================================================

model = joblib.load("final_cbr_model_with_pl.pkl")
imputer = joblib.load("imputer_with_pl.pkl")
features = joblib.load("features_with_pl.pkl")


# =====================================================
# LOAD DATA
# =====================================================

df = pd.read_excel("step2_structured.xlsx", engine="openpyxl")


# =====================================================
# CLEAN NUMERIC COLUMNS
# =====================================================

for col in df.columns:
    if col not in ["UCSCS", "AASHTO", "doi"]:
        df[col] = df[col].astype(str)
        df[col] = df[col].str.extract(r"([-+]?\d*\.?\d+)")
        df[col] = pd.to_numeric(df[col], errors="coerce")


# =====================================================
# RENAME COLUMNS
# =====================================================

df = df.rename(
    columns={
        "LL_": "LL",
        "PL_": "PL",
        "OMC_": "OMC",
        "MDDg/cm3": "MDD",
        "Sand_": "Sand",
        "Gravel_": "Gravel",
        "CBR_": "CBR",
    }
)


# =====================================================
# COHESIVE SOIL ONLY
# =====================================================


def classify_soil(value):
    text = str(value).lower()
    if "clay" in text or "silt" in text:
        return "Cohesive"
    return "Other"


df["Soil_Type"] = df["UCSCS"].apply(classify_soil)
df = df[df["Soil_Type"] == "Cohesive"].copy()


# =====================================================
# FEATURE ENGINEERING
# =====================================================

df["OMC_MDD"] = df["OMC"] / df["MDD"]
df["LL_MDD"] = df["LL"] / df["MDD"]
df["LL_OMC"] = df["LL"] * df["OMC"]

target = "CBR"
df_model = df[features + [target]].copy()


# =====================================================
# RECREATE RESIDUAL FILTERING DATASET
# =====================================================

X = df_model.drop(target, axis=1)
y = df_model[target]

temp_imputer = SimpleImputer(strategy="median")
X_imputed_temp = pd.DataFrame(temp_imputer.fit_transform(X), columns=X.columns)

temp_model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
temp_model.fit(X_imputed_temp, y)

initial_predictions = temp_model.predict(X_imputed_temp)
residuals = abs(y - initial_predictions)
threshold = residuals.quantile(0.90)

df_filtered = df_model[residuals < threshold].copy()


# =====================================================
# PREPARE FINAL MODEL INPUT
# =====================================================

X_filtered = df_filtered.drop(target, axis=1)
X_final = pd.DataFrame(imputer.transform(X_filtered), columns=features)


# =====================================================
# SHAP SUMMARY PLOT
# =====================================================

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_final)

plt.figure()
shap.summary_plot(
    shap_values,
    X_final,
    feature_names=features,
    show=False,
)

plt.tight_layout()
plt.savefig("Figure_14_SHAP_summary.png", dpi=300, bbox_inches="tight")
plt.close()


# =====================================================
# SHAP IMPORTANCE TABLE
# =====================================================

mean_abs_shap = np.abs(shap_values).mean(axis=0)

shap_importance = (
    pd.DataFrame(
        {
            "Feature": features,
            "Mean_ABS_SHAP": mean_abs_shap,
        }
    )
    .sort_values(by="Mean_ABS_SHAP", ascending=False)
    .reset_index(drop=True)
)

shap_importance.to_csv("Figure_14_SHAP_importance.csv", index=False)


# =====================================================
# RESULTS
# =====================================================

print("\n===== STEP 15 SHAP SUMMARY =====")
print("Original cohesive samples:", len(df_model))
print("Residual-filtered samples:", len(df_filtered))
print("Residual threshold:", round(float(threshold), 4))
print("\nSHAP files created:")
print("- Figure_14_SHAP_summary.png")
print("- Figure_14_SHAP_importance.csv")
print("\nMean absolute SHAP importance:")
print(shap_importance.to_string(index=False))
