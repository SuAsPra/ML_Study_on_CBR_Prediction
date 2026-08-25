import pandas as pd

# 🔹 Load structured dataset
df = pd.read_excel("step2_structured.xlsx", engine="openpyxl")

# 🔹 Rename columns cleanly (important)
df = df.rename(columns={
    'LL_': 'LL',
    'PL_': 'PL',
    'PI_': 'PI',
    'OMC_': 'OMC',
    'MDDg/cm3': 'MDD',
    'CBR_': 'CBR'
})

# 🔹 Select strong features only
core_cols = ['LL', 'PI', 'OMC', 'MDD', 'CBR']

df_base = df[core_cols].dropna()

print("\n===== BASE DATASET =====")
print("Shape:", df_base.shape)

# 🔹 Save dataset
df_base.to_excel("step3_base_dataset.xlsx", index=False)

print("\n✅ step3_base_dataset.xlsx created!")