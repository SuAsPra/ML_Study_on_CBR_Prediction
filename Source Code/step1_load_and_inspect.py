import pandas as pd

# 🔹 Load dataset
df = pd.read_excel("FINAL DATASET.xlsx", engine="openpyxl")

# 🔹 Basic info
print("\n===== BASIC INFO =====")
print("Shape:", df.shape)

print("\n===== COLUMN NAMES =====")
print(df.columns.tolist())

# 🔹 Clean column names (VERY IMPORTANT)
df.columns = df.columns.str.strip()
df.columns = df.columns.str.replace('%', '', regex=False)
df.columns = df.columns.str.replace('\n', '', regex=False)
df.columns = df.columns.str.replace('(', '', regex=False)
df.columns = df.columns.str.replace(')', '', regex=False)
df.columns = df.columns.str.replace(' ', '_', regex=False)

print("\n===== CLEANED COLUMN NAMES =====")
print(df.columns.tolist())

# 🔹 Check missing values
print("\n===== MISSING VALUES (%) =====")
missing = df.isnull().mean() * 100
print(missing.sort_values(ascending=False))

# 🔹 Save cleaned version (no data removed yet)
df.to_excel("step1_cleaned.xlsx", index=False)

print("\n✅ Step 1 completed → step1_cleaned.xlsx created")