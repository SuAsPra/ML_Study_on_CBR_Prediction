import pandas as pd

# 🔹 STEP 1: Load dataset with correct header (row 2)
df = pd.read_excel("FINAL DATASET.xlsx", engine="openpyxl", header=1)

print("\n===== ORIGINAL COLUMNS =====")
print(df.columns.tolist())

print("\nOriginal shape:", df.shape)

# 🔹 STEP 2: Drop completely empty columns
df = df.dropna(axis=1, how='all')

# 🔹 STEP 3: Remove useless unnamed columns
df = df.loc[:, ~df.columns.str.contains('Unnamed')]

# 🔹 STEP 4: Clean column names
df.columns = df.columns.str.strip()
df.columns = df.columns.str.replace('%', '', regex=False)
df.columns = df.columns.str.replace('\n', '', regex=False)
df.columns = df.columns.str.replace('(', '', regex=False)
df.columns = df.columns.str.replace(')', '', regex=False)
df.columns = df.columns.str.replace(' ', '_', regex=False)

print("\n===== CLEANED COLUMNS =====")
print(df.columns.tolist())

print("\nShape after cleaning:", df.shape)

# 🔹 STEP 5: Check missing values
print("\n===== MISSING % =====")
missing = df.isnull().mean() * 100
print(missing.sort_values(ascending=False))

# 🔹 STEP 6: Save clean structured dataset
df.to_excel("step2_structured.xlsx", index=False)

print("\n✅ step2_structured.xlsx created successfully!")