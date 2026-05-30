import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 🔹 Load dataset
df = pd.read_excel("step3_base_dataset.xlsx", engine="openpyxl")

print("\n===== ORIGINAL SHAPE =====")
print(df.shape)

# =====================================================
# 🔥 STEP 1: CLEAN NUMERIC DATA (IMPORTANT)
# =====================================================

for col in df.columns:
    df[col] = df[col].astype(str)  # convert everything to string
    df[col] = df[col].str.extract(r'([-+]?\d*\.?\d+)')  # extract numbers only
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Drop rows with invalid values
df = df.dropna()

print("\nAfter numeric cleaning:", df.shape)

# =====================================================
# 🔥 STEP 2: REMOVE UNREALISTIC VALUES
# =====================================================

# MDD should be ~1–3 g/cm³ → remove wrong units like 2000
df = df[(df['MDD'] > 0.5) & (df['MDD'] < 5)]

# CBR should be positive and reasonable
df = df[(df['CBR'] > 0) & (df['CBR'] < 100)]

print("\nAfter removing unrealistic values:", df.shape)

# =====================================================
# 🔹 STEP 3: BASIC STATS
# =====================================================

print("\n===== DATASET INFO =====")
print(df.describe())

# =====================================================
# 🔹 STEP 4: CBR DISTRIBUTION
# =====================================================

plt.figure()
df['CBR'].hist(bins=30)
plt.title("CBR Distribution")
plt.xlabel("CBR")
plt.ylabel("Frequency")
plt.show()

# =====================================================
# 🔹 STEP 5: CORRELATION MATRIX
# =====================================================

plt.figure()
corr = df.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()

print("\n===== CORRELATION WITH CBR =====")
print(corr['CBR'].sort_values(ascending=False))

# =====================================================
# 🔹 STEP 6: BOXPLOTS (OUTLIERS)
# =====================================================

for col in df.columns:
    plt.figure()
    sns.boxplot(x=df[col])
    plt.title(f"Boxplot of {col}")
    plt.show()

# =====================================================
# 🔹 STEP 7: SAVE CLEANED DATASET
# =====================================================

df.to_excel("step4_cleaned_dataset.xlsx", index=False)

print("\n✅ step4_cleaned_dataset.xlsx created!")