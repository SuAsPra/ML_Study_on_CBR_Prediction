import pandas as pd

# 🔹 Load dataset
df = pd.read_excel("soil final.xlsx", engine="openpyxl")

print("Original shape:", df.shape)

# 🔹 Calculate availability percentage
availability = 100 - (df.isnull().mean() * 100)

availability_df = pd.DataFrame({
    "Column": df.columns,
    "Available_%": availability.values
})

# 🔹 Select only HIGH availability columns (>70%)
high_cols = availability_df[availability_df["Available_%"] > 70]["Column"].tolist()

print("\nHigh availability columns (>70% data):")
print(high_cols)

# 🔹 Create new dataframe with only high availability columns
df_main = df[high_cols]

print("\nNew dataset shape (only HIGH columns):", df_main.shape)

# 🔹 Save to new Excel file
df_main.to_excel("mainsplitted.xlsx", index=False, engine="openpyxl")

print("\n✅ mainsplitted.xlsx file created successfully!")