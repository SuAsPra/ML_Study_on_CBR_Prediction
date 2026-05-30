import pandas as pd

# 🔹 Load your dataset
df = pd.read_excel("raw data soil.xlsx")   # change name if needed

# 🔹 Basic info
print("Shape of dataset:", df.shape)
print("\nColumn names:\n", df.columns.tolist())

# 🔹 Calculate missing percentage
missing_percent = df.isnull().mean() * 100

# 🔹 Create classification dataframe
availability_df = pd.DataFrame({
    "Column": df.columns,
    "Missing_%": missing_percent.values,
    "Available_%": 100 - missing_percent.values
})

# 🔹 Classify columns
def classify(avail):
    if avail > 70:
        return "HIGH (>70%)"
    elif avail >= 30:
        return "MEDIUM (30-70%)"
    else:
        return "LOW (<30%)"

availability_df["Category"] = availability_df["Available_%"].apply(classify)

# 🔹 Sort by availability
availability_df = availability_df.sort_values(by="Available_%", ascending=False)

print("\n=== Column Availability Classification ===\n")
print(availability_df)

# 🔹 Optional: Separate lists
high_cols = availability_df[availability_df["Category"] == "HIGH (>70%)"]["Column"].tolist()
medium_cols = availability_df[availability_df["Category"] == "MEDIUM (30-70%)"]["Column"].tolist()
low_cols = availability_df[availability_df["Category"] == "LOW (<30%)"]["Column"].tolist()

print("\nHigh Availability Columns (>70% data):")
print(high_cols)

print("\nMedium Availability Columns (30-70% data):")
print(medium_cols)

print("\nLow Availability Columns (<30% data):")
print(low_cols)