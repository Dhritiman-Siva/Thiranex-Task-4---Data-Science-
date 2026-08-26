import pandas as pd

# Column names for the Cleveland dataset
columns = [
    "Age",
    "Sex",
    "ChestPainType",
    "RestingBP",
    "Cholesterol",
    "FastingBS",
    "RestingECG",
    "MaxHR",
    "ExerciseAngina",
    "Oldpeak",
    "ST_Slope",
    "CA",
    "Thal",
    "HeartDisease"
]

# Load raw dataset
df = pd.read_csv(
    "processed.cleveland.data",
    header=None,
    names=columns,
    na_values="?"
)

# Save as CSV
df.to_csv("heart_disease.csv", index=False)
print("CSV file created successfully.")

# Load the CSV
df = pd.read_csv("heart_disease.csv")

print("\n--- First 5 Rows ---")
print(df.head())

print("\n--- Dataset Information ---")
df.info()

print("\n--- Dataset Shape ---")
print(df.shape)

# Check missing values before cleaning
print("\n--- Missing Values Before Cleaning ---")
print(df.isnull().sum())

print("\n--- Duplicate Rows ---")
print(df.duplicated().sum())

# Data Cleaning: Fill missing values using mode and remove duplicates
df["CA"] = df["CA"].fillna(df["CA"].mode()[0])
df["Thal"] = df["Thal"].fillna(df["Thal"].mode()[0])

# Remove duplicate rows
df = df.drop_duplicates()

print("\n--- Missing Values After Cleaning ---")
print(df.isnull().sum())

print("\n--- Dataset Shape After Cleaning ---")
print(df.shape)

print("\n--- Original Heart Disease Values ---")
print(df["HeartDisease"].value_counts().sort_index())

# Convert to binary target
df["HeartDisease"] = (df["HeartDisease"] > 0).astype(int)

print("\n--- Binary Heart Disease Values ---")
print(df["HeartDisease"].value_counts())

print("\n--- Statistical Summary ---")
print(df.describe())

print("\n--- Average Age ---")
print(df["Age"].mean())

print("\n--- Average Cholesterol ---")
print(df["Cholesterol"].mean())

print("\n--- Average Resting Blood Pressure ---")
print(df["RestingBP"].mean())

print("\n--- Average Maximum Heart Rate ---")
print(df["MaxHR"].mean())

