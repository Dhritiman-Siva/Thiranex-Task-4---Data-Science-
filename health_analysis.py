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

print("\n--- Average Values by Heart Disease ---")
print(
    df.groupby("HeartDisease")[
        ["Age", "RestingBP", "Cholesterol", "MaxHR"]
    ].mean()
)


import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(6, 4))
df["HeartDisease"].value_counts().plot(kind="bar")

plt.title("Heart Disease Distribution")
plt.xlabel("Heart Disease")
plt.ylabel("Number of Patients")

plt.show()

plt.figure(figsize=(6, 4))
sns.boxplot(
    x="HeartDisease",
    y="Age",
    data=df
)

plt.title("Age Distribution by Heart Disease")
plt.xlabel("Heart Disease")
plt.ylabel("Age")

plt.show()

plt.figure(figsize=(6, 4))
sns.boxplot(
    x="HeartDisease",
    y="Cholesterol",
    data=df
)

plt.title("Cholesterol Levels by Heart Disease")
plt.xlabel("Heart Disease")
plt.ylabel("Cholesterol")

plt.show()

plt.figure(figsize=(6, 4))
sns.boxplot(
    x="HeartDisease",
    y="MaxHR",
    data=df
)

plt.title("Maximum Heart Rate by Heart Disease")
plt.xlabel("Heart Disease")
plt.ylabel("Maximum Heart Rate")

plt.show()

plt.figure(figsize=(6, 4))
sns.countplot(
    x="ChestPainType",
    hue="HeartDisease",
    data=df
)

plt.title("Chest Pain Type and Heart Disease")
plt.xlabel("Chest Pain Type")
plt.ylabel("Number of Patients")

plt.show()

numeric_columns = [
    "Age",
    "RestingBP",
    "Cholesterol",
    "MaxHR",
    "Oldpeak",
    "HeartDisease"
]

correlation = df[numeric_columns].corr()

print("\n--- Correlation Matrix ---")
print(correlation)

plt.figure(figsize=(8, 6))
sns.heatmap(
    correlation,
    annot=True
)

plt.title("Correlation Between Health Variables")

plt.show()

# Prepare data for machine learning
X = df.drop("HeartDisease", axis=1)
y = df["HeartDisease"]

print("\n--- Features Shape ---")
print(X.shape)
print("\n--- Target Shape ---")
print(y.shape)

# Convert categorical variables into dummy/indicator variables
X = pd.get_dummies(X, drop_first=True)

print("\n--- Encoded Features Shape ---")
print(X.shape)
print("\n--- Encoded Features Preview ---")
print(X.head())

# Split dataset into training (80%) and testing (20%) sets
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\n--- Train Test Split ---")
print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)

# Train Logistic Regression model
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\n--- Model Training Complete ---")
print("Predictions preview:", y_pred[:10])
