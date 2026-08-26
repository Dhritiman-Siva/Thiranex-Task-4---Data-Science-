# 🫀 Heart Disease EDA & Machine Learning Classification

An end-to-end Data Science & Machine Learning project analyzing the **Cleveland Heart Disease Dataset**. This repository includes comprehensive Exploratory Data Analysis (EDA), statistical summary statistics, correlation matrices, data visualizations using Seaborn and Matplotlib, feature engineering, and a **Logistic Regression classification model** achieving **~88.5% accuracy** in predicting heart disease presence.

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Dataset Architecture](#-dataset-architecture)
- [Project Structure](#-project-structure)
- [Installation & Setup](#-installation--setup)
- [Usage](#-usage)
- [Model Performance & Evaluation](#-model-performance--evaluation)
- [Visualizations Generated](#-visualizations-generated)
- [Technologies Used](#-technologies-used)
- [License](#-license)

---

## 🩺 Overview

Heart disease is one of the leading causes of global mortality. Early diagnosis using clinical features—such as blood pressure, cholesterol, chest pain type, and maximum heart rate—can significantly improve patient outcomes.

This project performs:
1. **Data Ingestion & Cleaning**: Handles missing values (imputed via statistical mode) and converts non-binary targets into a clean binary classification target (`0` = No Disease, `1` = Disease).
2. **Exploratory Data Analysis (EDA)**: Computes statistical summaries, mean value comparisons across patient groups, and correlation coefficients.
3. **Data Visualization**: Generates distribution plots, boxplots, countplots, and heatmap representations.
4. **Machine Learning Pipeline**: Trains a Logistic Regression model on an 80/20 train-test split and evaluates model predictive power using accuracy, classification metrics, confusion matrix, and feature coefficient influence analysis.

---

## ✨ Key Features

- 🧹 **Robust Data Cleaning**: Fills missing values (`CA`, `Thal`) using mode imputation and removes duplicates.
- 📊 **Statistical Insights**: Calculates mean age, blood pressure, cholesterol, and maximum heart rate stratified by heart disease diagnosis.
- 📈 **8 Interactive/Exported Charts**: Saves high-resolution chart images (`.png`) for reporting.
- 🤖 **Predictive Machine Learning Model**: Scikit-Learn Logistic Regression model trained to predict heart disease presence.
- 🎯 **Feature Importance Analysis**: Measures the mathematical coefficient impact of each clinical attribute.

---

## 📋 Dataset Architecture

The analysis is conducted on the processed **Cleveland Dataset** with the following 14 clinical features:

| Feature Name | Description | Type |
| :--- | :--- | :--- |
| `Age` | Patient age in years | Continuous |
| `Sex` | Gender (1 = Male, 0 = Female) | Binary |
| `ChestPainType` | Chest pain type (1: Typical angina, 2: Atypical angina, 3: Non-anginal, 4: Asymptomatic) | Categorical |
| `RestingBP` | Resting blood pressure (in mm Hg on admission) | Continuous |
| `Cholesterol` | Serum cholesterol level in mg/dL | Continuous |
| `FastingBS` | Fasting blood sugar > 120 mg/dL (1 = True, 0 = False) | Binary |
| `RestingECG` | Resting electrocardiographic results (0, 1, 2) | Categorical |
| `MaxHR` | Maximum heart rate achieved during exercise | Continuous |
| `ExerciseAngina` | Exercise-induced angina (1 = Yes, 0 = No) | Binary |
| `Oldpeak` | ST depression induced by exercise relative to rest | Continuous |
| `ST_Slope` | Slope of peak exercise ST segment | Categorical |
| `CA` | Number of major vessels (0-3) colored by fluoroscopy | Discrete |
| `Thal` | Thalassemia (3 = Normal, 6 = Fixed defect, 7 = Reversable defect) | Categorical |
| `HeartDisease` | Diagnosis of heart disease (**Target**: 0 = Negative, 1 = Positive) | Binary |

---

## 📁 Project Structure

```
Task 4/
│
├── health_analysis.py            # Main end-to-end Python analysis & ML pipeline
├── processed.cleveland.data      # Raw dataset file
├── heart_disease.csv             # Cleaned CSV output dataset
├── requirements.txt              # Project Python package dependencies
├── README.md                     # Project documentation
│
└── Exported Visualizations/
    ├── heart_disease_distribution.png
    ├── age_distribution.png
    ├── cholesterol_distribution.png
    ├── max_hr_distribution.png
    ├── chest_pain_distribution.png
    ├── correlation_heatmap.png
    ├── confusion_matrix.png
    └── feature_influence.png
```

---

## ⚙️ Installation & Setup

### Prerequisites
- Python 3.8+ installed on your system.

### 1. Clone or Download Repository
Navigate to your project directory:
```bash
cd "Task 4"
```

### 2. Create a Virtual Environment (Recommended)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
Install all required libraries (`pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`):
```bash
pip install -r requirements.txt
```

---

## 🚀 Usage

To run the complete data pipeline, generate statistical summaries, display interactive plots, and train the Machine Learning model, execute:

```bash
python health_analysis.py
```

---

## 📊 Model Performance & Evaluation

### Model Evaluation Summary

| Metric | Score |
| :--- | :---: |
| **Model** | Logistic Regression (`max_iter=1000`) |
| **Train/Test Split** | 80% Train (242 samples) / 20% Test (61 samples) |
| **Accuracy Score** | **88.52%** |
| **Macro Average F1-Score** | **0.88** |

### Classification Report

```
              precision    recall  f1-score   support

           0       0.89      0.86      0.88        29
           1       0.88      0.91      0.89        32

    accuracy                           0.89        61
   macro avg       0.89      0.88      0.88        61
weighted avg       0.89      0.89      0.89        61
```

### Confusion Matrix Breakdown

| | Predicted: No Heart Disease (`0`) | Predicted: Heart Disease (`1`) |
| :--- | :---: | :---: |
| **Actual: No Heart Disease (`0`)** | **25** *(True Negative)* | **4** *(False Positive)* |
| **Actual: Heart Disease (`1`)** | **3** *(False Negative)* | **29** *(True Positive)* |

---

## 🖼️ Visualizations Generated

1. `heart_disease_distribution.png`: Bar chart showing patient count per heart disease class.
2. `age_distribution.png`: Boxplot comparing patient age distributions.
3. `cholesterol_distribution.png`: Boxplot analyzing serum cholesterol levels across diagnosis groups.
4. `max_hr_distribution.png`: Boxplot demonstrating the inverse relationship between maximum heart rate and heart disease risk.
5. `chest_pain_distribution.png`: Grouped countplot by Chest Pain Type.
6. `correlation_heatmap.png`: Annotated correlation matrix heat map.
7. `confusion_matrix.png`: Heatmap visualization of model prediction confusion matrix.
8. `feature_influence.png`: Horizontal bar chart ranking feature coefficients by magnitude and direction.

---

## 🛠️ Technologies Used

- **Language**: Python 3
- **Data Manipulation**: `pandas`, `numpy`
- **Data Visualization**: `matplotlib`, `seaborn`
- **Machine Learning**: `scikit-learn` (`train_test_split`, `LogisticRegression`, `accuracy_score`, `classification_report`, `confusion_matrix`)

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).
