import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Load the data
print("Loading and analyzing the IBM Telco Customer Churn dataset...\n")
data = pd.read_csv("telco.csv")

# 1. DATA OVERVIEW AND QUALITY CHECKS
print("\n=== DATA OVERVIEW ===")
print("\nDataset Shape:", data.shape)
print("\nColumns:", data.columns.tolist())
print("\nSample Data (first 5 rows):\n", data.head())
print("\nData Types:\n", data.dtypes)
print("\nMissing Values:\n", data.isnull().sum()[data.isnull().sum() > 0])
print("\nBasic Statistics for Numerical Columns:\n", data.describe())

# Basic Analysis
print("\n=== BASIC ANALYSIS ===")
print(f"\n1. Total number of customers: {len(data)}")
print(f"2. Churn Rate: {(data['Churn Label'] == 'Yes').mean():.2%}")
print(f"3. Average Customer Age: {data['Age'].mean():.1f} years")
print(f"4. Average CLTV: ${data['CLTV'].mean():.2f}")

# Customer Demographics
print("\n=== CUSTOMER DEMOGRAPHICS ===")
print("\nGender Distribution:")
print(data['Gender'].value_counts(normalize=True).apply(lambda x: f"{x:.2%}"))

print("\nAge Groups:")
age_bins = [0, 30, 45, 60, 100]
age_labels = ['Young Adult (0-30)', 'Middle Age (31-45)', 'Senior (46-60)', 'Elderly (60+)']
data['Age_Group'] = pd.cut(data['Age'], bins=age_bins, labels=age_labels)
print(data['Age_Group'].value_counts(normalize=True).apply(lambda x: f"{x:.2%}"))

# Churn Analysis
print("\n=== CHURN ANALYSIS ===")
print("\nChurn by Gender:")
churn_by_gender = pd.crosstab(data['Gender'], data['Churn Label'], normalize='index')
print(churn_by_gender.apply(lambda x: x.apply(lambda y: f"{y:.2%}")))

print("\nTop 5 Churn Reasons:")
churn_reasons = data[data['Churn Label'] == 'Yes']['Churn Reason'].value_counts().head()
for reason, count in churn_reasons.items():
    print(f"- {reason}: {count} customers ({count/len(data[data['Churn Label'] == 'Yes']):.2%} of churned customers)")

# Customer Value Analysis
print("\n=== CUSTOMER VALUE ANALYSIS ===")
print(f"\nAverage CLTV by Churn Status:")
print(data.groupby('Churn Label')['CLTV'].mean().apply(lambda x: f"${x:.2f}"))

print("\nChurn Score Statistics:")
print(data['Churn Score'].describe().apply(lambda x: f"{x:.2f}"))
