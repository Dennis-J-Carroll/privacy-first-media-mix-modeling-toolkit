import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('default')
sns.set()

# Load the data
data = pd.read_csv("telco.csv")

# Create directory for plots if it doesn't exist
import os
if not os.path.exists('plots'):
    os.makedirs('plots')

# 1. Churn Distribution and Demographics
plt.figure(figsize=(15, 5))

# Churn Distribution
plt.subplot(131)
sns.countplot(data=data, x='Churn Label')
plt.title('Churn Distribution')
plt.xlabel('Churn Status')
plt.ylabel('Count')

# Age Distribution by Churn
plt.subplot(132)
sns.boxplot(data=data, x='Churn Label', y='Age')
plt.title('Age Distribution by Churn Status')

# Gender Distribution by Churn
plt.subplot(133)
sns.countplot(data=data, x='Gender', hue='Churn Label')
plt.title('Gender Distribution by Churn')

plt.tight_layout()
plt.savefig('plots/churn_demographics.png')
plt.close()

# 2. Customer Value Analysis
plt.figure(figsize=(15, 5))

# CLTV by Churn
plt.subplot(131)
sns.boxplot(data=data, x='Churn Label', y='CLTV')
plt.title('Customer Lifetime Value by Churn')

# Monthly Charges by Churn
plt.subplot(132)
sns.boxplot(data=data, x='Churn Label', y='Monthly Charge')
plt.title('Monthly Charges by Churn')

# Tenure by Churn
plt.subplot(133)
sns.boxplot(data=data, x='Churn Label', y='Tenure in Months')
plt.title('Tenure by Churn Status')

plt.tight_layout()
plt.savefig('plots/value_analysis.png')
plt.close()

# 3. Service Usage Analysis
plt.figure(figsize=(15, 5))

# Internet Type Distribution
plt.subplot(131)
sns.countplot(data=data[data['Internet Type'].notna()], x='Internet Type', hue='Churn Label')
plt.title('Internet Type Distribution')
plt.xticks(rotation=45)

# Contract Type Distribution
plt.subplot(132)
sns.countplot(data=data, x='Contract', hue='Churn Label')
plt.title('Contract Type Distribution')
plt.xticks(rotation=45)

# Payment Method Distribution
plt.subplot(133)
sns.countplot(data=data, x='Payment Method', hue='Churn Label')
plt.title('Payment Method Distribution')
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig('plots/service_analysis.png')
plt.close()

# 4. Churn Reasons Analysis
plt.figure(figsize=(12, 6))
churn_reasons = data[data['Churn Label'] == 'Yes']['Churn Reason'].value_counts().head(10)
sns.barplot(x=churn_reasons.values, y=churn_reasons.index)
plt.title('Top 10 Reasons for Churn')
plt.xlabel('Count')
plt.tight_layout()
plt.savefig('plots/churn_reasons.png')
plt.close()

# 5. Correlation Analysis
numerical_cols = ['Age', 'Number of Dependents', 'Tenure in Months', 
                 'Monthly Charge', 'Total Charges', 'Churn Score', 'CLTV']
correlation_matrix = data[numerical_cols].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Matrix of Numerical Variables')
plt.tight_layout()
plt.savefig('plots/correlation_matrix.png')
plt.close()

# 6. Predictive Modeling
# Prepare features for modeling
features = ['Age', 'Tenure in Months', 'Monthly Charge', 'Total Charges', 
           'Churn Score', 'CLTV', 'Number of Dependents']
X = data[features]
y = (data['Churn Label'] == 'Yes').astype(int)

# Split and scale the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)

# Feature Importance Plot
feature_importance = pd.DataFrame({
    'feature': features,
    'importance': rf_model.feature_importances_
})
feature_importance = feature_importance.sort_values('importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='importance', y='feature', data=feature_importance)
plt.title('Feature Importance in Predicting Churn')
plt.tight_layout()
plt.savefig('plots/feature_importance.png')
plt.close()

# Print model performance
y_pred = rf_model.predict(X_test_scaled)
print("\n=== MODEL PERFORMANCE ===")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Print key insights
print("\n=== KEY INSIGHTS ===")
print("\n1. Churn Rate Analysis:")
print(f"- Overall churn rate: {(data['Churn Label'] == 'Yes').mean():.2%}")
print(f"- Average age of churned customers: {data[data['Churn Label'] == 'Yes']['Age'].mean():.1f} years")
print(f"- Average CLTV of churned customers: ${data[data['Churn Label'] == 'Yes']['CLTV'].mean():.2f}")

print("\n2. Top Churn Predictors:")
for idx, row in feature_importance.head(3).iterrows():
    print(f"- {row['feature']}: {row['importance']:.3f} importance score")

print("\n3. Service Analysis:")
print("- Internet Type impact on churn:")
internet_churn = data.groupby('Internet Type')['Churn Label'].apply(
    lambda x: (x == 'Yes').mean()
).sort_values(ascending=False)
for internet_type, churn_rate in internet_churn.items():
    if pd.notna(internet_type):
        print(f"  {internet_type}: {churn_rate:.2%} churn rate")

print("\nPlots have been saved in the 'plots' directory:")
print("1. churn_demographics.png - Demographics analysis")
print("2. value_analysis.png - Customer value analysis")
print("3. service_analysis.png - Service usage patterns")
print("4. churn_reasons.png - Top reasons for churn")
print("5. correlation_matrix.png - Correlation analysis")
print("6. feature_importance.png - Predictive modeling results")
