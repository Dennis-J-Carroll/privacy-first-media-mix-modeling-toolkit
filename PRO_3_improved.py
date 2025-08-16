import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from imblearn.combine import SMOTETomek
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('default')
sns.set_theme()

# Create directory for plots if it doesn't exist
import os
if not os.path.exists('plots'):
    os.makedirs('plots')

# Load the data
data = pd.read_csv("telco.csv")

# Feature Engineering
def create_features(df):
    # Create copy to avoid modifying original data
    df = df.copy()
    
    # Create ratio features
    df['Charge_per_Month'] = df['Total Charges'] / df['Tenure in Months']
    df['Dependents_per_Charge'] = df['Number of Dependents'] / df['Monthly Charge']
    df['CLTV_per_Tenure'] = df['CLTV'] / df['Tenure in Months']
    
    # Create interaction features
    df['Age_Tenure'] = df['Age'] * df['Tenure in Months']
    df['Charge_Score'] = df['Monthly Charge'] * df['Churn Score']
    
    return df

# Apply feature engineering
data = create_features(data)

# Define features
categorical_features = ['Gender', 'Internet Type', 'Contract', 'Payment Method']
numerical_features = ['Age', 'Tenure in Months', 'Monthly Charge', 'Total Charges', 
                     'Churn Score', 'CLTV', 'Number of Dependents',
                     'Charge_per_Month', 'Dependents_per_Charge', 'CLTV_per_Tenure',
                     'Age_Tenure', 'Charge_Score']

# Create preprocessing pipeline with polynomial features
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler()),
    ('poly', PolynomialFeatures(degree=2, include_bias=False))
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(drop='first', sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Prepare features and target
X = data[numerical_features + categorical_features]
y = (data['Churn Label'] == 'Yes').astype(int)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Apply preprocessing
preprocessor.fit(X_train)
X_train_preprocessed = preprocessor.transform(X_train)
X_test_preprocessed = preprocessor.transform(X_test)

# Apply SMOTETomek
smote_tomek = SMOTETomek(random_state=42)
X_train_balanced, y_train_balanced = smote_tomek.fit_resample(X_train_preprocessed, y_train)

# Define parameter grid for GridSearchCV
param_grid = {
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 4, 5, 6],
    'min_child_weight': [1, 3, 5],
    'gamma': [0, 0.1, 0.2],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0]
}

# Create and train model with GridSearchCV
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
model = XGBClassifier(random_state=42)
grid_search = GridSearchCV(model, param_grid, cv=cv, scoring='roc_auc', n_jobs=-1, verbose=1)
grid_search.fit(X_train_balanced, y_train_balanced)

# Print best parameters
print("\n=== BEST PARAMETERS ===")
print(grid_search.best_params_)

# Use best model for predictions
best_model = grid_search.best_estimator_
y_pred_proba = best_model.predict_proba(X_test_preprocessed)[:, 1]
y_pred = best_model.predict(X_test_preprocessed)

# Calculate and plot confusion matrix
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.savefig('plots/confusion_matrix_improved.png')
plt.close()

# Get feature names after polynomial transformation
num_features_poly = []
for feat in numerical_features:
    num_features_poly.extend([f"{feat}", f"{feat}^2"])
for i in range(len(numerical_features)):
    for j in range(i+1, len(numerical_features)):
        num_features_poly.append(f"{numerical_features[i]}_{numerical_features[j]}")

cat_features = (best_model.feature_names_in_[len(num_features_poly):]).tolist()
feature_names = num_features_poly + cat_features

# Calculate feature importance
feature_importance = pd.DataFrame({
    'feature': feature_names,
    'importance': best_model.feature_importances_
})
feature_importance = feature_importance.sort_values('importance', ascending=False)

# Plot feature importance
plt.figure(figsize=(12, 8))
sns.barplot(x='importance', y='feature', data=feature_importance.head(15))
plt.title('Top 15 Features Importance in Predicting Churn')
plt.tight_layout()
plt.savefig('plots/feature_importance_improved.png')
plt.close()

# Print comprehensive model evaluation
print("\n=== MODEL PERFORMANCE ===")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print(f"\nROC-AUC Score: {roc_auc_score(y_test, y_pred_proba):.3f}")

# Cross-validation on best model
cv_scores = cross_val_score(best_model, X_train_balanced, y_train_balanced, cv=cv, scoring='roc_auc')
print("\n=== CROSS-VALIDATION RESULTS ===")
print(f"Cross-validation ROC-AUC scores: {cv_scores}")
print(f"Average ROC-AUC: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")

# Print key insights
print("\n=== KEY INSIGHTS ===")
print("\n1. Most Important Features:")
for idx, row in feature_importance.head(5).iterrows():
    print(f"- {row['feature']}: {row['importance']:.3f} importance score")

print("\n2. Model Performance Summary:")
print(f"- Average Cross-validation ROC-AUC: {cv_scores.mean():.3f}")
print(f"- Test Set ROC-AUC: {roc_auc_score(y_test, y_pred_proba):.3f}")
print(f"- Overall Accuracy: {(y_pred == y_test).mean():.2%}")

# Save feature importance to CSV
feature_importance.to_csv('feature_importance_improved.csv', index=False)
