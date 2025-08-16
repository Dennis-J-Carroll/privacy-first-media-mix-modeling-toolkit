import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline  # type: ignore
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from imblearn.over_sampling import SMOTE
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

# Define features
categorical_features = ['Gender', 'Internet Type', 'Contract', 'Payment Method']
numerical_features = ['Age', 'Tenure in Months', 'Monthly Charge', 'Total Charges', 
                     'Churn Score', 'CLTV', 'Number of Dependents']

# Create preprocessing pipeline
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(drop='first', sparse=False))
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

# Create modeling pipeline
model = Pipeline([
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=42)),
    ('classifier', XGBClassifier(
        learning_rate=0.01,
        n_estimators=200,
        max_depth=5,
        min_child_weight=1,
        gamma=0,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    ))
])

# Cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc')
print("\n=== CROSS-VALIDATION RESULTS ===")
print(f"Cross-validation ROC-AUC scores: {cv_scores}")
print(f"Average ROC-AUC: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")

# Train final model
model.fit(X_train, y_train)

# Make predictions
y_pred_proba = model.predict_proba(X_test)[:, 1]
y_pred = model.predict(X_test)

# Calculate and plot confusion matrix
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.savefig('plots/confusion_matrix2.png')
plt.close()

# Calculate feature importance
feature_names = (numerical_features + 
                model.named_steps['preprocessor']
                .named_transformers_['cat']
                .named_steps['onehot']
                .get_feature_names(categorical_features).tolist())

feature_importance = pd.DataFrame({
    'feature': feature_names,
    'importance': model.named_steps['classifier'].feature_importances_
})
feature_importance = feature_importance.sort_values('importance', ascending=False)

# Plot feature importance
plt.figure(figsize=(12, 8))
sns.barplot(x='importance', y='feature', data=feature_importance.head(15))
plt.title('Top 15 Features Importance in Predicting Churn')
plt.tight_layout()
plt.savefig('plots/feature_importance2.png')
plt.close()

# Print comprehensive model evaluation
print("\n=== MODEL PERFORMANCE ===")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print(f"\nROC-AUC Score: {roc_auc_score(y_test, y_pred_proba):.3f}")

# Print key insights
print("\n=== KEY INSIGHTS ===")
print("\n1. Most Important Features:")
for idx, row in feature_importance.head(5).iterrows():
    print(f"- {row['feature']}: {row['importance']:.3f} importance score")

print("\n2. Model Performance Summary:")
print(f"- Average Cross-validation ROC-AUC: {cv_scores.mean():.3f}")
print(f"- Test Set ROC-AUC: {roc_auc_score(y_test, y_pred_proba):.3f}")
print(f"- Overall Accuracy: {(y_pred == y_test).mean():.2%}")
