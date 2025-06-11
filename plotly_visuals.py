import os
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Load data
data = pd.read_csv("telco.csv")

# Directory for interactive plots
plot_dir = os.path.join('plots', 'interactive')
os.makedirs(plot_dir, exist_ok=True)

# Churn Distribution
fig = px.histogram(data, x='Churn Label', color='Churn Label',
                   title='Churn Distribution')
fig.write_html(os.path.join(plot_dir, 'churn_distribution.html'))

# Age Distribution by Churn
fig = px.box(data, x='Churn Label', y='Age', color='Churn Label',
             title='Age Distribution by Churn')
fig.write_html(os.path.join(plot_dir, 'age_by_churn.html'))

# Gender Distribution by Churn
fig = px.histogram(data, x='Gender', color='Churn Label', barmode='group',
                   title='Gender Distribution by Churn')
fig.write_html(os.path.join(plot_dir, 'gender_by_churn.html'))

# CLTV by Churn
fig = px.box(data, x='Churn Label', y='CLTV', color='Churn Label',
             title='Customer Lifetime Value by Churn')
fig.write_html(os.path.join(plot_dir, 'cltv_by_churn.html'))

# Monthly Charges by Churn
fig = px.box(data, x='Churn Label', y='Monthly Charge', color='Churn Label',
             title='Monthly Charges by Churn')
fig.write_html(os.path.join(plot_dir, 'monthly_charge_by_churn.html'))

# Tenure by Churn
fig = px.box(data, x='Churn Label', y='Tenure in Months', color='Churn Label',
             title='Tenure by Churn Status')
fig.write_html(os.path.join(plot_dir, 'tenure_by_churn.html'))

# Internet Type Distribution
internet_data = data[data['Internet Type'].notna()]
fig = px.histogram(internet_data, x='Internet Type', color='Churn Label',
                   barmode='group', title='Internet Type Distribution')
fig.write_html(os.path.join(plot_dir, 'internet_type_distribution.html'))

# Contract Type Distribution
fig = px.histogram(data, x='Contract', color='Churn Label', barmode='group',
                   title='Contract Type Distribution')
fig.write_html(os.path.join(plot_dir, 'contract_distribution.html'))

# Payment Method Distribution
fig = px.histogram(data, x='Payment Method', color='Churn Label', barmode='group',
                   title='Payment Method Distribution')
fig.update_xaxes(tickangle=45)
fig.write_html(os.path.join(plot_dir, 'payment_method_distribution.html'))

# Top 10 Churn Reasons
churn_reasons = data[data['Churn Label'] == 'Yes']['Churn Reason'].value_counts().head(10)
fig = px.bar(x=churn_reasons.values, y=churn_reasons.index,
             orientation='h', labels={'x': 'Count', 'y': 'Churn Reason'},
             title='Top 10 Reasons for Churn')
fig.write_html(os.path.join(plot_dir, 'churn_reasons.html'))

# Correlation Matrix
numerical_cols = ['Age', 'Number of Dependents', 'Tenure in Months',
                 'Monthly Charge', 'Total Charges', 'Churn Score', 'CLTV']
corr = data[numerical_cols].corr()
fig = px.imshow(corr, text_auto=True,
                title='Correlation Matrix of Numerical Variables')
fig.write_html(os.path.join(plot_dir, 'correlation_matrix.html'))

# Feature Importance
features = ['Age', 'Tenure in Months', 'Monthly Charge', 'Total Charges',
           'Churn Score', 'CLTV', 'Number of Dependents']
X = data[features]
y = (data['Churn Label'] == 'Yes').astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_scaled, y_train)
feature_importance = pd.DataFrame({'feature': features,
                                  'importance': rf.feature_importances_})
feature_importance = feature_importance.sort_values('importance', ascending=False)
fig = px.bar(feature_importance, x='importance', y='feature', orientation='h',
             title='Feature Importance in Predicting Churn')
fig.write_html(os.path.join(plot_dir, 'feature_importance.html'))

# Additional Scatter Plot
fig = px.scatter(data, x='Age', y='CLTV', color='Churn Label',
                 size='Monthly Charge', hover_data=['Tenure in Months'],
                 title='Age vs CLTV by Churn')
fig.write_html(os.path.join(plot_dir, 'age_cltv_scatter.html'))

print(f"Plotly visualizations saved in {plot_dir}")
