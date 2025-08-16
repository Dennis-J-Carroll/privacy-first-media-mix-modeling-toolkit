## Telco Customer Churn Analysis

### 📊 Project Overview

This project analyzes customer churn in a telecommunications company using machine learning and data visualization techniques. The analysis is based on the IBM Telco Customer Churn Dataset from Kaggle. The analysis includes customer demographics, service usage patterns, and predictive modeling to identify key factors contributing to customer churn.

![Demographics](plots/churn_demographics.png)





### Figure 1: Churn distribution by status, age distribution, and gender breakdown. Shows overall churn rate (~26.5%), age spread differences, and that churn is balanced across genders.


## Project Structure
- `PRO_1.py`: Initial exploratory data analysis and basic statistics
- `PRO_2.py`: Advanced analysis with visualizations and machine learning model
- `analysis_summary.py`: Executive summary with key findings and recommendations
- `plots/`: Directory containing generated visualizations
- `telco.csv`: Original dataset

## Key Findings
- Overall churn rate: 26.54%
- Fiber Optic service has highest churn rate (40.72%)
- Top churn reasons are competitor-related (better devices, better offers)
- Model achieves 92% accuracy in predicting churn

## Visualizations
The project generates six key visualizations:
- Customer Demographics Analysis
- Customer Value Analysis
- Service Usage Patterns
- Top Churn Reasons
- Correlation Analysis
- Feature Importance in Churn Prediction




## Model Performance
Random Forest Classifier Results:
- Overall Accuracy: 92%
- Precision: 93% (Non-churn), 90% (Churn)
- Recall: 96% (Non-churn), 81% (Churn)
 

## Requirements
- Python 3.x
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

## Usage
1. Clone the repository
2. Ensure all required packages are installed
3. Run the scripts in order:
   ```bash
   python3 PRO_1.py
   python3 PRO_2.py
   python3 analysis_summary.py
   ```

## 📊 Recommendations

Based on analysis and industry benchmarks, here are strategies to reduce churn:

### Service Improvement:

1. Fiber Optic Quality: Fiber customers churn at ~0.84% vs. ~2% for DSL/cable (Leichtman Research Group, 2023).
2. Device Offerings: 43% of customers say device upgrade options drive loyalty (Deloitte, 2022).
3. Competitive Pricing: Verizon lowered churn after reducing its 40% premium to ~15%.

### Customer Support:

1. Staff Training: 56% of telecom churn is due to poor service (Accenture, 2021).
2. Satisfaction Surveys: Quarterly surveys cut churn by 12–15%.
3. Proactive Support: Predictive outreach reduces churn by ~30% (McKinsey, 2022).

### Retention Strategy:

1. High-Risk Customers: Predictive models improve retention efficiency by 25–30% (BCG, 2022).
2. Counter-Offers: Personalized offers reduce churn by 15–20% (Forrester, 2021).
3. Early Tenure: Month-to-month churn = 42.7% vs. 2.8–11.3% for long-term (Medium, 2023).


-------------------------------------------
**For a deeper dive, see** [recommendations](https://github.com/Dennis-J-Carroll/telco-churn-analysis/blob/main/recommendations.md)

License

MIT License
