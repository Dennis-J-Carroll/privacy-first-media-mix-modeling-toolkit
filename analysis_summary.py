"""
IBM Telco Customer Churn Analysis - Executive Summary
"""

def print_section_header(title):
    print("\n" + "="*50)
    print(title)
    print("="*50)

print_section_header("EXECUTIVE SUMMARY - TELCO CUSTOMER CHURN ANALYSIS")

print("""
Dataset Overview:
- Total Customers Analyzed: 7,043
- Overall Churn Rate: 26.54%
- Average Customer Age: 46.5 years
- Average Customer Lifetime Value: $4,400.30
""")

print_section_header("KEY FINDINGS")

print("""
1. Customer Demographics:
   - Gender distribution is balanced (Male: 50.48%, Female: 49.52%)
   - Age groups:
     * Middle Age (31-45): 27.59%
     * Senior (46-60): 27.10%
     * Elderly (60+): 23.60%
     * Young Adult (0-30): 21.71%
   - Churned customers tend to be older (avg. 49.7 years)

2. Churn Predictors (by importance):
   - Churn Score (0.576)
   - Tenure in Months (0.110)
   - Monthly Charge (0.093)

3. Service Analysis:
   - Internet Type Churn Rates:
     * Fiber Optic: 40.72% (Highest risk)
     * Cable: 25.66%
     * DSL: 18.58% (Lowest risk)

4. Top Churn Reasons:
   - Competitor had better devices (16.75%)
   - Competitor made better offer (16.64%)
   - Attitude of support person (11.77%)
   - Unknown reasons (6.96%)
   - Competitor offered more data (6.26%)

5. Financial Impact:
   - Churned customers have lower CLTV ($4,149 vs $4,491)
   - Higher monthly charges correlate with increased churn risk
""")

print_section_header("MODEL PERFORMANCE")

print("""
Random Forest Classifier Results:
- Overall Accuracy: 92%
- Precision: 
  * Non-churn (0): 93%
  * Churn (1): 90%
- Recall:
  * Non-churn (0): 96%
  * Churn (1): 81%
""")

print_section_header("RECOMMENDATIONS")

print("""
1. Service Improvement:
   - Focus on Fiber Optic service quality due to high churn rate
   - Review and enhance device offerings to compete with competitors
   - Implement competitive pricing strategy

2. Customer Support:
   - Enhance support staff training to improve customer interaction
   - Implement regular customer satisfaction surveys
   - Develop proactive support protocols for high-risk customers

3. Retention Strategy:
   - Develop targeted retention programs for:
     * Customers with high churn scores
     * New customers (low tenure)
     * Customers with high monthly charges
   - Create competitive counter-offers for at-risk customers

4. Product Development:
   - Enhance device offerings to match or exceed competitors
   - Review and optimize data plans
   - Consider bundled services with improved value proposition

5. Monitoring and Prevention:
   - Implement early warning system based on churn score
   - Regular competitive analysis of market offerings
   - Monitor customer usage patterns for early churn indicators
""")

print_section_header("VISUALIZATION REFERENCE")

print("""
Generated visualizations are saved in the 'plots' directory:
1. churn_demographics.png - Customer demographic analysis
2. value_analysis.png - Customer value and financial metrics
3. service_analysis.png - Service usage patterns
4. churn_reasons.png - Top reasons for customer churn
5. correlation_matrix.png - Relationship between variables
6. feature_importance.png - Key predictors of churn
""")
