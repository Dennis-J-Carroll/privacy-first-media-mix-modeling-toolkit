# Privacy Parameter Selection Guide

## Introduction

This guide helps you select an appropriate epsilon (ε) value for differential privacy in Media Mix Modeling (MMM). The privacy parameter epsilon controls the fundamental tradeoff between privacy protection and analytical utility.

**Key Principle**: Lower epsilon = stronger privacy guarantees but more noise and potentially lower model accuracy.

---

## Table of Contents

1. [Understanding Epsilon](#understanding-epsilon)
2. [Decision Framework](#decision-framework)
3. [Reference Tables](#reference-tables)
4. [Case Studies](#case-studies)
5. [Validation & Testing](#validation--testing)
6. [Documentation Template](#documentation-template)

---

## Understanding Epsilon

### What is Epsilon?

Epsilon (ε) is the **privacy budget** in differential privacy. It quantifies the privacy loss from a data analysis:

- **Formal Definition**: For any two datasets differing by one individual, the ratio of probabilities of observing any output is bounded by e^ε
- **Intuitive Meaning**: Smaller epsilon makes it harder to distinguish whether any individual's data was included in the analysis

### Privacy Guarantees

The toolkit implements **(ε, δ)-differential privacy** where:

- **ε (epsilon)**: Privacy loss parameter
  - ε = 0: Perfect privacy (no information released)
  - ε = ∞: No privacy (exact data released)
  - Typical values: 0.1 to 10.0

- **δ (delta)**: Failure probability (default: 1e-5)
  - Probability that privacy guarantee doesn't hold
  - Set very small (typically 1e-5 to 1e-7)

### The Privacy-Utility Tradeoff

**Privacy Side (Lower ε):**
- Stronger privacy protection for individuals
- More Laplace noise added to data
- Harder to recover exact patterns
- Lower model accuracy (R², parameter estimates)

**Utility Side (Higher ε):**
- Weaker privacy protection
- Less noise added to data
- Clearer signal in the data
- Higher model accuracy

**The Challenge**: Find the right balance for your specific use case.

---

## Decision Framework

### Four-Question Decision Tree

Use this framework to narrow down your epsilon range:

#### 1. What type of data are you analyzing?

| Data Type | Description | Recommended ε Range | Rationale |
|-----------|-------------|---------------------|-----------|
| **Individual Health/Financial** | Personal medical records, credit card transactions, salary data | **ε ≤ 0.5** | High sensitivity, severe consequences from breaches |
| **Individual Behavioral** | User-level clickstream, purchase history, location data | **ε = 0.5-1.0** | Moderate sensitivity, privacy regulations apply |
| **Aggregated Marketing** | Weekly spend, campaign revenue, channel metrics | **ε = 1.0-5.0** | Lower sensitivity due to aggregation, typical MMM use case |
| **Public/Non-Sensitive** | Public reports, anonymous surveys, synthetic data | **ε ≥ 5.0** | Minimal privacy risk, focus on utility |

**For this MMM toolkit**: Most use cases fall into "Aggregated Marketing" → Start with **ε = 1.0-3.0**

#### 2. What are your regulatory requirements?

| Regulation | Jurisdiction | Typical ε Requirements | Key Considerations |
|------------|--------------|------------------------|-------------------|
| **GDPR** | European Union | ε ≤ 1.0 | Strict "privacy by design" principle; document compliance |
| **CCPA** | California, USA | ε ≤ 3.0 | Focus on consumer rights; reasonable safeguards required |
| **HIPAA** | USA (Health) | ε ≤ 0.5 | Protected Health Information (PHI) requires strong protection |
| **PIPEDA** | Canada | ε ≤ 2.0 | Proportional to sensitivity; consent-based |
| **None Specific** | Various | ε based on utility needs | Still apply best practices; consider ethical obligations |

**Action**: Consult with your legal/compliance team to determine applicable regulations.

#### 3. What is your dataset size?

Dataset size affects how much noise your analysis can tolerate:

| Records (n) | Recommended ε Adjustment | Reasoning |
|-------------|-------------------------|-----------|
| **< 100** | Increase by 2-5x | Small datasets amplify noise impact; may need ε = 5-10 |
| **100-1,000** | Moderate range | ε = 2-5 typically viable |
| **1,000-10,000** | Lower range feasible | ε = 1-3 provides good balance |
| **> 10,000** | Can use low epsilon | ε = 0.5-2 still provides utility due to law of large numbers |

**For this toolkit**: Default is 104 weeks (2 years) of data → **ε = 1.0-3.0** is appropriate

#### 4. What are your accuracy requirements?

| Use Case | Accuracy Need | Recommended ε | Tradeoff |
|----------|--------------|---------------|----------|
| **High-Stakes Budget Decisions** | R² > 0.85, precise parameter estimates | ε = 3.0-5.0 | Sacrifice some privacy for confidence in decisions |
| **Strategic Planning** | R² > 0.75, directional insights | ε = 1.0-3.0 | Balanced approach |
| **Exploratory Analysis** | R² > 0.60, trend identification | ε = 0.5-1.0 | Prioritize privacy, tolerate more uncertainty |
| **Privacy Compliance Demo** | Privacy compliance > accuracy | ε ≤ 0.5 | Maximum privacy, accept lower utility |

**Action**: Define your minimum acceptable R² and MAE thresholds before selecting epsilon.

---

## Reference Tables

### Epsilon Value Interpretation

| Epsilon (ε) | Privacy Level | Noise Scale (Relative) | Typical R² Impact | Use Cases |
|-------------|---------------|------------------------|-------------------|-----------|
| **0.1** | Very High Privacy | 10x baseline | -20% to -40% | Medical research, highly sensitive data |
| **0.5** | High Privacy | 2x baseline | -10% to -20% | Financial data, GDPR compliance |
| **1.0** | Moderate-High Privacy | 1x (baseline) | -5% to -10% | Sensitive marketing data, recommended starting point |
| **2.0** | Moderate Privacy | 0.5x baseline | -2% to -5% | Standard marketing analytics |
| **3.0** | Low-Moderate Privacy | 0.33x baseline | -1% to -3% | Business intelligence, CCPA compliance |
| **5.0** | Low Privacy | 0.2x baseline | < -1% | Aggregated metrics, low sensitivity |
| **10.0** | Very Low Privacy | 0.1x baseline | Negligible | Public data, minimal privacy concerns |

*Note: R² impact is approximate and varies by dataset characteristics*

### Privacy Budget Composition

When multiple analyses are performed on the same dataset, privacy budgets **compose** (add up):

| Scenario | Total ε Used | Calculation |
|----------|-------------|-------------|
| Single analysis with ε=1.0 | **1.0** | Direct |
| Two analyses, each ε=1.0 | **2.0** | Sequential composition: 1.0 + 1.0 |
| Four queries, budget split | **1.0** | Parallel composition: ε/4 per query, total ε |
| Monthly reports (12 months), ε=1.0 each | **12.0** | Accumulated over time |

**Implication**: Plan your privacy budget carefully. If you need multiple analyses, either:
- Split epsilon across analyses (ε/k per analysis)
- Accept higher total epsilon from composition
- Use advanced composition theorems (consult expert)

---

## Case Studies

### Case Study 1: Health Tech MMM (ε=0.5)

**Context**: A digital health startup wanted to understand marketing effectiveness for user acquisition while protecting patient privacy.

**Requirements**:
- **Data**: 8,000 weekly records with user demographics linked to health conditions
- **Regulation**: HIPAA compliance required
- **Utility Need**: Identify top 2-3 performing channels (directional, not precise)

**Decision Process**:
1. HIPAA → Must use ε ≤ 0.5
2. Dataset size (8,000) → Large enough to tolerate high privacy
3. Exploratory analysis → Can accept lower accuracy
4. Tested ε = [0.1, 0.5, 1.0] → Selected ε=0.5 as best balance

**Results**:
- **Privacy**: Strong protection, (ε=0.5, δ=1e-5)-DP
- **Accuracy**: R² = 0.68 (down from 0.82 without privacy)
- **Utility**: Successfully identified top channels, directional ROI estimates
- **Noise**: Mean noise ~$800 on $15K average spend

**Lessons Learned**:
- High privacy was achievable with acceptable utility
- Stakeholder buy-in required clear privacy-utility tradeoff communication
- Validated results with A/B test using higher epsilon on subset

**Recommendation**: For health data, start with ε=0.5 and only increase if utility is insufficient.

---

### Case Study 2: E-commerce MMM (ε=2.0)

**Context**: Mid-size e-commerce company optimizing $2M/year marketing budget across 5 channels.

**Requirements**:
- **Data**: 156 weeks (3 years) of aggregated weekly data
- **Regulation**: CCPA applicable (California customers represent 30% of revenue)
- **Utility Need**: Precise parameter estimates for budget reallocation (high stakes)

**Decision Process**:
1. CCPA → ε ≤ 3.0 recommended
2. Aggregated data → Lower sensitivity than individual records
3. Large dataset (156 weeks) → Can tolerate moderate privacy
4. High-stakes decisions → Need R² > 0.80
5. Tested ε = [1.0, 2.0, 3.0, 5.0] → Selected ε=2.0

**Results**:
- **Privacy**: (ε=2.0, δ=1e-5)-DP, meets CCPA requirements
- **Accuracy**: R² = 0.84 (vs. 0.87 without privacy)
- **Utility**: Parameter estimates within 8% of true values
- **Business Impact**: Confidently reallocated $400K between channels

**Lessons Learned**:
- ε=2.0 provided excellent balance for aggregated marketing data
- Minimal accuracy sacrifice for meaningful privacy protection
- Privacy officer approved based on formal guarantee documentation

**Recommendation**: For most e-commerce MMM, ε=2.0 is a sweet spot.

---

### Case Study 3: Public Sector MMM (ε=1.0)

**Context**: Government agency promoting public health campaign, using taxpayer data for analysis.

**Requirements**:
- **Data**: 104 weeks of campaign spend and public health metrics
- **Regulation**: GDPR (EU citizens), strong ethical obligations
- **Utility Need**: Transparent, auditable analysis for public accountability

**Decision Process**:
1. GDPR + public trust → Must prioritize privacy
2. Ethical obligation to protect citizen data
3. Moderate utility needs (trend identification sufficient)
4. Selected ε=1.0 for strong privacy with acceptable utility

**Results**:
- **Privacy**: (ε=1.0, δ=1e-5)-DP, GDPR compliant
- **Accuracy**: R² = 0.76 (vs. 0.82 without privacy)
- **Transparency**: Published methodology including epsilon choice
- **Public Trust**: Positive reception for privacy-first approach

**Lessons Learned**:
- Public sector benefits from conservative epsilon choice
- Transparency about privacy methods builds trust
- ε=1.0 provided sufficient utility for policy decisions

**Recommendation**: For government/public sector, default to ε=1.0 unless specific utility needs justify higher.

---

## Validation & Testing

### How to Measure Utility Degradation

Before committing to an epsilon value, quantify the privacy cost:

#### 1. Run Parameter Sweep

```bash
python examples/privacy_parameter_sweep.py
```

This generates comparison across ε = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]

#### 2. Key Metrics to Evaluate

| Metric | What It Measures | Interpretation |
|--------|------------------|----------------|
| **R²** | Model fit quality | Higher = better predictions; track degradation vs. baseline |
| **MAE** | Prediction error | Lower = better; acceptable threshold depends on revenue scale |
| **Parameter Error** | Recovery of true parameters | For synthetic data; lower = more accurate parameter estimates |
| **Noise Statistics** | Amount of noise added | Higher = more privacy but more distortion |

#### 3. Define Acceptable Thresholds

Set thresholds based on business needs:

```
Minimum Acceptable R²: 0.75
Maximum Acceptable MAE: $2,000
Maximum Parameter Error: 15%
```

Then find the **lowest epsilon** (highest privacy) that meets these thresholds.

### Statistical Tests for Parameter Accuracy

If you have ground truth (synthetic data or holdout set):

#### Relative Error

```
Relative Error = |fitted_value - true_value| / |true_value|
```

- < 5%: Excellent recovery
- 5-15%: Good recovery
- 15-30%: Moderate recovery
- \> 30%: Poor recovery

#### Mean Absolute Percentage Error (MAPE)

```
MAPE = (1/n) * Σ |fitted_i - true_i| / |true_i| * 100%
```

Aggregate across all parameters to assess overall accuracy.

### Sensitivity Analysis Procedures

#### Test Multiple Epsilon Values

Don't rely on a single epsilon. Test range:

1. **Very Low** (ε=0.1): Understand maximum privacy scenario
2. **Low** (ε=0.5-1.0): High privacy range
3. **Moderate** (ε=2.0-3.0): Balanced range
4. **High** (ε=5.0-10.0): Low privacy, high utility

#### Analyze Sensitivity

- **Sharp drop in R² at specific epsilon?** → That's your threshold
- **Minimal improvement above certain epsilon?** → Diminishing returns, stop there
- **Consistent parameter errors across range?** → Noise isn't the limiting factor

### A/B Testing Privacy Levels

If possible, run parallel analyses:

1. **Control Group**: Analysis with ε=5.0 (low privacy)
2. **Treatment Group**: Analysis with ε=1.0 (high privacy)
3. **Compare**: Do business decisions differ?

If decisions are the same, choose higher privacy (ε=1.0). If different, assess whether the difference is material.

---

## Documentation Template

Always document your epsilon selection for audit trail and regulatory compliance.

### Template

```markdown
# Privacy Parameter Documentation

## Analysis Details
**Analysis Name**: [e.g., "Q1 2024 Marketing Mix Model"]
**Date**: [Date]
**Analyst**: [Your Name]
**Approvers**: [Privacy Officer, Data Owner]

## Epsilon Selection

### Selected Value
**Epsilon (ε)**: [value]
**Delta (δ)**: [value, typically 1e-5]

### Justification

#### 1. Data Sensitivity
- **Data Type**: [e.g., "Aggregated weekly marketing spend and revenue"]
- **Aggregation Level**: [e.g., "Weekly aggregates, no individual records"]
- **Sensitivity Assessment**: [e.g., "Medium - contains business metrics but no PII"]

#### 2. Regulatory Requirements
- **Applicable Regulations**: [e.g., "CCPA (30% California customers)"]
- **Compliance Requirements**: [e.g., "ε ≤ 3.0 per internal guidelines"]
- **Legal Review**: [Date and name]

#### 3. Dataset Characteristics
- **Number of Records**: [e.g., "104 weeks (2 years)"]
- **Temporal Range**: [e.g., "Jan 2022 - Dec 2023"]
- **Aggregation Method**: [e.g., "Sum of daily values per week"]

#### 4. Utility Requirements
- **Use Case**: [e.g., "Budget optimization for FY2025"]
- **Minimum Acceptable R²**: [e.g., "0.75"]
- **Maximum Acceptable MAE**: [e.g., "$2,000"]
- **Decision Stakes**: [High/Medium/Low]

#### 5. Testing Conducted
**Parameter Sweep Results**:
- Tested epsilon values: [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
- See attached: `epsilon_comparison_summary.csv`

**Selected Epsilon Performance**:
- R²: [value]
- MAE: [value]
- Mean Parameter Error: [value]

**Rationale for Selection**:
[Explain why this epsilon was chosen over alternatives]

## Privacy Guarantee

This analysis provides **(ε=[value], δ=[value])-differential privacy**.

**Interpretation**:
For any two datasets differing by one individual's weekly contribution, the probability of observing any specific analysis output changes by at most a factor of e^ε = [calculate e^ε].

**Practical Meaning**:
[Explain in plain language what this guarantees for your stakeholders]

## Impact Assessment

### Privacy Impact
- **Privacy Level**: [Very High / High / Moderate / Low / Very Low]
- **Noise Statistics**:
  - Mean noise added to spend: $[value]
  - Mean noise added to revenue: $[value]
  - Max noise observed: $[value]

### Utility Impact
- **Model Accuracy**:
  - R²: [value] (vs. [value] without privacy)
  - MAE: $[value] (vs. $[value] without privacy)
  - Relative degradation: [percentage]

- **Parameter Estimates**:
  - Mean parameter error: [percentage]
  - All parameters within [percentage] of true values (if known)

- **Business Decisions**:
  - [Describe how epsilon choice affects key business decisions]
  - [Note any decisions that would change with different epsilon]

## Risk Assessment

### Privacy Risks with This Epsilon
- [Assess residual privacy risks]
- [Consider composition if multiple analyses planned]
- [Note any limitations of differential privacy]

### Utility Risks with This Epsilon
- [Assess whether noise could lead to poor decisions]
- [Consider uncertainty in parameter estimates]
- [Note any business risks from reduced accuracy]

### Mitigation Strategies
- [Additional safeguards beyond differential privacy]
- [Access controls, data retention policies, etc.]
- [Monitoring and audit procedures]

## Approval

**Privacy Officer**:
- Name: [Name]
- Date: [Date]
- Signature: [Signature]
- Comments: [Any conditions or notes]

**Data Owner**:
- Name: [Name]
- Date: [Date]
- Signature: [Signature]
- Comments: [Any conditions or notes]

**Compliance Review** (if applicable):
- Name: [Name]
- Date: [Date]
- Signature: [Signature]
- Regulatory Compliance Confirmed: [Yes/No]

## Attachments

1. `epsilon_comparison_summary.csv` - Parameter sweep results
2. `privacy_utility_tradeoff.png` - Visual comparison
3. `parameter_accuracy_comparison.png` - Parameter recovery analysis
4. [Any additional supporting documents]

---

**Document Version**: 1.0
**Last Updated**: [Date]
**Next Review**: [Date - recommend annual or when regulations change]
```

---

## Best Practices Summary

### Quick Reference

1. **Start Conservative**: Begin with ε=1.0 and adjust based on testing
2. **Test Multiple Values**: Use parameter sweep script to compare options
3. **Document Everything**: Use template above for audit trail
4. **Get Sign-Off**: Involve privacy officer and legal team
5. **Monitor Composition**: Track cumulative epsilon if multiple analyses
6. **Review Regularly**: Reassess epsilon as regulations and data change
7. **Educate Stakeholders**: Use interactive notebook to build understanding
8. **Balance Thoughtfully**: Neither maximize privacy nor utility exclusively

### Common Pitfalls to Avoid

❌ **Setting epsilon too low without testing**: Can destroy utility unnecessarily
❌ **Setting epsilon too high "to be safe"**: Defeats the purpose of differential privacy
❌ **Ignoring composition**: Multiple analyses with ε=1.0 each → total ε = n
❌ **Assuming epsilon is universal**: Appropriate value is context-dependent
❌ **Skipping documentation**: Regulatory compliance requires audit trail
❌ **Not involving privacy officer**: Legal risk if epsilon isn't properly justified
❌ **Treating epsilon as purely technical**: It's a policy decision with ethical implications

### When to Seek Expert Help

Consult a differential privacy expert if:

- Your data involves individual-level sensitive information (not aggregated)
- You need to perform many related analyses (complex composition)
- Regulatory penalties for privacy breaches are severe
- You're implementing custom privacy mechanisms beyond this toolkit
- Stakeholders are unfamiliar with differential privacy (need training)
- You're using advanced techniques (Rényi DP, concentrated DP, etc.)

---

## Additional Resources

### Academic Papers

1. **Dwork et al. (2006)**: "Calibrating Noise to Sensitivity in Private Data Analysis"
   - Foundational paper introducing differential privacy

2. **Dwork & Roth (2014)**: "The Algorithmic Foundations of Differential Privacy"
   - Comprehensive textbook (free online)

3. **Lee & Clifton (2011)**: "How Much Is Enough? Choosing ε for Differential Privacy"
   - Practical guidance on epsilon selection

### Industry Guidelines

- **Google**: "Differentially Private Data Analysis at Scale"
- **Apple**: "Learning with Privacy at Scale" (iOS differential privacy)
- **U.S. Census Bureau**: "Disclosure Avoidance for the 2020 Census"

### Regulatory Resources

- **GDPR Article 25**: Privacy by Design and by Default
- **NIST Special Publication 800-188**: De-Identifying Government Datasets
- **ICO (UK)**: Anonymisation Code of Practice

### Tools

- **This Toolkit**: `privacy_parameter_sweep.py` and `interactive_privacy_tuning.ipynb`
- **Google DP Library**: Open-source DP primitives
- **IBM Diffprivlib**: Python library for differential privacy

---

## Conclusion

Selecting the right epsilon is both an art and a science. It requires:

- **Understanding** the mathematical guarantees
- **Assessing** your specific context and requirements
- **Testing** multiple values empirically
- **Documenting** your rationale and approval
- **Balancing** privacy protection with analytical utility

Use this guide, the interactive tools, and the parameter sweep script to make an informed, defensible epsilon choice for your Media Mix Modeling analysis.

**Remember**: The goal is not to maximize privacy or utility in isolation, but to find the optimal balance for your specific use case while meeting regulatory requirements and ethical obligations.

For questions or feedback on this guide, please consult with your privacy officer or differential privacy experts.

---

**Document Version**: 1.0
**Last Updated**: 2024
**Maintained by**: Privacy-First MMM Toolkit Team
