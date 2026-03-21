# Privacy-First Media Mix Modeling Toolkit

## Overview

This repository provides a **privacy-first toolkit for media mix modeling (MMM)** that demonstrates how to build effective marketing analytics while respecting individual privacy through formal privacy-preserving mechanisms. Marketing teams and analysts can estimate the incremental impact of different marketing channels (e.g., TV, search, social, email) on key outcomes such as conversions or revenue without relying on user-level tracking.

This project serves as a **proof-of-concept for building machine learning systems with privacy and safety constraints from the ground up**. It aims to demonstrate that sophisticated analytics can be performed in a way that is both effective and ethically responsible, aligning with the growing need for AI systems that are trustworthy and aligned with human values.

## Relevance to AI Safety

### Privacy-Preserving Analytics as a Pillar of AI Alignment

This toolkit exemplifies a fundamental principle of AI safety: **systems trained on data that has been ethically sourced and processed with strong privacy guarantees are inherently more aligned with human values**. Privacy violations themselves represent a form of negative outcome or harm that a safe AI system should be designed to avoid from the ground up.

By building models that operate exclusively on aggregated data and employ formal differential privacy mechanisms, we reduce the risk of creating systems that could be used for surveillance, discrimination, or other harmful purposes. This proactive approach to privacy implements several core AI safety principles:

- **Robustness**: The system behaves as intended even under adversarial conditions by design, as it cannot access individual-level data
- **Privacy and Confidentiality**: Formal mathematical guarantees protect fundamental individual rights
- **Transparency**: Privacy parameters (ε, δ) can be made public, allowing independent verification of privacy claims
- **Alignment by Design**: Safety constraints are integrated into the modeling process from the ground up, not bolted on as an afterthought

### Connecting Privacy by Design to AI Safety Principles

This toolkit implements **"privacy by design"** as a core architectural philosophy, where privacy considerations are embedded into the entire system lifecycle:

1. **Data Minimization (Layer 1)**: The system operates on aggregated summaries rather than individual records, inherently minimizing sensitive information processing. This reduces the attack surface and potential impact of data breaches.

2. **Differential Privacy (Layer 2)**: A formal method providing mathematical guarantees of privacy by adding carefully calibrated statistical noise to aggregated data, making it impossible to determine whether any single individual's data was included in the analysis.

3. **Proactive Risk Assessment**: Potential harms are anticipated and mitigated before they can occur, rather than relying on post-hoc auditing or patching.

This multi-layered approach demonstrates how AI safety principles can be translated into concrete engineering practices, making the toolkit a valuable reference implementation for the responsible AI community.

### Positioning as a Proof-of-Concept for Responsible AI

This toolkit is positioned as a **tangible proof-of-concept** demonstrating that valuable analytics can be performed without compromising individual privacy. It is designed to inspire and guide developers and researchers building privacy-preserving and safe AI systems by showing:

- How advanced statistical methods (Bayesian hierarchical models) can be adapted to work within a privacy-preserving framework
- The practical trade-offs between privacy and utility, and how to navigate them responsibly
- That "privacy-first" design is not about perfect privacy at any cost, but about finding a pragmatic balance that provides useful insights while maintaining strong privacy guarantees

We acknowledge that adding noise to data for privacy reasons impacts model accuracy—this is an inherent property of differential privacy, not a flaw. The goal is to manage this trade-off effectively and transparently, demonstrating a mature, responsible approach to AI development.

## Features

- **Privacy Preservation**: Built on a privacy-first architecture that operates exclusively on aggregated, non-user-level data. Employs formal differential privacy mechanisms (Laplace noise) with mathematically calibrated noise addition to provide quantifiable (ε, δ)-differential privacy guarantees. This ensures valuable marketing insights can be derived without compromising individual privacy.

- **Aggregated Data Pipelines**: Ingest channel-level spend, impressions, and conversions aggregated over time, ensuring no personal data is collected. Aggregation serves as the foundational privacy measure.

- **Advanced Modeling Frameworks**: Includes baseline linear models and sophisticated Bayesian hierarchical models that estimate channel contribution while accounting for:
  - **Saturation effects**: Diminishing returns as spend increases (Hill function)
  - **Ad-stock effects**: Carryover impact of advertising over time (geometric decay)
  - **Control variables**: Seasonality and promotions to avoid misattribution

- **Visualization Tools**: Generate charts showing marginal return curves, channel saturation, expected lift versus spend, and privacy-utility trade-offs to help stakeholders understand both media efficiency and privacy implications.

- **Extensible Design**: Modular codebase allowing analysts to plug in their own data sources, priors, model structures, and privacy parameters.

- **Privacy-Utility Trade-off Analysis**: Demonstrates how privacy parameter (ε) affects both privacy guarantees and model accuracy, with tools to find the optimal balance for your use case.

## Privacy-ML Connection: Technical Implementation

### Differential Privacy Mechanism

This toolkit implements **ε-differential privacy** using the **Laplace mechanism**, a well-established approach for providing formal privacy guarantees. The mathematical definition of ε-differential privacy is:

**A randomized algorithm M is ε-differentially private if for all pairs of neighboring datasets D and D' (differing by at most one record), and for all possible outputs S:**

```
Pr[M(D) ∈ S] ≤ e^ε × Pr[M(D') ∈ S]
```

This guarantee means that the presence or absence of any single individual's data has a bounded effect on the output of the analysis, making it nearly impossible to determine whether a specific individual's data was included.

### Application to Aggregated Input Data

Privacy protection in this toolkit is implemented through **input perturbation**: noise is added to aggregated data *before* it is used for modeling. The data flow is:

1. **Aggregation**: Individual-level data (e.g., user_id, timestamp, channel, conversion) is aggregated by summing total impressions, spend, and conversions for each marketing channel per day/week.

2. **Noise Addition**: The Laplace mechanism adds random noise to these aggregated sums. The noise is drawn from a Laplace distribution with scale parameter `b = sensitivity / ε`, where:
   - `sensitivity` is the maximum change in the aggregate from adding/removing one individual
   - `ε` (epsilon) is the privacy budget controlling the privacy-utility trade-off

3. **Modeling**: The resulting "noisy" aggregates serve as input to the Bayesian hierarchical model.

This approach is modular and allows the use of standard modeling algorithms without modification. The privacy guarantee is provided by the pre-processing step.

### Laplace Mechanism Details

The toolkit uses the **Laplace distribution** for adding noise because:
- It provides tight privacy guarantees for numerical data
- It has a sharp peak at its mean with exponential tails, making it well-suited for adding noise to counts or sums
- It is computationally efficient and straightforward to implement

The scale of the noise is calibrated to the **sensitivity** of the query (the maximum change in output from adding/removing one individual's data). For simple sums or counts, the sensitivity is typically 1 or a small constant.

### Formal Privacy Guarantees

This toolkit provides **(ε, δ)-differential privacy** where:

- **ε (epsilon)**: The privacy budget or privacy parameter. Smaller values mean stronger privacy but more noise. Typical values range from 0.1 (very strong privacy) to 10 (weaker privacy).
- **δ (delta)**: A small probability (e.g., 10⁻⁶) that the privacy guarantee does not hold, used in approximate differential privacy. In many practical applications, δ is set to a negligibly small value.

**Choosing ε**: The choice of ε involves balancing privacy and utility:
- **High privacy** (ε ≤ 1): Suitable for highly sensitive data (health, financial)
- **Moderate privacy** (ε = 1-5): Good balance for many applications
- **Lower privacy** (ε ≥ 5): More utility, weaker privacy guarantees

Users should experiment with different ε values and evaluate the impact on both privacy and model accuracy, documenting their reasoning for transparency and regulatory compliance.

### Privacy-First vs. Privacy-Only

This toolkit adopts a **"privacy-first"** philosophy, not "privacy-only":

- **Privacy-first** means privacy is a primary consideration from the beginning, influencing architectural decisions and feature design. The goal is to build systems that are both effective and ethical.
- **Privacy-only** would mean optimizing solely for privacy, potentially resulting in a system that is perfectly private but analytically useless.

The toolkit seeks a **practical balance**: performing valuable media mix modeling while maintaining strong privacy protections. This aligns with responsible AI principles that acknowledge inherent trade-offs and make them transparent.

### Alignment with Privacy-Enhancing Technologies (PETs)

This approach is part of the broader ecosystem of **Privacy-Enhancing Technologies (PETs)**, specifically:

- **Input Perturbation / Query-based Differential Privacy**: Adding noise to data before analysis
- Compared to other PETs:
  - **Secure Multi-Party Computation (SMPC)** and **Homomorphic Encryption**: Provide very strong guarantees but are computationally expensive
  - **Federated Learning**: Trains on decentralized data but adds complexity
  - **This toolkit's approach**: Simpler, more efficient, and practical for centralized aggregated data

## Privacy-Utility Trade-offs

### Understanding the Trade-off

A fundamental property of differential privacy is that **adding noise to data for privacy reasons inevitably impacts model accuracy**. This is not a flaw—it is an inherent characteristic of the technique. The goal is to manage this trade-off effectively:

- **Smaller ε** (more privacy) → More noise → Lower model accuracy
- **Larger ε** (less privacy) → Less noise → Higher model accuracy

### Impact on Model Accuracy

When noise is added to the aggregated input data, the Bayesian model is trained on a "noisy" version of the data, which can lead to less accurate predictions. The magnitude of this impact depends on:

- **Dataset size**: Larger datasets dilute the impact of noise
- **Model complexity**: More complex models may be more sensitive to noise
- **Amount of noise added**: Controlled by ε

The toolkit's use of Bayesian models helps mitigate noise impact by quantifying uncertainty in predictions, which is a key advantage of this design.

### Balancing Privacy and Utility

Finding the right balance is context-dependent. Consider:

1. **What is the potential harm if an individual's data is revealed?**
2. **What are the legal and regulatory requirements in your jurisdiction?**
3. **How accurate do predictions need to be for business needs?**
4. **What is the dataset size, and how does it affect noise impact?**

### Guidance on Tuning ε

**Practical Guidelines:**

- **Start with ε between 1 and 10** for a moderate balance
- **Highly sensitive data** (health, financial): Use ε ≤ 1
- **Less sensitive data**: Consider ε = 5-10
- **Experiment and evaluate**: Run the model with different ε values and measure impact on both privacy (the ε value itself) and utility (model accuracy metrics like R², MSE)
- **Document your choice**: Always explain your reasoning for transparency and regulatory compliance

The toolkit includes visualization tools to help understand the privacy-utility trade-off empirically.

## Repository Structure

**Core Privacy-First MMM Files:**
- `advanced_mmm.py` - Main privacy-first media mix modeling implementation with differential privacy
- `requirements.txt` - Python dependencies
- `effort_mix_modeling.md` - Additional MMM guidelines and methodology
- `mmm_output_advanced/` - Output directory for analysis results

**Interactive Privacy Tuning Tools:**
- `examples/privacy_parameter_sweep.py` - Systematic epsilon comparison script
- `examples/interactive_privacy_tuning.ipynb` - Jupyter notebook with interactive widgets
- `docs/privacy_parameter_guide.md` - Comprehensive epsilon selection guide
- `mmm_privacy_comparisons/` - Output directory for epsilon comparison results (generated)

**Note:** This repository also contains some customer churn analysis scripts (`PRO_*.py`, `analysis_summary.py`, `telco.csv`) from a separate analytics project. These are not part of the privacy-first MMM toolkit and can be ignored or removed for production use.

## Getting Started

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/privacy-first-media-mix-modeling-toolkit.git
   cd privacy-first-media-mix-modeling-toolkit
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Privacy-First MMM

1. **Run the advanced MMM example** with differential privacy enabled:
   ```bash
   python advanced_mmm.py
   ```

   This will:
   - Generate synthetic marketing data
   - Apply differential privacy with ε=1.0 (configurable in script)
   - Fit a Bayesian hierarchical model
   - Generate visualizations and summary statistics

2. **Customize privacy settings** by editing `advanced_mmm.py`:
   ```python
   CONFIG = {
       "enable_privacy": True,  # Set to False to disable
       "epsilon": 1.0,          # Adjust privacy budget
       "delta": 1e-5,          # Failure probability
       # ... other settings
   }
   ```

3. **Explore outputs** in the `mmm_output_advanced/` directory:
   - `mmm_summary.csv`: Estimated parameters and marginal ROI
   - `response_curves.png`: Saturation curves for each channel
   - `predicted_vs_actual.png`: Model fit quality
   - `contribution_breakdown.png`: Channel contribution over time

### Working with Your Own Data

To use your own marketing data:

1. Format your data as aggregated weekly/daily summaries with columns:
   - `week` or `date`: Time period
   - `spend_<channel>`: Aggregated spend per channel
   - `revenue`: Total revenue
   - `promotions`: Binary promotion indicator (optional)

2. Modify the `generate_weekly_data()` function to load your data instead of generating synthetic data

3. Adjust sensitivity parameters in CONFIG based on your data characteristics

## Interactive Privacy Parameter Tuning

This toolkit includes interactive tools for exploring the privacy-utility tradeoff and selecting an appropriate epsilon value for your use case.

### Quick Start: Parameter Sweep

Run a systematic comparison of multiple epsilon values to visualize the privacy-utility tradeoff:

```bash
python examples/privacy_parameter_sweep.py
```

This generates:
- **`privacy_utility_tradeoff.png`**: The main visualization showing how epsilon affects model accuracy
- **`parameter_accuracy_comparison.png`**: Heatmap showing parameter recovery across epsilon values
- **`noise_impact_by_epsilon.png`**: Analysis of noise distribution by epsilon
- **`epsilon_comparison_summary.csv`**: Tabular comparison of all tested values

The parameter sweep tests epsilon values [0.1, 0.5, 1.0, 2.0, 5.0, 10.0] and helps you understand:
- How much noise is added at each privacy level
- Impact on model fit quality (R²) and prediction error (MAE)
- Parameter accuracy compared to ground truth
- Recommended epsilon ranges for your needs

### Interactive Exploration

For hands-on learning with real-time parameter adjustment, use the Jupyter notebook:

```bash
jupyter notebook examples/interactive_privacy_tuning.ipynb
```

**Features:**
- **Interactive sliders** to adjust epsilon and see immediate impacts
- **Noise visualization** showing how privacy affects data quality
- **Side-by-side model comparison** for two different epsilon values
- **Guided exercises** to build intuition about the privacy-utility tradeoff
- **Best practices summary** with decision frameworks

**Recommended learning path:**
1. Start with the parameter sweep to see the full landscape
2. Explore the interactive notebook to develop hands-on intuition
3. Read the decision guide (below) for structured selection framework
4. Document your chosen epsilon with clear justification

### Decision Guide

For comprehensive guidance on selecting epsilon values, see:

```
docs/privacy_parameter_guide.md
```

This guide includes:
- **Decision trees** based on data type, regulations, and dataset size
- **Case studies** with real-world epsilon selections and rationales
- **Reference tables** mapping epsilon ranges to use cases and privacy levels
- **Validation procedures** for testing epsilon choices
- **Documentation templates** for regulatory compliance

### Choosing Your Epsilon Value

The privacy parameter (epsilon) controls the privacy-utility tradeoff:
- **Lower epsilon** = More privacy, more noise, less accuracy
- **Higher epsilon** = Less privacy, less noise, more accuracy

**General Guidelines:**

| Epsilon Range | Privacy Level | Typical Use Cases |
|---------------|---------------|-------------------|
| ε ≤ 1.0 | High Privacy | Sensitive health/financial data, GDPR compliance |
| ε = 1.0-3.0 | Moderate Privacy | **Recommended for most marketing MMM applications** |
| ε ≥ 5.0 | Low Privacy | Less sensitive aggregated data, focus on utility |

**For most marketing analytics, we recommend starting with ε = 1.0-3.0** and adjusting based on:
- Your specific privacy requirements and regulations
- Dataset size and characteristics
- Acceptable accuracy thresholds
- Results from the parameter sweep tool

Always document your epsilon choice with clear justification for transparency and compliance.

## Business Impact

By modeling marketing spend at an aggregated level and applying privacy-preserving techniques, this toolkit allows companies to:

- **Optimize media budgets** without violating consumer trust or privacy regulations
- **Identify the most efficient channels** and reallocate budgets to maximize ROI
- **Comply with privacy laws** (GDPR, CCPA, etc.) while maintaining analytical capabilities
- **Build trust** with customers through transparent, privacy-respecting analytics
- **Future-proof** marketing analytics as privacy regulations continue to evolve

This approach represents a paradigm shift toward a privacy-first engineering culture where systems are more resilient to data breaches, less susceptible to certain adversarial attacks, and more likely to maintain public trust—crucial for long-term AI adoption and success.

## Advanced Topics

### Composition and Privacy Budgets

When making multiple queries on the same dataset, privacy budgets compose:
- **Sequential composition**: If you run k queries each with ε/k, total budget is ε
- The toolkit implements this by splitting ε across channels and metrics
- Be mindful of privacy budget depletion when running multiple analyses

### Privacy Parameter Tuning and Selection

The toolkit provides comprehensive tools for selecting appropriate epsilon values:

**Systematic Testing:**
- Use `examples/privacy_parameter_sweep.py` to compare multiple epsilon values systematically
- Generates comparison visualizations and CSV export of metrics across epsilon values
- Helps identify the lowest epsilon (highest privacy) that meets your accuracy requirements

**Interactive Exploration:**
- The Jupyter notebook `examples/interactive_privacy_tuning.ipynb` provides hands-on exploration
- Adjust epsilon with sliders and see real-time impacts on noise and model quality
- Compare two epsilon values side-by-side to understand tradeoffs

**Decision Framework:**
- See `docs/privacy_parameter_guide.md` for comprehensive selection guidance
- Includes decision trees based on data sensitivity, regulations, and dataset characteristics
- Case studies showing real-world epsilon selections with documented rationales
- Documentation templates for regulatory compliance and audit trails

**Key Considerations:**
1. **Data Sensitivity**: Health/financial data requires ε ≤ 1.0; marketing aggregates can use ε = 1.0-5.0
2. **Regulatory Requirements**: GDPR typically requires ε ≤ 1.0; CCPA allows ε ≤ 3.0
3. **Dataset Size**: Larger datasets can tolerate lower epsilon; smaller datasets may need higher epsilon
4. **Utility Requirements**: High-stakes decisions may require higher epsilon for accuracy

**Best Practice**: Always test multiple epsilon values, document your selection rationale, and obtain privacy officer approval before production deployment.

### Extending the Toolkit

The modular design allows for:
- **Custom privacy mechanisms**: Implement Gaussian mechanism or advanced variants
- **Alternative models**: Plug in different statistical models (e.g., time-series models)
- **Additional PETs**: Combine with techniques like federated learning or secure aggregation
- **Enhanced visualizations**: Add custom plots for privacy-utility analysis

## Contributing

We welcome contributions that extend the toolkit's privacy-preserving capabilities, improve documentation, or add new analytical features. Please ensure any changes maintain the privacy-first philosophy and include appropriate documentation of privacy implications.

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## References and Further Reading

- Dwork, C., & Roth, A. (2014). "The Algorithmic Foundations of Differential Privacy"
- Differential Privacy: A Primer for a Non-technical Audience (NIST)
- Google's Differential Privacy Library: https://github.com/google/differential-privacy
- Privacy-Preserving Machine Learning: Threats and Solutions (IEEE)

## Acknowledgments

This toolkit demonstrates principles from the AI safety and privacy-preserving machine learning research communities. It is designed as an educational resource and proof-of-concept for responsible AI development.

---

**Note**: This is a proof-of-concept toolkit for educational and research purposes. For production deployments, consult with privacy and security experts, conduct thorough privacy audits, and ensure compliance with applicable regulations.
