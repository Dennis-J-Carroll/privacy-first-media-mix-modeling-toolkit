#!/usr/bin/env python3
"""
advanced_mmm.py
=================

This script provides an advanced sandbox example of a Privacy-First Media-Mix
Modeling (MMM) analysis. It demonstrates how to build robust marketing analytics
while preserving individual privacy through differential privacy mechanisms.

Key Features:

1.  **Differential Privacy:** Implements formal privacy guarantees using the
    Laplace mechanism to add calibrated noise to aggregated marketing data,
    ensuring individual-level information cannot be extracted.

2.  **Adstock (Carryover Effect):** The impact of advertising lingers and
    decays over subsequent weeks. This is modeled using a geometric decay
    function.

3.  **Diminishing Returns (Saturation):** Each additional dollar spent in a
    channel yields progressively less return. This is modeled using the
    Hill function, which creates a characteristic S-shaped response curve.

4.  **Control Variables:** It includes other business drivers like promotions
    and seasonality to avoid misattributing their effects to marketing spend.

5.  **Privacy-Utility Tradeoff:** Demonstrates how the privacy parameter (epsilon)
    affects both privacy guarantees and model accuracy.

The script first generates synthetic data based on these principles with known
"true" parameters. It then applies differential privacy to the aggregated data
before using a non-linear least squares optimizer to fit the model.

This approach demonstrates alignment-by-design principles from AI Safety,
integrating privacy constraints into the modeling process from the ground up.

Requirements:
* Python 3.8+
* pandas, numpy, matplotlib, scipy

Outputs (in `mmm_output_advanced` folder):
1. `mmm_summary.csv`: Table of estimated vs. true parameters and the
   calculated marginal ROI (mROI) for each channel.
2. `response_curves.png`: The S-shaped saturation curves for each channel,
   showing how revenue contribution changes with spend.
3. `predicted_vs_actual.png`: A plot comparing the model's predicted revenue
   against the actual generated revenue.
4. `contribution_breakdown.png`: A stacked area chart showing how much each
   channel, promotions, and seasonality contributed to revenue each week.
5. `privacy_utility_tradeoff.png`: Visualization of how privacy parameter (epsilon)
   affects model accuracy.

Version 2.0.0 Changes:
- Refactored into modular mmm package
- Fixed mROI calculation bug (Issue #1: added chain rule multiplier)
- Fixed clipping bias (Issue #9: shifted Laplace mechanism)
- Added convergence checks (Issue #2)
- Fixed channel colors (Issue #7)
- Removed delta parameter (Issue #3: not used in Laplace mechanism)
"""

import os
import random

import numpy as np
import pandas as pd

# Import from new mmm package
from mmm import (
    MMMConfig,
    OUTPUT_DIR,
    generate_weekly_data,
    apply_differential_privacy,
    fit_model,
    calculate_marginal_roi,
    generate_plots
)

# Create default configuration
config = MMMConfig()

# Set random seeds for reproducibility
random.seed(config.random_seed)
np.random.seed(config.random_seed)


def main():
    """Main function to run the MMM simulation and analysis."""
    print("=" * 70)
    print("PRIVACY-FIRST MEDIA MIX MODELING TOOLKIT v2.0.0")
    print("=" * 70)

    print("\n1. Generating synthetic data with known ground truth...")
    df_original = generate_weekly_data(config)

    print("\n2. Applying privacy-preserving mechanisms...")
    df = apply_differential_privacy(df_original, config.epsilon, config)

    print("\n3. Fitting the advanced MMM to the privatized data...")
    results = fit_model(df, config)
    fitted_params = results['fitted_params']

    print(f"\n   Convergence: {'✓ Success' if results['convergence']['success'] else '✗ Failed'}")
    print(f"   R² = {results['r_squared']:.3f}")
    print(f"   MAE = ${results['mae']:.2f}")

    print("\n4. Analyzing results and calculating mROI...")
    summary_data = []
    for ch in config.channels:
        true = getattr(config.true_params, ch)
        fitted = fitted_params[ch]
        avg_spend = df[f"spend_{ch}"].mean()

        # **Issue #1 Fix:** mROI now includes chain rule multiplier
        mroi = calculate_marginal_roi(
            avg_spend, fitted['adstock_decay'], fitted['hill_alpha'],
            fitted['hill_K'], fitted['hill_beta']
        )

        summary_data.append({
            "channel": ch,
            "parameter": "adstock_decay",
            "true_value": true.adstock_decay,
            "fitted_value": fitted["adstock_decay"]
        })
        summary_data.append({
            "channel": ch,
            "parameter": "hill_alpha (shape)",
            "true_value": true.hill_alpha,
            "fitted_value": fitted["hill_alpha"]
        })
        summary_data.append({
            "channel": ch,
            "parameter": "hill_K (saturation_point)",
            "true_value": true.hill_K,
            "fitted_value": fitted["hill_K"]
        })
        summary_data.append({
            "channel": ch,
            "parameter": "hill_beta (max_effect)",
            "true_value": true.hill_beta,
            "fitted_value": fitted["hill_beta"]
        })
        summary_data.append({
            "channel": ch,
            "parameter": "mROI_at_avg_spend",
            "true_value": None,
            "fitted_value": mroi
        })

    summary_df = pd.DataFrame(summary_data)
    summary_path = os.path.join(OUTPUT_DIR, "mmm_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"\nMMM summary written to {summary_path}")
    print(summary_df)

    print("\n5. Generating visualizations...")
    generate_plots(df, fitted_params, config)

    print("\n" + "=" * 70)
    print("PRIVACY-FIRST MMM ANALYSIS COMPLETE")
    print("=" * 70)
    # **Issue #3 Fix:** Removed delta (not used in Laplace mechanism)
    print(f"\nPrivacy Guarantee: ε={config.epsilon:.2f}-differential privacy (Laplace mechanism)")
    print(f"All outputs saved to: {OUTPUT_DIR}/")
    print("\nThis analysis demonstrates responsible AI development by:")
    print("  • Using only aggregated data (privacy by design)")
    print("  • Applying differential privacy for formal guarantees")
    print("  • Balancing privacy protection with analytical utility")
    print("\nVersion 2.0.0 Improvements:")
    print("  • Fixed mROI calculation (+chain rule for adstock)")
    print("  • Fixed clipping bias (shifted Laplace mechanism)")
    print("  • Added convergence checks and warnings")
    print("  • Improved channel visualization colors")
    print("=" * 70)


if __name__ == "__main__":
    main()
