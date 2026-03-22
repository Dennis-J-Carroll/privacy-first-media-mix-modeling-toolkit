"""
Visualization functions for MMM results.

This module generates all output charts and plots for model analysis.
"""

import os
from typing import Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .config import CONFIG, OUTPUT_DIR, CHANNEL_COLORS, CONTRIBUTION_COLORS
from .core import geometric_adstock, hill_function


def generate_plots(df: pd.DataFrame, fitted_params: Dict) -> None:
    """
    Generates and saves all output charts.

    Creates three visualization files:
    1. response_curves.png - Channel saturation curves
    2. predicted_vs_actual.png - Model fit quality
    3. contribution_breakdown.png - Revenue attribution over time

    **Issue #7 Fix:** Now uses explicit channel colors from CHANNEL_COLORS
    instead of matplotlib's default color cycle.

    Args:
        df: DataFrame with marketing data
        fitted_params: Fitted model parameters
    """

    # --- 1. Response Curves Plot ---
    plt.figure(figsize=(12, 7))
    for ch in CONFIG["channels"]:
        params = fitted_params[ch]
        spend_range = np.linspace(0, df[f"spend_{ch}"].max() * 1.2, 100)

        # Calculate revenue contribution (without adstock for curve visualization)
        revenue_contribution = hill_function(
            spend_range,
            params["hill_alpha"],
            params["hill_K"],
            params["hill_beta"]
        )

        # **Issue #7 Fix:** Use explicit channel color
        plt.plot(
            spend_range,
            revenue_contribution,
            label=f"{ch} Response Curve",
            color=CHANNEL_COLORS[ch],
            linewidth=2
        )

    plt.title("Fitted Channel Response Curves (Saturation)", fontsize=16)
    plt.xlabel("Weekly Spend ($)", fontsize=12)
    plt.ylabel("Expected Revenue Contribution ($)", fontsize=12)
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "response_curves.png"))
    plt.close()
    print("Saved response curves plot.")

    # --- 2. Predicted vs. Actual Revenue Plot ---
    # Re-calculate predicted revenue using fitted params
    predicted_revenue = CONFIG["base_revenue"] + \
                        CONFIG["true_params"]["seasonality_amplitude"] * \
                        np.sin(2 * np.pi * df['week'] / CONFIG["true_params"]["seasonality_period"])
    predicted_revenue += df["promotions"] * fitted_params["promo_effect"]

    for ch in CONFIG["channels"]:
        params = fitted_params[ch]
        adstocked_spend = geometric_adstock(
            df[f"spend_{ch}"].values,
            params["adstock_decay"]
        )
        predicted_revenue += hill_function(
            adstocked_spend,
            params["hill_alpha"],
            params["hill_K"],
            params["hill_beta"]
        )

    plt.figure(figsize=(12, 6))
    plt.plot(df['week'], df['revenue'], label="Actual Revenue", alpha=0.8)
    plt.plot(df['week'], predicted_revenue, label="Predicted Revenue", linestyle='--')
    plt.title("Model Fit: Predicted vs. Actual Revenue", fontsize=16)
    plt.xlabel("Week", fontsize=12)
    plt.ylabel("Revenue ($)", fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "predicted_vs_actual.png"))
    plt.close()
    print("Saved predicted vs. actual plot.")

    # --- 3. Contribution Breakdown Plot ---
    contributions = pd.DataFrame(index=df['week'])
    contributions['Base & Seasonality'] = CONFIG["base_revenue"] + \
                                          CONFIG["true_params"]["seasonality_amplitude"] * \
                                          np.sin(2 * np.pi * df['week'] / CONFIG["true_params"]["seasonality_period"])
    contributions['Promotions'] = df["promotions"] * fitted_params["promo_effect"]

    for ch in CONFIG["channels"]:
        params = fitted_params[ch]
        adstocked_spend = geometric_adstock(
            df[f"spend_{ch}"].values,
            params["adstock_decay"]
        )
        contributions[ch] = hill_function(
            adstocked_spend,
            params["hill_alpha"],
            params["hill_K"],
            params["hill_beta"]
        )

    plt.figure(figsize=(14, 8))

    # **Issue #7 Fix:** Use CONTRIBUTION_COLORS for stackplot
    colors_ordered = [CONTRIBUTION_COLORS[col] for col in contributions.columns]
    plt.stackplot(
        df['week'],
        contributions.T,
        labels=contributions.columns,
        colors=colors_ordered,
        alpha=0.8
    )

    plt.plot(df['week'], df['revenue'], label="Actual Revenue", color='black', linestyle=':')
    plt.title("Weekly Revenue Contribution Breakdown", fontsize=16)
    plt.xlabel("Week", fontsize=12)
    plt.ylabel("Revenue ($)", fontsize=12)
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "contribution_breakdown.png"))
    plt.close()
    print("Saved contribution breakdown plot.")
