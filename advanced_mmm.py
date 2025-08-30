#!/usr/bin/env python3
"""
advanced_mmm.py
=================

This script provides an advanced sandbox example of a Media-Mix Modeling
(MMM) analysis, building upon a simple linear model. It simulates a more
realistic marketing environment by incorporating key concepts:

1.  **Adstock (Carryover Effect):** The impact of advertising lingers and
    decays over subsequent weeks. This is modeled using a geometric decay
    function.

2.  **Diminishing Returns (Saturation):** Each additional dollar spent in a
    channel yields progressively less return. This is modeled using the
    Hill function, which creates a characteristic S-shaped response curve.

3.  **Control Variables:** It includes other business drivers like promotions
    and seasonality to avoid misattributing their effects to marketing spend.

The script first generates synthetic data based on these principles with known
"true" parameters. It then uses a non-linear least squares optimizer
(`scipy.optimize.minimize`) to fit the complex model and estimate the
parameters, attempting to recover the ground truth.

This approach is significantly more robust and realistic than a simple linear
regression.

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
"""

import os
import random
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.signal import lfilter

# -----------------------------------------------------------------------------
# Configuration & Ground Truth Parameters
# -----------------------------------------------------------------------------

OUTPUT_DIR = "mmm_output_advanced"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Use a config dictionary for clarity
CONFIG = {
    "num_weeks": 104,  # 2 years of data for better model stability
    "channels": ["Shopify", "TikTok", "Meta"],
    "base_revenue": 5000,
    "noise_std": 1000,
    "random_seed": 42,
    # "Ground Truth" parameters for the data simulation
    "true_params": {
        "Shopify": {"adstock_decay": 0.5, "hill_alpha": 2.0, "hill_K": 10000, "hill_beta": 15000},
        "TikTok":  {"adstock_decay": 0.2, "hill_alpha": 2.5, "hill_K": 12000, "hill_beta": 25000},
        "Meta":    {"adstock_decay": 0.7, "hill_alpha": 3.0, "hill_K": 8000, "hill_beta": 10000},
        "promo_effect": 8000,
        "seasonality_amplitude": 4000,
        "seasonality_period": 52, # Yearly seasonality
    }
}

random.seed(CONFIG["random_seed"])
np.random.seed(CONFIG["random_seed"])

# -----------------------------------------------------------------------------
# Core MMM Transformation Functions
# -----------------------------------------------------------------------------

def geometric_adstock(x: np.ndarray, theta: float) -> np.ndarray:
    """
    Applies geometric adstock decay to a marketing channel's spend series.
    Uses a filter for computational efficiency.
    """
    # y[t] = x[t] + theta * y[t-1]
    # This is equivalent to an IIR filter with coefficients a=[1, -theta] and b=[1]
    return lfilter([1], [1, -theta], x)

def hill_function(x: np.ndarray, alpha: float, k: float, beta: float) -> np.ndarray:
    """
    Calculates the revenue contribution based on the S-shaped Hill function.
    Represents diminishing returns and saturation.
    """
    return beta * (x**alpha) / (k**alpha + x**alpha)


# -----------------------------------------------------------------------------
# Data Generation
# -----------------------------------------------------------------------------

def generate_weekly_data() -> pd.DataFrame:
    """Generates synthetic weekly data based on the defined "true" parameters."""
    weeks = np.arange(1, CONFIG["num_weeks"] + 1)
    df = pd.DataFrame({'week': weeks})

    # 1. Generate Spend & Control Variables
    for ch in CONFIG["channels"]:
        df[f"spend_{ch}"] = np.random.uniform(2000, 20000, CONFIG["num_weeks"])

    df["promotions"] = (np.random.rand(CONFIG["num_weeks"]) < 0.15).astype(int)
    
    # More complex seasonality (e.g., yearly cycle)
    seasonality = CONFIG["true_params"]["seasonality_amplitude"] * \
                  np.sin(2 * np.pi * weeks / CONFIG["true_params"]["seasonality_period"])

    # 2. Calculate "True" Revenue using MMM principles
    total_revenue = CONFIG["base_revenue"] + seasonality
    total_revenue += df["promotions"] * CONFIG["true_params"]["promo_effect"]

    for ch in CONFIG["channels"]:
        params = CONFIG["true_params"][ch]
        adstocked_spend = geometric_adstock(df[f"spend_{ch}"].values, params["adstock_decay"])
        channel_contribution = hill_function(adstocked_spend, params["hill_alpha"], params["hill_K"], params["hill_beta"])
        df[f"contribution_{ch}"] = channel_contribution  # Store for later comparison
        total_revenue += channel_contribution

    # 3. Add random noise
    df["revenue"] = total_revenue + np.random.normal(0, CONFIG["noise_std"], CONFIG["num_weeks"])
    df.loc[df["revenue"] < 0, "revenue"] = 0  # Revenue can't be negative
    
    return df


# -----------------------------------------------------------------------------
# Modeling
# -----------------------------------------------------------------------------

def objective_function(params: np.ndarray, df: pd.DataFrame) -> float:
    """
    Function to be minimized. Calculates the sum of squared errors between
    predicted and actual revenue given a set of parameters.
    """
    num_channels = len(CONFIG["channels"])
    
    # Unpack parameters
    # 4 params per channel (adstock_decay, hill_alpha, hill_K, hill_beta)
    # + 1 for promo_effect
    channel_params = np.array(params[:num_channels * 4]).reshape((num_channels, 4))
    promo_effect = params[num_channels * 4]
    
    # We assume base revenue and seasonality are known or modeled separately
    # Here, we'll use the true values for simplicity in this example
    predicted_revenue = CONFIG["base_revenue"] + \
                        CONFIG["true_params"]["seasonality_amplitude"] * \
                        np.sin(2 * np.pi * df['week'] / CONFIG["true_params"]["seasonality_period"])
                        
    predicted_revenue += df["promotions"] * promo_effect

    for i, ch in enumerate(CONFIG["channels"]):
        adstock_decay, hill_alpha, hill_k, hill_beta = channel_params[i]
        adstocked_spend = geometric_adstock(df[f"spend_{ch}"].values, adstock_decay)
        predicted_revenue += hill_function(adstocked_spend, hill_alpha, hill_k, hill_beta)

    # Calculate Sum of Squared Errors (SSE)
    error = np.sum((df["revenue"] - predicted_revenue)**2)
    return error

def fit_model(df: pd.DataFrame) -> Dict:
    """
    Fits the non-linear MMM using scipy.optimize.minimize.
    """
    num_channels = len(CONFIG["channels"])
    
    # Define bounds to guide the optimizer to plausible values
    # (decay, alpha, K, beta)
    bounds = []
    for _ in CONFIG["channels"]:
        bounds.extend([
            (0.0, 0.9),      # adstock_decay
            (1.0, 5.0),      # hill_alpha
            (5000, 50000),   # hill_K
            (5000, 50000),   # hill_beta
        ])
    bounds.append((0, 20000))  # promo_effect

    # Initial guesses for the parameters (can be random or heuristic)
    initial_guesses = np.array([0.3, 2, 15000, 20000] * num_channels + [5000])

    print("Fitting model... This may take a moment.")
    result = minimize(
        objective_function,
        initial_guesses,
        args=(df,),
        bounds=bounds,
        method='L-BFGS-B'
    )
    print("Fitting complete.")

    # Unpack and return results in a structured dictionary
    fitted_params_list = result.x
    fitted_params = {}
    for i, ch in enumerate(CONFIG["channels"]):
        param_idx = i * 4
        fitted_params[ch] = {
            "adstock_decay": fitted_params_list[param_idx],
            "hill_alpha": fitted_params_list[param_idx + 1],
            "hill_K": fitted_params_list[param_idx + 2],
            "hill_beta": fitted_params_list[param_idx + 3],
        }
    fitted_params["promo_effect"] = fitted_params_list[num_channels * 4]

    return fitted_params

# -----------------------------------------------------------------------------
# Post-Modeling Analysis & Visualization
# -----------------------------------------------------------------------------

def calculate_marginal_roi(spend: float, adstock_decay: float, hill_alpha: float, hill_k: float, hill_beta: float) -> float:
    """
    Calculates the marginal ROI (mROI) for the next dollar spent.
    This is the derivative of the Hill function w.r.t. spend.
    Note: A full derivation would also account for the adstock effect chain rule.
    This is a simplified version for a single time-step response.
    """
    adstocked_spend = spend / (1 - adstock_decay)  # Approximation of steady state adstocked spend
    
    numerator = hill_beta * hill_alpha * (hill_k**hill_alpha) * (adstocked_spend**(hill_alpha - 1))
    denominator = ((hill_k**hill_alpha) + (adstocked_spend**hill_alpha))**2
    
    # Avoid division by zero if denominator is tiny
    return numerator / denominator if denominator > 1e-9 else 0.0

def generate_plots(df: pd.DataFrame, fitted_params: Dict) -> None:
    """Generates and saves all output charts."""
    
    # --- 1. Response Curves Plot ---
    plt.figure(figsize=(12, 7))
    for ch in CONFIG["channels"]:
        params = fitted_params[ch]
        spend_range = np.linspace(0, df[f"spend_{ch}"].max() * 1.2, 100)
        # Assuming no adstock for curve visualization for simplicity
        revenue_contribution = hill_function(spend_range, params["hill_alpha"], params["hill_K"], params["hill_beta"])
        plt.plot(spend_range, revenue_contribution, label=f"{ch} Response Curve")

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
        adstocked_spend = geometric_adstock(df[f"spend_{ch}"].values, params["adstock_decay"])
        predicted_revenue += hill_function(adstocked_spend, params["hill_alpha"], params["hill_K"], params["hill_beta"])

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
        adstocked_spend = geometric_adstock(df[f"spend_{ch}"].values, params["adstock_decay"])
        contributions[ch] = hill_function(adstocked_spend, params["hill_alpha"], params["hill_K"], params["hill_beta"])

    plt.figure(figsize=(14, 8))
    plt.stackplot(df['week'], contributions.T, labels=contributions.columns)
    plt.plot(df['week'], df['revenue'], label="Actual Revenue", color='black', linestyle=':')
    plt.title("Weekly Revenue Contribution Breakdown", fontsize=16)
    plt.xlabel("Week", fontsize=12)
    plt.ylabel("Revenue ($)", fontsize=12)
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "contribution_breakdown.png"))
    plt.close()
    print("Saved contribution breakdown plot.")


def main():
    """Main function to run the MMM simulation and analysis."""
    print("1. Generating synthetic data with known ground truth...")
    df = generate_weekly_data()

    print("\n2. Fitting the advanced MMM to the data...")
    fitted_params = fit_model(df)
    
    print("\n3. Analyzing results and calculating mROI...")
    summary_data = []
    for ch in CONFIG["channels"]:
        true = CONFIG["true_params"][ch]
        fitted = fitted_params[ch]
        avg_spend = df[f"spend_{ch}"].mean()
        mroi = calculate_marginal_roi(
            avg_spend, fitted['adstock_decay'], fitted['hill_alpha'],
            fitted['hill_K'], fitted['hill_beta']
        )
        summary_data.append({
            "channel": ch,
            "parameter": "adstock_decay",
            "true_value": true["adstock_decay"],
            "fitted_value": fitted["adstock_decay"]
        })
        summary_data.append({
            "channel": ch,
            "parameter": "hill_alpha (shape)",
            "true_value": true["hill_alpha"],
            "fitted_value": fitted["hill_alpha"]
        })
        summary_data.append({
            "channel": ch,
            "parameter": "hill_K (saturation_point)",
            "true_value": true["hill_K"],
            "fitted_value": fitted["hill_K"]
        })
        summary_data.append({
            "channel": ch,
            "parameter": "hill_beta (max_effect)",
            "true_value": true["hill_beta"],
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

    print("\n4. Generating visualizations...")
    generate_plots(df, fitted_params)
    
    print("\n--- MMM Analysis Complete ---")


if __name__ == "__main__":
    main()
