#!/usr/bin/env python3
"""
Privacy Parameter Sweep for Media Mix Modeling
===============================================

This script systematically compares multiple epsilon values to visualize
the privacy-utility tradeoff in differential privacy for MMM.

It generates:
1. privacy_utility_tradeoff.png - Main visualization showing epsilon vs accuracy
2. parameter_accuracy_comparison.png - Parameter recovery heatmap
3. noise_impact_by_epsilon.png - Noise distribution analysis
4. epsilon_comparison_summary.csv - Tabular results for all epsilon values

Usage:
    python examples/privacy_parameter_sweep.py
"""

import os
import sys

# Add parent directory to path to import advanced_mmm
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from advanced_mmm import (
    generate_weekly_data,
    apply_differential_privacy,
    fit_model,
    CONFIG,
    geometric_adstock,
    hill_function
)

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

EPSILON_VALUES = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
OUTPUT_DIR = "mmm_privacy_comparisons"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------

def calculate_model_predictions(df: pd.DataFrame, fitted_params: dict) -> np.ndarray:
    """
    Calculate predicted revenue using fitted parameters.
    """
    predicted_revenue = CONFIG["base_revenue"] + \
                        CONFIG["true_params"]["seasonality_amplitude"] * \
                        np.sin(2 * np.pi * df['week'] / CONFIG["true_params"]["seasonality_period"])

    predicted_revenue += df["promotions"] * fitted_params["promo_effect"]

    for ch in CONFIG["channels"]:
        params = fitted_params[ch]
        adstocked_spend = geometric_adstock(df[f"spend_{ch}"].values, params["adstock_decay"])
        predicted_revenue += hill_function(
            adstocked_spend,
            params["hill_alpha"],
            params["hill_K"],
            params["hill_beta"]
        )

    return predicted_revenue


def calculate_metrics(df: pd.DataFrame, fitted_params: dict, private_data: pd.DataFrame) -> dict:
    """
    Calculate accuracy metrics for a fitted model.

    Returns:
        Dictionary with R², MAE, parameter errors, and noise statistics
    """
    # Calculate predictions
    predictions = calculate_model_predictions(private_data, fitted_params)

    # R-squared
    ss_res = np.sum((private_data["revenue"] - predictions) ** 2)
    ss_tot = np.sum((private_data["revenue"] - private_data["revenue"].mean()) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    # Mean Absolute Error
    mae = np.mean(np.abs(private_data["revenue"] - predictions))

    # Parameter errors (compared to ground truth)
    param_errors = []
    for ch in CONFIG["channels"]:
        true_params = CONFIG["true_params"][ch]
        fitted = fitted_params[ch]

        # Normalized errors for each parameter
        for param_name in ["adstock_decay", "hill_alpha", "hill_K", "hill_beta"]:
            true_val = true_params[param_name]
            fitted_val = fitted[param_name]
            if true_val != 0:
                error = abs(fitted_val - true_val) / abs(true_val)
            else:
                error = abs(fitted_val)
            param_errors.append(error)

    # Promo effect error
    true_promo = CONFIG["true_params"]["promo_effect"]
    fitted_promo = fitted_params["promo_effect"]
    promo_error = abs(fitted_promo - true_promo) / abs(true_promo) if true_promo != 0 else abs(fitted_promo)
    param_errors.append(promo_error)

    mean_param_error = np.mean(param_errors)

    # Noise statistics (comparing private vs original data)
    noise_stats = []
    for ch in CONFIG["channels"]:
        spend_col = f"spend_{ch}"
        noise = (private_data[spend_col] - df[spend_col]).abs()
        noise_stats.append(noise.mean())

    revenue_noise = (private_data["revenue"] - df["revenue"]).abs()
    noise_stats.append(revenue_noise.mean())

    mean_noise = np.mean(noise_stats)
    max_noise = np.max(noise_stats)

    return {
        "r_squared": r_squared,
        "mae": mae,
        "mean_parameter_error": mean_param_error,
        "mean_noise_level": mean_noise,
        "max_noise_level": max_noise,
        "parameter_errors": param_errors
    }


def privacy_guarantee_description(epsilon: float) -> str:
    """
    Return a human-readable description of privacy level.
    """
    if epsilon <= 0.5:
        return "Very High Privacy"
    elif epsilon <= 1.0:
        return "High Privacy"
    elif epsilon <= 3.0:
        return "Moderate Privacy"
    elif epsilon <= 5.0:
        return "Low Privacy"
    else:
        return "Very Low Privacy"


# -----------------------------------------------------------------------------
# Main Comparison Logic
# -----------------------------------------------------------------------------

def run_epsilon_comparison():
    """
    Run MMM with multiple epsilon values and collect metrics.
    """
    print("=" * 80)
    print("Privacy Parameter Sweep for Media Mix Modeling")
    print("=" * 80)
    print(f"\nTesting epsilon values: {EPSILON_VALUES}")
    print(f"Output directory: {OUTPUT_DIR}\n")

    # Generate base data once for fair comparison
    print("Generating synthetic data...")
    base_data = generate_weekly_data()
    print(f"Generated {len(base_data)} weeks of data")

    # Store results
    results = []
    fitted_params_by_epsilon = {}

    # Run comparison for each epsilon
    for epsilon in EPSILON_VALUES:
        print(f"\n{'=' * 80}")
        print(f"Testing ε = {epsilon}")
        print(f"{'=' * 80}")

        # Apply differential privacy with current epsilon
        private_data = apply_differential_privacy(base_data, epsilon)

        # Fit model
        fitted_params = fit_model(private_data)
        fitted_params_by_epsilon[epsilon] = fitted_params

        # Calculate metrics
        metrics = calculate_metrics(base_data, fitted_params, private_data)

        # Store results
        result = {
            "epsilon": epsilon,
            "privacy_level": privacy_guarantee_description(epsilon),
            "r_squared": metrics["r_squared"],
            "mae": metrics["mae"],
            "mean_parameter_error": metrics["mean_parameter_error"],
            "mean_noise_level": metrics["mean_noise_level"],
            "max_noise_level": metrics["max_noise_level"]
        }
        results.append(result)

        print(f"\nResults for ε = {epsilon}:")
        print(f"  Privacy Level: {result['privacy_level']}")
        print(f"  R²: {result['r_squared']:.4f}")
        print(f"  MAE: {result['mae']:.2f}")
        print(f"  Mean Parameter Error: {result['mean_parameter_error']:.4f}")
        print(f"  Mean Noise Level: {result['mean_noise_level']:.2f}")

    # Convert to DataFrame
    results_df = pd.DataFrame(results)

    return results_df, fitted_params_by_epsilon, base_data


# -----------------------------------------------------------------------------
# Visualization Functions
# -----------------------------------------------------------------------------

def create_privacy_utility_tradeoff_plot(results_df: pd.DataFrame):
    """
    Create the main privacy-utility tradeoff visualization.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: R² vs Epsilon
    ax1.plot(results_df["epsilon"], results_df["r_squared"],
             marker='o', linewidth=2, markersize=8, color='#2E86AB')
    ax1.set_xlabel("Privacy Parameter (ε)", fontsize=12, fontweight='bold')
    ax1.set_ylabel("R² (Model Fit Quality)", fontsize=12, fontweight='bold')
    ax1.set_title("Privacy vs Model Accuracy", fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log')

    # Add privacy level annotations
    for idx, row in results_df.iterrows():
        if idx % 2 == 0:  # Annotate every other point to avoid crowding
            ax1.annotate(row['privacy_level'],
                        xy=(row['epsilon'], row['r_squared']),
                        xytext=(10, -15), textcoords='offset points',
                        fontsize=8, alpha=0.7)

    # Plot 2: MAE vs Epsilon
    ax2.plot(results_df["epsilon"], results_df["mae"],
             marker='s', linewidth=2, markersize=8, color='#A23B72')
    ax2.set_xlabel("Privacy Parameter (ε)", fontsize=12, fontweight='bold')
    ax2.set_ylabel("Mean Absolute Error", fontsize=12, fontweight='bold')
    ax2.set_title("Privacy vs Prediction Error", fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale('log')

    plt.tight_layout()

    output_path = os.path.join(OUTPUT_DIR, "privacy_utility_tradeoff.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved: {output_path}")
    plt.close()


def create_parameter_accuracy_plot(fitted_params_by_epsilon: dict):
    """
    Create heatmap showing parameter recovery accuracy by epsilon.
    """
    # Prepare data for heatmap
    param_names = []
    for ch in CONFIG["channels"]:
        param_names.extend([
            f"{ch}_decay",
            f"{ch}_alpha",
            f"{ch}_K",
            f"{ch}_beta"
        ])
    param_names.append("promo_effect")

    accuracy_matrix = []
    epsilon_labels = []

    for epsilon in sorted(fitted_params_by_epsilon.keys()):
        fitted = fitted_params_by_epsilon[epsilon]
        accuracies = []

        for ch in CONFIG["channels"]:
            true_params = CONFIG["true_params"][ch]
            fitted_ch = fitted[ch]

            for param_name in ["adstock_decay", "hill_alpha", "hill_K", "hill_beta"]:
                true_val = true_params[param_name]
                fitted_val = fitted_ch[param_name]
                # Calculate percentage accuracy (100% = perfect match)
                if true_val != 0:
                    accuracy = 100 * (1 - abs(fitted_val - true_val) / abs(true_val))
                else:
                    accuracy = 100 if fitted_val == 0 else 0
                accuracies.append(max(0, accuracy))  # Clip at 0%

        # Promo effect
        true_promo = CONFIG["true_params"]["promo_effect"]
        fitted_promo = fitted["promo_effect"]
        promo_accuracy = 100 * (1 - abs(fitted_promo - true_promo) / abs(true_promo))
        accuracies.append(max(0, promo_accuracy))

        accuracy_matrix.append(accuracies)
        epsilon_labels.append(f"ε={epsilon}")

    # Create heatmap
    fig, ax = plt.subplots(figsize=(14, 6))

    sns.heatmap(accuracy_matrix, annot=True, fmt='.1f', cmap='RdYlGn',
                xticklabels=param_names, yticklabels=epsilon_labels,
                cbar_kws={'label': 'Parameter Recovery Accuracy (%)'}, ax=ax)

    ax.set_title("Parameter Recovery Accuracy by Epsilon Value",
                fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel("Model Parameters", fontsize=12, fontweight='bold')
    ax.set_ylabel("Privacy Level", fontsize=12, fontweight='bold')

    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    output_path = os.path.join(OUTPUT_DIR, "parameter_accuracy_comparison.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def create_noise_impact_plot(results_df: pd.DataFrame):
    """
    Create visualization showing noise impact by epsilon.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(results_df))
    width = 0.35

    bars1 = ax.bar(x - width/2, results_df["mean_noise_level"], width,
                   label='Mean Noise', color='#F18F01', alpha=0.8)
    bars2 = ax.bar(x + width/2, results_df["max_noise_level"], width,
                   label='Max Noise', color='#C73E1D', alpha=0.8)

    ax.set_xlabel("Privacy Parameter (ε)", fontsize=12, fontweight='bold')
    ax.set_ylabel("Noise Level", fontsize=12, fontweight='bold')
    ax.set_title("Noise Impact by Epsilon Value", fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f"{eps}" for eps in results_df["epsilon"]])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Add privacy level labels
    for i, privacy_level in enumerate(results_df["privacy_level"]):
        ax.text(i, -50, privacy_level, ha='center', fontsize=8,
                rotation=0, alpha=0.7)

    plt.tight_layout()

    output_path = os.path.join(OUTPUT_DIR, "noise_impact_by_epsilon.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


# -----------------------------------------------------------------------------
# Main Execution
# -----------------------------------------------------------------------------

def main():
    """
    Main execution function.
    """
    # Run comparison
    results_df, fitted_params_by_epsilon, base_data = run_epsilon_comparison()

    # Generate visualizations
    print(f"\n{'=' * 80}")
    print("Generating Visualizations")
    print(f"{'=' * 80}")

    create_privacy_utility_tradeoff_plot(results_df)
    create_parameter_accuracy_plot(fitted_params_by_epsilon)
    create_noise_impact_plot(results_df)

    # Export CSV summary
    csv_path = os.path.join(OUTPUT_DIR, "epsilon_comparison_summary.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"✓ Saved: {csv_path}")

    # Print interpretation guide
    print(f"\n{'=' * 80}")
    print("Interpretation Guide")
    print(f"{'=' * 80}")
    print("\nEpsilon Selection Guidelines:")
    print("  • ε ≤ 1.0:  High privacy, suitable for sensitive data")
    print("  • ε = 1-3:  Balanced approach for most marketing applications")
    print("  • ε ≥ 5.0:  Lower privacy, higher utility for less sensitive data")
    print("\nKey Findings:")

    best_balance = results_df.iloc[(results_df["r_squared"] - results_df["r_squared"].max() * 0.95).abs().argmin()]
    print(f"  • Recommended starting point: ε = {best_balance['epsilon']}")
    print(f"    - Privacy Level: {best_balance['privacy_level']}")
    print(f"    - R²: {best_balance['r_squared']:.4f}")
    print(f"    - MAE: {best_balance['mae']:.2f}")

    print(f"\nAll outputs saved to: {OUTPUT_DIR}/")
    print("=" * 80)


if __name__ == "__main__":
    main()
