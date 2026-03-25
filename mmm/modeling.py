"""
Model fitting and optimization for Media Mix Modeling.

This module handles the non-linear optimization to fit MMM parameters
to observed data.
"""

import warnings
from typing import Dict

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .config import MMMConfig
from .core import geometric_adstock, hill_function


def objective_function(params: np.ndarray, df: pd.DataFrame, config: MMMConfig) -> float:
    """
    Objective function to minimize: Sum of Squared Errors (SSE).

    Calculates the SSE between predicted and actual revenue given
    a set of parameters.

    Args:
        params: Parameter array containing:
            - 4 parameters per channel (adstock_decay, hill_alpha, hill_K, hill_beta)
            - 1 promotion effect parameter
        df: DataFrame with spend and revenue data
        config: MMMConfig instance

    Returns:
        Sum of squared errors (lower is better)
    """
    num_channels = len(config.channels)

    # Unpack parameters
    # 4 params per channel (adstock_decay, hill_alpha, hill_K, hill_beta)
    # + 1 for promo_effect
    channel_params = np.array(params[:num_channels * 4]).reshape((num_channels, 4))
    promo_effect = params[num_channels * 4]

    # Base revenue + seasonality (using known ground truth for simplicity)
    predicted_revenue = config.base_revenue + \
                        config.true_params.seasonality_amplitude * \
                        np.sin(2 * np.pi * df['week'] / config.true_params.seasonality_period)

    predicted_revenue += df["promotions"] * promo_effect

    # Add channel contributions
    for i, ch in enumerate(config.channels):
        adstock_decay, hill_alpha, hill_k, hill_beta = channel_params[i]
        adstocked_spend = geometric_adstock(df[f"spend_{ch}"].values, adstock_decay)
        predicted_revenue += hill_function(adstocked_spend, hill_alpha, hill_k, hill_beta)

    # Calculate Sum of Squared Errors (SSE)
    error = np.sum((df["revenue"] - predicted_revenue)**2)
    return error


def fit_model(df: pd.DataFrame, config: MMMConfig = None) -> Dict:
    """
    Fits the non-linear MMM using scipy.optimize.minimize.

    **Issue #2 Fix:** Now includes convergence checks and warnings.

    Args:
        df: DataFrame with marketing data
        config: MMMConfig instance. If None, uses default configuration.

    Returns:
        Dictionary with:
        - fitted_params: Estimated parameters for each channel
        - convergence: Convergence status and diagnostics
        - predictions: Model predictions
        - r_squared: R² goodness-of-fit metric
        - mae: Mean absolute error
        - sse: Sum of squared errors
    """
    if config is None:
        config = MMMConfig()

    num_channels = len(config.channels)

    # Define bounds to guide optimizer to plausible values
    # (decay, alpha, K, beta) per channel + promo_effect
    bounds = []
    for _ in config.channels:
        bounds.extend([
            (0.0, 0.9),      # adstock_decay
            (1.0, 5.0),      # hill_alpha
            (5000, 50000),   # hill_K
            (5000, 50000),   # hill_beta
        ])
    bounds.append((0, 20000))  # promo_effect

    # Initial guesses for parameters
    initial_guesses = np.array([0.3, 2, 15000, 20000] * num_channels + [5000])

    print("Fitting model... This may take a moment.")
    result = minimize(
        objective_function,
        initial_guesses,
        args=(df, config),
        bounds=bounds,
        method='L-BFGS-B',
        options={'maxiter': 1000, 'ftol': 1e-6}
    )
    print("Fitting complete.")

    # **Issue #2: Convergence Checks**
    convergence_info = {
        'success': result.success,
        'message': result.message,
        'iterations': result.nit,
        'final_loss': result.fun,
    }

    # Warn on convergence issues
    if not result.success:
        warnings.warn(
            f"⚠️ Optimization did not converge: {result.message}\n"
            f"Results may be unreliable. Consider:\n"
            f"  - Increasing epsilon (less privacy noise)\n"
            f"  - Increasing num_weeks (more data)\n"
            f"  - Adjusting initial guesses or bounds",
            UserWarning
        )

    # Check for boundary hits
    fitted_params_list = result.x
    param_names = []
    for ch in config.channels:
        param_names.extend([
            f"{ch}_adstock_decay",
            f"{ch}_hill_alpha",
            f"{ch}_hill_K",
            f"{ch}_hill_beta"
        ])
    param_names.append("promo_effect")

    for i, (param_val, (lower, upper)) in enumerate(zip(fitted_params_list, bounds)):
        if abs(param_val - lower) < 1e-6 or abs(param_val - upper) < 1e-6:
            warnings.warn(
                f"⚠️ Parameter '{param_names[i]}' hit boundary: {param_val:.4f} "
                f"(bounds: [{lower}, {upper}]). Consider adjusting bounds.",
                UserWarning
            )

    # Unpack fitted parameters into structured dictionary
    fitted_params = {}
    for i, ch in enumerate(config.channels):
        param_idx = i * 4
        fitted_params[ch] = {
            "adstock_decay": fitted_params_list[param_idx],
            "hill_alpha": fitted_params_list[param_idx + 1],
            "hill_K": fitted_params_list[param_idx + 2],
            "hill_beta": fitted_params_list[param_idx + 3],
        }
    fitted_params["promo_effect"] = fitted_params_list[num_channels * 4]

    # Calculate predictions for model evaluation
    predicted_revenue = config.base_revenue + \
                        config.true_params.seasonality_amplitude * \
                        np.sin(2 * np.pi * df['week'].values / config.true_params.seasonality_period)
    predicted_revenue += df["promotions"].values * fitted_params["promo_effect"]

    for ch in config.channels:
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

    # Calculate model metrics
    actual_revenue = df["revenue"].values

    # R-squared
    ss_res = np.sum((actual_revenue - predicted_revenue)**2)
    ss_tot = np.sum((actual_revenue - actual_revenue.mean())**2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    # Mean Absolute Error
    mae = np.mean(np.abs(actual_revenue - predicted_revenue))

    # Store predictions in fitted_params for convenience
    fitted_params['predictions'] = predicted_revenue

    return {
        'fitted_params': fitted_params,
        'convergence': convergence_info,
        'predictions': predicted_revenue,
        'r_squared': r_squared,
        'mae': mae,
        'sse': result.fun,
    }
