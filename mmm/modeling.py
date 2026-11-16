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


def _predict_revenue(params: np.ndarray, df: pd.DataFrame, config: MMMConfig) -> np.ndarray:
    """Predicted revenue for a flat parameter array (see objective_function)."""
    num_channels = len(config.channels)
    channel_params = np.asarray(params[:num_channels * 4]).reshape((num_channels, 4))
    promo_effect = params[num_channels * 4]

    # Base revenue + seasonality (using known ground truth for simplicity)
    predicted = config.base_revenue + \
        config.true_params.seasonality_amplitude * \
        np.sin(2 * np.pi * df["week"].values / config.true_params.seasonality_period)
    predicted = predicted + df["promotions"].values * promo_effect

    for i, ch in enumerate(config.channels):
        adstock_decay, hill_alpha, hill_k, hill_beta = channel_params[i]
        adstocked_spend = geometric_adstock(df[f"spend_{ch}"].values, adstock_decay)
        predicted = predicted + hill_function(adstocked_spend, hill_alpha, hill_k, hill_beta)

    return predicted


def objective_function(params: np.ndarray, df: pd.DataFrame, config: MMMConfig) -> float:
    """
    Objective function to minimize: Sum of Squared Errors (SSE).

    Args:
        params: Parameter array containing:
            - 4 parameters per channel (adstock_decay, hill_alpha, hill_K, hill_beta)
            - 1 promotion effect parameter
        df: DataFrame with spend and revenue data
        config: MMMConfig instance

    Returns:
        Sum of squared errors (lower is better)
    """
    residuals = df["revenue"].values - _predict_revenue(params, df, config)
    return float(np.sum(residuals ** 2))


def fit_model(df: pd.DataFrame, config: MMMConfig = None, n_starts: int = 8) -> Dict:
    """
    Fits the non-linear MMM by bounded least squares (L-BFGS-B).

    The optimizer works on parameters rescaled to [0, 1] and on a rescaled
    loss. The raw parameters span five orders of magnitude (decay ~0.5,
    hill_K ~10,000), and on that scale L-BFGS-B barely moves hill_K and
    hill_beta from their starting values. The problem is also non-convex, so
    the fit is run from several starting points and the lowest-SSE result is
    kept.

    Args:
        df: DataFrame with marketing data
        config: MMMConfig instance. If None, uses default configuration.
        n_starts: Number of starting points. The first is a fixed mid-range
            guess; the rest are drawn uniformly within the bounds from a
            generator seeded with config.random_seed, so results are
            reproducible and the global NumPy random state is not touched.

    Returns:
        Dictionary with:
        - fitted_params: Estimated parameters for each channel
        - convergence: Convergence status and diagnostics of the best start
        - predictions: Model predictions
        - r_squared: R² goodness-of-fit metric
        - mae: Mean absolute error
        - sse: Sum of squared errors

    Interpreting the parameters:
        hill_K and hill_beta are only weakly identified: different pairs give
        nearly the same response over the observed spend range, so either can
        be far from the value that generated the data even when the fit is
        good. Quantities evaluated inside that range, such as marginal ROI at
        a typical spend level, are much more stable.
    """
    if config is None:
        config = MMMConfig()
    if n_starts < 1:
        raise ValueError("n_starts must be at least 1")

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

    lower = np.array([b[0] for b in bounds], dtype=float)
    upper = np.array([b[1] for b in bounds], dtype=float)
    span = upper - lower

    # Scale the loss so its magnitude is O(1) regardless of revenue units
    revenue = df["revenue"].values
    loss_scale = float(np.sum((revenue - revenue.mean()) ** 2)) or 1.0

    def scaled_objective(z: np.ndarray) -> float:
        return objective_function(lower + z * span, df, config) / loss_scale

    initial_guesses = np.array([0.3, 2, 15000, 20000] * num_channels + [5000], dtype=float)
    rng = np.random.default_rng(config.random_seed)
    starts = [(initial_guesses - lower) / span]
    starts += [rng.uniform(0.05, 0.95, len(bounds)) for _ in range(n_starts - 1)]

    print("Fitting model... This may take a moment.")
    result = None
    for z0 in starts:
        candidate = minimize(
            scaled_objective,
            z0,
            bounds=[(0.0, 1.0)] * len(bounds),
            method='L-BFGS-B',
            options={'maxiter': 1000}
        )
        if result is None or candidate.fun < result.fun:
            result = candidate
    print("Fitting complete.")

    fitted_params_list = lower + result.x * span
    sse = objective_function(fitted_params_list, df, config)

    convergence_info = {
        'success': bool(result.success),
        'message': result.message,
        'iterations': result.nit,
        'final_loss': sse,
        'n_starts': n_starts,
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
    param_names = []
    for ch in config.channels:
        param_names.extend([
            f"{ch}_adstock_decay",
            f"{ch}_hill_alpha",
            f"{ch}_hill_K",
            f"{ch}_hill_beta"
        ])
    param_names.append("promo_effect")

    for i, (param_val, (lo, hi)) in enumerate(zip(fitted_params_list, bounds)):
        if result.x[i] < 1e-6 or result.x[i] > 1 - 1e-6:
            warnings.warn(
                f"⚠️ Parameter '{param_names[i]}' hit boundary: {param_val:.4f} "
                f"(bounds: [{lo}, {hi}]). Consider adjusting bounds.",
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
    predicted_revenue = _predict_revenue(fitted_params_list, df, config)

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
        'sse': sse,
    }
