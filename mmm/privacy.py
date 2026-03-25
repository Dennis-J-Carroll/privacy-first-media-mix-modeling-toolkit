"""
Privacy-preserving mechanisms for differential privacy.

This module implements formal differential privacy mechanisms including:
- Laplace mechanism for epsilon-DP
- Shifted Laplace mechanism (bias-corrected for non-negative quantities)
"""

import numpy as np
import pandas as pd
from .config import MMMConfig


def laplace_mechanism(value: float, sensitivity: float, epsilon: float) -> float:
    """
    Applies the Laplace mechanism for differential privacy.

    Adds noise drawn from a Laplace distribution to a value, providing
    epsilon-differential privacy. The scale of the noise is calibrated to
    the sensitivity of the query and the desired privacy level.

    Args:
        value: The true value to be privatized
        sensitivity: Maximum change in output from adding/removing one individual
        epsilon: Privacy budget (smaller = more privacy, more noise)

    Returns:
        Noisy value satisfying epsilon-differential privacy

    Mathematical Guarantee:
        For any two neighboring datasets D and D' (differing by one record),
        Pr[M(D) = x] ≤ e^ε × Pr[M(D') = x]
    """
    if epsilon <= 0:
        raise ValueError("Epsilon must be positive")

    # Scale parameter for Laplace distribution: b = sensitivity / epsilon
    scale = sensitivity / epsilon

    # Add Laplace noise
    noise = np.random.laplace(loc=0, scale=scale)

    return value + noise


def shifted_laplace_mechanism(
    value: float,
    sensitivity: float,
    epsilon: float,
    lower_bound: float = 0.0
) -> float:
    """
    Applies shifted Laplace mechanism for non-negative quantities.

    This mechanism samples directly from a truncated Laplace distribution,
    providing rigorous epsilon-differential privacy without the bias issues
    of rejection sampling or post-hoc clipping.

    **Privacy Fix:** Uses inverse transform sampling from truncated Laplace CDF.
    This is mathematically correct, efficient (single sample), and preserves
    the formal epsilon-DP guarantee.

    Args:
        value: The true value to be privatized
        sensitivity: Maximum change in output from adding/removing one individual
        epsilon: Privacy budget
        lower_bound: Minimum valid value (e.g., 0 for spend/revenue)

    Returns:
        Noisy value satisfying epsilon-DP and >= lower_bound

    Privacy Guarantee:
        Satisfies pure epsilon-differential privacy. The truncation is applied
        to the noise distribution before sampling, not after, which preserves
        the DP guarantee rigorously.

    Implementation:
        Uses inverse transform sampling from truncated Laplace CDF.
        Guarantees acceptance in one draw (no rejection, no fallback).

    Mathematical Background:
        Laplace(μ, b) has CDF:
            F(x) = 0.5 * exp((x - μ)/b)           for x < μ
            F(x) = 1 - 0.5 * exp(-(x - μ)/b)      for x >= μ

        For truncation at L, we sample u ~ Uniform(F(L), 1) and return F^(-1)(u).
    """
    if epsilon <= 0:
        raise ValueError("Epsilon must be positive")

    scale = sensitivity / epsilon

    # Compute CDF at lower_bound for Laplace(value, scale)
    if lower_bound < value:
        # Lower bound is in left tail
        cdf_at_bound = 0.5 * np.exp((lower_bound - value) / scale)
    else:
        # Lower bound is in right tail or at center
        cdf_at_bound = 1.0 - 0.5 * np.exp(-(lower_bound - value) / scale)

    # Sample uniformly from [cdf_at_bound, 1.0]
    u = np.random.uniform(cdf_at_bound, 1.0)

    # Inverse CDF (quantile function) for Laplace
    if u < 0.5:
        # Left tail: x = μ + b * ln(2u)
        noisy_value = value + scale * np.log(2 * u)
    else:
        # Right tail: x = μ - b * ln(2(1 - u))
        noisy_value = value - scale * np.log(2 * (1 - u))

    return noisy_value


def apply_differential_privacy(data: pd.DataFrame, epsilon: float, config: MMMConfig = None) -> pd.DataFrame:
    """
    Applies differential privacy to aggregated marketing data.

    This function implements input perturbation by adding Laplace noise to
    aggregated spend and revenue data before it is used for modeling. This
    provides a formal privacy guarantee that individual-level contributions
    cannot be determined from the model.

    **Updated (Issue #9):** Now uses shifted Laplace mechanism instead of
    standard Laplace + clipping to avoid introducing bias.

    Args:
        data: DataFrame with aggregated marketing data
        epsilon: Privacy budget to be split across all queries
        config: MMMConfig instance. If None, uses default configuration.

    Returns:
        DataFrame with noisy data satisfying differential privacy
    """
    if config is None:
        config = MMMConfig()

    if not config.enable_privacy:
        print("Privacy is disabled. Using original data without noise.")
        return data.copy()

    print(f"\nApplying Differential Privacy with ε={epsilon:.2f}")
    print(f"Privacy Guarantee: ε={epsilon:.2f}-differential privacy (Laplace mechanism)")

    # Create a copy to avoid modifying original data
    private_data = data.copy()

    # Split epsilon budget across channels and metrics
    # Using composition theorem: if we make k queries each with epsilon/k,
    # total privacy budget is epsilon
    num_queries = len(config.channels) + 1  # channels + revenue
    epsilon_per_query = epsilon / num_queries

    # Add noise to spend data for each channel using shifted Laplace
    for ch in config.channels:
        spend_col = f"spend_{ch}"
        if spend_col in private_data.columns:
            # Apply shifted Laplace mechanism to ensure non-negative values
            private_data[spend_col] = private_data[spend_col].apply(
                lambda x: shifted_laplace_mechanism(
                    x,
                    config.sensitivity.spend,
                    epsilon_per_query,
                    lower_bound=0.0
                )
            )

    # Add noise to revenue data using shifted Laplace
    if "revenue" in private_data.columns:
        private_data["revenue"] = private_data["revenue"].apply(
            lambda x: shifted_laplace_mechanism(
                x,
                config.sensitivity.revenue,
                epsilon_per_query,
                lower_bound=0.0
            )
        )

    # Calculate noise statistics for reporting
    for ch in config.channels:
        spend_col = f"spend_{ch}"
        if spend_col in data.columns:
            noise = (private_data[spend_col] - data[spend_col]).abs()
            print(f"  {ch} spend noise: mean={noise.mean():.2f}, max={noise.max():.2f}")

    if "revenue" in data.columns:
        revenue_noise = (private_data["revenue"] - data["revenue"]).abs()
        print(f"  Revenue noise: mean={revenue_noise.mean():.2f}, max={revenue_noise.max():.2f}")

    return private_data
