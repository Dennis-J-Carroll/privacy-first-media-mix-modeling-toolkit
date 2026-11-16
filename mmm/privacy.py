"""
Privacy-preserving mechanisms for differential privacy.

This module implements formal differential privacy mechanisms including:
- Laplace mechanism for epsilon-DP
- Clamped Laplace mechanism (for quantities with a lower bound, e.g. spend)
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


def clamped_laplace_mechanism(
    value: float,
    sensitivity: float,
    epsilon: float,
    lower_bound: float = 0.0
) -> float:
    """
    Laplace mechanism followed by a clamp at ``lower_bound``.

    Draws standard Laplace noise and then raises any result below
    ``lower_bound`` up to it. The clamp is a fixed function of the noisy
    output that never looks at the true value, so it is post-processing and
    the result keeps the epsilon-DP guarantee of the Laplace mechanism.

    Args:
        value: The true value to be privatized
        sensitivity: Maximum change in output from adding/removing one individual
        epsilon: Privacy budget
        lower_bound: Minimum valid value (e.g., 0 for spend/revenue)

    Returns:
        Noisy value satisfying epsilon-DP and >= lower_bound

    Bias:
        Clamping is NOT unbiased. Mass that would fall below the bound is
        moved onto it, so the expected output exceeds the true value by
        (scale / 2) * exp(-(value - lower_bound) / scale) for value >=
        lower_bound, where scale = sensitivity / epsilon. This is negligible
        when the value is several scales above the bound and substantial
        when it is within about one scale of it.

    Why not sample a truncated Laplace instead:
        Conditioning the noise on landing above the bound renormalises the
        density by a constant that depends on the true value. That constant
        differs between neighbouring datasets, so the privacy loss can exceed
        epsilon (up to ln(2*e^eps - 1) at the bound: 1.49 for eps = 1). It is
        also more biased than clamping. An earlier version of this module did
        this and claimed pure epsilon-DP; that claim was wrong.
    """
    return max(laplace_mechanism(value, sensitivity, epsilon), lower_bound)


# Deprecated name, kept so existing imports keep working. It used to sample a
# truncated Laplace, which did not satisfy epsilon-DP (see above).
shifted_laplace_mechanism = clamped_laplace_mechanism


def apply_differential_privacy(data: pd.DataFrame, epsilon: float, config: MMMConfig = None) -> pd.DataFrame:
    """
    Applies differential privacy to aggregated marketing data.

    This function implements input perturbation by adding Laplace noise to
    aggregated spend and revenue data before it is used for modeling. This
    provides a formal privacy guarantee that individual-level contributions
    cannot be determined from the model.

    Each value gets Laplace noise and is then clamped at zero. Clamping is
    post-processing, so the guarantee below holds, but it biases values near
    zero upward (see ``clamped_laplace_mechanism``).

    Scope of the guarantee: within one week, the budget is split evenly over
    the released columns, so that week's release is epsilon-DP for a person
    whose contribution to each column is bounded by ``config.sensitivity``.
    Weeks are released independently. A person who contributes in k weeks
    is therefore protected at k * epsilon, not epsilon.

    Args:
        data: DataFrame with aggregated marketing data
        epsilon: Privacy budget per week, split across the released columns
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
    print(f"Privacy Guarantee: ε={epsilon:.2f}-differential privacy per weekly release (Laplace mechanism)")

    # Create a copy to avoid modifying original data
    private_data = data.copy()

    # Split epsilon budget across channels and metrics
    # Using composition theorem: if we make k queries each with epsilon/k,
    # total privacy budget is epsilon
    num_queries = len(config.channels) + 1  # channels + revenue
    epsilon_per_query = epsilon / num_queries

    # Add noise to spend data for each channel using clamped Laplace
    for ch in config.channels:
        spend_col = f"spend_{ch}"
        if spend_col in private_data.columns:
            # Laplace noise, clamped at zero (post-processing)
            private_data[spend_col] = private_data[spend_col].apply(
                lambda x: clamped_laplace_mechanism(
                    x,
                    config.sensitivity.spend,
                    epsilon_per_query,
                    lower_bound=0.0
                )
            )

    # Add noise to revenue data using clamped Laplace
    if "revenue" in private_data.columns:
        private_data["revenue"] = private_data["revenue"].apply(
            lambda x: clamped_laplace_mechanism(
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
