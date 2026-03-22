"""
Core mathematical transformations for Media Mix Modeling.

This module contains the fundamental mathematical functions used in MMM:
- Geometric adstock (carryover effect)
- Hill function (saturation/diminishing returns)
"""

import numpy as np
from scipy.signal import lfilter


def geometric_adstock(x: np.ndarray, theta: float) -> np.ndarray:
    """
    Applies geometric adstock decay to a marketing channel's spend series.

    The adstock effect models how advertising impact carries over from one
    period to the next, decaying geometrically over time.

    Formula: y[t] = x[t] + theta * y[t-1]

    Args:
        x: Input spend series (e.g., weekly spend)
        theta: Decay rate (0 to 0.9). Higher values mean longer carryover.
               theta=0.5 means 50% of impact carries to next period.

    Returns:
        Adstocked spend series incorporating carryover effects

    Implementation:
        Uses scipy.signal.lfilter for computational efficiency.
        Equivalent to IIR filter with coefficients a=[1, -theta] and b=[1]
    """
    # y[t] = x[t] + theta * y[t-1]
    # This is equivalent to an IIR filter with coefficients a=[1, -theta] and b=[1]
    return lfilter([1], [1, -theta], x)


def hill_function(x: np.ndarray, alpha: float, k: float, beta: float) -> np.ndarray:
    """
    Calculates revenue contribution based on the S-shaped Hill function.

    The Hill function models diminishing returns and saturation effects,
    creating a characteristic S-curve where additional spend yields
    progressively less incremental revenue.

    Formula: f(x) = beta * (x^alpha) / (k^alpha + x^alpha)

    Args:
        x: Input variable (typically adstocked spend)
        alpha: Shape parameter controlling curve steepness (typically 1.5-4.0)
        k: Saturation point (spend level at 50% of maximum effect)
        beta: Maximum revenue contribution (asymptotic ceiling)

    Returns:
        Revenue contribution from the channel

    Properties:
        - f(0) = 0
        - f(k) = beta/2 (half-saturation at k)
        - f(∞) → beta (asymptotic maximum)
        - Higher alpha → steeper transition
    """
    return beta * (x**alpha) / (k**alpha + x**alpha)
