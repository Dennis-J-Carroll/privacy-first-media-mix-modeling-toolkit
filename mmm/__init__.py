"""
Privacy-First Media Mix Modeling Toolkit
=========================================

A modular toolkit for performing Media Mix Modeling with formal
differential privacy guarantees.

This package provides:
- Core MMM mathematical functions (adstock, saturation)
- Differential privacy mechanisms (Laplace, shifted Laplace)
- Synthetic data generation
- Model fitting and optimization
- Analysis and metrics calculation
- Visualization functions

Example usage:
    >>> from mmm import generate_weekly_data, apply_differential_privacy, fit_model
    >>> from mmm import CONFIG
    >>>
    >>> # Generate synthetic data
    >>> df = generate_weekly_data()
    >>>
    >>> # Apply privacy
    >>> df_private = apply_differential_privacy(df, epsilon=1.0)
    >>>
    >>> # Fit model
    >>> results = fit_model(df_private)
    >>> print(f"R² = {results['r_squared']:.3f}")
"""

__version__ = "2.0.0"
__author__ = "Privacy-First MMM Contributors"

# Configuration
from .config import (
    CONFIG,
    OUTPUT_DIR,
    CHANNEL_COLORS,
    CONTRIBUTION_COLORS,
    MMMConfig,
    SensitivityConfig,
    ChannelParams,
    TrueParams
)

# Core mathematical functions
from .core import geometric_adstock, hill_function

# Privacy mechanisms
from .privacy import (
    laplace_mechanism,
    shifted_laplace_mechanism,
    apply_differential_privacy
)

# Data generation
from .data import generate_weekly_data

# Model fitting
from .modeling import fit_model, objective_function

# Analysis functions
from .analysis import (
    calculate_marginal_roi,
    calculate_r_squared,
    calculate_mae,
    calculate_rmse
)

# Visualization
from .visualization import generate_plots

# Public API
__all__ = [
    # Configuration
    "CONFIG",
    "MMMConfig",
    "SensitivityConfig",
    "ChannelParams",
    "TrueParams",
    "OUTPUT_DIR",
    "CHANNEL_COLORS",
    "CONTRIBUTION_COLORS",

    # Core functions
    "geometric_adstock",
    "hill_function",

    # Privacy
    "laplace_mechanism",
    "shifted_laplace_mechanism",
    "apply_differential_privacy",

    # Data
    "generate_weekly_data",

    # Modeling
    "fit_model",
    "objective_function",

    # Analysis
    "calculate_marginal_roi",
    "calculate_r_squared",
    "calculate_mae",
    "calculate_rmse",

    # Visualization
    "generate_plots",
]
