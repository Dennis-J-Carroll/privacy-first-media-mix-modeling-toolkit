"""
Configuration and constants for Privacy-First MMM.

This module contains all configuration parameters, ground truth values,
and constants used throughout the MMM toolkit.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

# Output directory for results
OUTPUT_DIR = "mmm_output_advanced"
os.makedirs(OUTPUT_DIR, exist_ok=True)


@dataclass(frozen=True)
class SensitivityConfig:
    """Sensitivity values for differential privacy."""
    spend: int = 1000
    revenue: int = 500
    promotions: int = 1


@dataclass(frozen=True)
class ChannelParams:
    """Parameters for a single marketing channel."""
    adstock_decay: float
    hill_alpha: float
    hill_K: float
    hill_beta: float


@dataclass(frozen=True)
class TrueParams:
    """Ground truth parameters for data simulation."""
    Shopify: ChannelParams
    TikTok: ChannelParams
    Meta: ChannelParams
    promo_effect: float
    seasonality_amplitude: float
    seasonality_period: int


@dataclass(frozen=True)
class MMMConfig:
    """
    Immutable configuration for Media Mix Modeling.

    This frozen dataclass replaces the mutable CONFIG dictionary,
    preventing order-dependent bugs and improving test isolation.

    Usage:
        # Create config
        config = MMMConfig()

        # Create modified config
        config2 = MMMConfig(num_weeks=52, epsilon=0.5)

        # Use in functions
        data = generate_weekly_data(config)
        results = fit_model(data, config)
    """
    num_weeks: int = 104
    channels: Tuple[str, ...] = ("Shopify", "TikTok", "Meta")
    base_revenue: int = 5000
    noise_std: int = 1000
    random_seed: int = 42
    enable_privacy: bool = True
    epsilon: float = 1.0
    sensitivity: SensitivityConfig = field(default_factory=SensitivityConfig)
    true_params: TrueParams = field(default_factory=lambda: TrueParams(
        Shopify=ChannelParams(adstock_decay=0.5, hill_alpha=2.0, hill_K=10000, hill_beta=15000),
        TikTok=ChannelParams(adstock_decay=0.2, hill_alpha=2.5, hill_K=12000, hill_beta=25000),
        Meta=ChannelParams(adstock_decay=0.7, hill_alpha=3.0, hill_K=8000, hill_beta=10000),
        promo_effect=8000,
        seasonality_amplitude=4000,
        seasonality_period=52
    ))


# Main configuration dictionary (DEPRECATED - use MMMConfig dataclass)
CONFIG = {
    "num_weeks": 104,  # 2 years of data for better model stability
    "channels": ["Shopify", "TikTok", "Meta"],
    "base_revenue": 5000,
    "noise_std": 1000,
    "random_seed": 42,

    # Privacy parameters
    "enable_privacy": True,  # Set to False to disable differential privacy
    "epsilon": 1.0,  # Privacy budget (lower = more privacy, more noise)

    # Sensitivity for different metrics (max change from one individual)
    "sensitivity": {
        "spend": 1000,      # Max spend contribution from one individual
        "revenue": 500,     # Max revenue contribution from one individual
        "promotions": 1,    # Binary indicator
    },

    # "Ground Truth" parameters for the data simulation
    "true_params": {
        "Shopify": {"adstock_decay": 0.5, "hill_alpha": 2.0, "hill_K": 10000, "hill_beta": 15000},
        "TikTok":  {"adstock_decay": 0.2, "hill_alpha": 2.5, "hill_K": 12000, "hill_beta": 25000},
        "Meta":    {"adstock_decay": 0.7, "hill_alpha": 3.0, "hill_K": 8000, "hill_beta": 10000},
        "promo_effect": 8000,
        "seasonality_amplitude": 4000,
        "seasonality_period": 52,  # Yearly seasonality
    }
}

# Channel color palette for visualizations (Issue #7)
CHANNEL_COLORS = {
    'Shopify': '#95BF47',  # Shopify green
    'TikTok': '#FF0050',   # TikTok brand pink/red
    'Meta': '#0668E1',     # Meta blue
}

# Extended color palette for contribution plots
CONTRIBUTION_COLORS = {
    'Base & Seasonality': '#808080',  # Gray
    'Promotions': '#FFA500',          # Orange
    'Shopify': '#95BF47',
    'TikTok': '#FF0050',
    'Meta': '#0668E1',
}
