"""
Synthetic data generation for Media Mix Modeling.

This module generates synthetic marketing data with known ground truth
parameters for testing and demonstration purposes.
"""

import numpy as np
import pandas as pd

from .config import CONFIG
from .core import geometric_adstock, hill_function


def generate_weekly_data() -> pd.DataFrame:
    """
    Generates synthetic weekly marketing data based on defined "true" parameters.

    Creates a realistic marketing dataset with:
    - Multiple advertising channels (Shopify, TikTok, Meta)
    - Adstock (carryover) effects
    - Saturation (diminishing returns)
    - Promotions and seasonality
    - Random noise

    Returns:
        DataFrame with columns:
        - week: Week number
        - spend_{channel}: Weekly spend per channel
        - revenue: Total revenue (with all effects)
        - contribution_{channel}: True revenue contribution per channel
        - promotions: Binary promotion indicator

    Ground Truth:
        Uses parameters from CONFIG["true_params"] to generate data
        with known relationships, enabling validation of model estimates.
    """
    weeks = np.arange(1, CONFIG["num_weeks"] + 1)
    df = pd.DataFrame({'week': weeks})

    # 1. Generate Spend & Control Variables
    for ch in CONFIG["channels"]:
        # Realistic spend range: $2k-$20k per week
        df[f"spend_{ch}"] = np.random.uniform(2000, 20000, CONFIG["num_weeks"])

    # Promotions occur ~15% of weeks
    df["promotions"] = (np.random.rand(CONFIG["num_weeks"]) < 0.15).astype(int)

    # Yearly seasonality cycle
    seasonality = CONFIG["true_params"]["seasonality_amplitude"] * \
                  np.sin(2 * np.pi * weeks / CONFIG["true_params"]["seasonality_period"])

    # 2. Calculate "True" Revenue using MMM principles
    total_revenue = CONFIG["base_revenue"] + seasonality
    total_revenue += df["promotions"] * CONFIG["true_params"]["promo_effect"]

    # Add channel contributions with adstock and saturation
    for ch in CONFIG["channels"]:
        params = CONFIG["true_params"][ch]

        # Apply adstock transformation
        adstocked_spend = geometric_adstock(
            df[f"spend_{ch}"].values,
            params["adstock_decay"]
        )

        # Apply saturation (Hill function)
        channel_contribution = hill_function(
            adstocked_spend,
            params["hill_alpha"],
            params["hill_K"],
            params["hill_beta"]
        )

        # Store contribution for later comparison
        df[f"contribution_{ch}"] = channel_contribution
        total_revenue += channel_contribution

    # 3. Add random noise to simulate natural variation
    df["revenue"] = total_revenue + np.random.normal(
        0,
        CONFIG["noise_std"],
        CONFIG["num_weeks"]
    )

    # Ensure non-negative revenue
    df.loc[df["revenue"] < 0, "revenue"] = 0

    return df
