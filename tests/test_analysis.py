"""
Tests for mROI calculation (Issue #1 Fix)

Verifies that the marginal ROI calculation correctly includes the
chain rule multiplier for the adstock transformation.
"""

import numpy as np
import pytest

from mmm.analysis import calculate_marginal_roi


def test_marginal_roi_has_chain_rule_multiplier():
    """
    **Issue #1 Fix Verification:** Verify that mROI includes the 1/(1-decay) multiplier.

    This is the key fix - the original version was missing this multiplier.
    """
    spend = 100  # Use low spend to avoid saturation effects
    alpha = 2.0
    k = 10000
    beta = 15000

    # Without chain rule: result would be the Hill derivative only
    # With chain rule: result is Hill derivative × 1/(1-decay)

    # For decay=0 (no adstock), multiplier = 1/(1-0) = 1
    # For decay=0.5, multiplier = 1/(1-0.5) = 2
    # So mROI with decay=0.5 should be approximately 2x the mROI with decay=0

    mroi_no_decay = calculate_marginal_roi(spend, 0.0, alpha, k, beta)
    mroi_half_decay = calculate_marginal_roi(spend, 0.5, alpha, k, beta)

    # Ratio should be close to 2.0 (the chain rule multiplier)
    ratio = mroi_half_decay / mroi_no_decay

    # Allow 10% tolerance for numerical precision
    assert 1.8 < ratio < 2.2, f"Ratio {ratio:.2f} should be close to 2.0 (1/(1-0.5))"


def test_marginal_roi_increases_with_decay():
    """
    Verify that higher decay (longer carryover) leads to higher mROI.

    This makes economic sense: if advertising effects last longer,
    each dollar spent has more total impact.
    """
    spend = 100  # Use low spend to avoid saturation effects
    alpha = 2.0
    k = 15000
    beta = 20000

    mroi_02 = calculate_marginal_roi(spend, 0.2, alpha, k, beta)
    mroi_05 = calculate_marginal_roi(spend, 0.5, alpha, k, beta)
    mroi_07 = calculate_marginal_roi(spend, 0.7, alpha, k, beta)

    # Higher decay should give higher mROI (due to chain rule multiplier)
    assert mroi_05 > mroi_02, "mROI should increase with decay"
    assert mroi_07 > mroi_05, "mROI should increase with decay"


def test_marginal_roi_decreases_with_spend():
    """
    Verify mROI decreases with higher spend (saturation effect).

    At low spend: no saturation, high mROI
    At high spend: saturation, low mROI
    """
    decay = 0.5
    alpha = 2.0
    k = 10000
    beta = 15000

    # Use spend levels all above K/√3 ≈ 5774 to ensure we're in the saturating region
    mroi_low = calculate_marginal_roi(10000, decay, alpha, k, beta)
    mroi_mid = calculate_marginal_roi(30000, decay, alpha, k, beta)
    mroi_high = calculate_marginal_roi(50000, decay, alpha, k, beta)

    # mROI should decrease as spend increases (diminishing returns)
    assert mroi_low > mroi_mid, "mROI should decrease with higher spend"
    assert mroi_mid > mroi_high, "mROI should decrease with higher spend"


def test_marginal_roi_positive():
    """Verify mROI is always positive for valid parameters."""
    test_cases = [
        (1000, 0.3, 2.0, 10000, 15000),
        (10000, 0.5, 3.0, 8000, 20000),
        (20000, 0.7, 2.5, 15000, 25000),
    ]

    for spend, decay, alpha, k, beta in test_cases:
        mroi = calculate_marginal_roi(spend, decay, alpha, k, beta)
        assert mroi > 0, f"mROI should be positive for spend={spend}, decay={decay}"


def test_marginal_roi_chain_rule_magnitude():
    """
    Test the magnitude of the chain rule effect for known parameter values.

    This verifies that the fix had a significant impact (2-3x for typical decay values).
    """
    spend = 100  # Use low spend to avoid saturation effects
    alpha = 2.0
    k = 10000
    beta = 15000

    # Test different decay values
    test_cases = [
        (0.2, 1.25),  # 1/(1-0.2) = 1.25
        (0.5, 2.0),   # 1/(1-0.5) = 2.0
        (0.7, 3.33),  # 1/(1-0.7) = 3.33
    ]

    baseline_mroi = calculate_marginal_roi(spend, 0.0, alpha, k, beta)

    for decay, expected_multiplier in test_cases:
        mroi = calculate_marginal_roi(spend, decay, alpha, k, beta)
        actual_multiplier = mroi / baseline_mroi

        # Allow 15% tolerance (Hill function nonlinearity affects the ratio)
        lower_bound = expected_multiplier * 0.85
        upper_bound = expected_multiplier * 1.15

        assert lower_bound < actual_multiplier < upper_bound, \
            f"For decay={decay}, multiplier {actual_multiplier:.2f} should be near {expected_multiplier:.2f}"


def test_marginal_roi_with_ground_truth_params():
    """
    Test mROI with the ground truth parameters from CONFIG.

    This verifies the fix works with realistic parameter values.
    """
    # Shopify parameters
    mroi_shopify = calculate_marginal_roi(10000, 0.5, 2.0, 10000, 15000)
    assert mroi_shopify > 0, "Shopify mROI should be positive"

    # TikTok parameters
    mroi_tiktok = calculate_marginal_roi(10000, 0.2, 2.5, 12000, 25000)
    assert mroi_tiktok > 0, "TikTok mROI should be positive"

    # Meta parameters
    mroi_meta = calculate_marginal_roi(10000, 0.7, 3.0, 8000, 10000)
    assert mroi_meta > 0, "Meta mROI should be positive"

    # Meta has highest decay, so should have highest multiplier effect
    # (though absolute mROI depends on other parameters too)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
