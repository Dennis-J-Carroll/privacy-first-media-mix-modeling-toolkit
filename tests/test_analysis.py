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

    At realistic spend levels (spend ≈ k), the ratio is NOT simply 1/(1-decay) because
    we're also moving along the Hill curve. The steady-state adstock with decay=0.5 is
    2x higher, which puts us deeper into saturation, so the Hill derivative is lower.
    The combined effect: chain rule gives 2x, Hill derivative gives ~0.32x, net ~0.64x.
    """
    spend = 10000  # Realistic spend level (at half-saturation point k)
    alpha = 2.0
    k = 10000
    beta = 15000

    # For decay=0: adstock = 10000/(1-0) = 10000
    # For decay=0.5: adstock = 10000/(1-0.5) = 20000 (2x k, heavily saturated)

    mroi_no_decay = calculate_marginal_roi(spend, 0.0, alpha, k, beta)
    mroi_half_decay = calculate_marginal_roi(spend, 0.5, alpha, k, beta)

    # Ratio should be ~0.64 (chain rule 2x × Hill derivative drop ~0.32x)
    ratio = mroi_half_decay / mroi_no_decay

    # Allow 20% tolerance for numerical effects
    assert 0.5 < ratio < 0.8, f"Ratio {ratio:.2f} should be ~0.64 at spend=k"


def test_marginal_roi_increases_with_decay():
    """
    Verify that higher decay (longer carryover) leads to higher mROI.

    NOTE: This only holds at LOW spend levels where saturation is minimal.
    At high spend, higher decay increases saturation faster than the chain
    rule multiplier increases mROI, so mROI can actually DECREASE with decay.

    This test uses spend << k to validate the chain rule in the linear regime.
    """
    spend = 1000  # Low spend (well below k=15000, minimal saturation)
    alpha = 2.0
    k = 15000
    beta = 20000

    mroi_02 = calculate_marginal_roi(spend, 0.2, alpha, k, beta)
    mroi_05 = calculate_marginal_roi(spend, 0.5, alpha, k, beta)
    mroi_07 = calculate_marginal_roi(spend, 0.7, alpha, k, beta)

    # At low spend, higher decay → higher mROI (chain rule dominates)
    assert mroi_05 > mroi_02, "mROI should increase with decay at low spend"
    assert mroi_07 > mroi_05, "mROI should increase with decay at low spend"


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

    At realistic spend levels (spend ≈ k), the ratio includes both the chain rule
    multiplier 1/(1-decay) AND the Hill derivative change from moving along the curve.
    These test cases were empirically verified against the corrected formula.
    """
    spend = 10000  # Realistic spend level (at k)
    alpha = 2.0
    k = 10000
    beta = 15000

    # Test different decay values with REALISTIC expected ratios
    # (not naive 1/(1-decay) because we're also moving on Hill curve)
    test_cases = [
        (0.2, 0.95),   # Chain rule 1.25x × Hill drop ~0.76x ≈ 0.95x
        (0.5, 0.64),   # Chain rule 2.0x × Hill drop ~0.32x ≈ 0.64x
        (0.7, 0.30),   # Chain rule 3.33x × Hill drop ~0.09x ≈ 0.30x
    ]

    baseline_mroi = calculate_marginal_roi(spend, 0.0, alpha, k, beta)

    for decay, expected_multiplier in test_cases:
        mroi = calculate_marginal_roi(spend, decay, alpha, k, beta)
        actual_multiplier = mroi / baseline_mroi

        # Allow 20% tolerance for numerical effects
        lower_bound = expected_multiplier * 0.80
        upper_bound = expected_multiplier * 1.20

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


def test_mroi_numerical_ground_truth():
    """
    Regression guard: Validate mROI against ground-truth simulation values.

    These values were verified against analytical formula with correct steady-state
    adstock: A = S/(1-θ). This test prevents formula regressions like Sprint 2
    which changed to A = S and had 14-3,280% errors.

    The key insight: we evaluate Hill derivative at steady-state adstock level,
    NOT at raw spend.
    """
    # Verified analytical values (using adstocked_spend = spend / (1 - decay))
    test_cases = [
        # (spend, decay, alpha, k, beta, expected_mroi, tolerance_pct)
        (10000, 0.5, 2.0, 10000, 15000, 0.480, 1.0),    # Shopify - exact match
        (5000, 0.2, 2.5, 12000, 25000, 1.711, 2.0),     # TikTok - analytical value
        (2000, 0.7, 3.0, 8000, 10000, 3.483, 2.0),      # Meta - analytical value
        (10000, 0.8, 2.0, 10000, 15000, 0.111, 5.0),    # High decay edge case
    ]

    for spend, decay, alpha, k, beta, expected, tolerance in test_cases:
        mroi = calculate_marginal_roi(spend, decay, alpha, k, beta)
        error_pct = abs(mroi - expected) / expected * 100

        assert error_pct < tolerance, \
            f"mROI regression detected at spend={spend}, decay={decay}\n" \
            f"  Got: {mroi:.3f}, Expected: {expected:.3f}, Error: {error_pct:.1f}%\n" \
            f"  Formula likely changed. Correct: adstocked_spend = spend / (1 - decay)"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
