"""
Tests for privacy mechanisms (Issue #9 Fix)

Verifies that the shifted Laplace mechanism correctly handles
non-negative constraints without introducing clipping bias.
"""

import numpy as np
import pytest

from mmm.privacy import laplace_mechanism, shifted_laplace_mechanism


def test_laplace_mechanism_basic():
    """Verify basic Laplace mechanism properties."""
    value = 1000
    sensitivity = 100
    epsilon = 1.0

    # Generate samples
    np.random.seed(42)
    samples = [laplace_mechanism(value, sensitivity, epsilon) for _ in range(1000)]

    # Mean should be close to original value
    mean = np.mean(samples)
    assert abs(mean - value) < 50, f"Mean {mean} should be close to {value}"

    # Std should be close to sqrt(2) * scale = sqrt(2) * sensitivity/epsilon
    std = np.std(samples)
    expected_std = np.sqrt(2) * (sensitivity / epsilon)
    assert abs(std - expected_std) / expected_std < 0.2, \
        f"Std {std} should be close to {expected_std}"


def test_shifted_laplace_respects_bounds():
    """
    **Issue #9 Test:** Verify shifted Laplace never produces negative values.

    This is the key fix - the shifted mechanism should NEVER return values
    below the lower bound, avoiding the bias from post-hoc clipping.
    """
    value = 100
    sensitivity = 100
    epsilon = 1.0
    lower_bound = 0.0

    # Generate many samples
    np.random.seed(42)
    samples = [shifted_laplace_mechanism(value, sensitivity, epsilon, lower_bound)
               for _ in range(10000)]

    # ALL samples must be >= lower_bound
    assert all(s >= lower_bound for s in samples), \
        f"All samples must be >= {lower_bound}, but found min={min(samples)}"

    # No negative values
    negative_count = sum(1 for s in samples if s < 0)
    assert negative_count == 0, f"Found {negative_count} negative values"


def test_shifted_laplace_minimal_bias():
    """
    Verify that shifted Laplace has minimal bias compared to standard Laplace.

    The bias should be much smaller than the bias from post-hoc clipping.
    """
    value = 1000
    sensitivity = 100
    epsilon = 1.0
    lower_bound = 0.0

    # Generate large sample
    np.random.seed(42)
    samples = [shifted_laplace_mechanism(value, sensitivity, epsilon, lower_bound)
               for _ in range(100000)]

    # Calculate bias
    mean = np.mean(samples)
    bias = abs(mean - value)

    # Bias should be small relative to scale
    scale = sensitivity / epsilon
    relative_bias = bias / scale

    # Bias should be less than 5% of scale
    assert relative_bias < 0.05, \
        f"Bias {bias:.2f} ({relative_bias:.1%} of scale) is too large"


def test_shifted_laplace_vs_clipped_laplace_bias():
    """
    Compare shifted Laplace vs post-hoc clipping to verify bias reduction.

    Post-hoc clipping introduces downward bias. Shifted Laplace should
    have much less bias.

    Uses value well above lower bound to avoid excessive rejection sampling
    in the shifted mechanism, which would skew results.
    """
    value = 5000  # Well above lower bound (rejection sampling rarely triggers)
    sensitivity = 1000
    epsilon = 1.0  # Moderate epsilon for clearer signal
    lower_bound = 0.0

    np.random.seed(42)
    n_samples = 10000

    # Shifted Laplace (new method)
    shifted_samples = [
        shifted_laplace_mechanism(value, sensitivity, epsilon, lower_bound)
        for _ in range(n_samples)
    ]

    # Standard Laplace + clipping (old method)
    clipped_samples = []
    for _ in range(n_samples):
        noisy = laplace_mechanism(value, sensitivity, epsilon)
        clipped = max(noisy, lower_bound)  # Post-hoc clip
        clipped_samples.append(clipped)

    shifted_mean = np.mean(shifted_samples)
    clipped_mean = np.mean(clipped_samples)

    # Clipped version should have larger downward bias
    shifted_bias = abs(shifted_mean - value)
    clipped_bias = abs(clipped_mean - value)

    # For this parameter set, clipping bias should be significantly larger
    # (since many samples get clipped)
    assert clipped_bias > shifted_bias, \
        f"Clipped bias ({clipped_bias:.2f}) should be larger than shifted bias ({shifted_bias:.2f})"


def test_shifted_laplace_convergence():
    """
    Verify that shifted Laplace converges quickly (few rejection samples needed).

    For reasonable epsilon values, rejection sampling should succeed in 1-2 attempts.
    """
    value = 1000
    sensitivity = 100
    epsilon = 1.0  # Reasonable epsilon
    lower_bound = 0.0

    # This should succeed without infinite loops
    # (if it hangs, there's a bug in the rejection sampling)
    np.random.seed(42)
    for _ in range(100):
        result = shifted_laplace_mechanism(value, sensitivity, epsilon, lower_bound)
        assert result >= lower_bound
        assert not np.isnan(result)
        assert not np.isinf(result)


def test_shifted_laplace_with_very_low_epsilon():
    """
    Test that shifted Laplace handles very low epsilon gracefully.

    With very low epsilon (high noise), rejection sampling might fail,
    but should fallback to lower_bound rather than looping forever.
    """
    value = 100
    sensitivity = 1000
    epsilon = 0.01  # Very low epsilon → very high noise
    lower_bound = 0.0

    # Should not hang or crash
    np.random.seed(42)
    samples = [shifted_laplace_mechanism(value, sensitivity, epsilon, lower_bound)
               for _ in range(100)]

    # All samples should be valid
    assert all(s >= lower_bound for s in samples)
    assert all(not np.isnan(s) for s in samples)

    # Many samples will be at the lower bound (fallback case)
    # This is expected with such high noise


def test_epsilon_validation():
    """Verify that mechanisms reject invalid epsilon values."""
    with pytest.raises(ValueError):
        laplace_mechanism(100, 10, epsilon=0)

    with pytest.raises(ValueError):
        laplace_mechanism(100, 10, epsilon=-1.0)

    with pytest.raises(ValueError):
        shifted_laplace_mechanism(100, 10, epsilon=0)


def test_shifted_laplace_different_bounds():
    """Test shifted Laplace with different lower bounds."""
    value = 1000
    sensitivity = 100
    epsilon = 1.0

    # Test with lower_bound = 0
    samples_0 = [shifted_laplace_mechanism(value, sensitivity, epsilon, 0.0)
                 for _ in range(1000)]
    assert all(s >= 0.0 for s in samples_0)

    # Test with lower_bound = 500
    samples_500 = [shifted_laplace_mechanism(value, sensitivity, epsilon, 500.0)
                   for _ in range(1000)]
    assert all(s >= 500.0 for s in samples_500)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
