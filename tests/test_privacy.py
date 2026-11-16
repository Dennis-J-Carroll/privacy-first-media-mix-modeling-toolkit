"""
Tests for privacy mechanisms.

Verifies that the clamped Laplace mechanism respects its lower bound, keeps
the epsilon-DP guarantee of the Laplace mechanism, and has the bias its
docstring states.
"""

import numpy as np
import pandas as pd
import pytest

from mmm.config import MMMConfig
from mmm.privacy import (
    apply_differential_privacy,
    clamped_laplace_mechanism,
    laplace_mechanism,
    shifted_laplace_mechanism,
)


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


def test_clamped_laplace_respects_bounds():
    """Outputs are never below the lower bound, for several bounds."""
    np.random.seed(42)
    for value, lower_bound in [(100, 0.0), (1000, 500.0), (0, 0.0)]:
        samples = [clamped_laplace_mechanism(value, 100, 1.0, lower_bound)
                   for _ in range(5000)]
        assert min(samples) >= lower_bound
        assert np.all(np.isfinite(samples))


def test_clamped_laplace_with_very_low_epsilon():
    """Very high noise still yields finite, in-bounds outputs."""
    np.random.seed(42)
    samples = [clamped_laplace_mechanism(100, 1000, 0.01, 0.0) for _ in range(1000)]
    assert min(samples) >= 0.0
    assert np.all(np.isfinite(samples))


@pytest.mark.parametrize("epsilon", [0.25, 1.0])
def test_clamped_laplace_privacy_loss_within_epsilon(epsilon):
    """
    The empirical privacy loss between neighbouring inputs stays within epsilon.

    Uses the worst case for a bounded mechanism: one neighbour sits on the
    bound, the other is one sensitivity above it. Outputs are binned (the
    clamp's point mass at the bound gets its own bin) and every bin's
    probability ratio must be at most e^epsilon, in both directions.

    A truncated-Laplace sampler fails this: its ratio just above the bound is
    2*e^eps - 1 (4.44 instead of 2.72 at eps = 1).
    """
    sensitivity = 1000.0
    scale = sensitivity / epsilon
    n = 400_000

    np.random.seed(42)
    a = np.array([clamped_laplace_mechanism(0.0, sensitivity, epsilon) for _ in range(n)])
    b = np.array([clamped_laplace_mechanism(sensitivity, sensitivity, epsilon) for _ in range(n)])

    # Bin 0 holds exactly the clamped outputs. The next bin is narrow because
    # the worst case for a bounded mechanism sits right at the bound.
    edges = np.concatenate(([-1.0, 1e-9, 0.05 * scale], np.linspace(0.25 * scale, 4 * scale, 16)))
    pa = np.histogram(a, edges)[0] / n
    pb = np.histogram(b, edges)[0] / n
    assert pa.min() > 0.002 and pb.min() > 0.002, "bins too sparse to compare"

    worst = max((pa / pb).max(), (pb / pa).max())
    # 5% slack covers sampling error at this n; truncation overshoots by 20%+.
    assert worst <= np.exp(epsilon) * 1.05, \
        f"Privacy loss ratio {worst:.3f} exceeds e^eps = {np.exp(epsilon):.3f}"


def test_clamped_laplace_bias_matches_documented_formula():
    """
    Clamping biases outputs upward by (scale/2) * exp(-(value - bound)/scale).

    Checked near the bound, where the bias is large, and far from it, where
    it is negligible.
    """
    sensitivity, epsilon = 1000.0, 0.25
    scale = sensitivity / epsilon
    n = 200_000

    np.random.seed(42)
    for value in (2000.0, 20000.0):
        samples = [clamped_laplace_mechanism(value, sensitivity, epsilon) for _ in range(n)]
        bias = np.mean(samples) - value
        expected = 0.5 * scale * np.exp(-value / scale)
        # Standard error of the mean is about scale*sqrt(2/n) ~= 13.
        assert abs(bias - expected) < 60, \
            f"value={value}: bias {bias:.1f}, expected {expected:.1f}"


def test_shifted_name_is_alias_for_clamped():
    """The deprecated name must not resurrect the truncated sampler."""
    assert shifted_laplace_mechanism is clamped_laplace_mechanism


def test_apply_differential_privacy_outputs_non_negative(capsys):
    """The pipeline releases non-negative spend and revenue, other columns untouched."""
    config = MMMConfig()
    np.random.seed(42)
    data = pd.DataFrame({
        "week": np.arange(1, 201),
        **{f"spend_{ch}": np.full(200, 500.0) for ch in config.channels},
        "revenue": np.full(200, 300.0),
        "promotions": np.zeros(200, dtype=int),
    })

    private = apply_differential_privacy(data, epsilon=1.0, config=config)
    capsys.readouterr()

    for col in [f"spend_{ch}" for ch in config.channels] + ["revenue"]:
        assert (private[col] >= 0).all()
        assert not private[col].equals(data[col])
    assert private["week"].equals(data["week"])
    assert private["promotions"].equals(data["promotions"])


def test_epsilon_validation():
    """Verify that mechanisms reject invalid epsilon values."""
    with pytest.raises(ValueError):
        laplace_mechanism(100, 10, epsilon=0)

    with pytest.raises(ValueError):
        laplace_mechanism(100, 10, epsilon=-1.0)

    with pytest.raises(ValueError):
        clamped_laplace_mechanism(100, 10, epsilon=0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
