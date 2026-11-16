"""
Smoke tests for the user-facing entry points.

These guard against the entry points drifting from the mmm package API:
the app once wrote its slider values to a config dict that nothing read,
and the sweep example imported a name that no longer existed.
"""

import importlib
import os

import matplotlib

matplotlib.use("Agg")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_mobile_app_uses_slider_values():
    """The weeks and noise sliders must reach data generation."""
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(os.path.join(REPO_ROOT, "mmm_mobile_app.py"), default_timeout=120)
    app.run()
    assert not app.exception

    sliders = {s.label: s for s in app.slider}
    sliders["Weeks of data"].set_value(60)
    sliders["Baseline noise (σ)"].set_value(1500)
    sliders["Privacy Budget (ε)"].set_value(2.0)
    app.button[0].click().run()
    assert not app.exception

    state = app.session_state
    assert len(state["df_original"]) == 60
    assert len(state["results"]["predictions"]) == 60
    assert state["config"].noise_std == 1500
    assert state["config"].epsilon == 2.0


def test_sweep_example_imports_and_scores_a_fit(tmp_path, monkeypatch):
    """The sweep example imports cleanly and its metrics accept a real fit."""
    monkeypatch.chdir(tmp_path)  # the module creates its output directory on import
    sweep = importlib.import_module("examples.privacy_parameter_sweep")

    from mmm import MMMConfig, apply_differential_privacy, fit_model, generate_weekly_data

    config = MMMConfig()
    base = generate_weekly_data(config)
    private = apply_differential_privacy(base, 5.0, config)
    fitted = fit_model(private, config, n_starts=1)["fitted_params"]

    metrics = sweep.calculate_metrics(base, fitted, private)
    assert 0 <= metrics["r_squared"] <= 1
    assert metrics["mae"] > 0
