import pytest

from scripts.load_balance_metrics import (
    empirical_cdf,
    load_balance_coefficient,
    load_variance,
    normalize_load_balance_metrics,
    summarize_load_variance_samples,
)


def test_load_variance_uses_only_active_satellites():
    loads = [0.0, 0.25, 0.75]
    active = [False, True, True]

    assert load_variance(loads, active_mask=active) == pytest.approx(0.0625)


def test_load_balance_coefficient_matches_paper_mapping():
    assert load_balance_coefficient(0.0) == pytest.approx(1.0)
    assert load_balance_coefficient(0.0625) == pytest.approx(0.6)
    assert load_balance_coefficient(0.25) == pytest.approx(0.0)


def test_empirical_cdf_returns_sorted_variance_distribution():
    points = empirical_cdf([0.03, 0.01, 0.02])

    assert points == [
        {"x": pytest.approx(0.01), "cdf": pytest.approx(1 / 3)},
        {"x": pytest.approx(0.02), "cdf": pytest.approx(2 / 3)},
        {"x": pytest.approx(0.03), "cdf": pytest.approx(1.0)},
    ]


def test_summary_converts_mean_variance_to_balance_score():
    summary = summarize_load_variance_samples([0.0, 0.0625, 0.125])

    assert summary["load_balance_variance"] == pytest.approx(0.0625)
    assert summary["load_balance_coefficient"] == pytest.approx(0.6)
    assert summary["load_variance_sample_count"] == 3


def test_normalize_load_balance_metrics_replaces_legacy_aliases_when_variance_exists():
    normalized = normalize_load_balance_metrics(
        {
            "load_balance_variance": 0.0625,
            "mec_load_fairness": 0.99,
            "avg_load_balance_score": 0.99,
            "active_load_balance_score": 0.99,
        }
    )

    assert normalized["load_balance_coefficient"] == pytest.approx(0.6)
    assert normalized["mec_load_fairness"] == pytest.approx(0.6)
    assert normalized["avg_load_balance_score"] == pytest.approx(0.6)
    assert normalized["active_load_balance_score"] == pytest.approx(0.6)
