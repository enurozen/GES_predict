import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from imbalance_cost import evaluate_financial, imbalance_cost, optimal_bid_quantile


# --------------------------------------------------------------------------
# optimal_bid_quantile
# --------------------------------------------------------------------------

def test_optimal_bid_quantile_symmetric_costs_is_median():
    assert optimal_bid_quantile(10.0, 10.0) == pytest.approx(0.5)


def test_optimal_bid_quantile_asymmetric_costs():
    # Being short is 3x more expensive than being long -> bid higher (0.75).
    assert optimal_bid_quantile(30.0, 10.0) == pytest.approx(0.75)


def test_optimal_bid_quantile_zero_costs_defaults_to_median():
    assert optimal_bid_quantile(0.0, 0.0) == pytest.approx(0.5)


def test_optimal_bid_quantile_negative_cost_raises():
    with pytest.raises(ValueError):
        optimal_bid_quantile(-1.0, 10.0)


# --------------------------------------------------------------------------
# imbalance_cost
# --------------------------------------------------------------------------

def test_imbalance_cost_zero_when_bid_matches_actual():
    cost = imbalance_cost([10.0, 20.0], [10.0, 20.0], price_short_mwh=100.0, price_long_mwh=50.0)
    assert list(cost) == pytest.approx([0.0, 0.0])


def test_imbalance_cost_uses_short_price_when_actual_below_bid():
    # actual=8, bid=10 -> under by 2 MWh -> short price applies.
    cost = imbalance_cost([8.0], [10.0], price_short_mwh=100.0, price_long_mwh=50.0)
    assert cost[0] == pytest.approx(2.0 * 100.0)


def test_imbalance_cost_uses_long_price_when_actual_above_bid():
    # actual=12, bid=10 -> over by 2 MWh -> long price applies.
    cost = imbalance_cost([12.0], [10.0], price_short_mwh=100.0, price_long_mwh=50.0)
    assert cost[0] == pytest.approx(2.0 * 50.0)


def test_imbalance_cost_accepts_per_hour_price_arrays():
    cost = imbalance_cost([8.0, 12.0], [10.0, 10.0],
                           price_short_mwh=[100.0, 999.0], price_long_mwh=[999.0, 50.0])
    assert list(cost) == pytest.approx([200.0, 100.0])


# --------------------------------------------------------------------------
# evaluate_financial
# --------------------------------------------------------------------------

def test_evaluate_financial_ranks_perfect_prediction_cheapest():
    actual = pd.Series([10.0, 20.0, 30.0])
    predictions = {
        "mükemmel": actual.copy(),
        "kötü": pd.Series([5.0, 25.0, 10.0]),
    }

    result = evaluate_financial(actual, predictions, price_short_mwh=100.0, price_long_mwh=100.0)

    assert result.iloc[0]["model"] == "mükemmel"
    assert result.iloc[0]["toplam_maliyet"] == pytest.approx(0.0)
    assert result.iloc[1]["toplam_maliyet"] > 0
