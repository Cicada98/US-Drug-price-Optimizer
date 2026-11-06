import json

import pandas as pd
import pytest

from drug_price_optimization.agents.data_foundation_agent import DataFoundationAgent


def _rows(desc, ndc, cls, prices, generics):
    dates = pd.date_range("2025-01-01", periods=len(prices), freq="7D").strftime("%m/%d/%Y")
    return [
        {
            "NDC Description": desc, "NDC": ndc, "NADAC Per Unit": p, "Effective Date": d,
            "Pricing Unit": "EA", "Pharmacy Type Indicator": "C/I", "OTC": "N",
            "Explanation Code": "1", "Classification for Rate Setting": cls,
            "Corresponding Generic Drug NADAC Per Unit": g,
            "Corresponding Generic Drug Effective Date": d, "As of Date": d,
        }
        for p, g, d in zip(prices, generics, dates)
    ]


@pytest.fixture(scope="module")
def result():
    df = pd.DataFrame(
        _rows("ABILIFY 10 MG TABLET", 1, "B", [20.0] * 6, [0.10] * 5 + [0.20])          # premium ratio falls
        + _rows("FORTEO 600 MCG/2.4 ML PEN INJ", 2, "B-ANDA", [1726.0] * 6, [800.0] * 6)  # > $1,000 is real
        + _rows("DAYTRANA 10 MG/9 HR PATCH", 3, "B", [5.0] * 6, [9.0] * 6)              # brand below generic
    )
    return DataFoundationAgent().process_nadac_data(df)


def _profile(result, ndc):
    return next(p for p in result["drug_profiles"] if p["ndc"] == ndc)


def test_output_is_json_serialisable(result):
    json.dumps(result, allow_nan=False)


def test_b_anda_is_kept_but_brand_generic_is_unchanged(result):
    forteo = _profile(result, 2)["classification"]
    assert forteo["rate_setting_class"] == "B-ANDA"
    assert forteo["brand_generic"] == "Brand"  # downstream agents still rely on this value
    assert result["summary_statistics"]["rate_setting_class_split"] == {"B": 2, "B-ANDA": 1}


def test_expensive_prices_are_not_treated_as_errors(result):
    assert _profile(result, 2)["generic_pricing"]["brand_nadac_per_unit"] == 1726.0


def test_premium_uses_latest_record(result):
    gp = _profile(result, 1)["generic_pricing"]
    assert gp["generic_nadac_per_unit"] == 0.20
    assert gp["premium_ratio"] == pytest.approx(100.0)
    assert gp["premium_per_unit"] == pytest.approx(19.80)
    assert gp["cut_to_generic_parity_pct"] == pytest.approx(99.0)
    assert gp["brand_below_generic"] is False
    assert gp["premium_ratio_trend"] == "decreasing"


def test_brand_below_generic_is_flagged(result):
    gp = _profile(result, 3)["generic_pricing"]
    assert gp["brand_below_generic"] is True
    assert gp["cut_to_generic_parity_pct"] < 0
    assert result["summary_statistics"]["generic_premium"]["ndcs_brand_below_generic"] == 1
