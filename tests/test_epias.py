import sys
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import epias
from shared import ApiError


def _mock_response(status_code, json_data=None, text=""):
    resp = MagicMock()
    resp.status_code = status_code
    resp.ok = status_code < 400
    resp.text = text
    if json_data is not None:
        resp.json.return_value = json_data
    return resp


# --------------------------------------------------------------------------
# fetch_system_marginal_price_for_date
# --------------------------------------------------------------------------

def test_fetch_smf_success_with_recognized_field_name():
    body = {"items": [{"date": "2026-08-01T00:00:00+03:00", "price": 1850.5}]}
    resp = _mock_response(200, json_data=body)

    with patch("shared.requests.request", return_value=resp) as mock_request:
        rows = epias.fetch_system_marginal_price_for_date("TGT", date(2026, 8, 1))

    assert rows == body["items"]
    called_args = mock_request.call_args.args
    called_kwargs = mock_request.call_args.kwargs
    assert called_args[1] == epias.SMF_URL
    assert called_kwargs["headers"]["TGT"] == "TGT"
    assert called_kwargs["json"]["startDate"] == "2026-08-01T00:00:00+03:00"
    assert called_kwargs["json"]["endDate"] == "2026-08-01T23:59:59+03:00"


def test_fetch_smf_unrecognized_field_names_raises_diagnostic_error():
    # Simulates the real EPİAŞ response using different field names than
    # our guessed candidates - must fail loudly with the real keys shown,
    # not silently return garbage.
    body = {"items": [{"tarih": "2026-08-01T00:00:00+03:00", "fiyat": 1850.5}]}
    resp = _mock_response(200, json_data=body)

    with patch("shared.requests.request", return_value=resp):
        with pytest.raises(ApiError, match="alan adları tanınmadı"):
            epias.fetch_system_marginal_price_for_date("TGT", date(2026, 8, 1))


def test_fetch_smf_empty_response_does_not_raise():
    resp = _mock_response(200, json_data={"items": []})
    with patch("shared.requests.request", return_value=resp):
        rows = epias.fetch_system_marginal_price_for_date("TGT", date(2026, 8, 1))
    assert rows == []


def test_fetch_smf_expired_token_raises_token_expired_error():
    resp = _mock_response(401)
    with patch("shared.requests.request", return_value=resp):
        with pytest.raises(epias.TokenExpiredError):
            epias.fetch_system_marginal_price_for_date("TGT", date(2026, 8, 1))


def test_fetch_smf_http_error():
    # 4xx (other than 401/403) is returned as-is by request_with_retries
    # (no retry), so epias.py's own status check should raise EpiasError.
    resp = _mock_response(400)
    with patch("shared.requests.request", return_value=resp):
        with pytest.raises(epias.EpiasError, match="EPİAŞ API"):
            epias.fetch_system_marginal_price_for_date("TGT", date(2026, 8, 1))


# --------------------------------------------------------------------------
# fetch_imbalance_amount_for_date
# --------------------------------------------------------------------------

def test_fetch_imbalance_amount_success_with_recognized_field_name():
    body = {"items": [{"date": "2026-08-01T00:00:00+03:00", "imbalanceAmount": -12.3}]}
    resp = _mock_response(200, json_data=body)

    with patch("shared.requests.request", return_value=resp) as mock_request:
        rows = epias.fetch_imbalance_amount_for_date("TGT", date(2026, 8, 1))

    assert rows == body["items"]
    called_args = mock_request.call_args.args
    assert called_args[1] == epias.IMBALANCE_AMOUNT_URL


def test_fetch_imbalance_amount_unrecognized_field_names_raises_diagnostic_error():
    body = {"items": [{"saat": "00:00", "miktar": -12.3}]}
    resp = _mock_response(200, json_data=body)

    with patch("shared.requests.request", return_value=resp):
        with pytest.raises(ApiError, match="alan adları tanınmadı"):
            epias.fetch_imbalance_amount_for_date("TGT", date(2026, 8, 1))


# --------------------------------------------------------------------------
# fetch_imbalance_price_for_date (compat wrapper)
# --------------------------------------------------------------------------

def test_fetch_imbalance_price_for_date_wraps_smf():
    body = {"items": [{"date": "2026-08-01T00:00:00+03:00", "price": 1850.5}]}
    resp = _mock_response(200, json_data=body)

    with patch("shared.requests.request", return_value=resp) as mock_request:
        rows = epias.fetch_imbalance_price_for_date("TGT", date(2026, 8, 1))

    assert rows == body["items"]
    assert mock_request.call_args.args[1] == epias.SMF_URL
