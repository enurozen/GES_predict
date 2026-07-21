import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shared import ApiError, request_with_retries


def _mock_response(status_code, headers=None):
    resp = MagicMock()
    resp.status_code = status_code
    resp.headers = headers or {}
    return resp


def test_retries_on_429_then_succeeds():
    responses = [_mock_response(429), _mock_response(429), _mock_response(200)]
    with patch("shared.requests.request", side_effect=responses) as m_req, patch("shared.time.sleep") as m_sleep:
        response = request_with_retries("GET", "https://example.com", timeout=5, error_context="ctx")

    assert response.status_code == 200
    assert m_req.call_count == 3
    assert m_sleep.call_count == 2


def test_429_honors_retry_after_header():
    responses = [_mock_response(429, headers={"Retry-After": "7"}), _mock_response(200)]
    with patch("shared.requests.request", side_effect=responses), patch("shared.time.sleep") as m_sleep:
        request_with_retries("GET", "https://example.com", timeout=5, error_context="ctx")

    m_sleep.assert_called_once_with(7)


def test_400_is_not_retried():
    resp = _mock_response(400)
    with patch("shared.requests.request", return_value=resp) as m_req:
        response = request_with_retries("GET", "https://example.com", timeout=5, error_context="ctx")

    assert response.status_code == 400
    assert m_req.call_count == 1


def test_429_exhausted_raises_api_error():
    resp = _mock_response(429)
    with patch("shared.requests.request", return_value=resp), patch("shared.time.sleep"):
        with pytest.raises(ApiError, match="ctx"):
            request_with_retries("GET", "https://example.com", timeout=5, error_context="ctx")
