"""
Shared HTTP helpers for modules that call external APIs.

Provides a small retry/backoff wrapper around `requests`, following the same
pattern used by EPİAŞ integrations elsewhere: retry on connection errors,
timeouts, 429 (rate limit), and 5xx responses (transient problems, likely to
pass on retry), but return other client errors (4xx) as-is on the first try
since retrying won't change them.
"""

import time
from typing import Any, Optional

import requests

MAX_RETRIES = 6
BACKOFF_BASE_SECONDS = 1.0


class ApiError(Exception):
    """A user-facing error while talking to an external API."""


def request_with_retries(
    method: str, url: str, *, timeout: float, error_context: str, **kwargs: Any
) -> requests.Response:
    """Issue an HTTP request with exponential-backoff retries for transient failures.

    Retries on connection errors, timeouts, 429 (rate limit), and 5xx
    responses. Other client errors like 400/404 are returned as-is on the
    first try, since retrying won't change them.
    """
    last_exc: Optional[Exception] = None

    for attempt in range(MAX_RETRIES + 1):
        sleep_seconds = BACKOFF_BASE_SECONDS * (2**attempt)
        try:
            response = requests.request(method, url, timeout=timeout, **kwargs)
        except requests.exceptions.RequestException as exc:
            last_exc = exc
        else:
            if response.status_code != 429 and response.status_code < 500:
                return response
            last_exc = ApiError(
                f"{error_context}: server error (HTTP {response.status_code})."
            )
            if response.status_code == 429 and response.headers.get("Retry-After", "").isdigit():
                sleep_seconds = max(sleep_seconds, int(response.headers["Retry-After"]))

        if attempt < MAX_RETRIES:
            time.sleep(sleep_seconds)

    if isinstance(last_exc, requests.exceptions.RequestException):
        raise ApiError(f"{error_context}: {last_exc}") from last_exc
    raise last_exc  # the ApiError built above for repeated 5xx responses
