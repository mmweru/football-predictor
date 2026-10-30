"""
Tiny shared HTTP layer for every external data source: one place for the
User-Agent, timeouts, retry-with-backoff on 429/5xx, and per-host rate limiting
(football-data.org's free tier allows only 10 requests/minute, BigBalls free
allows 250/day). Clients take a `session` argument so tests can inject a fake.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional

import requests

USER_AGENT = "football-predictor-app (mwerumaryann@gmail.com)"


class DataSourceError(RuntimeError):
    """Raised for any non-retryable failure talking to an external source."""


class RateLimiter:
    def __init__(self, min_interval_s: float = 0.0):
        self.min_interval_s = min_interval_s
        self._last = 0.0

    def wait(self) -> None:
        if self.min_interval_s <= 0:
            return
        delta = time.monotonic() - self._last
        if delta < self.min_interval_s:
            time.sleep(self.min_interval_s - delta)
        self._last = time.monotonic()


def get(
    session: requests.Session,
    url: str,
    *,
    headers: Optional[Dict[str, str]] = None,
    params: Optional[Dict[str, Any]] = None,
    limiter: Optional[RateLimiter] = None,
    retries: int = 3,
    backoff_s: float = 2.0,
    timeout_s: float = 30.0,
) -> requests.Response:
    """GET with retries on 429 / 5xx / connection errors. 4xx (other than 429) fail immediately."""
    hdrs = {"User-Agent": USER_AGENT}
    hdrs.update(headers or {})
    last_exc: Optional[Exception] = None
    for attempt in range(retries + 1):
        if limiter:
            limiter.wait()
        try:
            resp = session.get(url, headers=hdrs, params=params, timeout=timeout_s)
        except requests.RequestException as e:  # connection reset, DNS, timeout...
            last_exc = e
        else:
            if resp.status_code == 200:
                return resp
            if resp.status_code in (429, 500, 502, 503, 504):
                last_exc = DataSourceError(f"{url} -> HTTP {resp.status_code}")
                retry_after = resp.headers.get("Retry-After")
                if retry_after and retry_after.isdigit():
                    time.sleep(min(int(retry_after), 65))
            else:
                raise DataSourceError(f"{url} -> HTTP {resp.status_code}: {resp.text[:200]}")
        if attempt < retries:
            time.sleep(backoff_s * (2 ** attempt))
    raise DataSourceError(f"{url} failed after {retries + 1} attempts: {last_exc}")
