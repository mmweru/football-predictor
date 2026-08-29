"""
Additional test verifying the Nominatim geocoder timeout fix — confirms
_get_geocode_fn() actually configures a longer timeout than geopy's
default, without needing to hit the real network.
"""

from unittest.mock import MagicMock, patch

import pytest

from app import venue


def test_get_geocode_fn_uses_extended_timeout(monkeypatch):
    # Reset the module-level cache so this test builds a fresh instance
    # regardless of what other tests/imports may have already triggered.
    monkeypatch.setattr(venue, "_geocode_fn", None)
    # Use a valid (non-placeholder) user agent for this test — the real
    # placeholder-rejection behavior is covered separately below.
    monkeypatch.setattr(venue, "USER_AGENT", "test-app/1.0")

    captured_kwargs = {}

    def fake_nominatim(*args, **kwargs):
        captured_kwargs.update(kwargs)
        return MagicMock()

    with patch.object(venue, "Nominatim", side_effect=fake_nominatim):
        venue._get_geocode_fn()

    assert captured_kwargs.get("timeout") == venue.GEOCODE_TIMEOUT_SECONDS
    assert venue.GEOCODE_TIMEOUT_SECONDS > 1  # must exceed geopy's spurious-timeout-prone 1s default


def test_get_geocode_fn_rejects_placeholder_user_agent(monkeypatch):
    monkeypatch.setattr(venue, "_geocode_fn", None)
    monkeypatch.setattr(venue, "USER_AGENT", "REPLACE_ME_something")

    with pytest.raises(ValueError, match="placeholder"):
        venue._get_geocode_fn()


def test_get_geocode_fn_rejects_example_dot_com_user_agent(monkeypatch):
    monkeypatch.setattr(venue, "_geocode_fn", None)
    monkeypatch.setattr(venue, "USER_AGENT", "my-app (contact: someone@example.com)")

    with pytest.raises(ValueError, match="placeholder"):
        venue._get_geocode_fn()
