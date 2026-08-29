"""
Venue location resolution — three ways to get coordinates for a match venue:

1. geocode_address()   — turn a typed address/place name into lat/lon (Nominatim, free)
2. get_ip_based_location() — coarse auto-detect fallback (free, no key, but approximate — see warning below)
3. Live GPS — NOT possible from this Python script at all. See static/detect_location.html,
   which uses the browser's Geolocation API on whatever device opens it (phone/laptop) to
   get genuine GPS coordinates, which you then feed into app.set_match_venue manually.

IMPORTANT about get_ip_based_location(): this returns the approximate location
of whatever network connection is making the request — often accurate only to
city level, and can be quite wrong on mobile data, VPNs, or corporate networks.
Treat it as a rough starting point to confirm/correct, never as ground truth
for exactly which field a match is being played on.
"""

from __future__ import annotations

import time
from typing import Optional, Tuple

import requests
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter

USER_AGENT = "football-predictor-app (mwerumaryann@gmail.com)"

# When a team's stadium genuinely can't be geocoded by name (common for
# small local clubs — e.g. "Maili Saba Combined" has no findable venue on
# Nominatim or Photon), fall back to an approximate central location for
# that team's league/region rather than leaving travel_distance_km
# permanently null for every one of that team's matches. This is a coarse
# approximation, not a real venue — every team using a fallback is logged
# clearly (see geocode_teams.py) so you know which coordinates are exact
# vs approximate.
LEAGUE_FALLBACK_LOCATIONS = {
    "KPL": (-1.286389, 36.817223),  # Nairobi, Kenya — center-of-country approximation
}

# geopy's Nominatim defaults to a 1-second timeout, which is too short for
# real-world network latency and causes spurious GeocoderUnavailable /
# ReadTimeoutError failures even when the service is working fine. 10
# seconds is a safer default for a free public service.
GEOCODE_TIMEOUT_SECONDS = 10

_geocode_fn = None  # lazily built, so importing this module doesn't require network access
_fallback_geocode_fn = None


def _get_geocode_fn():
    """
    Builds (and caches) the primary Nominatim geocoder.

    IMPORTANT — Nominatim actively rejects requests with a generic or
    unfilled placeholder User-Agent (returns HTTP 403 /
    GeocoderInsufficientPrivileges), even though the RateLimiter will
    happily retry a 403 a couple of times before giving up — those retries
    are wasted, since a 403 from a bad User-Agent will never succeed no
    matter how many times you ask. This check fails fast and tells you
    exactly what to fix, instead of burning through your whole team list
    getting the same rejection 18 times.

    Fix: open this file and change USER_AGENT to something that actually
    identifies your app — e.g. "my-football-predictor/1.0". No need for a
    real email address; Nominatim's policy just wants *something*
    non-generic, not personal contact info specifically.
    Policy: https://operations.osmfoundation.org/policies/nominatim/
    """
    global _geocode_fn
    if _geocode_fn is None:
        if USER_AGENT.startswith("REPLACE_ME") or "example.com" in USER_AGENT:
            raise ValueError(
                "app/venue.py: USER_AGENT is still set to its placeholder value. "
                "Nominatim rejects generic/placeholder user agents with a 403 error. "
                "Edit USER_AGENT in app/venue.py to something identifying your app "
                "(e.g. 'my-football-predictor/1.0') before geocoding."
            )
        geolocator = Nominatim(user_agent=USER_AGENT, timeout=GEOCODE_TIMEOUT_SECONDS)
        _geocode_fn = RateLimiter(geolocator.geocode, min_delay_seconds=1, max_retries=2, error_wait_seconds=2.0)
    return _geocode_fn


def _get_fallback_geocode_fn():
    """
    Photon (https://photon.komoot.io) — a free, OSM-data-based geocoder run
    by komoot, used as a fallback when Nominatim is unavailable, rate-limited,
    or blocked. No API key required. Its usage policy is more lenient than
    Nominatim's (no mandatory User-Agent requirement), but it's still a free
    shared service — don't hammer it either.
    """
    global _fallback_geocode_fn
    if _fallback_geocode_fn is None:
        from geopy.geocoders import Photon

        geolocator = Photon(user_agent=USER_AGENT, timeout=GEOCODE_TIMEOUT_SECONDS)
        _fallback_geocode_fn = RateLimiter(geolocator.geocode, min_delay_seconds=1, max_retries=1)
    return _fallback_geocode_fn


def geocode_address(address: str, geocode_fn=None, use_fallback: bool = True) -> Optional[Tuple[float, float]]:
    """
    Converts a typed address or place name (e.g. "Kasarani Stadium, Nairobi"
    or "Church Road playing field, Reading") into (lat, lon).

    Tries Nominatim first; if that raises an error (blocked, rate-limited,
    timed out) and use_fallback=True (the default), falls back to Photon.
    Returns None only if both fail, or if the address genuinely isn't found
    by either.

    `geocode_fn` is exposed as a parameter (rather than always building one
    internally) specifically so tests can inject a mock instead of hitting
    the real network — when provided, no fallback is attempted (this is
    purely for isolated unit testing of the primary path).
    """
    if geocode_fn is not None:
        location = geocode_fn(address)
        return (location.latitude, location.longitude) if location else None

    try:
        location = _get_geocode_fn()(address)
        if location is not None:
            return (location.latitude, location.longitude)
    except Exception as e:
        print(f"  Nominatim lookup failed for {address!r} ({e}); trying fallback geocoder...")

    if not use_fallback:
        return None

    try:
        location = _get_fallback_geocode_fn()(address)
        if location is not None:
            return (location.latitude, location.longitude)
    except Exception as e:
        print(f"  Fallback geocoder also failed for {address!r}: {e}")

    return None


def get_ip_based_location(http_get=None) -> Optional[Tuple[float, float, str]]:
    """
    Coarse auto-detect via IP geolocation (ipapi.co, free, no API key
    required for reasonable usage volumes). Returns (lat, lon, description)
    or None on failure.

    `http_get` is exposed as a parameter so tests can inject a mock instead
    of hitting the real network — same pattern as geocode_address.

    Free tier limits (check current terms before heavy use):
    https://ipapi.co/api/#introduction
    """
    getter = http_get or requests.get
    try:
        response = getter("https://ipapi.co/json/", timeout=5)
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        print(f"IP geolocation lookup failed: {e}")
        return None

    lat, lon = data.get("latitude"), data.get("longitude")
    if lat is None or lon is None:
        return None

    description = f"{data.get('city', 'Unknown city')}, {data.get('country_name', 'Unknown country')}"
    return (lat, lon, description)
