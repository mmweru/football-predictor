"""Geocoding and coarse IP-based venue location helpers."""

from __future__ import annotations

from typing import Optional, Tuple

import requests
from geopy.extra.rate_limiter import RateLimiter
from geopy.geocoders import Nominatim, Photon

USER_AGENT = "football-predictor-app (mwerumaryann@gmail.com)"
GEOCODE_TIMEOUT_SECONDS: float = 10.0
UK_COORDINATE_BOUNDS = (49.5, 61.0, -8.7, 2.2)
LEAGUE_FALLBACK_LOCATIONS = {
    "KPL": (-1.286389, 36.817223),
}

_geocode_fn = None
_fallback_geocode_fn = None


def is_uk_coordinate(latitude: float, longitude: float) -> bool:
    min_lat, max_lat, min_lon, max_lon = UK_COORDINATE_BOUNDS
    return min_lat <= latitude <= max_lat and min_lon <= longitude <= max_lon


def _get_geocode_fn():
    global _geocode_fn
    if _geocode_fn is None:
        if USER_AGENT.startswith("REPLACE_ME") or "example.com" in USER_AGENT:
            raise ValueError("Set a descriptive USER_AGENT in app/venue.py before using Nominatim.")
        geolocator = Nominatim(user_agent=USER_AGENT)
        setattr(geolocator, "timeout", GEOCODE_TIMEOUT_SECONDS)
        _geocode_fn = RateLimiter(geolocator.geocode, min_delay_seconds=1, max_retries=2, error_wait_seconds=2)
    return _geocode_fn


def _get_fallback_geocode_fn():
    global _fallback_geocode_fn
    if _fallback_geocode_fn is None:
        geolocator = Photon(user_agent=USER_AGENT)
        setattr(geolocator, "timeout", GEOCODE_TIMEOUT_SECONDS)
        _fallback_geocode_fn = RateLimiter(geolocator.geocode, min_delay_seconds=1, max_retries=1)
    return _fallback_geocode_fn


def geocode_address(address: str, geocode_fn=None, use_fallback: bool = True,
                    country_codes: Optional[str] = None) -> Optional[Tuple[float, float]]:
    if geocode_fn is not None:
        location = geocode_fn(address)
        return (location.latitude, location.longitude) if location else None

    try:
        if country_codes:
            location = _get_geocode_fn()(address, country_codes=country_codes)
        else:
            location = _get_geocode_fn()(address)
        if location is not None:
            return (location.latitude, location.longitude)
    except Exception as error:
        print(f"  Nominatim lookup failed for {address!r} ({error}); trying fallback geocoder...")

    if not use_fallback:
        return None
    try:
        location = _get_fallback_geocode_fn()(address)
        if location is not None:
            return (location.latitude, location.longitude)
    except Exception as error:
        print(f"  Fallback geocoder also failed for {address!r}: {error}")
    return None


def get_ip_based_location(http_get=None) -> Optional[Tuple[float, float, str]]:
    getter = http_get or requests.get
    try:
        response = getter("https://ipapi.co/json/", timeout=5)
        response.raise_for_status()
        data = response.json()
    except Exception as error:
        print(f"IP geolocation lookup failed: {error}")
        return None
    lat, lon = data.get("latitude"), data.get("longitude")
    if lat is None or lon is None:
        return None
    description = f"{data.get('city', 'Unknown city')}, {data.get('country_name', 'Unknown country')}"
    return (lat, lon, description)