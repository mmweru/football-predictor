"""
Travel distance feature: straight-line (great-circle) distance between two
stadiums, in kilometers, using the haversine formula.

This is deliberately NOT road/flight distance — haversine gives "as the
crow flies" distance, which is a reasonable proxy for travel burden and
needs zero external API calls once you have stadium coordinates (see
geocode_teams.py for how those get populated).
"""

import math

EARTH_RADIUS_KM = 6371.0


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Great-circle distance between two lat/lon points, in kilometers.
    """
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)

    a = (
        math.sin(d_phi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return EARTH_RADIUS_KM * c
