"""
Tests for app.distance — verified against known real-world distances.
"""

import pytest

from app.distance import haversine_km


def test_same_point_is_zero_distance():
    assert haversine_km(51.5074, -0.1278, 51.5074, -0.1278) == pytest.approx(0.0, abs=0.01)


def test_london_to_manchester_known_distance():
    # London (Emirates area) to Manchester (Old Trafford area) is ~262km straight-line.
    london = (51.5549, -0.1084)
    manchester = (53.4631, -2.2913)
    distance = haversine_km(*london, *manchester)
    assert 255 < distance < 270  # allow a small margin around the known ~262km


def test_distance_is_symmetric():
    a = (51.5549, -0.1084)
    b = (40.7128, -74.0060)  # New York, for a large-distance sanity check
    assert haversine_km(*a, *b) == pytest.approx(haversine_km(*b, *a))


def test_london_to_new_york_known_distance():
    # Known great-circle distance is ~5570km.
    london = (51.5074, -0.1278)
    new_york = (40.7128, -74.0060)
    distance = haversine_km(*london, *new_york)
    assert 5500 < distance < 5650
