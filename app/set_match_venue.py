"""
Sets the venue (venue_lat, venue_lon, venue_name) for a specific match,
overriding the default assumption that a match is played at the home
team's registered stadium. Needed for local matches where the pitch isn't
the team's usual venue, or isn't geocodable by team name at all.

Three ways to provide the location:

  --address "Kasarani Stadium, Nairobi"
      Geocodes a typed address/place name via Nominatim (free).

  --lat -1.2921 --lon 36.8219
      Use exact coordinates directly — e.g. copied from
      static/detect_location.html (genuine GPS, see that file), or from
      Google Maps ("what's here?" gives you lat/lon directly).

  --auto-detect
      Coarse IP-based location as a rough starting point (free, no setup).
      WARNING: approximates your network's location, not the actual venue
      GPS position — often only accurate to city level. You'll be asked to
      confirm before it's saved. Prefer --lat/--lon from
      static/detect_location.html when you can.

Usage:
    python -m app.set_match_venue --match-id 42 --address "Kasarani Stadium, Nairobi"
    python -m app.set_match_venue --match-id 42 --lat -1.2921 --lon 36.8219 --name "Kasarani Stadium"
    python -m app.set_match_venue --match-id 42 --auto-detect
"""

from __future__ import annotations

import argparse

from app.database import SessionLocal
from app.models import Match
from app.venue import geocode_address, get_ip_based_location


def set_venue_from_address(match_id: int, address: str) -> None:
    result = geocode_address(address)
    if result is None:
        print(f"Could not geocode address: {address!r}. Try --lat/--lon instead.")
        return
    lat, lon = result
    _save_venue(match_id, lat, lon, venue_name=address)
    print(f"Set match {match_id} venue to {address!r} -> ({lat:.4f}, {lon:.4f})")


def set_venue_from_coordinates(match_id: int, lat: float, lon: float, name: str = None) -> None:
    _save_venue(match_id, lat, lon, venue_name=name)
    print(f"Set match {match_id} venue to ({lat:.4f}, {lon:.4f})" + (f" [{name}]" if name else ""))


def set_venue_from_ip_autodetect(match_id: int) -> None:
    result = get_ip_based_location()
    if result is None:
        print("IP-based location lookup failed. Try --address or --lat/--lon instead.")
        return
    lat, lon, description = result
    print(f"Approximate location detected: {description} ({lat:.4f}, {lon:.4f})")
    print("WARNING: this is your network's approximate location, not necessarily the match venue.")
    confirm = input("Use this as the match venue? [y/N]: ").strip().lower()
    if confirm != "y":
        print("Cancelled. Try --address or --lat/--lon for an accurate location instead.")
        return
    _save_venue(match_id, lat, lon, venue_name=f"Auto-detected: {description}")
    print(f"Set match {match_id} venue to ({lat:.4f}, {lon:.4f})")


def _save_venue(match_id: int, lat: float, lon: float, venue_name: str = None) -> None:
    db = SessionLocal()
    try:
        match = db.query(Match).filter_by(id=match_id).one_or_none()
        if match is None:
            print(f"No match found with id={match_id}")
            return
        match.venue_lat = lat
        match.venue_lon = lon
        if venue_name:
            match.venue_name = venue_name
        db.commit()
    finally:
        db.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Set the venue for a specific match.")
    parser.add_argument("--match-id", type=int, required=True)
    parser.add_argument("--address", default=None, help="Address/place name to geocode")
    parser.add_argument("--lat", type=float, default=None, help="Latitude (use with --lon)")
    parser.add_argument("--lon", type=float, default=None, help="Longitude (use with --lat)")
    parser.add_argument("--name", default=None, help="Optional display name for the venue")
    parser.add_argument("--auto-detect", action="store_true", help="Coarse IP-based auto-detect (approximate)")
    args = parser.parse_args()

    if args.address:
        set_venue_from_address(args.match_id, args.address)
    elif args.lat is not None and args.lon is not None:
        set_venue_from_coordinates(args.match_id, args.lat, args.lon, name=args.name)
    elif args.auto_detect:
        set_venue_from_ip_autodetect(args.match_id)
    else:
        parser.error("Provide one of: --address, --lat/--lon, or --auto-detect")


if __name__ == "__main__":
    main()
