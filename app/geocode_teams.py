"""
Fills in stadium_lat / stadium_lon for every team that doesn't have them
yet, using OpenStreetMap Nominatim (free, no API key required).

IMPORTANT — Nominatim's usage policy requires:
  - Max 1 request per second (enforced below via RateLimiter)
  - A real, descriptive User-Agent identifying your app (set below —
    change the contact email to your own before running this for real)
  - No heavy/bulk use — this is fine for a few hundred teams, not millions
Policy: https://operations.osmfoundation.org/policies/nominatim/

Usage:
    python -m app.geocode_teams
    python -m app.geocode_teams --team "Arsenal"   # geocode just one team

NOTE: this requires network access to nominatim.openstreetmap.org, which
isn't reachable from the sandbox this project was built in — the geocoding
LOGIC below is unit-tested against a mocked geocoder (see
tests/test_geocode.py), but you'll be the first to run it against the real
service. Run it once on your machine and check a few results against a map
before trusting the rest.
"""

from __future__ import annotations

import argparse
from typing import Optional

from app.database import SessionLocal
from app.models import Team
from app.venue import LEAGUE_FALLBACK_LOCATIONS, geocode_address


def build_geocoder():
    """
    Preflight check + returns the primary geocoder. Kept for backwards
    compatibility with existing tests. Note: geocode_all_teams below does
    NOT pass this return value into geocode_stadium for actual lookups —
    it's called here purely so the placeholder-User-Agent check inside
    _get_geocode_fn() fires immediately (before processing any teams)
    rather than failing on the very first team. Real lookups go through
    geocode_stadium(None, ...) so the automatic Nominatim -> Photon
    fallback in app.venue.geocode_address is actually used.
    """
    from app.venue import _get_geocode_fn
    return _get_geocode_fn()


def geocode_stadium(geocode_fn, team_name: str) -> Optional[tuple]:
    """
    Looks up "<team name> stadium". Returns (lat, lon) or None if nothing
    was found by either the primary or fallback geocoder. Thin wrapper
    around app.venue.geocode_address, kept separate so team-specific query
    phrasing ("<name> stadium") lives here rather than in the
    general-purpose venue module.

    Pass geocode_fn=None (the default in normal use) to get the automatic
    Nominatim -> Photon fallback behavior. Tests pass an explicit mock here
    to isolate the primary path only.
    """
    return geocode_address(f"{team_name} stadium", geocode_fn=geocode_fn)


def geocode_all_teams(only_team: Optional[str] = None) -> None:
    db = SessionLocal()
    build_geocoder()  # preflight: fails fast with a clear error if USER_AGENT wasn't customized

    try:
        query = db.query(Team).filter(Team.stadium_lat.is_(None))
        if only_team:
            query = query.filter(Team.name == only_team)
        teams = query.all()

        if not teams:
            print("No teams need geocoding (all already have coordinates, or team not found).")
            return

        print(f"Geocoding {len(teams)} team(s)...")
        exact_count, fallback_count, not_found_count = 0, 0, 0
        for team in teams:
            result = geocode_stadium(None, team.name)  # None -> automatic Nominatim -> Photon fallback
            if result is not None:
                lat, lon = result
                team.stadium_lat, team.stadium_lon = lat, lon
                db.commit()
                print(f"  [OK] {team.name!r} -> ({lat:.4f}, {lon:.4f})")
                exact_count += 1
                continue

            # Exact geocoding failed — try a league-level fallback location
            # instead of leaving this team permanently without coordinates.
            fallback = LEAGUE_FALLBACK_LOCATIONS.get(team.league)
            if fallback is not None:
                lat, lon = fallback
                team.stadium_lat, team.stadium_lon = lat, lon
                db.commit()
                print(f"  [APPROXIMATE] {team.name!r} -> ({lat:.4f}, {lon:.4f}) "
                      f"— exact stadium not found, using {team.league} regional fallback")
                fallback_count += 1
            else:
                print(f"  [NOT FOUND] {team.name!r} — you'll need to set this one manually")
                not_found_count += 1

        print(f"\nDone. {exact_count} exact, {fallback_count} approximate (regional fallback), {not_found_count} not found.")
        if fallback_count > 0:
            print("Teams marked [APPROXIMATE] are using a rough regional location, not their real stadium —")
            print("travel_distance_km for their matches will be a coarse estimate, not exact.")
        print("Spot-check a few [OK] results against a map before fully trusting them —")
        print("stadium name searches occasionally return the wrong venue or a training ground.")
    finally:
        db.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Geocode team stadiums via OpenStreetMap Nominatim.")
    parser.add_argument("--team", default=None, help="Geocode only this team (by exact name)")
    args = parser.parse_args()
    geocode_all_teams(only_team=args.team)


if __name__ == "__main__":
    main()
