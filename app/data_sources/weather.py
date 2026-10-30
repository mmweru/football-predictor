"""
Open-Meteo — free, no API key. Daily mean temperature, total precipitation, max wind at the stadium.
  history  : https://archive-api.open-meteo.com/v1/archive      (matches already played)
  forecast : https://api.open-meteo.com/v1/forecast             (matches up to ~14 days ahead)
Needs stadium coordinates on the team (run `python -m app.geocode_teams` once). Daily granularity, not kick-off hour:
good enough to flag gales and downpours, which is all the model uses.
"""

from __future__ import annotations

import datetime as dt
from collections import defaultdict
from typing import Dict, Optional, Tuple

import requests
from sqlalchemy.orm import Session

from app.data_sources import http
from app.models import Match, Team

ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"
FORECAST = "https://api.open-meteo.com/v1/forecast"
DAILY = "temperature_2m_mean,precipitation_sum,wind_speed_10m_max"


def fetch_daily(lat: float, lon: float, start: dt.date, end: dt.date, session: Optional[requests.Session] = None,
                today: Optional[dt.date] = None) -> Dict[dt.date, Tuple[float, float, float]]:
    today = today or dt.date.today()
    url = FORECAST if start >= today - dt.timedelta(days=1) else ARCHIVE
    params = {"latitude": round(lat, 4), "longitude": round(lon, 4), "start_date": start.isoformat(), "end_date": end.isoformat(),
              "daily": DAILY, "timezone": "Europe/London", "wind_speed_unit": "kmh"}
    d = http.get(session or requests.Session(), url, params=params, retries=2).json().get("daily", {})
    out = {}
    for i, day in enumerate(d.get("time", [])):
        t, p, w = (d.get(k, [None])[i] for k in ("temperature_2m_mean", "precipitation_sum", "wind_speed_10m_max"))
        out[dt.date.fromisoformat(day)] = (t, p, w)
    return out


def fill_weather(db: Session, session: Optional[requests.Session] = None, today: Optional[dt.date] = None,
                 horizon_days: int = 14) -> int:
    """Fills temp/wind/precip for matches missing them (one request per stadium per date-range). Returns matches updated."""
    today = today or dt.date.today()
    teams = {t.id: t for t in db.query(Team).all()}
    todo = defaultdict(list)
    for m in db.query(Match).filter(Match.temp_c.is_(None), Match.date <= today + dt.timedelta(days=horizon_days)).all():
        home = teams.get(m.home_team_id)
        lat, lon = (m.venue_lat, m.venue_lon) if m.venue_lat is not None else (home.stadium_lat if home else None, home.stadium_lon if home else None)
        if lat is None or lon is None or not (home and home.is_active):
            continue
        todo[(round(lat, 3), round(lon, 3))].append(m)
    n = 0
    for (lat, lon), matches in todo.items():
        past = [m for m in matches if m.date < today - dt.timedelta(days=1)]
        future = [m for m in matches if m.date >= today - dt.timedelta(days=1)]
        for group in (past, future):
            if not group:
                continue
            try:
                data = fetch_daily(lat, lon, min(m.date for m in group), max(m.date for m in group), session, today)
            except http.DataSourceError as e:
                print(f"  [weather skip] {lat},{lon}: {e}")
                continue
            for m in group:
                t, p, w = data.get(m.date, (None, None, None))
                if t is not None:
                    m.temp_c, m.precip_mm, m.wind_kph = t, p, w
                    n += 1
        db.commit()
    return n
