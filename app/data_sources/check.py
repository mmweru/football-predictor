"""
Connectivity + shape check for every external source. Run this FIRST on your machine (the sandbox this code was
written in had no internet, so the live responses could not be inspected):

    python -m app.data_sources.check
"""

from __future__ import annotations

import json

import requests

from app.config import settings
from app.data_sources import http
from app.data_sources.footballdata_org import FootballDataOrg
from app.data_sources.bigballs import BigBalls
from app.data_sources.api_football import APIFootball
from app.leagues import BY_NAME


def _try(name, fn):
    try:
        res = fn()
        print(f"[ OK ] {name}: {res}")
    except Exception as e:  # noqa: BLE001 - diagnostic tool, show everything
        print(f"[FAIL] {name}: {type(e).__name__}: {e}")


def main() -> None:
    s = requests.Session()
    _try("football-data.co.uk E1 2025/26 csv", lambda: len(http.get(s, "https://www.football-data.co.uk/mmz4281/2526/E1.csv").text.splitlines()) - 1)
    _try("football-data.co.uk fixtures.csv", lambda: len(http.get(s, "https://www.football-data.co.uk/fixtures.csv").text.splitlines()) - 1)
    _try("FPL bootstrap-static", lambda: len(http.get(s, "https://fantasy.premierleague.com/api/bootstrap-static/").json()["elements"]))
    _try("Open-Meteo forecast", lambda: http.get(s, "https://api.open-meteo.com/v1/forecast", params={"latitude": 51.55, "longitude": -0.1, "daily": "temperature_2m_mean", "forecast_days": 1}).json()["daily"])
    if settings.football_data_org_token:
        fd = FootballDataOrg(settings.football_data_org_token, s)
        _try("football-data.org PL teams (expect 20)", lambda: len(fd.teams(BY_NAME["Premier League"], 2026)))
        _try("football-data.org ELC teams (expect 24)", lambda: len(fd.teams(BY_NAME["EFL Championship"], 2026)))
    else:
        print("[SKIP] football-data.org: FOOTBALL_DATA_ORG_TOKEN not set in .env")
    if settings.bigballs_api_key:
        bb = BigBalls(settings.bigballs_api_key, s)
        _try("BigBalls /v1/coverage", lambda: json.dumps(bb._get("/v1/coverage", sport="soccer"))[:200])
        _try("BigBalls EPL injuries", lambda: {"rows": len(bb.epl_absences())})
    else:
        print("[SKIP] BigBalls: BIGBALLS_API_KEY not set in .env")
    if settings.football_api_key:
        api = APIFootball(settings.football_api_key, s)
        _try("API-Football EPL 2024 fixtures", lambda: len(api.fixtures(BY_NAME["Premier League"], 2024)))
    else:
        print("[SKIP] API-Football: FOOTBALL_API_KEY not set in .env")


if __name__ == "__main__":
    main()
