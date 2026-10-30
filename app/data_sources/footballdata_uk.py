"""
football-data.co.uk — free CSVs, NO API key, covers ALL FOUR leagues we need:
    E0 Premier League   E1 Championship   E2 League One   E3 League Two

This is the backbone for training because it is the only free source that has,
for every league and many seasons: results, shots, shots on target, corners,
fouls, cards AND pre-match bookmaker odds.

URLs (stable for 20+ years):
    season results : https://www.football-data.co.uk/mmz4281/<SSYY>/<DIV>.csv   e.g. .../mmz4281/2627/E1.csv
    upcoming games : https://www.football-data.co.uk/fixtures.csv   (all divisions, with odds)
Notes: https://www.football-data.co.uk/notes.txt
"""

from __future__ import annotations

import datetime as dt
import io
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd
import requests

from app.data_sources import http
from app.leagues import BY_DIV, LEAGUES, League, season_code

BASE = "https://www.football-data.co.uk"
CACHE_DIR = Path("data_cache")

# raw column -> our column
STAT_COLUMNS = {
    "HS": "home_shots", "AS": "away_shots", "HST": "home_sot", "AST": "away_sot",
    "HC": "home_corners", "AC": "away_corners", "HF": "home_fouls", "AF": "away_fouls",
    "HY": "home_yellow", "AY": "away_yellow", "HR": "home_red", "AR": "away_red",
}
# Pre-match odds, in order of preference. We deliberately do NOT use the
# "closing" (…C…) columns: closing odds don't exist yet when you predict an
# upcoming match, so training on them would make the model look better than it
# can ever be live.
ODDS_PREFERENCE = [
    ("AvgH", "AvgD", "AvgA"),       # market average (2019/20+)
    ("BbAvH", "BbAvD", "BbAvA"),    # market average (older files)
    ("B365H", "B365D", "B365A"),    # Bet365
    ("PSH", "PSD", "PSA"),          # Pinnacle
]

CANONICAL_COLUMNS = [
    "date", "league", "season", "home_team", "away_team", "home_score", "away_score",
    *STAT_COLUMNS.values(), "odds_home", "odds_draw", "odds_away", "referee",
]


def parse_date_series(s: pd.Series) -> pd.Series:
    """dd/mm/yyyy or dd/mm/yy -> datetime.date (NaT-safe)."""
    out = pd.to_datetime(s, format="%d/%m/%Y", errors="coerce")
    missing = out.isna() & s.notna()
    if missing.any():
        out[missing] = pd.to_datetime(s[missing], format="%d/%m/%y", errors="coerce")
    return out


def _pick_odds(df: pd.DataFrame) -> pd.DataFrame:
    """First odds family with real numbers in each row (preference order above)."""
    n = len(df)
    h, d, a = (np.full(n, np.nan) for _ in range(3))
    for ch, cd, ca in ODDS_PREFERENCE:
        if not {ch, cd, ca} <= set(df.columns):
            continue
        oh, od, oa = (pd.to_numeric(df[c], errors="coerce").to_numpy() for c in (ch, cd, ca))
        use = np.isnan(h) & (oh > 1) & (od > 1) & (oa > 1)
        h[use], d[use], a[use] = oh[use], od[use], oa[use]
    return pd.DataFrame({"odds_home": h, "odds_draw": d, "odds_away": a}, index=df.index)


def parse_csv_text(text: str, league: Optional[League] = None, season_start: Optional[int] = None) -> pd.DataFrame:
    """
    Parses one football-data.co.uk CSV into our canonical frame. Rows with an
    unparseable date or missing team names are dropped (these files often end with blank lines).
    Unplayed fixtures keep NaN scores. If `league` is None it is inferred from the `Div` column per row.
    """
    raw = pd.read_csv(io.StringIO(text.lstrip("\ufeff")), on_bad_lines="skip")
    raw.columns = [c.strip().lstrip("\ufeff") for c in raw.columns]
    need = {"Date", "HomeTeam", "AwayTeam"}
    if not need <= set(raw.columns):
        raise ValueError(f"Not a football-data.co.uk results file — missing {need - set(raw.columns)}; got {list(raw.columns)[:12]}")

    raw = raw.dropna(subset=["Date", "HomeTeam", "AwayTeam"]).copy()
    dates = parse_date_series(raw["Date"].astype(str).str.strip())
    raw = raw[dates.notna()].copy()
    dates = dates[dates.notna()]

    out = pd.DataFrame(index=raw.index)
    out["date"] = dates.dt.date
    if league is not None:
        out["league"] = league.name
    else:
        out["league"] = raw["Div"].map(lambda d: BY_DIV[d].name if d in BY_DIV else None) if "Div" in raw else None
    out["home_team"] = raw["HomeTeam"].astype(str).str.strip()
    out["away_team"] = raw["AwayTeam"].astype(str).str.strip()
    out["home_score"] = pd.to_numeric(raw.get("FTHG"), errors="coerce")
    out["away_score"] = pd.to_numeric(raw.get("FTAG"), errors="coerce")
    for src, dst in STAT_COLUMNS.items():
        out[dst] = pd.to_numeric(raw[src], errors="coerce") if src in raw else np.nan
    out = out.join(_pick_odds(raw))
    out["referee"] = raw["Referee"] if "Referee" in raw else None
    # Season: explicit if given, else derived from each match date (works for fixtures.csv, which spans a season boundary).
    if season_start is not None:
        out["season"] = season_start
    else:
        out["season"] = [d.year if d.month >= 7 else d.year - 1 for d in out["date"]]
    out = out[out["league"].notna()]
    return out[CANONICAL_COLUMNS].reset_index(drop=True)


def _cache_path(url: str) -> Path:
    return CACHE_DIR / (url.replace("https://", "").replace("/", "_"))


def fetch_text(url: str, session: Optional[requests.Session] = None, use_cache_if_offline: bool = True) -> str:
    """Downloads a CSV. On network failure falls back to the last cached copy (if any) so a flaky connection doesn't break a refresh."""
    session = session or requests.Session()
    CACHE_DIR.mkdir(exist_ok=True)
    try:
        resp = http.get(session, url, retries=2, backoff_s=1.5)
        _cache_path(url).write_text(resp.text, encoding="utf-8")
        return resp.text
    except http.DataSourceError:
        cached = _cache_path(url)
        if use_cache_if_offline and cached.exists():
            print(f"  [offline] using cached copy of {url}")
            return cached.read_text(encoding="utf-8")
        raise


def fetch_season(league: League, season_start: int, session: Optional[requests.Session] = None) -> pd.DataFrame:
    url = f"{BASE}/mmz4281/{season_code(season_start)}/{league.fduk_div}.csv"
    return parse_csv_text(fetch_text(url, session), league=league, season_start=season_start)


def fetch_fixtures(session: Optional[requests.Session] = None, today: Optional[dt.date] = None) -> pd.DataFrame:
    """Upcoming matches for the 4 leagues (fixtures.csv covers every division; we keep E0-E3, dates >= today)."""
    today = today or dt.date.today()
    df = parse_csv_text(fetch_text(f"{BASE}/fixtures.csv", session))
    return df[df["date"] >= today].reset_index(drop=True)


def fetch_history(
    seasons: Iterable[int], session: Optional[requests.Session] = None, leagues: Optional[List[League]] = None
) -> pd.DataFrame:
    """Several seasons x all leagues, concatenated. A season file that doesn't exist yet (404) is skipped with a note."""
    frames = []
    for lg in leagues or LEAGUES:
        for yr in seasons:
            try:
                frames.append(fetch_season(lg, yr, session))
            except (http.DataSourceError, ValueError) as e:
                print(f"  [skip] {lg.name} {yr}: {e}")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=CANONICAL_COLUMNS)


def load_local_csv(path: str, league: League, season_start: Optional[int] = None) -> pd.DataFrame:
    """For CSV files you already downloaded by hand (like your E0_*.csv files)."""
    text = Path(path).read_text(encoding="utf-8-sig", errors="replace")
    df = parse_csv_text(text, league=league, season_start=None)
    if season_start is None and len(df):
        # A whole file is one season. Infer it from the MOST COMMON per-match season, so a COVID-delayed
        # 2019/20 file (finished in July 2020) doesn't get split across two seasons.
        season_start = int(df["season"].mode().iloc[0])
    if season_start is not None:
        df["season"] = season_start
    return df
