"""
Unified feature engine.

ONE chronological pass over every match (played or upcoming) produces the
feature row for that match using ONLY what was known before kickoff. Played
matches then update the running state; upcoming fixtures do not. Because
training rows and live-prediction rows come out of the very same code path,
there is no train/serve skew (the #1 silent killer of football models), and a
whole 4-league history builds in seconds instead of ~12 SQL queries per match.

Feature groups (the numbers in brackets are YOUR requested metrics):

  elo          team strength, tier-aware, goal-margin weighted
  form         recent results, short + long memory, venue-split form
  team_stats   [8] shots / shots on target / corners / discipline, for & against (xG proxy)
  situation    [9] attack & defence strength vs league -> expected goals for each side
  home_adv     [2] league-wide AND team-specific home advantage
  fatigue      [3] rest days + games in the last 7/14/30 days (league + any extra games supplied)
  context      [4] derby, travel distance, weather, midweek, season progress, league tier
  h2h          [5] recency-weighted head-to-head
  suspension   [1] proxy for unavailability that exists for EVERY league/season: red card last game, yellow-card load
  market       pre-match bookmaker odds, de-margined (this is how the model "knows" about lineups/injuries/news)
  injury       [1] OPTIONAL real injury counts (only trainable once injury history has accumulated)

Public API:
    engine = FeatureEngine(teams, extra_games=None, injury_lookup=None)
    frame  = engine.run(matches_df)   # one row per input match, features + identifiers
"""

from __future__ import annotations

import datetime as dt
import math
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from app.derbies import is_derby
from app.distance import haversine_km
from app.leagues import TIER_BY_NAME, canonical_league
from app.models import Match, Team

# ----------------------------------------------------------------------------
# Tunables (sensible defaults; all were chosen up-front, none fitted to the test set)
# ----------------------------------------------------------------------------
ELO_K = 20.0
ELO_HOME_ADV = 65.0
ELO_BASE_BY_TIER = {1: 1550.0, 2: 1480.0, 3: 1420.0, 4: 1370.0}   # initial rating + season mean-reversion target
ELO_SEASON_KEEP = 0.75        # fraction of rating kept into a new season
ALPHA_SHORT = 0.25            # EWMA weight of the newest match (~ last 3-4 matches matter most)
ALPHA_LONG = 0.08             # ~ last 12 matches
SHRINK_MATCHES = 6.0          # attack/defence ratings are shrunk toward league average with this many "pseudo matches"
LOCAL_DERBY_KM = 40.0         # two clubs this close are treated as local rivals even if not in derbies.py
MAX_REST_DAYS = 14
H2H_MEETINGS = 6
H2H_DECAY = 0.8

FEATURE_GROUPS: Dict[str, List[str]] = {
    "elo": ["home_elo", "away_elo", "elo_diff", "home_elo_matches", "away_elo_matches"],
    "form": [
        "home_ppg_s", "away_ppg_s", "home_ppg_l", "away_ppg_l", "home_gd_s", "away_gd_s", "home_gd_l", "away_gd_l",
        "home_win5", "away_win5", "home_venue_ppg", "away_venue_ppg", "home_venue_gd", "away_venue_gd",
        "ppg_diff_s", "ppg_diff_l",
    ],
    "team_stats": [
        "home_sot_for", "home_sot_against", "away_sot_for", "away_sot_against",
        "home_shots_for", "home_shots_against", "away_shots_for", "away_shots_against",
        "home_corners_for", "away_corners_for", "home_xgp_for", "home_xgp_against", "away_xgp_for", "away_xgp_against",
        "home_sot_share", "away_sot_share", "xgp_diff",
    ],
    "situation": [
        "home_attack", "home_defence", "away_attack", "away_defence", "exp_home_goals", "exp_away_goals", "exp_goal_diff",
    ],
    "home_adv": ["league_home_win_rate", "league_home_gd", "home_team_hfa", "away_team_hfa"],
    "fatigue": [
        "home_rest_days", "away_rest_days", "rest_diff", "home_games_7d", "away_games_7d",
        "home_games_14d", "away_games_14d", "home_games_30d", "away_games_30d", "home_short_rest", "away_short_rest",
    ],
    "context": [
        "is_derby", "travel_km", "temp_c", "wind_kph", "precip_mm", "bad_weather", "is_midweek",
        "season_progress", "league_tier",
    ],
    "h2h": ["h2h_n", "h2h_home_score", "h2h_goal_diff", "h2h_days_since"],
    "suspension": ["home_red_last", "away_red_last", "home_yellow_load", "away_yellow_load"],
    "market": ["mkt_p_home", "mkt_p_draw", "mkt_p_away"],
    "injury": ["home_inj_count", "away_inj_count", "home_inj_importance", "away_inj_importance"],
}
# Used by default. "injury" is opt-in because historical injury rows only exist once you have accumulated them.
DEFAULT_GROUPS = ["elo", "form", "team_stats", "situation", "home_adv", "fatigue", "context", "h2h", "suspension"]
CATEGORICAL_FEATURES: List[str] = []   # league tier is numeric on purpose (ordered 1-4), so no category handling is needed


def feature_columns(groups: Iterable[str]) -> List[str]:
    cols: List[str] = []
    for g in groups:
        cols.extend(FEATURE_GROUPS[g])
    return cols


def implied_probabilities(oh: float, od: float, oa: float) -> Tuple[float, float, float]:
    """Decimal odds -> probabilities with the bookmaker margin removed proportionally."""
    if not all(x is not None and not (isinstance(x, float) and math.isnan(x)) and x > 1.0 for x in (oh, od, oa)):
        return (np.nan, np.nan, np.nan)
    ih, idr, ia = 1 / oh, 1 / od, 1 / oa
    s = ih + idr + ia
    return (ih / s, idr / s, ia / s)


def goal_margin_multiplier(goal_diff: int) -> float:
    gd = abs(goal_diff)
    if gd <= 1:
        return 1.0
    if gd == 2:
        return 1.5
    return (11.0 + gd) / 8.0


def elo_expected(home_elo: float, away_elo: float, home_adv: float = ELO_HOME_ADV) -> float:
    return 1.0 / (1.0 + 10 ** (-((home_elo + home_adv) - away_elo) / 400.0))


def _ew(store: Dict[str, float], key: str, value: Optional[float], alpha: float) -> None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return
    store[key] = value if key not in store else alpha * value + (1 - alpha) * store[key]


def _get(store: Dict[str, float], key: str) -> float:
    return store.get(key, np.nan)


@dataclass
class TeamState:
    elo: float
    league_tier: int = 1
    n: int = 0
    season: Optional[int] = None
    season_games: int = 0
    last_date: Optional[dt.date] = None
    dates: List[dt.date] = field(default_factory=list)
    s: Dict[str, float] = field(default_factory=dict)         # short-memory EWMAs
    l: Dict[str, float] = field(default_factory=dict)         # long-memory EWMAs
    home: Dict[str, float] = field(default_factory=dict)      # EWMAs of matches played at home
    away: Dict[str, float] = field(default_factory=dict)      # EWMAs of matches played away
    last5: deque = field(default_factory=lambda: deque(maxlen=5))
    last_red: float = 0.0


@dataclass
class LeagueState:
    home_goals: float = 1.50
    away_goals: float = 1.20
    home_win_rate: float = 0.44
    home_gd: float = 0.30
    n: int = 0


def _nan(v) -> bool:
    return v is None or (isinstance(v, float) and math.isnan(v))


class FeatureEngine:
    def __init__(
        self,
        teams: Dict[int, dict],
        extra_games: Optional[Dict[int, List[dt.date]]] = None,
        injury_lookup: Optional[Callable[[int, dt.date], Tuple[float, float]]] = None,
    ):
        """
        teams        : {team_id: {"name", "lat", "lon"}}  (lat/lon may be None)
        extra_games  : {team_id: [dates]} cup / European games to count toward fatigue
        injury_lookup: f(team_id, date) -> (injured_count, importance_sum) or (nan, nan) when unknown
        """
        self.teams = teams
        self.extra_games = {tid: sorted(ds) for tid, ds in (extra_games or {}).items()}
        self.injury_lookup = injury_lookup
        self.state: Dict[int, TeamState] = {}
        self.league_state: Dict[str, LeagueState] = {}
        self.h2h: Dict[Tuple[int, int], List[Tuple[dt.date, int, int, int]]] = {}  # pair -> [(date, home_id, hg, ag)]

    # ------------------------------------------------------------------ helpers
    def _team(self, tid: int, tier: int) -> TeamState:
        st = self.state.get(tid)
        if st is None:
            st = TeamState(elo=ELO_BASE_BY_TIER[tier], league_tier=tier)
            self.state[tid] = st
        return st

    def _league(self, league: str) -> LeagueState:
        return self.league_state.setdefault(league, LeagueState())

    @staticmethod
    def _season_roll(st: TeamState, season: int, tier: int) -> None:
        """First match of a new season: pull the rating part-way toward the mean of the division the club is in NOW."""
        if st.season is not None and season != st.season:
            base = ELO_BASE_BY_TIER[tier]
            st.elo = ELO_SEASON_KEEP * st.elo + (1 - ELO_SEASON_KEEP) * base
            st.season_games = 0
        st.season = season
        st.league_tier = tier

    def _games_in_window(self, tid: int, st: TeamState, d: dt.date, days: int) -> int:
        lo = d - dt.timedelta(days=days)
        n = sum(1 for x in st.dates[-14:] if lo <= x < d)
        extra = self.extra_games.get(tid)
        if extra:
            n += sum(1 for x in extra if lo <= x < d and x not in st.dates[-14:])
        return n

    def _rest_days(self, tid: int, st: TeamState, d: dt.date) -> float:
        prev = [x for x in st.dates[-3:] if x < d]
        extra = [x for x in self.extra_games.get(tid, []) if x < d][-3:]
        cand = prev + extra
        # Capped: beyond ~2 weeks a team is simply "fully rested" (and a 90-day summer gap is not comparable to a 3-day turnaround).
        return float(min((d - max(cand)).days, MAX_REST_DAYS)) if cand else np.nan

    def _distance(self, home_id: int, away_id: int, vlat, vlon) -> float:
        h, a = self.teams.get(home_id, {}), self.teams.get(away_id, {})
        dest = (vlat, vlon) if not _nan(vlat) and not _nan(vlon) else (h.get("lat"), h.get("lon"))
        if None in dest or _nan(a.get("lat")) or a.get("lat") is None or a.get("lon") is None:
            return np.nan
        return haversine_km(a["lat"], a["lon"], dest[0], dest[1])

    def _is_derby(self, home_id: int, away_id: int) -> float:
        h, a = self.teams.get(home_id, {}), self.teams.get(away_id, {})
        if is_derby(h.get("name", ""), a.get("name", "")):
            return 1.0
        if all(not _nan(x) and x is not None for x in (h.get("lat"), h.get("lon"), a.get("lat"), a.get("lon"))):
            return 1.0 if haversine_km(h["lat"], h["lon"], a["lat"], a["lon"]) <= LOCAL_DERBY_KM else 0.0
        return 0.0

    # ------------------------------------------------------------------ features
    def _h2h_features(self, home_id: int, away_id: int, d: dt.date) -> Dict[str, float]:
        meetings = [m for m in self.h2h.get(tuple(sorted((home_id, away_id))), []) if m[0] < d][-H2H_MEETINGS:]
        if not meetings:
            return {"h2h_n": 0.0, "h2h_home_score": np.nan, "h2h_goal_diff": np.nan, "h2h_days_since": np.nan}
        num = den = gd_num = 0.0
        for k, (md, mh, hg, ag) in enumerate(reversed(meetings)):
            w = H2H_DECAY ** k
            gd = (hg - ag) if mh == home_id else (ag - hg)       # from the CURRENT home team's perspective
            pts = 1.0 if gd > 0 else 0.5 if gd == 0 else 0.0
            num += w * pts
            gd_num += w * gd
            den += w
        return {
            "h2h_n": float(len(meetings)),
            "h2h_home_score": num / den,
            "h2h_goal_diff": gd_num / den,
            "h2h_days_since": float((d - meetings[-1][0]).days),
        }

    def _features(self, row, hs: TeamState, as_: TeamState, lg: LeagueState, tier: int) -> Dict[str, float]:
        d = row.date
        home_id, away_id = row.home_id, row.away_id
        f: Dict[str, float] = {}

        # --- elo
        f.update(home_elo=hs.elo, away_elo=as_.elo, elo_diff=hs.elo - as_.elo,
                 home_elo_matches=float(hs.n), away_elo_matches=float(as_.n))

        # --- form
        for side, st in (("home", hs), ("away", as_)):
            f[f"{side}_ppg_s"], f[f"{side}_ppg_l"] = _get(st.s, "pts"), _get(st.l, "pts")
            f[f"{side}_gd_s"], f[f"{side}_gd_l"] = _get(st.s, "gd"), _get(st.l, "gd")
            f[f"{side}_win5"] = (sum(1 for p in st.last5 if p == 3) / len(st.last5)) if st.last5 else np.nan
        f["home_venue_ppg"], f["home_venue_gd"] = _get(hs.home, "pts"), _get(hs.home, "gd")
        f["away_venue_ppg"], f["away_venue_gd"] = _get(as_.away, "pts"), _get(as_.away, "gd")
        f["ppg_diff_s"] = f["home_ppg_s"] - f["away_ppg_s"]
        f["ppg_diff_l"] = f["home_ppg_l"] - f["away_ppg_l"]

        # --- team stats (xG proxy comes from shots data; present for every league/season in the CSVs)
        for side, st in (("home", hs), ("away", as_)):
            f[f"{side}_sot_for"], f[f"{side}_sot_against"] = _get(st.l, "sot_f"), _get(st.l, "sot_a")
            f[f"{side}_shots_for"], f[f"{side}_shots_against"] = _get(st.l, "shots_f"), _get(st.l, "shots_a")
            f[f"{side}_xgp_for"], f[f"{side}_xgp_against"] = _get(st.l, "xgp_f"), _get(st.l, "xgp_a")
            sf, sa = f[f"{side}_sot_for"], f[f"{side}_sot_against"]
            f[f"{side}_sot_share"] = sf / (sf + sa) if not _nan(sf) and not _nan(sa) and (sf + sa) > 0 else np.nan
        f["home_corners_for"], f["away_corners_for"] = _get(hs.l, "corners_f"), _get(as_.l, "corners_f")
        f["xgp_diff"] = (f["home_xgp_for"] - f["home_xgp_against"]) - (f["away_xgp_for"] - f["away_xgp_against"])

        # --- situation: attack / defence strength relative to league, shrunk when few matches, -> expected goals
        def strength(st: TeamState, key: str, league_avg: float) -> float:
            v = _get(st.l, key)
            if _nan(v):
                return 1.0
            w = st.n / (st.n + SHRINK_MATCHES)
            return w * (v / league_avg) + (1 - w) * 1.0

        lg_avg = max((lg.home_goals + lg.away_goals) / 2.0, 0.3)
        f["home_attack"], f["home_defence"] = strength(hs, "gf", lg_avg), strength(hs, "ga", lg_avg)
        f["away_attack"], f["away_defence"] = strength(as_, "gf", lg_avg), strength(as_, "ga", lg_avg)
        # defence value > 1 means LEAKY. Expected goals = league scoring level x own attack x opponent leakiness
        f["exp_home_goals"] = lg.home_goals * f["home_attack"] * f["away_defence"]
        f["exp_away_goals"] = lg.away_goals * f["away_attack"] * f["home_defence"]
        f["exp_goal_diff"] = f["exp_home_goals"] - f["exp_away_goals"]

        # --- home advantage
        f["league_home_win_rate"], f["league_home_gd"] = lg.home_win_rate, lg.home_gd
        f["home_team_hfa"] = self._hfa(hs)
        f["away_team_hfa"] = self._hfa(as_)

        # --- fatigue
        f["home_rest_days"], f["away_rest_days"] = self._rest_days(home_id, hs, d), self._rest_days(away_id, as_, d)
        f["rest_diff"] = f["home_rest_days"] - f["away_rest_days"]
        for side, tid, st in (("home", home_id, hs), ("away", away_id, as_)):
            f[f"{side}_games_7d"] = float(self._games_in_window(tid, st, d, 7))
            f[f"{side}_games_14d"] = float(self._games_in_window(tid, st, d, 14))
            f[f"{side}_games_30d"] = float(self._games_in_window(tid, st, d, 30))
            rd = f[f"{side}_rest_days"]
            f[f"{side}_short_rest"] = 1.0 if (not _nan(rd) and rd <= 3) else 0.0

        # --- context
        f["is_derby"] = self._is_derby(home_id, away_id)
        f["travel_km"] = self._distance(home_id, away_id, getattr(row, "venue_lat", np.nan), getattr(row, "venue_lon", np.nan))
        f["temp_c"], f["wind_kph"], f["precip_mm"] = (float(getattr(row, k, np.nan)) if not _nan(getattr(row, k, np.nan)) else np.nan
                                                      for k in ("temp_c", "wind_kph", "precip_mm"))
        f["bad_weather"] = (1.0 if ((not _nan(f["wind_kph"]) and f["wind_kph"] > 35) or (not _nan(f["precip_mm"]) and f["precip_mm"] > 5)) else
                            (0.0 if not (_nan(f["wind_kph"]) and _nan(f["precip_mm"])) else np.nan))
        f["is_midweek"] = 1.0 if d.weekday() in (1, 2, 3) else 0.0
        games = 46 if tier >= 2 else 38
        f["season_progress"] = min(1.0, hs.season_games / games)
        f["league_tier"] = float(tier)

        # --- head to head
        f.update(self._h2h_features(home_id, away_id, d))

        # --- suspension / availability proxy
        f["home_red_last"], f["away_red_last"] = hs.last_red, as_.last_red
        f["home_yellow_load"], f["away_yellow_load"] = _get(hs.s, "yellow"), _get(as_.s, "yellow")

        # --- market
        f["mkt_p_home"], f["mkt_p_draw"], f["mkt_p_away"] = implied_probabilities(
            getattr(row, "odds_home", np.nan), getattr(row, "odds_draw", np.nan), getattr(row, "odds_away", np.nan))

        # --- injuries (optional)
        if self.injury_lookup is not None:
            hc, hi = self.injury_lookup(home_id, d)
            ac, ai = self.injury_lookup(away_id, d)
            f.update(home_inj_count=hc, away_inj_count=ac, home_inj_importance=hi, away_inj_importance=ai)
        else:
            f.update(home_inj_count=np.nan, away_inj_count=np.nan, home_inj_importance=np.nan, away_inj_importance=np.nan)
        return f

    @staticmethod
    def _hfa(st: TeamState) -> float:
        """How many more points per game this club takes at home than away (shrunk toward 0.35, the typical gap)."""
        h, a = _get(st.home, "pts"), _get(st.away, "pts")
        if _nan(h) or _nan(a):
            return 0.35
        w = min(1.0, st.n / 20.0)
        return w * (h - a) + (1 - w) * 0.35

    # ------------------------------------------------------------------ state updates
    def _update(self, row, hs: TeamState, as_: TeamState, lg: LeagueState) -> None:
        hg, ag = int(row.home_score), int(row.away_score)
        d = row.date

        # elo (uses the same goal-margin weighting + home advantage everywhere)
        exp_h = elo_expected(hs.elo, as_.elo)
        res_h = 1.0 if hg > ag else 0.5 if hg == ag else 0.0
        delta = ELO_K * goal_margin_multiplier(hg - ag) * (res_h - exp_h)
        hs.elo += delta
        as_.elo -= delta

        for st, is_home, gf, ga in ((hs, True, hg, ag), (as_, False, ag, hg)):
            side, opp = ("home", "away") if is_home else ("away", "home")
            g = lambda k: getattr(row, f"{side}_{k}", np.nan)
            o = lambda k: getattr(row, f"{opp}_{k}", np.nan)
            pts = 3.0 if gf > ga else 1.0 if gf == ga else 0.0
            sot_f, sot_a, sh_f, sh_a = g("sot"), o("sot"), g("shots"), o("shots")
            xgp_f = 0.30 * sot_f + 0.04 * (sh_f - sot_f) if not (_nan(sot_f) or _nan(sh_f)) else np.nan
            xgp_a = 0.30 * sot_a + 0.04 * (sh_a - sot_a) if not (_nan(sot_a) or _nan(sh_a)) else np.nan
            vals = {"pts": pts, "gf": float(gf), "ga": float(ga), "gd": float(gf - ga),
                    "sot_f": sot_f, "sot_a": sot_a, "shots_f": sh_f, "shots_a": sh_a,
                    "corners_f": g("corners"), "yellow": g("yellow"), "xgp_f": xgp_f, "xgp_a": xgp_a}
            for k, v in vals.items():
                _ew(st.s, k, v, ALPHA_SHORT)
                _ew(st.l, k, v, ALPHA_LONG)
            venue = st.home if is_home else st.away
            for k in ("pts", "gd", "gf", "ga"):
                _ew(venue, k, vals[k], 0.15)
            st.last5.append(int(pts))
            st.last_red = 0.0 if _nan(g("red")) else float(g("red"))
            st.n += 1
            st.season_games += 1
            st.last_date = d
            st.dates.append(d)

        # league level running averages
        # Running averages start from a prior worth ~20 matches (never snaps to 0 after an early 0-0) and settle to a ~200-match memory.
        a = 1.0 / min(lg.n + 20.0, 200.0)
        lg.home_goals += a * (hg - lg.home_goals)
        lg.away_goals += a * (ag - lg.away_goals)
        lg.home_win_rate += a * ((1.0 if hg > ag else 0.0) - lg.home_win_rate)
        lg.home_gd += a * ((hg - ag) - lg.home_gd)
        lg.n += 1

        self.h2h.setdefault(tuple(sorted((row.home_id, row.away_id))), []).append((d, row.home_id, hg, ag))

    # ------------------------------------------------------------------ main entry
    def run(self, matches: pd.DataFrame) -> pd.DataFrame:
        """
        matches: columns match_id, date, season, league, home_id, away_id, home_team, away_team,
                 home_score, away_score (NaN when unplayed), stats/odds/weather columns (may be NaN).
        Returns one row per input match: identifiers + every feature + `result` (H/D/A or None).
        """
        df = matches.sort_values(["date", "match_id"]).reset_index(drop=True)
        out: List[dict] = []
        for row in df.itertuples(index=False):
            tier = TIER_BY_NAME.get(canonical_league(row.league) or "", 1)
            hs, as_ = self._team(row.home_id, tier), self._team(row.away_id, tier)
            self._season_roll(hs, int(row.season), tier)
            self._season_roll(as_, int(row.season), tier)
            lg = self._league(canonical_league(row.league) or row.league)

            feats = self._features(row, hs, as_, lg, tier)
            played = not _nan(row.home_score) and not _nan(row.away_score)
            rec = {
                "match_id": row.match_id, "date": row.date, "season": int(row.season), "league": canonical_league(row.league) or row.league,
                "home_id": row.home_id, "away_id": row.away_id, "home_team": row.home_team, "away_team": row.away_team,
                "home_score": row.home_score if played else np.nan, "away_score": row.away_score if played else np.nan,
                "result": ("H" if row.home_score > row.away_score else "A" if row.home_score < row.away_score else "D") if played else None,
                "played": played,
            }
            rec.update(feats)
            out.append(rec)
            if played:
                self._update(row, hs, as_, lg)
        return pd.DataFrame(out)


# ----------------------------------------------------------------------------
# Database glue
# ----------------------------------------------------------------------------
MATCH_COLUMNS = [
    "match_id", "date", "season", "league", "home_id", "away_id", "home_team", "away_team", "home_score", "away_score",
    "home_shots", "away_shots", "home_sot", "away_sot", "home_corners", "away_corners", "home_fouls", "away_fouls",
    "home_yellow", "away_yellow", "home_red", "away_red", "odds_home", "odds_draw", "odds_away",
    "temp_c", "wind_kph", "precip_mm", "venue_lat", "venue_lon",
]


def load_matches_frame(db: Session, supported_only: bool = True) -> pd.DataFrame:
    """All matches (played + upcoming) in the 4 supported leagues as a DataFrame."""
    q = db.query(Match, Team).join(Team, Team.id == Match.home_team_id)
    away_names = {t.id: t.name for t in db.query(Team).all()}
    rows = []
    for m, home in q.all():
        league = canonical_league(m.competition)
        if supported_only and league is None:
            continue
        season = m.season if m.season is not None else (m.date.year if m.date.month >= 7 else m.date.year - 1)
        rows.append([
            m.id, m.date, season, league or m.competition, m.home_team_id, m.away_team_id, home.name, away_names.get(m.away_team_id, "?"),
            m.home_score, m.away_score, m.home_shots, m.away_shots, m.home_sot, m.away_sot, m.home_corners, m.away_corners,
            m.home_fouls, m.away_fouls, m.home_yellow, m.away_yellow, m.home_red, m.away_red,
            m.odds_home, m.odds_draw, m.odds_away, m.temp_c, m.wind_kph, m.precip_mm, m.venue_lat, m.venue_lon,
        ])
    df = pd.DataFrame(rows, columns=MATCH_COLUMNS)
    num = [c for c in MATCH_COLUMNS if c not in ("match_id", "date", "league", "home_team", "away_team")]
    for c in num:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def team_info(db: Session) -> Dict[int, dict]:
    return {t.id: {"name": t.name, "lat": t.stadium_lat, "lon": t.stadium_lon} for t in db.query(Team).all()}


def build_feature_frame(db: Session, extra_games: Optional[Dict[int, List[dt.date]]] = None, injury_lookup=None) -> pd.DataFrame:
    """Features for EVERY match (played = training rows, unplayed = fixtures to predict), all from one engine pass."""
    matches = load_matches_frame(db)
    if matches.empty:
        return pd.DataFrame()
    return FeatureEngine(team_info(db), extra_games=extra_games, injury_lookup=injury_lookup).run(matches)
