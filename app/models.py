"""
SQLAlchemy ORM models.

Design note: the original schema sketch had `matches.home_team` /
`away_team` as plain text. Instead, this links matches, players, injuries
and elo_history to `teams.id` via foreign keys. This is standard relational
design — it avoids storing the same team name as a string in five places,
prevents typos like "Man Utd" vs "Manchester United" from silently creating
duplicate teams, and makes joins/aggregations (e.g. "all matches for this
team") trivial and fast via indexed foreign keys instead of string matching.
"""

from __future__ import annotations

import datetime as dt
from typing import List, Optional

from sqlalchemy import Boolean, Date, DateTime, Float, ForeignKey, Integer, String, UniqueConstraint, true
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


class Team(Base):
    __tablename__ = "teams"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(120), unique=True, nullable=False, index=True)
    stadium_lat: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    stadium_lon: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    league: Mapped[Optional[str]] = mapped_column(String(120), nullable=True, index=True)

    # NEW: True only for clubs that play in one of the 4 supported divisions
    # THIS season. app.sync_data flips this every run, which is what makes the
    # Premier League list show exactly 20 clubs instead of every club that
    # was ever in the data (relegated teams are kept for history, just hidden).
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True, server_default=true())
    # NEW: ids in external APIs, used to join injuries / player data back to a team.
    fdorg_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    fpl_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    players: Mapped[List["Player"]] = relationship(back_populates="team", cascade="all, delete-orphan")
    elo_history: Mapped[List["EloHistory"]] = relationship(back_populates="team", cascade="all, delete-orphan")
    home_matches: Mapped[List["Match"]] = relationship(
        back_populates="home_team", foreign_keys="Match.home_team_id"
    )
    away_matches: Mapped[List["Match"]] = relationship(
        back_populates="away_team", foreign_keys="Match.away_team_id"
    )

    def __repr__(self) -> str:
        return f"<Team id={self.id} name={self.name!r} league={self.league!r}>"


class Match(Base):
    __tablename__ = "matches"
    __table_args__ = (
        UniqueConstraint("date", "home_team_id", "away_team_id", name="uq_match_fixture"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    date: Mapped[dt.date] = mapped_column(Date, nullable=False, index=True)
    home_team_id: Mapped[int] = mapped_column(ForeignKey("teams.id"), nullable=False)
    away_team_id: Mapped[int] = mapped_column(ForeignKey("teams.id"), nullable=False)
    home_score: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    away_score: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    competition: Mapped[Optional[str]] = mapped_column(String(120), nullable=True, index=True)

    # Optional per-match venue override. Falls back to home_team's stadium
    # coordinates when None — see app.features.get_travel_distance_km.
    # Needed for local/lower-league matches where the actual pitch isn't
    # the team's usual registered stadium (or isn't geocodable by team name
    # at all), or for one-off neutral-venue fixtures.
    venue_lat: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    venue_lon: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    venue_name: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)

    # --- NEW: season + match statistics (football-data.co.uk) ---
    season: Mapped[Optional[int]] = mapped_column(Integer, nullable=True, index=True)  # start year, 2026 = 2026/27
    home_shots: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    away_shots: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    home_sot: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)   # shots on target
    away_sot: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    home_corners: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    away_corners: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    home_fouls: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    away_fouls: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    home_yellow: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    away_yellow: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    home_red: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    away_red: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    # Pre-match market odds (decimal, bookmaker average where available). Used ONLY as a model input
    # (de-margined to probabilities) — they're how the model "knows" about lineups/injuries/news.
    odds_home: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    odds_draw: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    odds_away: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    # --- NEW: weather at kickoff day for the venue (Open-Meteo) ---
    temp_c: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    wind_kph: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    precip_mm: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    home_team: Mapped["Team"] = relationship(back_populates="home_matches", foreign_keys=[home_team_id])
    away_team: Mapped["Team"] = relationship(back_populates="away_matches", foreign_keys=[away_team_id])
    injuries_missed: Mapped[List["Injury"]] = relationship(back_populates="match_missed")

    def __repr__(self) -> str:
        return f"<Match id={self.id} date={self.date} home={self.home_team_id} away={self.away_team_id}>"


class Player(Base):
    __tablename__ = "players"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(120), nullable=False, index=True)
    team_id: Mapped[int] = mapped_column(ForeignKey("teams.id"), nullable=False)
    position: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    importance_weight: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)

    # --- NEW: player stats + live availability (FPL API for Premier League) ---
    fpl_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True, index=True)
    minutes: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    goals: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    assists: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    xg: Mapped[Optional[float]] = mapped_column(Float, nullable=True)   # expected goals, season to date
    xa: Mapped[Optional[float]] = mapped_column(Float, nullable=True)   # expected assists, season to date
    status: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)  # a=available d=doubtful i=injured s=suspended u=unavailable
    chance_of_playing: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)  # 0-100, next match
    news: Mapped[Optional[str]] = mapped_column(String(300), nullable=True)

    team: Mapped["Team"] = relationship(back_populates="players")
    injuries: Mapped[List["Injury"]] = relationship(back_populates="player", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<Player id={self.id} name={self.name!r} team_id={self.team_id}>"


class PlayerSeasonStats(Base):
    __tablename__ = "player_season_stats"
    __table_args__ = (UniqueConstraint("api_player_id", "team_id", "season", "league", name="uq_player_season_team"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    api_player_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    team_id: Mapped[int] = mapped_column(ForeignKey("teams.id"), nullable=False, index=True)
    season: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    league: Mapped[str] = mapped_column(String(120), nullable=False)
    player_name: Mapped[str] = mapped_column(String(120), nullable=False)
    position: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    appearances: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    lineups: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    minutes: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    goals: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    assists: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    shots: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    shots_on_target: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    yellow_cards: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    red_cards: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    rating: Mapped[Optional[float]] = mapped_column(Float, nullable=True)


class Injury(Base):
    __tablename__ = "injuries"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    player_id: Mapped[int] = mapped_column(ForeignKey("players.id"), nullable=False)
    date_out: Mapped[dt.date] = mapped_column(Date, nullable=False)
    date_back: Mapped[Optional[dt.date]] = mapped_column(Date, nullable=True)
    match_missed_id: Mapped[Optional[int]] = mapped_column(ForeignKey("matches.id"), nullable=True)
    reason: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)   # NEW: e.g. "Hamstring", "Suspended"
    source: Mapped[Optional[str]] = mapped_column(String(30), nullable=True)    # NEW: fpl / bigballs / manual

    player: Mapped["Player"] = relationship(back_populates="injuries")
    match_missed: Mapped[Optional["Match"]] = relationship(back_populates="injuries_missed")

    def __repr__(self) -> str:
        return f"<Injury id={self.id} player_id={self.player_id} date_out={self.date_out}>"


class EloHistory(Base):
    __tablename__ = "elo_history"
    __table_args__ = (UniqueConstraint("team_id", "date", name="uq_elo_team_date"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    team_id: Mapped[int] = mapped_column(ForeignKey("teams.id"), nullable=False)
    date: Mapped[dt.date] = mapped_column(Date, nullable=False, index=True)
    rating: Mapped[float] = mapped_column(Float, nullable=False)

    team: Mapped["Team"] = relationship(back_populates="elo_history")

    def __repr__(self) -> str:
        return f"<EloHistory team_id={self.team_id} date={self.date} rating={self.rating}>"


class Prediction(Base):
    """
    NEW: one stored prediction per (match, model run). Upcoming fixtures get a
    row when `app.predict_fixtures` runs; once the real result arrives,
    `app.grade_predictions` fills `actual_result`, so you get a live,
    honest track record (see GET /api/accuracy).
    """
    __tablename__ = "predictions"
    __table_args__ = (UniqueConstraint("match_id", "model_version", name="uq_prediction_match_model"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    match_id: Mapped[int] = mapped_column(ForeignKey("matches.id"), nullable=False, index=True)
    model_version: Mapped[str] = mapped_column(String(40), nullable=False)
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime, nullable=False, default=lambda: dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)
    )
    p_home: Mapped[float] = mapped_column(Float, nullable=False)
    p_draw: Mapped[float] = mapped_column(Float, nullable=False)
    p_away: Mapped[float] = mapped_column(Float, nullable=False)
    exp_home_goals: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    exp_away_goals: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    predicted: Mapped[str] = mapped_column(String(1), nullable=False)          # H / D / A
    actual_result: Mapped[Optional[str]] = mapped_column(String(1), nullable=True)  # filled after the match

    match: Mapped["Match"] = relationship()

    def __repr__(self) -> str:
        return f"<Prediction match_id={self.match_id} {self.predicted} H={self.p_home:.2f} D={self.p_draw:.2f} A={self.p_away:.2f}>"
