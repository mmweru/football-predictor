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

from sqlalchemy import Date, Float, ForeignKey, Integer, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


class Team(Base):
    __tablename__ = "teams"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(120), unique=True, nullable=False, index=True)
    stadium_lat: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    stadium_lon: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    league: Mapped[Optional[str]] = mapped_column(String(120), nullable=True, index=True)

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

    team: Mapped["Team"] = relationship(back_populates="players")
    injuries: Mapped[List["Injury"]] = relationship(back_populates="player", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<Player id={self.id} name={self.name!r} team_id={self.team_id}>"


class Injury(Base):
    __tablename__ = "injuries"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    player_id: Mapped[int] = mapped_column(ForeignKey("players.id"), nullable=False)
    date_out: Mapped[dt.date] = mapped_column(Date, nullable=False)
    date_back: Mapped[Optional[dt.date]] = mapped_column(Date, nullable=True)
    match_missed_id: Mapped[Optional[int]] = mapped_column(ForeignKey("matches.id"), nullable=True)

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
