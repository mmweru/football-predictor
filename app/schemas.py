"""
Pydantic schemas.

These are separate from the SQLAlchemy models on purpose: SQLAlchemy models
describe how data is *stored*; Pydantic schemas describe how data is
*validated on the way in* (e.g. from a CSV row or an API request) and
*shaped on the way out* (e.g. as JSON). Keeping them separate means you can
change your DB structure without breaking your validation layer, and vice
versa.

Naming convention used throughout:
  - `<Entity>Base`   -> shared fields
  - `<Entity>Create` -> fields required to insert a new row (no id)
  - `<Entity>Read`   -> fields returned when reading from the DB (includes id)
"""

from __future__ import annotations

import datetime as dt
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# Team
# ---------------------------------------------------------------------------
class TeamBase(BaseModel):
    name: str = Field(..., min_length=1, max_length=120)
    stadium_lat: Optional[float] = Field(None, ge=-90, le=90)
    stadium_lon: Optional[float] = Field(None, ge=-180, le=180)
    league: Optional[str] = Field(None, max_length=120)


class TeamCreate(TeamBase):
    pass


class TeamRead(TeamBase):
    model_config = ConfigDict(from_attributes=True)
    id: int


# ---------------------------------------------------------------------------
# Match
# ---------------------------------------------------------------------------
class MatchBase(BaseModel):
    date: dt.date
    home_team_id: int
    away_team_id: int
    home_score: Optional[int] = Field(None, ge=0)
    away_score: Optional[int] = Field(None, ge=0)
    competition: Optional[str] = Field(None, max_length=120)
    venue_lat: Optional[float] = Field(None, ge=-90, le=90)
    venue_lon: Optional[float] = Field(None, ge=-180, le=180)
    venue_name: Optional[str] = Field(None, max_length=200)


class MatchCreate(MatchBase):
    pass


class MatchRead(MatchBase):
    model_config = ConfigDict(from_attributes=True)
    id: int


# ---------------------------------------------------------------------------
# Player
# ---------------------------------------------------------------------------
class PlayerBase(BaseModel):
    name: str = Field(..., min_length=1, max_length=120)
    team_id: int
    position: Optional[str] = Field(None, max_length=50)
    importance_weight: float = Field(0.0, ge=0.0, le=1.0)


class PlayerCreate(PlayerBase):
    pass


class PlayerRead(PlayerBase):
    model_config = ConfigDict(from_attributes=True)
    id: int


# ---------------------------------------------------------------------------
# Injury
# ---------------------------------------------------------------------------
class InjuryBase(BaseModel):
    player_id: int
    date_out: dt.date
    date_back: Optional[dt.date] = None
    match_missed_id: Optional[int] = None


class InjuryCreate(InjuryBase):
    pass


class InjuryRead(InjuryBase):
    model_config = ConfigDict(from_attributes=True)
    id: int


# ---------------------------------------------------------------------------
# EloHistory
# ---------------------------------------------------------------------------
class EloHistoryBase(BaseModel):
    team_id: int
    date: dt.date
    rating: float = Field(..., gt=0)


class EloHistoryCreate(EloHistoryBase):
    pass


class EloHistoryRead(EloHistoryBase):
    model_config = ConfigDict(from_attributes=True)
    id: int
