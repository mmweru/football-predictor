"""Pydantic request/response schemas kept separate from ORM models."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class TeamCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=120)
    stadium_lat: Optional[float] = Field(None, ge=-90, le=90)
    stadium_lon: Optional[float] = Field(None, ge=-180, le=180)
    league: Optional[str] = Field(None, max_length=120)