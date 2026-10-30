"""
League registry — the SINGLE source of truth for which competitions this app
supports. Everything that filters, ingests or displays leagues imports from
here, so "only EPL + EFL Championship/League One/League Two" is enforced in
exactly one place.

Naming: `Team.league` stores the *display name* (e.g. "Premier League").
football-data.co.uk calls these divisions E0..E3.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass(frozen=True)
class League:
    name: str               # stored in Team.league / Match.competition and shown in the UI
    badge: str              # short label for the UI card
    tier: int               # 1 = top flight ... 4 = League Two
    fduk_div: str           # football-data.co.uk division code
    fdorg_code: Optional[str]    # football-data.org competition code (None = not on the free tier)
    bigballs_code: Optional[str] # BigBalls league key (None = not covered)
    expected_teams: int     # how many clubs play in this division in a season


LEAGUES: List[League] = [
    League("Premier League",     "EPL", 1, "E0", "PL",  "epl", 20),
    League("EFL Championship",   "ELC", 2, "E1", "ELC", None,  24),
    League("EFL League One",     "L1",  3, "E2", None,  None,  24),
    League("EFL League Two",     "L2",  4, "E3", None,  None,  24),
]

LEAGUE_NAMES: List[str] = [l.name for l in LEAGUES]
BY_NAME: Dict[str, League] = {l.name: l for l in LEAGUES}
BY_DIV: Dict[str, League] = {l.fduk_div: l for l in LEAGUES}
TIER_BY_NAME: Dict[str, int] = {l.name: l.tier for l in LEAGUES}

# Older data in an existing database may carry these labels (what you passed
# as --competition when ingesting). They are mapped onto the canonical names.
LEGACY_LABELS: Dict[str, str] = {
    "premier league": "Premier League",
    "epl": "Premier League",
    "english premier league": "Premier League",
    "championship": "EFL Championship",
    "efl championship": "EFL Championship",
    "league one": "EFL League One",
    "league 1": "EFL League One",
    "efl league one": "EFL League One",
    "league two": "EFL League Two",
    "league 2": "EFL League Two",
    "efl league two": "EFL League Two",
}


def canonical_league(label: Optional[str]) -> Optional[str]:
    """Maps any known spelling to the canonical league name, or None if it isn't one of the 4 supported leagues."""
    if not label:
        return None
    return LEGACY_LABELS.get(label.strip().lower())


def is_supported(label: Optional[str]) -> bool:
    return canonical_league(label) is not None


# ---------------------------------------------------------------------------
# Seasons
# ---------------------------------------------------------------------------
def season_start_year(on: Optional[dt.date] = None) -> int:
    """
    English seasons run Aug -> May. A date in Jul-Dec belongs to the season
    that STARTED that calendar year; Jan-Jun belongs to the one that started
    the year before. (July is treated as the new season so pre-season/fixture
    release weeks already point at the upcoming campaign.)
    """
    on = on or dt.date.today()
    return on.year if on.month >= 7 else on.year - 1


def season_code(start_year: int) -> str:
    """2026 -> '2627' (football-data.co.uk folder name)."""
    return f"{start_year % 100:02d}{(start_year + 1) % 100:02d}"


def season_label(start_year: int) -> str:
    return f"{start_year}/{(start_year + 1) % 100:02d}"


def season_of_date(d: dt.date) -> int:
    return season_start_year(d)
