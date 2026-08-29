"""
Team name normalization.

Different data sources spell the same team differently: football-data.co.uk
uses "Man United", FootyStats might use "Manchester Utd", a local league
site might use something else entirely. Without normalizing these, you'll
silently end up with multiple `Team` rows for the same real-world team,
which quietly corrupts Elo ratings, form calculations, and everything else
that groups data by team.

This module is intentionally small and manual — team aliases don't follow
a pattern you can algorithmically detect, they have to be curated. Extend
TEAM_NAME_ALIASES as you encounter new variants across whatever sources
you ingest from.
"""

from typing import Dict

# Maps a raw name AS IT APPEARS IN SOURCE DATA -> canonical name to store in `teams.name`.
# Keys are matched case-insensitively (see normalize_team_name below).
TEAM_NAME_ALIASES: Dict[str, str] = {
    "man united": "Manchester United",
    "man utd": "Manchester United",
    "manchester utd": "Manchester United",
    "man city": "Manchester City",
    "spurs": "Tottenham Hotspur",
    "tottenham": "Tottenham Hotspur",
    "wolves": "Wolverhampton Wanderers",
    "nott'm forest": "Nottingham Forest",
    "nottm forest": "Nottingham Forest",
    "newcastle": "Newcastle United",
    "leicester": "Leicester City",
    "west brom": "West Bromwich Albion",
    "west ham": "West Ham United",
}


def normalize_team_name(raw_name: str) -> str:
    """
    Returns the canonical team name for a raw name from any data source.
    Falls back to the raw name (title-cased/stripped) if no alias is known —
    this means new/unmapped teams still get inserted (not dropped), but you
    should periodically check for accidental near-duplicates in `teams`,
    e.g. via:
        SELECT name FROM teams ORDER BY name;
    and add any missed aliases to TEAM_NAME_ALIASES above.
    """
    cleaned = raw_name.strip()
    key = cleaned.lower()
    return TEAM_NAME_ALIASES.get(key, cleaned)
