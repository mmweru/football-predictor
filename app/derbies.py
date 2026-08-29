"""
Derby / rivalry flags.

There's no dataset for "which teams are rivals" — this has to be manually
curated, same as team_aliases.py. Uses canonical team names (the ones
stored in `teams.name`, post-normalization), so make sure you're checking
this AFTER names have gone through app.team_aliases.normalize_team_name.

Extend DERBY_PAIRS as you add teams/leagues — each entry is an unordered
pair, so {"A", "B"} matches both A-vs-B and B-vs-A automatically.
"""

from typing import FrozenSet, Set, Tuple

DERBY_PAIRS: Set[FrozenSet[str]] = {
    frozenset({"Arsenal", "Tottenham Hotspur"}),
    frozenset({"Liverpool", "Everton"}),
    frozenset({"Manchester United", "Manchester City"}),
    frozenset({"Manchester United", "Liverpool"}),
    frozenset({"Chelsea", "Tottenham Hotspur"}),
    frozenset({"West Ham United", "Millwall"}),
    frozenset({"Newcastle United", "Sunderland"}),
    frozenset({"Aston Villa", "Birmingham City"}),
}


def is_derby(team_a_name: str, team_b_name: str) -> bool:
    return frozenset({team_a_name, team_b_name}) in DERBY_PAIRS
