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
    frozenset({"Arsenal", "Chelsea"}),
    frozenset({"Liverpool", "Everton"}),
    frozenset({"Manchester United", "Manchester City"}),
    frozenset({"Manchester United", "Liverpool"}),
    frozenset({"Manchester United", "Leeds"}),
    frozenset({"Chelsea", "Tottenham Hotspur"}),
    frozenset({"Chelsea", "Fulham"}),
    frozenset({"West Ham United", "Millwall"}),
    frozenset({"West Ham United", "Tottenham Hotspur"}),
    frozenset({"Newcastle United", "Sunderland"}),
    frozenset({"Aston Villa", "Birmingham"}),
    frozenset({"Aston Villa", "Wolverhampton Wanderers"}),
    frozenset({"Birmingham", "West Bromwich Albion"}),
    frozenset({"West Bromwich Albion", "Wolverhampton Wanderers"}),
    frozenset({"Brighton", "Crystal Palace"}),
    frozenset({"Crystal Palace", "Millwall"}),
    frozenset({"Sheffield United", "Sheffield Wednesday"}),
    frozenset({"Nottingham Forest", "Derby"}),
    frozenset({"Nottingham Forest", "Leicester City"}),
    frozenset({"Nottingham Forest", "Notts County"}),
    frozenset({"Leicester City", "Derby"}),
    frozenset({"Bristol City", "Bristol Rovers"}),
    frozenset({"Cardiff", "Swansea"}),
    frozenset({"Middlesbrough", "Sunderland"}),
    frozenset({"Middlesbrough", "Newcastle United"}),
    frozenset({"Portsmouth", "Southampton"}),
    frozenset({"Norwich", "Ipswich"}),
    frozenset({"Stoke", "Port Vale"}),
    frozenset({"Preston", "Blackburn"}),
    frozenset({"Preston", "Blackpool"}),
    frozenset({"Bolton", "Wigan"}),
    frozenset({"Huddersfield", "Barnsley"}),
    frozenset({"Leeds", "Huddersfield"}),
    frozenset({"Rotherham", "Barnsley"}),
    frozenset({"Rotherham", "Sheffield United"}),
    frozenset({"Lincoln", "Grimsby"}),
    frozenset({"Plymouth", "Exeter"}),
    frozenset({"Charlton", "Millwall"}),
    frozenset({"Watford", "Luton"}),
    frozenset({"Reading", "Oxford"}),
    frozenset({"Milton Keynes Dons", "AFC Wimbledon"}),
    frozenset({"Burton", "Derby"}),
    frozenset({"Blackpool", "Fleetwood Town"}),
    frozenset({"Crewe", "Port Vale"}),
    frozenset({"Notts County", "Mansfield"}),
    frozenset({"Doncaster", "Rotherham"}),
    frozenset({"Colchester", "Gillingham"}),
    frozenset({"Cheltenham", "Bristol Rovers"}),
    frozenset({"Swindon", "Oxford"}),
    frozenset({"Bradford", "Leeds"}),
    frozenset({"Tranmere", "Wrexham"}),
}


def is_derby(team_a_name: str, team_b_name: str) -> bool:
    return frozenset({team_a_name, team_b_name}) in DERBY_PAIRS
