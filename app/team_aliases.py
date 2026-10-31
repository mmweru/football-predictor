"""
Team name normalization.

Four different sources name the same club four different ways:

    football-data.co.uk :  "Man United"      "Nott'm Forest"   "Sheffield Weds"   "Peterboro"
    football-data.org   :  "Manchester United FC"  "Nottingham Forest FC"  "Sheffield Wednesday FC"
    FPL (fantasy PL)    :  "Man Utd"         "Spurs"           "Nott'm Forest"
    BigBalls            :  "Manchester United"  ...

Without normalizing, every source silently creates its own duplicate `Team`
row, which corrupts Elo, form and head-to-head. Everything funnels through
`normalize_team_name` (via get_or_create_team in the ingest modules).

Matching is done on a *key*: lowercase, "&" -> "and", punctuation removed,
and the pure-suffix tokens FC / AFC dropped. So "Manchester United FC",
"Manchester United" and "manchester united" all collapse to the same key.
The alias table maps KEYS -> the canonical name stored in `teams.name`.

Canonical names for clubs that were already in your database (e.g. "Brighton",
"Leeds", "Burnley") are deliberately unchanged, so upgrading never creates a
second copy of an existing team. Extend ALIASES whenever
`python -m app.sync_data` prints a "possible duplicate teams" warning.
"""

from __future__ import annotations

import re
from typing import Dict

_DROP_TOKENS = {"fc", "afc"}


def team_key(raw: str) -> str:
    s = raw.strip().lower().replace("&", " and ")
    s = re.sub(r"[^a-z0-9 ]", "", s)          # drop apostrophes, dots, hyphens
    tokens = [t for t in s.split() if t not in _DROP_TOKENS]
    return " ".join(tokens)


# key -> canonical name. Keys are written human-readably and passed through team_key() below.
_RAW_ALIASES: Dict[str, str] = {
    # ---- Premier League (existing canonical names preserved) ----
    "man united": "Manchester United", "man utd": "Manchester United", "manchester utd": "Manchester United",
    "manchester united": "Manchester United",
    "man city": "Manchester City", "manchester city": "Manchester City",
    "spurs": "Tottenham Hotspur", "tottenham": "Tottenham Hotspur", "tottenham hotspur": "Tottenham Hotspur",
    "wolves": "Wolverhampton Wanderers", "wolverhampton": "Wolverhampton Wanderers",
    "wolverhampton wanderers": "Wolverhampton Wanderers",
    "nott'm forest": "Nottingham Forest", "nottm forest": "Nottingham Forest", "nottingham forest": "Nottingham Forest",
    "newcastle": "Newcastle United", "newcastle utd": "Newcastle United", "newcastle united": "Newcastle United",
    "leicester": "Leicester City", "leicester city": "Leicester City",
    "west brom": "West Bromwich Albion", "west bromwich albion": "West Bromwich Albion",
    "west ham": "West Ham United", "west ham united": "West Ham United",
    "brighton and hove albion": "Brighton", "brighton hove albion": "Brighton", "brighton": "Brighton",
    "leeds united": "Leeds", "leeds": "Leeds",
    "bournemouth": "Bournemouth",
    "ipswich town": "Ipswich", "ipswich": "Ipswich",
    "burnley": "Burnley", "sunderland": "Sunderland", "southampton": "Southampton",
    "luton town": "Luton", "luton": "Luton",
    "norwich city": "Norwich", "norwich": "Norwich",
    "sheffield united": "Sheffield United", "sheffield utd": "Sheffield United",
    "hull city": "Hull", "hull": "Hull",
    "coventry city": "Coventry", "coventry": "Coventry",
    # ---- Championship ----
    "sheffield weds": "Sheffield Wednesday", "sheffield wednesday": "Sheffield Wednesday",
    "qpr": "QPR", "queens park rangers": "QPR",
    "preston north end": "Preston", "preston": "Preston",
    "stoke city": "Stoke", "stoke": "Stoke",
    "swansea city": "Swansea", "swansea": "Swansea",
    "cardiff city": "Cardiff", "cardiff": "Cardiff",
    "derby county": "Derby", "derby": "Derby",
    "birmingham city": "Birmingham", "birmingham": "Birmingham",
    "blackburn rovers": "Blackburn", "blackburn": "Blackburn",
    "bristol city": "Bristol City",
    "charlton athletic": "Charlton", "charlton": "Charlton",
    "oxford united": "Oxford", "oxford": "Oxford",
    "plymouth argyle": "Plymouth", "plymouth": "Plymouth",
    "portsmouth": "Portsmouth", "middlesbrough": "Middlesbrough", "millwall": "Millwall", "watford": "Watford",
    "wrexham": "Wrexham",
    # ---- League One ----
    "peterborough united": "Peterborough United", "peterborough": "Peterborough United", "peterboro": "Peterborough United",
    "bristol rovers": "Bristol Rovers", "bristol rvs": "Bristol Rovers",
    "wimbledon": "AFC Wimbledon",  # football-data.co.uk drops the "AFC"; key drops it too
    "milton keynes dons": "Milton Keynes Dons", "mk dons": "Milton Keynes Dons",
    "wigan athletic": "Wigan", "wigan": "Wigan",
    "bolton wanderers": "Bolton", "bolton": "Bolton",
    "huddersfield town": "Huddersfield", "huddersfield": "Huddersfield",
    "rotherham united": "Rotherham", "rotherham": "Rotherham",
    "lincoln city": "Lincoln", "lincoln": "Lincoln",
    "northampton town": "Northampton", "northampton": "Northampton",
    "exeter city": "Exeter", "exeter": "Exeter",
    "doncaster rovers": "Doncaster", "doncaster": "Doncaster",
    "cambridge united": "Cambridge", "cambridge": "Cambridge",
    "burton albion": "Burton", "burton": "Burton",
    "reading": "Reading", "barnsley": "Barnsley", "blackpool": "Blackpool", "bradford city": "Bradford",
    "bradford": "Bradford", "stevenage": "Stevenage", "stockport county": "Stockport", "stockport": "Stockport",
    "wycombe wanderers": "Wycombe", "wycombe": "Wycombe", "leyton orient": "Leyton Orient",
    "mansfield town": "Mansfield", "mansfield": "Mansfield", "shrewsbury town": "Shrewsbury",
    "shrewsbury": "Shrewsbury", "crawley town": "Crawley Town", "port vale": "Port Vale",
    # ---- League Two ----
    "accrington stanley": "Accrington", "accrington st": "Accrington", "accrington": "Accrington",
    "barrow": "Barrow", "bromley": "Bromley", "carlisle united": "Carlisle", "carlisle": "Carlisle",
    "cheltenham town": "Cheltenham", "cheltenham": "Cheltenham", "chesterfield": "Chesterfield",
    "colchester united": "Colchester", "colchester": "Colchester", "crewe alexandra": "Crewe", "crewe": "Crewe",
    "fleetwood town": "Fleetwood Town", "gillingham": "Gillingham", "grimsby town": "Grimsby", "grimsby": "Grimsby",
    "harrogate town": "Harrogate", "harrogate": "Harrogate", "newport county": "Newport County",
    "notts county": "Notts County", "oldham athletic": "Oldham", "oldham": "Oldham",
    "salford city": "Salford", "salford": "Salford", "swindon town": "Swindon", "swindon": "Swindon",
    "tranmere rovers": "Tranmere", "tranmere": "Tranmere", "walsall": "Walsall", "morecambe": "Morecambe",
    "barnet": "Barnet", "forest green rovers": "Forest Green", "forest green": "Forest Green",
}

# Public, human-readable table (kept under the old name for backwards compatibility with existing imports/tests).
TEAM_NAME_ALIASES: Dict[str, str] = _RAW_ALIASES

_BY_KEY: Dict[str, str] = {team_key(k): v for k, v in _RAW_ALIASES.items()}


def normalize_team_name(raw_name: str) -> str:
    """
    Canonical name for a raw name from any data source. Unknown names are
    returned stripped but otherwise unchanged (so new clubs are inserted, not
    dropped) — python -m app.sync_data then reports likely duplicates.
    """
    cleaned = raw_name.strip()
    alias = _BY_KEY.get(team_key(cleaned))
    if alias:
        return alias
    tokens = cleaned.split()
    while tokens and re.sub(r"[^a-z0-9]", "", tokens[-1].lower()) in _DROP_TOKENS:
        tokens.pop()
    return " ".join(tokens)
