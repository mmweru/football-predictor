"""Repair duplicate club rows created by inconsistent source names."""

from __future__ import annotations

from collections import defaultdict

from sqlalchemy import or_

from app.models import EloHistory, Match, Player, PlayerSeasonStats, Prediction, Team
from app.team_aliases import normalize_team_name


def _played(match: Match) -> bool:
    return match.home_score is not None and match.away_score is not None


def _copy_missing_match_fields(target: Match, source: Match) -> None:
    for field in (
        "competition", "season", "home_score", "away_score", "home_shots", "away_shots", "home_sot", "away_sot",
        "home_corners", "away_corners", "home_fouls", "away_fouls", "home_yellow", "away_yellow", "home_red",
        "away_red", "odds_home", "odds_draw", "odds_away", "temp_c", "wind_kph", "precip_mm",
    ):
        if getattr(target, field) is None:
            setattr(target, field, getattr(source, field))


def merge_duplicate_teams(db) -> dict:
    """Normalize team suffixes, merge duplicate references, and collapse duplicate fixtures."""
    stats = {"teams_merged": 0, "teams_renamed": 0, "matches_merged": 0}
    groups = defaultdict(list)
    for team in db.query(Team).order_by(Team.id).all():
        groups[normalize_team_name(team.name)].append(team)

    replacements = {}
    for canonical, group in groups.items():
        if len(group) == 1:
            if group[0].name != canonical:
                group[0].name = canonical
                stats["teams_renamed"] += 1
            continue

        keeper = next((team for team in group if team.name == canonical), None)
        if keeper is None:
            keeper = max(group, key=lambda team: (team.fdorg_id is not None, team.fpl_id is not None, -team.id))
            keeper.name = canonical
            stats["teams_renamed"] += 1

        for duplicate in group:
            if duplicate.id == keeper.id:
                continue
            replacements[duplicate.id] = keeper.id
            keeper.fdorg_id = keeper.fdorg_id or duplicate.fdorg_id
            keeper.fpl_id = keeper.fpl_id or duplicate.fpl_id
            keeper.stadium_lat = keeper.stadium_lat if keeper.stadium_lat is not None else duplicate.stadium_lat
            keeper.stadium_lon = keeper.stadium_lon if keeper.stadium_lon is not None else duplicate.stadium_lon
            keeper.league = keeper.league or duplicate.league
            keeper.is_active = bool(keeper.is_active or duplicate.is_active)
            stats["teams_merged"] += 1

    if not replacements:
        db.commit()
        return stats

    duplicate_ids = set(replacements)
    affected_ids = duplicate_ids | set(replacements.values())
    matches = db.query(Match).filter(or_(Match.home_team_id.in_(affected_ids), Match.away_team_id.in_(affected_ids))).all()
    by_exact = defaultdict(list)
    deleted_match_ids = set()
    for match in matches:
        home_id = replacements.get(match.home_team_id, match.home_team_id)
        away_id = replacements.get(match.away_team_id, match.away_team_id)
        if home_id == away_id:
            deleted_match_ids.add(match.id)
            db.delete(match)
            stats["matches_merged"] += 1
            continue
        by_exact[(match.date, home_id, away_id)].append((match, home_id, away_id))

    survivors = []
    for group in by_exact.values():
        group.sort(key=lambda item: (_played(item[0]), -item[0].id), reverse=True)
        keep, home_id, away_id = group[0]
        for duplicate, _, _ in group[1:]:
            _copy_missing_match_fields(keep, duplicate)
            deleted_match_ids.add(duplicate.id)
            db.delete(duplicate)
            stats["matches_merged"] += 1
        survivors.append((keep, home_id, away_id))

    if deleted_match_ids:
        db.query(Prediction).filter(Prediction.match_id.in_(deleted_match_ids)).delete(synchronize_session=False)
    db.flush()

    played_pairs = set()
    for match, home_id, away_id in survivors:
        match.home_team_id, match.away_team_id = home_id, away_id
        if _played(match):
            played_pairs.add((home_id, away_id, match.season, match.competition, match.date))
    for match, home_id, away_id in survivors:
        if _played(match) or match.season is None:
            continue
        duplicate_played = any(
            (h, a, season, competition) == (home_id, away_id, match.season, match.competition)
            and abs((date - match.date).days) <= 1
            for h, a, season, competition, date in played_pairs
        )
        if duplicate_played:
            deleted_match_ids.add(match.id)
            db.delete(match)
            stats["matches_merged"] += 1
    db.flush()

    for duplicate_id, keeper_id in replacements.items():
        db.query(Player).filter_by(team_id=duplicate_id).update({Player.team_id: keeper_id}, synchronize_session=False)
        for stats_row in db.query(PlayerSeasonStats).filter_by(team_id=duplicate_id).all():
            existing_stats = db.query(PlayerSeasonStats).filter_by(
                api_player_id=stats_row.api_player_id, team_id=keeper_id,
                season=stats_row.season, league=stats_row.league,
            ).one_or_none()
            if existing_stats is None:
                stats_row.team_id = keeper_id
            else:
                for field in (
                    "player_name", "position", "appearances", "lineups", "minutes", "goals", "assists",
                    "shots", "shots_on_target", "yellow_cards", "red_cards", "rating",
                ):
                    if getattr(existing_stats, field) is None:
                        setattr(existing_stats, field, getattr(stats_row, field))
                db.delete(stats_row)
        for history in db.query(EloHistory).filter_by(team_id=duplicate_id).all():
            existing = db.query(EloHistory).filter_by(team_id=keeper_id, date=history.date).one_or_none()
            if existing is None:
                history.team_id = keeper_id
            else:
                db.delete(history)
        duplicate = db.get(Team, duplicate_id)
        if duplicate is not None:
            db.delete(duplicate)

    db.commit()
    return stats