"""
Tests for app.features.

The most important property being tested throughout: every function must
use ONLY matches strictly before the given date. Several tests exist
specifically to catch leakage bugs (a match on-or-after the cutoff
accidentally being included).
"""

import datetime as dt

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database import Base
from app.derbies import is_derby
from app.features import (
    get_current_elo_rating,
    get_current_injury_impact,
    get_elo_rating,
    get_head_to_head,
    get_injury_impact,
    get_rest_days,
    get_rolling_form,
    get_travel_distance_km,
)
from app.models import EloHistory, Injury, Match, Player, Team


@pytest.fixture()
def db():
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine, future=True)
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


def _team(db, name, **kwargs):
    t = Team(name=name, **kwargs)
    db.add(t)
    db.commit()
    return t


def _played_match(db, home, away, home_score, away_score, date):
    m = Match(date=date, home_team_id=home.id, away_team_id=away.id, home_score=home_score, away_score=away_score)
    db.add(m)
    db.commit()
    return m


# ---------------------------------------------------------------------------
# Rest days
# ---------------------------------------------------------------------------
def test_rest_days_none_when_no_prior_match(db):
    team = _team(db, "New FC")
    opponent = _team(db, "Opponent FC")
    assert get_rest_days(db, team.id, dt.date(2024, 1, 1)) is None


def test_rest_days_counts_correctly(db):
    home = _team(db, "Home FC")
    away = _team(db, "Away FC")
    _played_match(db, home, away, 1, 0, dt.date(2024, 1, 1))

    rest = get_rest_days(db, home.id, dt.date(2024, 1, 8))
    assert rest == 7


def test_rest_days_ignores_matches_on_or_after_cutoff(db):
    """Leakage check: a match ON the cutoff date must not count as 'previous'."""
    home = _team(db, "Home FC")
    away = _team(db, "Away FC")
    earlier = _played_match(db, home, away, 1, 0, dt.date(2024, 1, 1))
    # A match exactly on the cutoff date should be excluded (not "in the past").
    _played_match(db, home, away, 2, 2, dt.date(2024, 1, 8))

    rest = get_rest_days(db, home.id, dt.date(2024, 1, 8))
    assert rest == 7  # must be measured from Jan 1, NOT from the Jan 8 match itself


# ---------------------------------------------------------------------------
# Rolling form
# ---------------------------------------------------------------------------
def test_rolling_form_no_history_returns_neutral_default(db):
    team = _team(db, "Fresh FC")
    form = get_rolling_form(db, team.id, dt.date(2024, 1, 1))
    assert form["matches_played"] == 0
    assert form["win_rate"] == 0.5


def test_rolling_form_computes_win_rate_and_goals_from_both_home_and_away_games(db):
    team = _team(db, "Team A")
    opp1 = _team(db, "Opp 1")
    opp2 = _team(db, "Opp 2")

    _played_match(db, team, opp1, 3, 1, dt.date(2024, 1, 1))   # Team A home win, scored 3 conceded 1
    _played_match(db, opp2, team, 0, 2, dt.date(2024, 1, 8))   # Team A away win, scored 2 conceded 0

    form = get_rolling_form(db, team.id, dt.date(2024, 1, 15), num_matches=5)
    assert form["matches_played"] == 2
    assert form["win_rate"] == pytest.approx(1.0)
    assert form["avg_goals_scored"] == pytest.approx((3 + 2) / 2)
    assert form["avg_goals_conceded"] == pytest.approx((1 + 0) / 2)


def test_rolling_form_respects_num_matches_limit(db):
    team = _team(db, "Team A")
    opp = _team(db, "Opp")
    for i in range(10):
        _played_match(db, team, opp, 1, 0, dt.date(2024, 1, 1) + dt.timedelta(days=i * 7))

    form = get_rolling_form(db, team.id, dt.date(2024, 12, 1), num_matches=3)
    assert form["matches_played"] == 3  # only the most recent 3, not all 10


def test_rolling_form_excludes_matches_on_or_after_cutoff(db):
    team = _team(db, "Team A")
    opp = _team(db, "Opp")
    _played_match(db, team, opp, 5, 0, dt.date(2024, 1, 1))  # big win, before cutoff
    _played_match(db, team, opp, 0, 5, dt.date(2024, 1, 10))  # big loss, ON the cutoff date

    form = get_rolling_form(db, team.id, dt.date(2024, 1, 10), num_matches=5)
    # Only the Jan 1 win should count — the Jan 10 loss is on the cutoff itself.
    assert form["matches_played"] == 1
    assert form["win_rate"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Head-to-head
# ---------------------------------------------------------------------------
def test_head_to_head_no_history(db):
    a = _team(db, "Team A")
    b = _team(db, "Team B")
    h2h = get_head_to_head(db, a.id, b.id, dt.date(2024, 1, 1))
    assert h2h["h2h_matches_played"] == 0
    assert h2h["h2h_win_rate"] == 0.5


def test_head_to_head_counts_meetings_in_either_venue_configuration(db):
    a = _team(db, "Team A")
    b = _team(db, "Team B")
    _played_match(db, a, b, 2, 1, dt.date(2024, 1, 1))  # A home win
    _played_match(db, b, a, 0, 3, dt.date(2024, 6, 1))  # A away win (B home, A away)

    h2h = get_head_to_head(db, a.id, b.id, dt.date(2024, 12, 1))
    assert h2h["h2h_matches_played"] == 2
    assert h2h["h2h_win_rate"] == pytest.approx(1.0)  # A won both, regardless of venue
    assert h2h["h2h_avg_goal_diff"] == pytest.approx(((2 - 1) + (3 - 0)) / 2)


def test_head_to_head_ignores_unrelated_matches(db):
    a = _team(db, "Team A")
    b = _team(db, "Team B")
    c = _team(db, "Team C")
    _played_match(db, a, b, 1, 0, dt.date(2024, 1, 1))
    _played_match(db, a, c, 5, 0, dt.date(2024, 2, 1))  # not a meeting between A and B

    h2h = get_head_to_head(db, a.id, b.id, dt.date(2024, 12, 1))
    assert h2h["h2h_matches_played"] == 1


# ---------------------------------------------------------------------------
# Travel distance
# ---------------------------------------------------------------------------
def test_travel_distance_returns_none_without_coordinates(db):
    home = _team(db, "Home FC")  # no stadium_lat/lon set
    away = _team(db, "Away FC")
    assert get_travel_distance_km(home, away) is None


def test_travel_distance_computed_with_coordinates(db):
    home = _team(db, "Arsenal", stadium_lat=51.5549, stadium_lon=-0.1084)
    away = _team(db, "Man United", stadium_lat=53.4631, stadium_lon=-2.2913)

    distance = get_travel_distance_km(home, away)
    assert 255 < distance < 270  # matches the known ~262km London-Manchester distance


def test_travel_distance_prefers_match_venue_override(db):
    """A local match played away from the home team's registered stadium
    should use the override, not the home team's usual coordinates."""
    home = _team(db, "Home FC", stadium_lat=51.5549, stadium_lon=-0.1084)  # London
    away = _team(db, "Away FC", stadium_lat=53.4631, stadium_lon=-2.2913)  # Manchester

    # Match is actually played at a neutral venue much closer to the away team, e.g. Leeds.
    match = Match(
        date=dt.date(2024, 1, 1), home_team_id=home.id, away_team_id=away.id,
        venue_lat=53.8008, venue_lon=-1.5491,  # Leeds
    )
    db.add(match)
    db.commit()

    distance_with_override = get_travel_distance_km(home, away, match=match)
    distance_without_override = get_travel_distance_km(home, away, match=None)

    # Leeds is much closer to Manchester than London is — override should reduce distance a lot.
    assert distance_with_override < distance_without_override
    assert distance_with_override < 100  # Manchester-Leeds is roughly 60km


def test_travel_distance_falls_back_to_home_stadium_when_no_override(db):
    home = _team(db, "Arsenal", stadium_lat=51.5549, stadium_lon=-0.1084)
    away = _team(db, "Man United", stadium_lat=53.4631, stadium_lon=-2.2913)
    match = Match(date=dt.date(2024, 1, 1), home_team_id=home.id, away_team_id=away.id)  # no venue override
    db.add(match)
    db.commit()

    distance = get_travel_distance_km(home, away, match=match)
    assert 255 < distance < 270  # same as the no-match-arg case


# ---------------------------------------------------------------------------
# Injuries
# ---------------------------------------------------------------------------
def test_injury_impact_no_injuries(db):
    team = _team(db, "Healthy FC")
    opp = _team(db, "Opp")
    match = _played_match(db, team, opp, 1, 0, dt.date(2024, 1, 1))

    impact = get_injury_impact(db, team.id, match.id)
    assert impact["injury_count"] == 0
    assert impact["injury_importance_sum"] == 0


def test_injury_impact_counts_and_weights_correctly(db):
    team = _team(db, "Injured FC")
    opp = _team(db, "Opp")
    match = _played_match(db, team, opp, 1, 0, dt.date(2024, 1, 1))

    star = Player(name="Star Player", team_id=team.id, importance_weight=0.9)
    bench = Player(name="Bench Player", team_id=team.id, importance_weight=0.2)
    db.add_all([star, bench])
    db.commit()

    db.add(Injury(player_id=star.id, date_out=dt.date(2023, 12, 20), match_missed_id=match.id))
    db.add(Injury(player_id=bench.id, date_out=dt.date(2023, 12, 25), match_missed_id=match.id))
    db.commit()

    impact = get_injury_impact(db, team.id, match.id)
    assert impact["injury_count"] == 2
    assert impact["injury_importance_sum"] == pytest.approx(1.1)


def test_injury_impact_ignores_other_teams_injuries(db):
    team = _team(db, "Team A")
    other_team = _team(db, "Team B")
    opp = _team(db, "Opp")
    match = _played_match(db, team, opp, 1, 0, dt.date(2024, 1, 1))

    other_player = Player(name="Not Relevant", team_id=other_team.id, importance_weight=0.9)
    db.add(other_player)
    db.commit()
    db.add(Injury(player_id=other_player.id, date_out=dt.date(2023, 12, 20), match_missed_id=match.id))
    db.commit()

    impact = get_injury_impact(db, team.id, match.id)
    assert impact["injury_count"] == 0


# ---------------------------------------------------------------------------
# Current injuries (for live/upcoming match predictions, not tied to a match_id)
# ---------------------------------------------------------------------------
def test_current_injury_impact_counts_player_within_window(db):
    team = _team(db, "Team A")
    player = Player(name="Injured Player", team_id=team.id, importance_weight=0.8)
    db.add(player)
    db.commit()
    db.add(Injury(player_id=player.id, date_out=dt.date(2024, 1, 1), date_back=dt.date(2024, 2, 1)))
    db.commit()

    impact = get_current_injury_impact(db, team.id, dt.date(2024, 1, 15))
    assert impact["injury_count"] == 1
    assert impact["injury_importance_sum"] == pytest.approx(0.8)


def test_current_injury_impact_excludes_player_before_injury_started(db):
    team = _team(db, "Team A")
    player = Player(name="Injured Player", team_id=team.id, importance_weight=0.8)
    db.add(player)
    db.commit()
    db.add(Injury(player_id=player.id, date_out=dt.date(2024, 2, 1), date_back=dt.date(2024, 3, 1)))
    db.commit()

    impact = get_current_injury_impact(db, team.id, dt.date(2024, 1, 15))  # before date_out
    assert impact["injury_count"] == 0


def test_current_injury_impact_excludes_player_after_recovery(db):
    team = _team(db, "Team A")
    player = Player(name="Recovered Player", team_id=team.id, importance_weight=0.8)
    db.add(player)
    db.commit()
    db.add(Injury(player_id=player.id, date_out=dt.date(2024, 1, 1), date_back=dt.date(2024, 1, 31)))
    db.commit()

    impact = get_current_injury_impact(db, team.id, dt.date(2024, 2, 15))  # after date_back
    assert impact["injury_count"] == 0


def test_current_injury_impact_open_ended_injury_still_counts(db):
    """An injury with no date_back (unresolved/long-term) should still count as 'currently out'."""
    team = _team(db, "Team A")
    player = Player(name="Long Term Injury", team_id=team.id, importance_weight=0.6)
    db.add(player)
    db.commit()
    db.add(Injury(player_id=player.id, date_out=dt.date(2024, 1, 1), date_back=None))
    db.commit()

    impact = get_current_injury_impact(db, team.id, dt.date(2025, 1, 1))  # a year later, still no date_back
    assert impact["injury_count"] == 1


# ---------------------------------------------------------------------------
# Elo lookup
# ---------------------------------------------------------------------------
def test_get_elo_rating_returns_stored_value(db):
    team = _team(db, "Team A")
    db.add(EloHistory(team_id=team.id, date=dt.date(2024, 1, 1), rating=1623.5))
    db.commit()

    assert get_elo_rating(db, team.id, dt.date(2024, 1, 1)) == pytest.approx(1623.5)


def test_get_elo_rating_falls_back_to_base_when_missing(db):
    team = _team(db, "No Elo Yet FC")
    assert get_elo_rating(db, team.id, dt.date(2024, 1, 1)) == pytest.approx(1500.0)


def test_get_current_elo_rating_returns_most_recent_on_or_before_date(db):
    team = _team(db, "Team A")
    db.add(EloHistory(team_id=team.id, date=dt.date(2024, 1, 1), rating=1500.0))
    db.add(EloHistory(team_id=team.id, date=dt.date(2024, 6, 1), rating=1600.0))
    db.commit()

    # A future date with no exact elo_history row should return the LATEST prior rating.
    assert get_current_elo_rating(db, team.id, dt.date(2024, 12, 1)) == pytest.approx(1600.0)


def test_get_current_elo_rating_ignores_future_entries(db):
    team = _team(db, "Team A")
    db.add(EloHistory(team_id=team.id, date=dt.date(2024, 1, 1), rating=1500.0))
    db.add(EloHistory(team_id=team.id, date=dt.date(2025, 1, 1), rating=1700.0))  # after the as_of_date below
    db.commit()

    assert get_current_elo_rating(db, team.id, dt.date(2024, 6, 1)) == pytest.approx(1500.0)


def test_get_current_elo_rating_falls_back_to_base_when_no_history(db):
    team = _team(db, "Brand New FC")
    assert get_current_elo_rating(db, team.id, dt.date(2024, 1, 1)) == pytest.approx(1500.0)


# ---------------------------------------------------------------------------
# Derbies (no DB needed — pure function)
# ---------------------------------------------------------------------------
def test_is_derby_known_rivalry():
    assert is_derby("Arsenal", "Tottenham Hotspur") is True
    assert is_derby("Tottenham Hotspur", "Arsenal") is True  # order shouldn't matter


def test_is_derby_unrelated_teams():
    assert is_derby("Arsenal", "Bournemouth") is False
