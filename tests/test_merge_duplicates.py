import datetime as dt

import numpy as np
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database import Base
from app.merge_duplicates import merge_duplicate_teams
from app.models import Match, Player, Prediction, Team
from app.sync_core import refresh_membership, upsert_matches
from app.team_aliases import normalize_team_name
from tests.helpers import fixture_frame


@pytest.fixture()
def db():
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    s = sessionmaker(bind=engine, future=True)()
    yield s
    s.close()


def test_fc_suffix_is_stripped_even_for_clubs_missing_from_the_alias_table():
    for raw in ("Crystal Palace FC", "Aston Villa FC", "Arsenal FC", "Hull City AFC", "Sunderland AFC", "Brentford F.C."):
        assert not normalize_team_name(raw).endswith(("FC", "AFC", "F.C."))
    assert normalize_team_name("Crystal Palace FC") == "Crystal Palace" == normalize_team_name("Crystal Palace")
    assert normalize_team_name("AFC Wimbledon") == "AFC Wimbledon"            # leading AFC is part of the real name
    assert normalize_team_name("FC Halifax Town") == "FC Halifax Town"        # leading FC untouched


def _row(date, h, a, hs=np.nan, as_=np.nan, **kw):
    return dict(date=date, league="Premier League", season=2026, home_team=h, away_team=a, home_score=hs, away_score=as_, **kw)


def test_merge_repairs_the_exact_situation_from_a_real_run(db):
    # football-data.co.uk ingested first, with plain names + results + stats
    upsert_matches(db, fixture_frame([_row(dt.date(2026, 8, 22), "Crystal Palace", "Arsenal", 1, 2, home_shots=9)]))
    # old buggy normaliser then created "...FC" clubs from football-data.org, incl. the same fixture a day later (UTC date) + a new one
    db.add_all([Team(name="Crystal Palace FC", league="Premier League", fdorg_id=354, is_active=True),
                Team(name="Arsenal FC", league="Premier League", fdorg_id=57, is_active=True)])
    db.flush()
    cp, ar = db.query(Team).filter_by(name="Crystal Palace FC").one(), db.query(Team).filter_by(name="Arsenal FC").one()
    db.add_all([Match(date=dt.date(2026, 8, 23), home_team_id=cp.id, away_team_id=ar.id, competition="Premier League", season=2026),
                Match(date=dt.date(2026, 11, 1), home_team_id=ar.id, away_team_id=cp.id, competition="Premier League", season=2026)])
    db.add(Player(name="Star", team_id=cp.id, importance_weight=0.9))
    db.commit()
    assert db.query(Team).count() == 4

    stats = merge_duplicate_teams(db)

    names = sorted(t.name for t in db.query(Team).all())
    assert names == ["Arsenal", "Crystal Palace"]
    assert db.query(Match).count() == 2                                       # duplicate fixture collapsed, new fixture kept
    played = db.query(Match).filter(Match.home_score.isnot(None)).one()
    assert (played.home_score, played.away_score, played.home_shots) == (1, 2, 9)   # the real result survived
    assert db.query(Team).filter_by(name="Crystal Palace").one().fdorg_id == 354   # external id carried over
    assert db.query(Player).one().team_id == db.query(Team).filter_by(name="Crystal Palace").one().id
    assert stats["teams_merged"] == 2


def test_merge_is_idempotent_and_membership_is_correct_afterwards(db):
    rows = [_row(dt.date(2026, 8, 10 + i), f"Club {i}", f"Club {i + 10}", 1, 0) for i in range(10)]
    upsert_matches(db, fixture_frame(rows))
    for i in range(3):                                                         # three buggy duplicates
        t = Team(name=f"Club {i} FC", league="Premier League", is_active=True)
        db.add(t); db.flush()
        db.add(Match(date=dt.date(2026, 11, 1 + i), home_team_id=t.id, away_team_id=db.query(Team).filter_by(name=f"Club {i + 10}").one().id,
                     competition="Premier League", season=2026))
    db.commit()
    merge_duplicate_teams(db)
    assert merge_duplicate_teams(db) == {"teams_merged": 0, "teams_renamed": 0, "matches_merged": 0}
    rep = refresh_membership(db, 2026)
    assert rep["per_league"]["Premier League"] == 20


def test_rename_when_no_canonical_team_exists(db):
    db.add(Team(name="Some Town FC", league="EFL League Two", is_active=True))
    db.commit()
    assert merge_duplicate_teams(db)["teams_renamed"] == 1
    assert db.query(Team).one().name == "Some Town"


def test_two_real_results_between_same_pair_are_both_kept(db):
    """League game + play-off between the same clubs in one season must not be merged."""
    upsert_matches(db, fixture_frame([_row(dt.date(2026, 5, 3), "Leeds", "Hull", 1, 0), _row(dt.date(2026, 5, 4), "Leeds", "Hull", 2, 2)]))
    merge_duplicate_teams(db)
    assert db.query(Match).count() == 2
