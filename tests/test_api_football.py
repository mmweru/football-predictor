import datetime as dt
from typing import cast

import requests

from app.data_sources.api_football import APIFootball, SeasonUnavailableError, parse_fixtures, parse_player_stats, sync_api_football
from app.data_sources.bigballs import BigBalls
from app.leagues import BY_NAME
from app.team_aliases import normalize_team_name


def test_fixture_parser_keeps_finished_scores_and_leaves_upcoming_scores_empty():
    league = BY_NAME["Premier League"]
    rows = [
        {"fixture": {"date": "2024-08-16T19:00:00+00:00", "status": {"short": "FT"}},
         "teams": {"home": {"name": "Manchester United"}, "away": {"name": "Fulham"}},
         "goals": {"home": 1, "away": 0}},
        {"fixture": {"date": "2024-08-17T14:00:00+00:00", "status": {"short": "NS"}},
         "teams": {"home": {"name": "Arsenal"}, "away": {"name": "Wolves"}}, "goals": {"home": None, "away": None}},
    ]

    frame = parse_fixtures(league, 2024, rows)

    assert frame.iloc[0]["date"] == dt.date(2024, 8, 16)
    assert frame.iloc[0]["home_score"] == 1 and frame.iloc[0]["away_score"] == 0
    assert frame.iloc[1]["home_score"] != frame.iloc[1]["home_score"]


def test_player_stats_parser_keeps_season_and_team_scoped_totals():
    league = BY_NAME["EFL Championship"]
    rows = [{
        "player": {"id": 123, "name": "Example Player"},
        "statistics": [{
            "team": {"name": "Leeds United"},
            "games": {"position": "Attacker", "appearances": 32, "lineups": 25, "minutes": 2200, "rating": "7.2"},
            "goals": {"total": 12, "assists": 4}, "shots": {"total": 60, "on": 25},
            "cards": {"yellow": 3, "red": 0},
        }],
    }]

    result = parse_player_stats(rows, league, 2024)

    assert result == [{
        "api_player_id": 123, "player_name": "Example Player", "team_name": "Leeds United",
        "league": "EFL Championship", "season": 2024, "position": "Attacker", "appearances": 32,
        "lineups": 25, "minutes": 2200, "goals": 12, "assists": 4, "shots": 60,
        "shots_on_target": 25, "yellow_cards": 3, "red_cards": 0, "rating": 7.2,
    }]


def test_bigballs_injury_parser_unwraps_feed_and_keeps_unmapped_team_uuid():
    class Response:
        status_code = 200
        headers = {}
        text = ""

        def __init__(self, payload):
            self.payload = payload

        def json(self):
            return self.payload

    class Session:
        def __init__(self):
            self.params = []

        def get(self, url, **kwargs):
            self.params.append(kwargs["params"])
            if url.endswith("/v1/injuries"):
                return Response({"data": {"injuries": {"value": [{
                    "id": "injury-1", "full_name": "Example Player", "display_name": "Example",
                    "current_team_id": "team-uuid",
                }]}}})
            return Response({"data": {"standings": [{"rows": [{"team_id": "different-id", "team_name": "Example FC"}]}]}})

    session = Session()
    rows = BigBalls("test-key", cast(requests.Session, session)).epl_absences(today=dt.date(2026, 10, 30))

    assert session.params[0] == {"sport": "football", "limit": 10}
    assert rows == [{
        "player": "Example Player", "team": None, "team_id": "team-uuid",
        "date": dt.date(2026, 10, 30), "reason": None, "status": "i",
    }]


def test_api_football_player_stats_use_auth_header_and_fetch_all_pages():
    class Response:
        status_code = 200
        headers = {}
        text = ""

        def __init__(self, page):
            self.page = page

        def json(self):
            return {"response": [], "paging": {"current": self.page, "total": 5}}

    class Session:
        def __init__(self):
            self.pages = []
            self.headers = None

        def get(self, url, **kwargs):
            self.pages.append(kwargs["params"]["page"])
            self.headers = kwargs["headers"]
            return Response(kwargs["params"]["page"])

    session = Session()
    rows = APIFootball("test-key", cast(requests.Session, session), min_request_interval_s=0).player_stats(
        BY_NAME["Premier League"], 2024,
    )

    assert rows == []
    assert session.pages == [1, 2, 3]
    assert session.headers is not None
    assert session.headers["x-apisports-key"] == "test-key"


def test_sync_skips_an_unavailable_season_after_first_plan_error():
    class API:
        def __init__(self):
            self.calls = []

        def fixtures(self, league, season):
            self.calls.append((league.name, season))
            raise SeasonUnavailableError("not available")

    api = API()
    totals = sync_api_football(None, cast(APIFootball, api), [2025])

    assert api.calls == [("Premier League", 2025)]
    assert totals["matches_inserted"] == 0


def test_suffix_normalization_keeps_one_name_for_epl_fc_variants():
    assert normalize_team_name("Crystal Palace FC") == "Crystal Palace"
    assert normalize_team_name("Aston Villa F.C.") == "Aston Villa"