"""
Batch jobs for the season-long loop:

    python -m app.predict_fixtures     # predict EVERY upcoming fixture (all 4 leagues) and store it
    python -m app.grade_predictions    # after results arrive, score the stored predictions

Together with `sync_data` (which writes real results over the old "upcoming" rows) this is "all subsequent games
predicted, all subsequent games played": each result updates Elo / form / h2h for the NEXT prediction automatically,
because every prediction is computed from the state after all results to date.
"""

from __future__ import annotations

import datetime as dt
from typing import Optional

import joblib
import pandas as pd
from sqlalchemy.orm import Session

from app.models import Match, Prediction, Team
from app.modeling import LABELS, scoreline_summary
from app.predict_live import apply_injury_adjustment, get_engine, matchup_features
from app.injuries import team_injury_report


def predict_upcoming(db: Session, bundle_path: str = "model_bundle.joblib", horizon_days: int = 60,
                     today: Optional[dt.date] = None) -> int:
    today = today or dt.date.today()
    saved = joblib.load(bundle_path)
    model, version = saved["model"], saved["metadata"]["model_version"]
    fixtures = (db.query(Match).filter(Match.home_score.is_(None), Match.date >= today,
                                       Match.date <= today + dt.timedelta(days=horizon_days)).order_by(Match.date).all())
    teams = {t.id: t for t in db.query(Team).all()}
    n = 0
    for m in fixtures:
        home, away = teams[m.home_team_id], teams[m.away_team_id]
        if not (home.is_active and away.is_active):
            continue
        ov = {k: v for k, v in dict(temp_c=m.temp_c, wind_kph=m.wind_kph, precip_mm=m.precip_mm,
                                    odds_home=m.odds_home, odds_draw=m.odds_draw, odds_away=m.odds_away).items() if v is not None}
        feats = matchup_features(db, home, away, m.date, ov)
        feats = apply_injury_adjustment(feats, team_injury_report(db, home.id)["elo_penalty"], team_injury_report(db, away.id)["elo_penalty"])
        p = model.predict(pd.DataFrame([feats]))[0]
        row = db.query(Prediction).filter_by(match_id=m.id, model_version=version).one_or_none() or Prediction(match_id=m.id, model_version=version)
        row.created_at = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)
        row.p_away, row.p_draw, row.p_home = float(p[0]), float(p[1]), float(p[2])
        row.exp_home_goals, row.exp_away_goals = float(feats["exp_home_goals"]), float(feats["exp_away_goals"])
        row.predicted = LABELS[int(p.argmax())]
        db.add(row)
        n += 1
    db.commit()
    print(f"Stored predictions for {n} upcoming fixtures (model {version}).")
    return n


def main() -> None:
    import argparse
    from app.database import SessionLocal
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="model_bundle.joblib")
    ap.add_argument("--horizon-days", type=int, default=60)
    a = ap.parse_args()
    db = SessionLocal()
    try:
        predict_upcoming(db, a.model, a.horizon_days)
    finally:
        db.close()


if __name__ == "__main__":
    main()
