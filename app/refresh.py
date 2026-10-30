"""
The whole weekly loop in one command (put it in cron / Task Scheduler / a GitHub Action):

    python -m app.refresh              # sync -> grade last week's predictions -> retrain if due -> predict all fixtures
    python -m app.refresh --retrain    # force a retrain now
"""

from __future__ import annotations

import argparse
import os
import time

from app.sync_data import sync
from app.config import settings


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--retrain", action="store_true")
    ap.add_argument("--model", default="model_bundle.joblib")
    ap.add_argument("--retrain-every-days", type=float, default=7.0)
    a = ap.parse_args()

    sync(seasons=1, use_fdorg=bool(settings.football_data_org_token), use_fpl=True, use_weather=True)

    from app.database import SessionLocal
    from app.grade_predictions import grade, track_record
    db = SessionLocal()
    try:
        print("graded:", grade(db), "| track record:", track_record(db))
        stale = (not os.path.exists(a.model)) or (time.time() - os.path.getmtime(a.model)) > a.retrain_every_days * 86400
        if a.retrain or stale:
            from app.train_ensemble import train
            train(db, a.model, market_weight=settings.market_weight)
        from app.predict_fixtures import predict_upcoming
        predict_upcoming(db, a.model)
    finally:
        db.close()


if __name__ == "__main__":
    main()
