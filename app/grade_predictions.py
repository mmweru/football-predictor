"""Fills `actual_result` on stored predictions whose match has now been played, and prints the running track record."""

from __future__ import annotations

from typing import Dict

import numpy as np
from sqlalchemy.orm import Session

from app.models import Match, Prediction


def grade(db: Session) -> int:
    n = 0
    for pred, m in db.query(Prediction, Match).join(Match, Match.id == Prediction.match_id).filter(
            Prediction.actual_result.is_(None), Match.home_score.isnot(None)).all():
        pred.actual_result = "H" if m.home_score > m.away_score else "A" if m.home_score < m.away_score else "D"
        n += 1
    db.commit()
    return n


def track_record(db: Session) -> Dict[str, object]:
    rows = db.query(Prediction).filter(Prediction.actual_result.isnot(None)).all()
    if not rows:
        return {"graded": 0}
    idx = {"A": 0, "D": 1, "H": 2}
    P = np.array([[r.p_away, r.p_draw, r.p_home] for r in rows])
    y = np.array([idx[r.actual_result] for r in rows])
    ll = float(-np.mean(np.log(np.clip(P[np.arange(len(y)), y], 1e-9, 1))))
    return {"graded": len(rows), "accuracy": float(np.mean(P.argmax(1) == y)), "log_loss": ll,
            "brier": float(np.mean(np.sum((P - np.eye(3)[y]) ** 2, axis=1)))}


def main() -> None:
    from app.database import SessionLocal
    db = SessionLocal()
    try:
        print(f"Graded {grade(db)} new predictions.  Track record: {track_record(db)}")
    finally:
        db.close()


if __name__ == "__main__":
    main()
