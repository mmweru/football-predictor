"""
Trains the production model (see app/modeling.py) on every played match in the
4 supported leagues, prints an honest evaluation, and saves a model bundle.

    python -m app.train_ensemble                          # build features from the DB, evaluate, save model_bundle.joblib
    python -m app.train_ensemble --walk-forward 4         # also test on each of the last 4 whole seasons
    python -m app.train_ensemble --ablate                 # which of your metric groups actually help?
    python -m app.train_ensemble --market-weight 0.7      # how much to trust bookmaker odds when they exist
    python -m app.train_ensemble --use-injuries           # add real injury features (needs injury history in the DB)

What gets saved is trained on ALL played matches (more data = better live
predictions); the numbers printed come from models that never saw the data they
are scored on.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from app.feature_engine import DEFAULT_GROUPS, FEATURE_GROUPS, build_feature_frame, feature_columns
from app.modeling import (
    EnsembleModel, chronological_split, decay_weights, encode, evaluate, score_table, walk_forward,
)

BUNDLE_PATH = "model_bundle.joblib"


def _print_table(title: str, table: Dict[str, Dict[str, float]]) -> None:
    print(f"\n=== {title} ===")
    for name, m in table.items():
        print(f"  {name:34s} log_loss={m['log_loss']:.4f}  brier={m['brier']:.4f}  acc={m['accuracy']:.3f}  (n={m['n']})")


def _per_league(model: EnsembleModel, test: pd.DataFrame) -> None:
    print("\n=== Per-league (test window) ===")
    for lg, sub in test.groupby("league"):
        if len(sub) < 30:
            continue
        y = encode(sub["result"])
        e, f = evaluate(y, model.predict_fundamental(sub)), None
        print(f"  {lg:18s} n={len(sub):4d}  ensemble log_loss={e['log_loss']:.4f} acc={e['accuracy']:.3f}")


def ablate(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame, groups: List[str]) -> None:
    """Retrains with one group removed at a time. Positive delta = removing it made things WORSE = the group helps."""
    y = encode(test["result"])
    base = EnsembleModel(xgb_groups=groups).fit(train, val)
    base_ll = evaluate(y, base.predict_fundamental(test))["log_loss"]
    print(f"\n=== Ablation (ensemble, no odds). Baseline log_loss={base_ll:.4f} ===")
    print("  delta > 0  => the group HELPS (model gets worse without it)")
    for g in groups:
        if g == "elo":
            continue                                    # the LR + Poisson members depend on elo/form; skip the trivial case
        rest = [x for x in groups if x != g]
        m = EnsembleModel(xgb_groups=rest, lr_groups=[x for x in ("elo", "form", "team_stats") if x != g]).fit(train, val)
        ll = evaluate(y, m.predict_fundamental(test))["log_loss"]
        print(f"  without {g:11s} log_loss={ll:.4f}  delta={ll - base_ll:+.4f}")


def train(
    db: Session, model_out: str = BUNDLE_PATH, groups: Optional[List[str]] = None, market_weight: float = 0.8,
    walk_forward_seasons: int = 0, run_ablation: bool = False, half_life_days: float = 0.0, extra_games=None,
    injury_lookup=None,
) -> Optional[dict]:
    groups = list(groups or DEFAULT_GROUPS)
    frame = build_feature_frame(db, extra_games=extra_games, injury_lookup=injury_lookup)
    if frame.empty:
        print("No matches in the database for the 4 supported leagues. Run: python -m app.sync_data")
        return None
    P = frame[frame["played"]].copy().sort_values(["date", "match_id"]).reset_index(drop=True)
    print(f"Played matches: {len(P)}  |  leagues: {dict(P['league'].value_counts())}")
    print(f"Seasons: {sorted(P['season'].unique())}")
    if len(P) < 400:
        print("WARNING: <400 matches. Fine for a smoke test, but sync more seasons (python -m app.sync_data --seasons 8) before trusting this.")

    train_df, val_df, test_df = chronological_split(P)
    print(f"Split  train={len(train_df)} ({train_df['date'].min()}..{train_df['date'].max()})  "
          f"val={len(val_df)}  test={len(test_df)} ({test_df['date'].min()}..{test_df['date'].max()})")

    w = decay_weights(list(train_df["date"]), half_life_days) if half_life_days > 0 else None
    model = EnsembleModel(xgb_groups=groups, market_weight=market_weight).fit(train_df, val_df, sample_weight=w)
    print(f"XGBoost early-stopped at {model.best_iteration} trees (max 500)")
    _print_table("Held-out TEST window (models never saw these matches)", score_table(model, train_df, test_df))
    _per_league(model, test_df)

    if walk_forward_seasons > 0:
        seasons = sorted(P["season"].unique())[-walk_forward_seasons:]
        wf = walk_forward(P, seasons, market_weight=market_weight)
        print("\n=== Walk-forward: train on everything before each season, then predict that WHOLE season ===")
        print(wf.round(4).to_string(index=False))
        if len(wf):
            print(f"  mean ensemble log_loss={wf['ensemble'].mean():.4f}  vs mean Elo={wf['elo'].mean():.4f}"
                  + (f"  vs mean bookmaker={wf['market'].mean():.4f}" if "market" in wf else ""))

    if run_ablation:
        ablate(train_df, val_df, test_df, groups)

    # ---- Final production fit: ALL played matches, fixed tree count found above ----
    w_all = decay_weights(list(P["date"]), half_life_days) if half_life_days > 0 else None
    final = EnsembleModel(xgb_groups=groups, market_weight=market_weight, best_iteration=model.best_iteration)
    final.fit(P, None, n_estimators=int(model.best_iteration * 1.15) + 1, sample_weight=w_all)
    version = hashlib.sha1(f"{len(P)}-{P['date'].max()}-{groups}".encode()).hexdigest()[:8]
    metadata = {
        "model_version": version, "trained_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "n_matches": len(P), "last_match_date": str(P["date"].max()), "groups": groups, "market_weight": market_weight,
        "leagues": sorted(P["league"].unique()),
        "test_metrics": {k.strip(): v for k, v in score_table(model, train_df, test_df).items()},
    }
    joblib.dump({"model": final, "metadata": metadata}, model_out)
    print(f"\nSaved production model (version {version}, trained on {len(P)} matches) -> {model_out}")
    return metadata


def main() -> None:
    from app.database import SessionLocal

    ap = argparse.ArgumentParser(description="Train the ensemble match-outcome model.")
    ap.add_argument("--model-out", default=BUNDLE_PATH)
    ap.add_argument("--groups", default=None, help=f"comma list from {list(FEATURE_GROUPS)}; default = the 9 metric groups")
    ap.add_argument("--use-injuries", action="store_true", help="add the 'injury' feature group (needs historical injury rows)")
    ap.add_argument("--market-weight", type=float, default=0.8, help="0-1; weight on bookmaker odds when available (0 = ignore odds)")
    ap.add_argument("--walk-forward", type=int, default=0, metavar="N", help="also evaluate on each of the last N whole seasons")
    ap.add_argument("--ablate", action="store_true", help="show which feature groups help")
    ap.add_argument("--half-life-days", type=float, default=0.0, help="down-weight older matches (0 = off)")
    args = ap.parse_args()

    groups = args.groups.split(",") if args.groups else list(DEFAULT_GROUPS)
    injury_lookup = None
    if args.use_injuries:
        from app.injuries import make_injury_lookup
        groups.append("injury")
        db0 = SessionLocal()
        try:
            injury_lookup = make_injury_lookup(db0)
        finally:
            db0.close()
    db = SessionLocal()
    try:
        train(db, args.model_out, groups, args.market_weight, args.walk_forward, args.ablate, args.half_life_days,
              injury_lookup=injury_lookup)
    finally:
        db.close()


if __name__ == "__main__":
    main()
