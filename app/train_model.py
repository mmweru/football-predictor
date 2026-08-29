"""
Trains an XGBoost multiclass model (Home win / Draw / Away win) on the
feature table from app.build_feature_table, evaluates it against the Elo
and naive baselines using a time-based split, and saves the trained model.

Usage:
    python -m app.train_model --input features.csv
    python -m app.train_model --input features.csv --model-out model.joblib

If --input is omitted, builds the feature table fresh from the database
(same as running app.build_feature_table).
"""

from __future__ import annotations

import argparse
from typing import List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from app.baselines import (
    OUTCOME_LABELS,
    elo_baseline_probabilities,
    estimate_draw_probability,
    naive_baseline_probabilities,
)
from app.data_split import print_split_summary, time_based_split, time_based_split_grouped
from app.evaluation import evaluate_predictions, print_evaluation

# Columns that are identifiers/labels/leakage risks, never fed to the model as features.
NON_FEATURE_COLUMNS = [
    "match_id", "date", "home_team", "away_team", "competition", "venue_name",
    "home_score", "away_score", "result",
]

# Columns that ARE features but are categorical (strings), not numeric —
# handled separately from the numeric-coercion loop below, and require
# XGBoost's enable_categorical=True to train on directly.
CATEGORICAL_FEATURE_COLUMNS = ["league"]


def get_feature_columns(df: pd.DataFrame, exclude: Optional[List[str]] = None) -> List[str]:
    exclude = exclude or []
    return [c for c in df.columns if c not in NON_FEATURE_COLUMNS and c not in exclude]


def prepare_features(
    df: pd.DataFrame, exclude: Optional[List[str]] = None,
    category_dtypes: Optional[dict] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Selects feature columns, converts booleans to int (XGBoost wants
    numeric), casts known categorical columns (e.g. "league") to pandas
    'category' dtype, and leaves missing numeric values as NaN — XGBoost
    handles missing values natively (it learns a default split direction
    for them during training), so we deliberately do NOT impute here. This
    preserves the "missing = genuinely unknown" signal (e.g. a team's
    first-ever match having no rolling form) rather than pretending it's a
    specific value.

    category_dtypes: if provided, categorical columns are cast to these
    EXACT pandas CategoricalDtype objects (same categories, same order)
    instead of inferring categories fresh from this particular DataFrame.
    This matters because train/val/test/live-prediction must all use the
    identical category encoding — inferring separately per split risks a
    category appearing in val but not train (or vice versa) getting
    encoded inconsistently. Get category_dtypes from
    get_category_dtypes(train_df) and reuse it everywhere else.
    """
    feature_cols = get_feature_columns(df, exclude=exclude)
    X = df[feature_cols].copy()
    for col in X.columns:
        if col in CATEGORICAL_FEATURE_COLUMNS:
            if category_dtypes and col in category_dtypes:
                X[col] = X[col].astype(category_dtypes[col])
            else:
                X[col] = X[col].astype("category")
        elif X[col].dtype == bool:
            X[col] = X[col].astype(int)
        else:
            X[col] = pd.to_numeric(X[col], errors="coerce")
    return X, feature_cols


def get_category_dtypes(train_df: pd.DataFrame) -> dict:
    """
    Builds the canonical category set for each categorical feature, from
    the TRAINING data only (never val/test — that would leak information
    about categories only seen later). Save this alongside the model so
    live predictions use the exact same encoding.
    """
    dtypes = {}
    for col in CATEGORICAL_FEATURE_COLUMNS:
        if col in train_df.columns:
            dtypes[col] = pd.CategoricalDtype(categories=sorted(train_df[col].dropna().unique()))
    return dtypes


def train_xgboost(
    X_train: pd.DataFrame, y_train: pd.Series,
    X_val: Optional[pd.DataFrame] = None, y_val: Optional[pd.Series] = None,
    early_stopping_rounds: Optional[int] = 20,
    **xgb_kwargs,
) -> XGBClassifier:
    """
    If X_val/y_val are provided, uses early stopping: training halts once
    validation log loss stops improving for `early_stopping_rounds` rounds,
    rather than always running the full n_estimators. This is the standard
    guard against XGBoost overfitting a small/noisy dataset — without it,
    the model can easily end up worse than a simple baseline on unseen
    data despite fitting the training set very well (check
    model.best_iteration after training; if it's much lower than
    n_estimators, that's a sign your data doesn't support this many trees).
    """
    label_to_idx = {label: i for i, label in enumerate(OUTCOME_LABELS)}
    y_train_encoded = y_train.map(label_to_idx)

    default_params = dict(
        objective="multi:softprob",
        num_class=len(OUTCOME_LABELS),
        eval_metric="mlogloss",
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=42,
    )
    # Auto-enable categorical support if any categorical dtype columns are present
    # (e.g. "league") — without this, XGBoost rejects category-dtype columns outright.
    if any(X_train[col].dtype.name == "category" for col in X_train.columns):
        default_params["enable_categorical"] = True
    default_params.update(xgb_kwargs)

    use_early_stopping = X_val is not None and y_val is not None and early_stopping_rounds is not None
    if use_early_stopping:
        default_params["early_stopping_rounds"] = early_stopping_rounds

    model = XGBClassifier(**default_params)

    if use_early_stopping:
        y_val_encoded = y_val.map(label_to_idx)
        model.fit(X_train, y_train_encoded, eval_set=[(X_val, y_val_encoded)], verbose=False)
    else:
        model.fit(X_train, y_train_encoded)

    return model


def predict_probabilities(model: XGBClassifier, X: pd.DataFrame) -> List[dict]:
    """Returns model predictions as a list of {'H':.., 'D':.., 'A':..} dicts, matching the baseline functions' format."""
    proba = model.predict_proba(X)  # columns in OUTCOME_LABELS order (0=A, 1=D, 2=H), since that's how we encoded y_train
    return [dict(zip(OUTCOME_LABELS, row)) for row in proba]


def get_feature_importance(model: XGBClassifier, feature_cols: List[str]) -> List[Tuple[str, float]]:
    """
    Returns (feature_name, importance) pairs, sorted descending by
    importance. Uses XGBoost's default 'gain' importance type — the
    average improvement in the loss function each feature contributes
    when it's used in a split, which is generally more informative than
    raw split-count ('weight') importance for judging whether a feature
    is actually useful vs just frequently (but weakly) used.
    """
    importances = model.feature_importances_
    pairs = list(zip(feature_cols, importances))
    return sorted(pairs, key=lambda p: p[1], reverse=True)


def print_feature_importance(model: XGBClassifier, feature_cols: List[str], top_n: int = 20) -> None:
    pairs = get_feature_importance(model, feature_cols)
    print(f"\n=== Feature importance (top {min(top_n, len(pairs))}) ===")
    max_importance = max((p[1] for p in pairs), default=1.0) or 1.0
    for name, importance in pairs[:top_n]:
        bar_len = int(30 * importance / max_importance)
        bar = "#" * bar_len
        print(f"  {name:32s} {importance:.4f}  {bar}")


def elo_baseline_for_df(df: pd.DataFrame, draw_prob: float) -> List[dict]:
    return [
        elo_baseline_probabilities(row["home_elo"], row["away_elo"], draw_prob=draw_prob)
        for _, row in df.iterrows()
    ]


def naive_baseline_for_df(df: pd.DataFrame, naive_probs: dict) -> List[dict]:
    return [naive_probs] * len(df)


def run_training_pipeline(
    df: pd.DataFrame, train_frac: float = 0.7, val_frac: float = 0.15, model_out: Optional[str] = None,
    exclude_features: Optional[List[str]] = None, split_by_league: bool = False, **xgb_kwargs,
) -> XGBClassifier:
    print(f"Full dataset: {len(df)} matches\n")

    if split_by_league and "league" in df.columns:
        print("Using PER-LEAGUE chronological split (--split-by-league) — each league gets its own")
        print("70/15/15 split of its own history, so no league is excluded from training.\n")
        train_df, val_df, test_df = time_based_split_grouped(df, group_col="league", train_frac=train_frac, val_frac=val_frac)
    else:
        train_df, val_df, test_df = time_based_split(df, train_frac=train_frac, val_frac=val_frac)
    print("Split summary:")
    print_split_summary(train_df, val_df, test_df)
    if "league" in df.columns:
        print("League distribution per split:")
        print(f"  train: {dict(train_df['league'].value_counts())}")
        print(f"  val:   {dict(val_df['league'].value_counts())}")
        print(f"  test:  {dict(test_df['league'].value_counts())}")
        train_leagues = set(train_df["league"].unique())
        all_leagues = set(df["league"].unique())
        missing = all_leagues - train_leagues
        if missing:
            print(f"  WARNING: {missing} present in the data but MISSING from training entirely. "
                  f"The model has never seen these leagues — consider --split-by-league.")
    print()

    if len(train_df) < 20:
        print(
            "WARNING: fewer than 20 training rows. This is fine for testing the pipeline "
            "logic, but nowhere near enough data to train a meaningful real model — "
            "ingest more seasons before trusting any of the numbers below.\n"
        )

    X_train, feature_cols = prepare_features(train_df, exclude=exclude_features)
    category_dtypes = get_category_dtypes(X_train)  # derived from TRAIN only, reused everywhere below
    X_val, _ = prepare_features(val_df, exclude=exclude_features, category_dtypes=category_dtypes)
    X_test, _ = prepare_features(test_df, exclude=exclude_features, category_dtypes=category_dtypes)
    y_train, y_val, y_test = train_df["result"], val_df["result"], test_df["result"]

    print(f"Features used ({len(feature_cols)}): {feature_cols}\n")

    # --- Baselines, computed from TRAIN set statistics only (no leakage from val/test) ---
    draw_prob = estimate_draw_probability(train_df)
    naive_probs = naive_baseline_probabilities(train_df)
    print(f"Training set empirical draw rate: {draw_prob:.3f}")
    print(f"Training set empirical outcome distribution: {naive_probs}\n")

    # --- Train the real model, using validation set for early stopping ---
    model = train_xgboost(X_train, y_train, X_val=X_val, y_val=y_val, **xgb_kwargs)
    print(f"Early stopping halted at iteration {model.best_iteration} (of {model.n_estimators} max) — "
          f"if this is much lower than the max, the extra trees were overfitting, not helping.\n")

    print_feature_importance(model, feature_cols)

    # --- Overfitting check: compare train vs test performance ---
    train_metrics = evaluate_predictions(y_train.tolist(), predict_probabilities(model, X_train))
    print("\n=== Overfitting check (train vs test) ===")
    print_evaluation("XGBoost (train)", train_metrics)

    # --- Evaluate everything on the TEST set (the true held-out measure) ---
    print("\n=== Test set evaluation ===")
    print_evaluation("Naive baseline", evaluate_predictions(y_test.tolist(), naive_baseline_for_df(test_df, naive_probs)))
    print_evaluation("Elo baseline", evaluate_predictions(y_test.tolist(), elo_baseline_for_df(test_df, draw_prob)))
    print_evaluation("XGBoost model", evaluate_predictions(y_test.tolist(), predict_probabilities(model, X_test)))
    print(
        "  (If XGBoost's train log_loss is much lower than its test log_loss, "
        "and/or XGBoost doesn't beat the Elo baseline above, the model is "
        "overfitting or the features aren't adding real signal yet — both are "
        "worth investigating before trusting this model.)"
    )

    print("\n=== Validation set evaluation (for reference/tuning) ===")
    print_evaluation("XGBoost model", evaluate_predictions(y_val.tolist(), predict_probabilities(model, X_val)))

    if model_out:
        joblib.dump({"model": model, "feature_columns": feature_cols, "category_dtypes": category_dtypes}, model_out)
        print(f"\nModel saved to {model_out}")

    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the XGBoost match outcome model.")
    parser.add_argument("--input", default=None, help="Path to a feature CSV (from app.build_feature_table). If omitted, builds fresh from the database.")
    parser.add_argument("--model-out", default="model.joblib", help="Path to save the trained model")
    parser.add_argument("--train-frac", type=float, default=0.7)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--max-depth", type=int, default=4, help="Max tree depth — lower values (2-3) regularize harder against overfitting")
    parser.add_argument("--min-child-weight", type=float, default=1.0, help="Minimum sum of instance weight in a leaf — higher values prevent splits on small/anomalous subsets")
    parser.add_argument("--reg-lambda", type=float, default=1.0, help="L2 regularization strength — higher values penalize complex trees more")
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--n-estimators", type=int, default=200, help="Max trees (early stopping usually halts well before this)")
    parser.add_argument(
        "--exclude-features", default=None,
        help="Comma-separated feature columns to drop before training, e.g. "
             "'travel_distance_km,home_injury_count,home_injury_importance_sum,away_injury_count,away_injury_importance_sum' "
             "to test an Elo+form-only model without the noisier features."
    )
    parser.add_argument(
        "--split-by-league", action="store_true",
        help="Split train/val/test PER LEAGUE (each league gets its own chronological 70/15/15) "
             "instead of one global split. Use this if a league is clustered in a particular time "
             "window relative to others — a global split can otherwise exclude that league from "
             "training entirely. The training output will warn you if this is happening."
    )
    args = parser.parse_args()

    if args.input:
        df = pd.read_csv(args.input, parse_dates=["date"])
        df["date"] = df["date"].dt.date
    else:
        from app.build_feature_table import build_feature_table
        from app.database import SessionLocal

        db = SessionLocal()
        try:
            df = build_feature_table(db)
        finally:
            db.close()

    exclude_features = args.exclude_features.split(",") if args.exclude_features else None

    run_training_pipeline(
        df, train_frac=args.train_frac, val_frac=args.val_frac, model_out=args.model_out,
        exclude_features=exclude_features, split_by_league=args.split_by_league,
        max_depth=args.max_depth, min_child_weight=args.min_child_weight,
        reg_lambda=args.reg_lambda, learning_rate=args.learning_rate, n_estimators=args.n_estimators,
    )


if __name__ == "__main__":
    main()
