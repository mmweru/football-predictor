"""
Tests for app.train_model.

IMPORTANT CONTEXT: app.synthetic_data generates outcomes using the exact
same logistic formula as app.baselines.elo_baseline_probabilities. This
makes the Elo baseline close to mathematically optimal FOR THIS SYNTHETIC
DATA SPECIFICALLY — it's not a fair "can XGBoost beat a simple heuristic"
test, since the heuristic IS the answer by construction here. Real football
data won't have that property (outcomes aren't a clean logistic function of
Elo alone), so these tests check the achievable, honest bar instead:
XGBoost should clearly beat the NAIVE baseline (predicting the training
set's average outcome distribution for every match) — that's a property
any correctly-working model should have regardless of how the synthetic
data happens to be generated.
"""

import numpy as np
import pandas as pd
import pytest

from app.baselines import naive_baseline_probabilities, estimate_draw_probability
from app.data_split import time_based_split
from app.evaluation import evaluate_predictions
from app.synthetic_data import generate_synthetic_feature_table
from app.train_model import (
    get_feature_columns,
    naive_baseline_for_df,
    predict_probabilities,
    prepare_features,
    train_xgboost,
)


@pytest.fixture(scope="module")
def synthetic_df():
    return generate_synthetic_feature_table(n_matches=1500, seed=123)


@pytest.fixture(scope="module")
def trained_pipeline(synthetic_df):
    """Runs split + feature prep + training once, reused across several tests for speed."""
    train_df, val_df, test_df = time_based_split(synthetic_df, train_frac=0.7, val_frac=0.15)
    X_train, feature_cols = prepare_features(train_df)
    X_val, _ = prepare_features(val_df)
    X_test, _ = prepare_features(test_df)
    model = train_xgboost(X_train, train_df["result"], X_val=X_val, y_val=val_df["result"])
    return {
        "model": model, "feature_cols": feature_cols,
        "train_df": train_df, "val_df": val_df, "test_df": test_df,
        "X_train": X_train, "X_val": X_val, "X_test": X_test,
    }


# ---------------------------------------------------------------------------
# Feature selection
# ---------------------------------------------------------------------------
def test_get_feature_columns_excludes_identifiers_and_target():
    df = generate_synthetic_feature_table(n_matches=5)
    feature_cols = get_feature_columns(df)
    assert "result" not in feature_cols
    assert "match_id" not in feature_cols
    assert "date" not in feature_cols
    assert "home_team" not in feature_cols
    assert "home_score" not in feature_cols
    assert "home_elo" in feature_cols  # a real feature must still be included


def test_prepare_features_converts_booleans_to_int():
    df = generate_synthetic_feature_table(n_matches=20)
    X, _ = prepare_features(df)
    assert X["is_derby"].dtype in (np.int64, np.int32, int)


# ---------------------------------------------------------------------------
# Training produces a usable model
# ---------------------------------------------------------------------------
def test_predict_probabilities_sum_to_one(trained_pipeline):
    probs = predict_probabilities(trained_pipeline["model"], trained_pipeline["X_test"])
    for p in probs:
        assert sum(p.values()) == pytest.approx(1.0, abs=1e-4)


def test_predict_probabilities_all_non_negative(trained_pipeline):
    probs = predict_probabilities(trained_pipeline["model"], trained_pipeline["X_test"])
    for p in probs:
        assert all(v >= 0 for v in p.values())


def test_early_stopping_actually_stops_before_max_iterations(trained_pipeline):
    """With a validation set provided, best_iteration should be found and typically well under the max."""
    model = trained_pipeline["model"]
    assert model.best_iteration is not None
    assert model.best_iteration < model.n_estimators


# ---------------------------------------------------------------------------
# The achievable, honest bar: beats the naive baseline
# ---------------------------------------------------------------------------
def test_model_beats_naive_baseline_on_test_set(trained_pipeline):
    test_df = trained_pipeline["test_df"]
    naive_probs = naive_baseline_probabilities(trained_pipeline["train_df"])

    model_metrics = evaluate_predictions(
        test_df["result"].tolist(), predict_probabilities(trained_pipeline["model"], trained_pipeline["X_test"])
    )
    naive_metrics = evaluate_predictions(
        test_df["result"].tolist(), naive_baseline_for_df(test_df, naive_probs)
    )

    assert model_metrics["log_loss"] < naive_metrics["log_loss"]
    assert model_metrics["accuracy"] > naive_metrics["accuracy"]


def test_model_train_performance_is_at_least_as_good_as_test():
    """Sanity check on the direction of overfitting — train should never be
    worse than test (a model that fits training data poorly is broken, not
    just 'not overfitting'). This does NOT assert the gap is small — some
    train/test gap is normal and expected."""
    df = generate_synthetic_feature_table(n_matches=1500, seed=123)
    train_df, val_df, test_df = time_based_split(df, train_frac=0.7, val_frac=0.15)
    X_train, _ = prepare_features(train_df)
    X_val, _ = prepare_features(val_df)
    X_test, _ = prepare_features(test_df)
    model = train_xgboost(X_train, train_df["result"], X_val=X_val, y_val=val_df["result"])

    train_metrics = evaluate_predictions(train_df["result"].tolist(), predict_probabilities(model, X_train))
    test_metrics = evaluate_predictions(test_df["result"].tolist(), predict_probabilities(model, X_test))

    assert train_metrics["log_loss"] <= test_metrics["log_loss"]


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
def test_same_seed_produces_deterministic_predictions():
    df = generate_synthetic_feature_table(n_matches=300, seed=7)
    train_df, val_df, test_df = time_based_split(df, train_frac=0.7, val_frac=0.15)
    X_train, _ = prepare_features(train_df)
    X_val, _ = prepare_features(val_df)
    X_test, _ = prepare_features(test_df)

    model_a = train_xgboost(X_train, train_df["result"], X_val=X_val, y_val=val_df["result"], random_state=42)
    model_b = train_xgboost(X_train, train_df["result"], X_val=X_val, y_val=val_df["result"], random_state=42)

    preds_a = predict_probabilities(model_a, X_test)
    preds_b = predict_probabilities(model_b, X_test)

    for pa, pb in zip(preds_a, preds_b):
        for label in pa:
            assert pa[label] == pytest.approx(pb[label])


# ---------------------------------------------------------------------------
# Model save/load round-trip
# ---------------------------------------------------------------------------
def test_model_save_and_load_roundtrip(trained_pipeline, tmp_path):
    import joblib

    model_path = tmp_path / "test_model.joblib"
    joblib.dump({"model": trained_pipeline["model"], "feature_columns": trained_pipeline["feature_cols"]}, model_path)

    loaded = joblib.load(model_path)
    loaded_model = loaded["model"]
    assert loaded["feature_columns"] == trained_pipeline["feature_cols"]

    original_preds = predict_probabilities(trained_pipeline["model"], trained_pipeline["X_test"])
    loaded_preds = predict_probabilities(loaded_model, trained_pipeline["X_test"])
    for orig, loaded_p in zip(original_preds, loaded_preds):
        for label in orig:
            assert orig[label] == pytest.approx(loaded_p[label])
