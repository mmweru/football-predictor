"""
Tests for the "league" categorical feature — added because mixing EPL and
KPL data means the model needs a way to distinguish leagues with different
scoring/competitiveness baselines. Covers: correct categorical dtype,
consistent category encoding across train/val/test, and that training
actually succeeds with a mixed-league dataset.
"""

import numpy as np
import pandas as pd
import pytest

from app.data_split import time_based_split
from app.synthetic_data import generate_synthetic_feature_table
from app.train_model import CATEGORICAL_FEATURE_COLUMNS, get_category_dtypes, prepare_features, train_xgboost


@pytest.fixture()
def mixed_league_df():
    df = generate_synthetic_feature_table(n_matches=400, seed=11)
    rng = np.random.default_rng(11)
    df["league"] = rng.choice(["EPL", "KPL"], size=len(df), p=[0.7, 0.3])
    return df


def test_league_is_included_as_a_feature_column(mixed_league_df):
    X, feature_cols = prepare_features(mixed_league_df)
    assert "league" in feature_cols


def test_league_column_gets_category_dtype(mixed_league_df):
    X, _ = prepare_features(mixed_league_df)
    assert X["league"].dtype.name == "category"


def test_get_category_dtypes_captures_categories_from_train_only():
    train_df = pd.DataFrame({"league": ["EPL", "EPL", "KPL"]})
    dtypes = get_category_dtypes(train_df)
    assert "league" in dtypes
    assert set(dtypes["league"].categories) == {"EPL", "KPL"}


def test_prepare_features_reuses_provided_category_dtypes(mixed_league_df):
    train_df, val_df, test_df = time_based_split(mixed_league_df, train_frac=0.7, val_frac=0.15)
    X_train, _ = prepare_features(train_df)
    category_dtypes = get_category_dtypes(X_train)

    X_val, _ = prepare_features(val_df, category_dtypes=category_dtypes)

    # Both must share the EXACT same category set/order for consistent XGBoost encoding.
    assert list(X_train["league"].cat.categories) == list(X_val["league"].cat.categories)


def test_training_succeeds_with_mixed_league_data(mixed_league_df):
    train_df, val_df, test_df = time_based_split(mixed_league_df, train_frac=0.7, val_frac=0.15)
    X_train, _ = prepare_features(train_df)
    category_dtypes = get_category_dtypes(X_train)
    X_val, _ = prepare_features(val_df, category_dtypes=category_dtypes)

    model = train_xgboost(X_train, train_df["result"], X_val=X_val, y_val=val_df["result"])
    # enable_categorical should have been auto-detected and set:
    assert model.get_params().get("enable_categorical") is True


def test_single_row_prediction_uses_saved_category_dtype():
    """
    Simulates the live-prediction path: a single-row DataFrame must be cast
    using the SAME category dtype captured from training, not inferred
    fresh (which for a single row would only ever have one category).
    """
    train_df = pd.DataFrame({"league": ["EPL", "EPL", "KPL", "KPL"]})
    category_dtypes = get_category_dtypes(train_df)

    single_row = pd.DataFrame({"league": ["KPL"]})
    single_row["league"] = single_row["league"].astype(category_dtypes["league"])

    # Even though this single row only contains "KPL", the dtype must still
    # know about "EPL" as a valid category (from training) — this is what
    # makes predictions for either league consistent with how the model was trained.
    assert set(single_row["league"].cat.categories) == {"EPL", "KPL"}
