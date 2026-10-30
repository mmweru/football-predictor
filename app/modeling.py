"""
The prediction model: a small, deliberately simple ENSEMBLE of three different
views of a match, averaged in probability space, then (when bookmaker odds
exist) blended with the market.

    1. Regularised logistic regression on elo + form + team stats     (stable, hard to overfit)
    2. XGBoost (depth 2, strong regularisation) on ALL feature groups (captures interactions)
    3. Independent-Poisson model from each side's expected goals       (a different model family entirely)

Why averaging: on real EPL data each member alone scored ~0.965-0.970 log loss
while their average scored 0.959 — models that make *different* mistakes cancel
each other's noise. Why NOT a calibrator on top: tested, it overfits on a
~570-row validation set and made things worse, so none is used.

Probability order is ALWAYS [A, D, H] (away, draw, home), matching
app.baselines.OUTCOME_LABELS.
"""

from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import poisson
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from app.feature_engine import DEFAULT_GROUPS, elo_expected, feature_columns

LABELS = ["A", "D", "H"]
LABEL_TO_IDX = {l: i for i, l in enumerate(LABELS)}
LR_GROUPS = ["elo", "form", "team_stats"]
MARKET_COLS = ["mkt_p_away", "mkt_p_draw", "mkt_p_home"]

DEFAULT_XGB_PARAMS = dict(
    objective="multi:softprob", num_class=3, eval_metric="mlogloss", n_estimators=500, max_depth=2,
    learning_rate=0.03, subsample=0.8, colsample_bytree=0.7, min_child_weight=8, reg_lambda=5.0, random_state=1,
)


# ----------------------------------------------------------------------------
# Poisson scoreline maths (also powers the UI's expected goals / over-under / BTTS)
# ----------------------------------------------------------------------------
def scoreline_matrix(exp_home: float, exp_away: float, max_goals: int = 8) -> np.ndarray:
    """M[i, j] = P(home scores i, away scores j) under independent Poisson goals."""
    g = np.arange(0, max_goals + 1)
    m = np.outer(poisson.pmf(g, np.clip(exp_home, 0.15, 5.0)), poisson.pmf(g, np.clip(exp_away, 0.15, 5.0)))
    return m / m.sum()


def poisson_1x2(exp_home: float, exp_away: float) -> np.ndarray:
    """[P(away), P(draw), P(home)]"""
    m = scoreline_matrix(exp_home, exp_away)
    return np.array([np.triu(m, 1).sum(), np.trace(m), np.tril(m, -1).sum()])


def scoreline_summary(exp_home: float, exp_away: float, top: int = 3) -> dict:
    m = scoreline_matrix(exp_home, exp_away)
    idx = np.dstack(np.unravel_index(np.argsort(-m.ravel()), m.shape))[0][:top]
    total = np.add.outer(np.arange(m.shape[0]), np.arange(m.shape[1]))
    return {
        "exp_home_goals": float(exp_home), "exp_away_goals": float(exp_away),
        "over_2_5": float(m[total >= 3].sum()), "btts": float(m[1:, 1:].sum()),
        "most_likely_scores": [{"score": f"{i}-{j}", "prob": float(m[i, j])} for i, j in idx],
    }


def poisson_probs_for_frame(df: pd.DataFrame) -> np.ndarray:
    return np.array([poisson_1x2(h, a) for h, a in zip(df["exp_home_goals"].fillna(1.4), df["exp_away_goals"].fillna(1.1))])


def encode(result: pd.Series) -> np.ndarray:
    return result.map(LABEL_TO_IDX).to_numpy()


def evaluate(y_idx: np.ndarray, proba: np.ndarray) -> Dict[str, float]:
    onehot = np.eye(3)[y_idx]
    return {
        "log_loss": float(log_loss(y_idx, proba, labels=[0, 1, 2])),
        "brier": float(np.mean(np.sum((proba - onehot) ** 2, axis=1))),
        "accuracy": float(accuracy_score(y_idx, proba.argmax(1))),
        "n": int(len(y_idx)),
    }


def elo_baseline(df: pd.DataFrame, draw_rate: float) -> np.ndarray:
    ex = np.array([elo_expected(h, a) for h, a in zip(df["home_elo"], df["away_elo"])])
    return np.c_[(1 - draw_rate) * (1 - ex), np.full(len(df), draw_rate), (1 - draw_rate) * ex]


# ----------------------------------------------------------------------------
# The ensemble
# ----------------------------------------------------------------------------
@dataclass
class EnsembleModel:
    xgb_groups: List[str] = field(default_factory=lambda: list(DEFAULT_GROUPS))
    lr_groups: List[str] = field(default_factory=lambda: list(LR_GROUPS))
    market_weight: float = 0.8
    xgb_params: Dict = field(default_factory=dict)
    lr: Optional[object] = None
    xgb: Optional[XGBClassifier] = None
    best_iteration: Optional[int] = None
    draw_rate: float = 0.25

    # -- training -----------------------------------------------------------
    def fit(
        self, train: pd.DataFrame, val: Optional[pd.DataFrame] = None,
        n_estimators: Optional[int] = None, sample_weight: Optional[np.ndarray] = None,
    ) -> "EnsembleModel":
        """
        With `val`: XGBoost early-stops on it (finds how many trees the data supports).
        With `n_estimators` and no `val`: refit on everything with a fixed tree count (final production fit).
        """
        y_tr = encode(train["result"])
        self.draw_rate = float((train["result"] == "D").mean())

        lr_cols = feature_columns(self.lr_groups)
        self.lr = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(C=0.3, max_iter=3000))
        self.lr.fit(train[lr_cols], y_tr, logisticregression__sample_weight=sample_weight)

        xgb_cols = feature_columns(self.xgb_groups)
        params = {**DEFAULT_XGB_PARAMS, **self.xgb_params}
        if val is not None:
            params["early_stopping_rounds"] = 30
            self.xgb = XGBClassifier(**params)
            self.xgb.fit(train[xgb_cols], y_tr, sample_weight=sample_weight,
                         eval_set=[(val[xgb_cols], encode(val["result"]))], verbose=False)
            self.best_iteration = int(self.xgb.best_iteration) + 1
        else:
            params["n_estimators"] = n_estimators or self.best_iteration or 150
            self.xgb = XGBClassifier(**params)
            self.xgb.fit(train[xgb_cols], y_tr, sample_weight=sample_weight, verbose=False)
        return self

    # -- inference ----------------------------------------------------------
    def components(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        return {
            "logreg": self.lr.predict_proba(df[feature_columns(self.lr_groups)]),
            "xgboost": self.xgb.predict_proba(df[feature_columns(self.xgb_groups)]),
            "poisson": poisson_probs_for_frame(df),
        }

    def predict_fundamental(self, df: pd.DataFrame) -> np.ndarray:
        comps = self.components(df)
        p = np.mean(list(comps.values()), axis=0)
        return p / p.sum(axis=1, keepdims=True)

    def predict(self, df: pd.DataFrame, use_market: bool = True) -> np.ndarray:
        """Fundamental ensemble, blended with the de-margined bookmaker probabilities on rows that have odds."""
        p = self.predict_fundamental(df)
        if not use_market or self.market_weight <= 0 or not set(MARKET_COLS) <= set(df.columns):
            return p
        mk = df[MARKET_COLS].to_numpy(dtype=float)
        has = ~np.isnan(mk).any(axis=1)
        out = p.copy()
        out[has] = self.market_weight * mk[has] + (1 - self.market_weight) * p[has]
        return out / out.sum(axis=1, keepdims=True)


# ----------------------------------------------------------------------------
# Time-ordered evaluation helpers
# ----------------------------------------------------------------------------
def chronological_split(P: pd.DataFrame, train_frac: float = 0.7, val_frac: float = 0.15):
    P = P.sort_values(["date", "match_id"]).reset_index(drop=True)
    a, b = int(len(P) * train_frac), int(len(P) * (train_frac + val_frac))
    return P.iloc[:a], P.iloc[a:b], P.iloc[b:]


def score_table(model: EnsembleModel, train: pd.DataFrame, test: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    y = encode(test["result"])
    counts = np.bincount(encode(train["result"]), minlength=3) / len(train)
    out = {
        "Naive (train outcome rates)": evaluate(y, np.tile(counts, (len(test), 1))),
        "Elo baseline": evaluate(y, elo_baseline(test, model.draw_rate)),
    }
    comps = model.components(test)
    out["  Logistic regression"] = evaluate(y, comps["logreg"])
    out["  XGBoost"] = evaluate(y, comps["xgboost"])
    out["  Poisson (exp. goals)"] = evaluate(y, comps["poisson"])
    out["ENSEMBLE (fundamental)"] = evaluate(y, model.predict_fundamental(test))
    if test[MARKET_COLS].notna().all(axis=1).any():
        sub = test[test[MARKET_COLS].notna().all(axis=1)]
        ys = encode(sub["result"])
        out["Bookmaker odds (raw)"] = evaluate(ys, sub[MARKET_COLS].to_numpy(dtype=float))
        out[f"FINAL = {model.market_weight:.0%} market + model"] = evaluate(ys, model.predict(sub))
    return out


def walk_forward(P: pd.DataFrame, test_seasons: Sequence[int], market_weight: float = 0.8) -> pd.DataFrame:
    """
    Train on every match BEFORE a season, test on that whole season, repeat. The most trustworthy number in this
    project: each fold is genuinely "predict a season the model has never seen, knowing only the past".
    """
    rows = []
    for s in test_seasons:
        train_all = P[P["season"] < s]
        test = P[P["season"] == s]
        if len(train_all) < 300 or len(test) < 50:
            continue
        cut = int(len(train_all) * 0.85)
        tr, va = train_all.iloc[:cut], train_all.iloc[cut:]
        m = EnsembleModel(market_weight=market_weight).fit(tr, va)
        y = encode(test["result"])
        row = {"season": f"{s}/{(s + 1) % 100:02d}", "n": len(test),
               "elo": evaluate(y, elo_baseline(test, m.draw_rate))["log_loss"],
               "ensemble": evaluate(y, m.predict_fundamental(test))["log_loss"],
               "ensemble_acc": evaluate(y, m.predict_fundamental(test))["accuracy"]}
        if test[MARKET_COLS].notna().all(axis=1).all():
            row["market"] = evaluate(y, test[MARKET_COLS].to_numpy(dtype=float))["log_loss"]
            row["final"] = evaluate(y, m.predict(test))["log_loss"]
        rows.append(row)
    return pd.DataFrame(rows, columns=None if rows else ["season", "n", "elo", "ensemble", "ensemble_acc"])


def decay_weights(dates: Sequence[dt.date], half_life_days: float) -> np.ndarray:
    """Newer matches count more: weight = 0.5 ** (age_in_days / half_life). Off by default (not validated on your data)."""
    newest = max(dates)
    return np.array([0.5 ** ((newest - d).days / half_life_days) for d in dates])
