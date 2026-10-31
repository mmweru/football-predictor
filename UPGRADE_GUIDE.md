# Upgrade guide — EPL + EFL predictor

Unzip this OVER your project folder (same paths; new files are added, changed files replaced). Then:

```bash
pip install -r requirements.txt
# 1) .env: add FOOTBALL_DATA_ORG_TOKEN= and BIGBALLS_API_KEY=  (both optional, both free)
# Also set FOOTBALL_API_KEY= in the project-root .env to enable API-Football imports.
python -m app.data_sources.check        # FIRST: confirms each source is reachable from your machine
python -m app.migrate                   # adds new columns to your existing database (safe to re-run)
python -m app.sync_data --seasons 8 --all    # 8 seasons x 4 leagues, fixtures, injuries, weather, who-plays-where
python -m app.sync_data --fpl --bigballs --api-football --api-football-seasons 2024
python -m app.geocode_teams             # stadium coordinates (travel distance, weather, local derbies) - slow, once
python -m app.sync_data --weather       # after geocoding
python -m app.train_ensemble --walk-forward 4 --ablate    # train + honest evaluation
python -m app.predict_fixtures          # predict every upcoming fixture
uvicorn app.api:app --reload
# weekly (cron / Task Scheduler):  python -m app.refresh
```
Your own CSVs still work: `python -m app.sync_data --local-csv "E0*.csv" --local-div E0`.

## What each file does
| File | Status | Why |
|---|---|---|
| `app/leagues.py` | NEW | Single source of truth for the 4 allowed leagues + season logic |
| `app/models.py` | CHANGED | `Team.is_active`, match stats/odds/weather columns, player stats/status, `Prediction` table |
| `app/migrate.py` | NEW | Adds those columns to your existing Postgres/SQLite without data loss |
| `app/team_aliases.py` | CHANGED | Matches names across 4 sources (football-data.co.uk/.org, FPL, BigBalls); no duplicate clubs |
| `app/sync_core.py` | NEW | Upserts results (so "upcoming" rows become real results), reschedules, **assigns league by CURRENT season** |
| `app/sync_data.py` | NEW | One command to update everything; each source failure is isolated |
| `app/data_sources/*` | NEW | football-data.co.uk (all 4 leagues), football-data.org (PL+ELC), FPL (EPL injuries + player stats), Open-Meteo (weather), BigBalls |
| `app/data_sources/api_football.py` | NEW | API-Football fixtures and season-scoped player totals for the four EPL/EFL leagues |
| `app/feature_engine.py` | NEW | All 9 metric groups in ONE chronological pass; same code for training and live (0.8 s for 3,810 matches) |
| `app/modeling.py`, `app/train_ensemble.py` | NEW | Ensemble (logistic + XGBoost + Poisson), market blend, walk-forward, ablation |
| `app/injuries.py` | NEW | Player importance, availability, Elo penalty, manual injuries |
| `app/predict_live.py`, `predict_fixtures.py`, `grade_predictions.py`, `refresh.py` | NEW | Predict one match / all fixtures, grade results, weekly loop |
| `app/api.py` | CHANGED | Only 4 leagues, only active clubs, new endpoints (`/api/fixtures`, `/api/accuracy`, `/api/injuries`, `/api/model`) |
| `frontend/index.html` | CHANGED | League badges only (the UI already renders the new `key_factors`) |
| `app/derbies.py`, `app/geocode_teams.py` | CHANGED | 50 derby pairs incl. EFL; geocoder skips inactive clubs |
| `app/train_model.py`, `build_feature_table.py`, `features.py`, `predict.py`, `elo.py` | UNCHANGED | Legacy path kept working (and its tests) — new code does not depend on it |

## Your 9 metrics -> where they live
1. Injuries: `injuries.py` (FPL feed for EPL, `POST /api/injuries` for EFL) as an Elo penalty + red-card suspension proxy for all leagues
2. Home advantage: `home_adv` group (league rate + team-specific home/away gap)
3. Fatigue: rest days, games in last 7/14/30 days (`fatigue` group; supports extra cup games)
4. Game factors: derby (curated + <40 km), travel km, weather, midweek (`context`)
5. Head-to-head: recency-weighted, last 6 meetings (`h2h`)
6. Subsequent games: `predict_fixtures` -> `sync_data` (results overwrite fixtures) -> `grade_predictions`
7. Player stats: FPL xG/xA/minutes -> importance weights + key players shown in the result
8. Team stats: shots, shots on target, corners, cards, xG-proxy (`team_stats`)
9. Match situation: attack/defence strength -> expected goals, scorelines, over 2.5, BTTS (`situation`)

## Honest limits (read these)
* **Sandbox had no internet.** Parsers were tested on your real E0 CSVs and on fake HTTP responses, never on the live APIs. Run `app.data_sources.check`; BigBalls field names in particular are guessed defensively.
* **football-data.org free = PL + Championship only. BigBalls = EPL only.** League One / Two come only from football-data.co.uk, so they have no injuries (use `/api/injuries`) and no full-season schedule beyond fixtures.csv.
* **API-Football quotas and season access depend on your plan.** The configured free plan allows at most three player pages, 10 requests/minute, and seasons through 2024. Page results beyond the first three are omitted; 2025 requires a plan that supports that season. Historical player totals are stored separately and do not replace current FPL availability.
* **`sync_data --all` defaults to 8 history seasons.** Use `--seasons 1` for a routine current-season refresh. Production training refuses to replace `model_bundle.joblib` until there are at least 1,500 played matches across 3 seasons; `--allow-small-data` is for deliberate test models only.
* **Injury Elo penalty (`ELO_PER_IMPORTANCE=18`) is an assumption**, not fitted. Each sync stores an injury history; after ~1 season, train with `--use-injuries` and tune it.
* **Fatigue only sees league games** unless you pass cup/European dates (`extra_games`).
* **Bookmaker odds beat every model built from free data.** The final prediction is 80% odds + 20% model when odds exist (set `--market-weight`). Odds exist for fixtures from `fixtures.csv`; for hypothetical matchups the model runs alone.
* Realistic ceiling: ~54-56% outcome accuracy, even for bookmakers. Do not expect 70%.
