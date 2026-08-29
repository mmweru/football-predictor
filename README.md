# Football Predictor — Database Layer

SQLAlchemy + PostgreSQL + Pydantic setup for the five core tables: `teams`,
`matches`, `players`, `injuries`, `elo_history`.

Every piece of code in this project was run and tested before being handed
to you — the 13 automated tests all pass, and `init_db.py` / `seed_sample_data.py`
were both verified to run end-to-end.

---

## Project structure

```
football_predictor/
├── README.md
├── requirements.txt
├── .env.example          # copy to .env and fill in your real values
├── .gitignore
├── conftest.py            # marks project root for pytest (needed for `from app...` imports)
├── sample_data/
│   └── sample_results.csv # tiny example CSV for testing ingestion, matches football-data.co.uk format
├── static/
│   └── detect_location.html # standalone browser page for live GPS venue detection, no server needed
├── frontend/
│   └── index.html                    # football-themed web UI, served by app/api.py
├── app/
│   ├── __init__.py
│   ├── config.py                # reads settings from .env
│   ├── database.py              # SQLAlchemy engine/session/Base
│   ├── models.py                # the 5 ORM tables + relationships
│   ├── schemas.py                # Pydantic validation schemas
│   ├── init_db.py                # creates tables
│   ├── seed_sample_data.py       # inserts sample rows for manual eyeballing
│   ├── team_aliases.py           # canonical team name mapping (prevents duplicate teams)
│   ├── ingest_csv.py             # loads football-data.co.uk CSVs into the database
│   ├── ingest_local_csv.py       # loads local/Kenyan-format CSVs (different columns, yyyy-mm-dd dates)
│   ├── elo.py                     # computes Elo ratings from matches, chronologically
│   ├── distance.py                # haversine great-circle distance
│   ├── geocode_teams.py           # fills in stadium coordinates via OpenStreetMap Nominatim
│   ├── derbies.py                  # manually curated rivalry pairs
│   ├── features.py                 # per-feature functions (rest days, form, h2h, injuries, elo lookup)
│   ├── build_feature_table.py      # assembles all features into one row-per-match table
│   ├── venue.py                     # address geocoding + IP-based coarse location lookup
│   ├── set_match_venue.py           # CLI to override a match's venue (address / lat-lon / IP auto-detect)
│   ├── baselines.py                  # naive + Elo-derived baseline predictors
│   ├── evaluation.py                  # log loss, Brier score, accuracy
│   ├── data_split.py                   # time-based train/val/test split (never random)
│   ├── synthetic_data.py                # generates realistic-shaped fake data for testing the pipeline
│   ├── train_model.py                    # trains + evaluates the XGBoost model, saves it
│   ├── predict.py                         # builds a feature row for a hypothetical upcoming match
│   └── api.py                              # FastAPI backend: /api/teams, /api/predict, serves the frontend
└── tests/
    ├── test_db.py           # 13 tests, one set per table + a full workflow test
    ├── test_elo.py           # 12 tests: Elo math + full chronological build
    ├── test_ingest.py         # 9 tests: date parsing, alias normalization, full ingestion flow
    ├── test_distance.py        # 4 tests: haversine against known real-world distances
    ├── test_geocode.py          # 4 tests: geocoding logic against a mocked service
    ├── test_features.py          # 22 tests: every feature function, with explicit leakage checks
    ├── test_venue.py              # 14 tests: address/IP geocoding + venue override CLI logic
    ├── test_venue_timeout.py       # 3 tests: geocoder timeout config + placeholder User-Agent rejection
    ├── test_baselines.py            # 9 tests: naive + Elo baseline probability math
    ├── test_evaluation.py            # 6 tests: log loss / Brier score against hand-computed values
    ├── test_data_split.py             # 10 tests: chronological split + per-league grouped split fix
    └── test_train_model.py             # 9 tests: full training pipeline against synthetic data
    ├── test_predict.py                  # 13 tests: live-prediction feature builder (current Elo, current injuries)
    └── test_api.py                       # 21 tests: full API stack via TestClient, incl. league filtering, team creation, manual overrides
    ├── test_ingest_local.py               # 5 tests: local/Kenyan CSV format parsing and ingestion
    └── test_categorical_features.py        # 6 tests: league-as-model-feature encoding consistency
```

**Design choice worth knowing:** the original schema sketch had `matches.home_team`
as a plain text column. This implementation instead links `matches`, `players`,
`elo_history`, and `injuries` to `teams.id` via foreign keys. This is standard
relational design — it stops "Man Utd" and "Manchester United" from silently
becoming two different teams, and makes joins fast and reliable instead of
matching on strings.

---

## Part 1 — Install PostgreSQL & pgAdmin

If you don't already have them:

- **PostgreSQL**: https://www.postgresql.org/download/
- **pgAdmin** (usually bundled with the Postgres installer, or standalone): https://www.pgadmin.org/download/

During Postgres install, you'll set a password for the default `postgres`
superuser — remember it, you'll need it once in the next step.

---

## Part 2 — Create the database and user in pgAdmin

1. Open pgAdmin. In the left sidebar, right-click **Servers → Register → Server**.
   - **General tab** → Name: `Local Postgres` (anything you like)
   - **Connection tab** → Host: `localhost`, Port: `5432`, Username: `postgres`,
     Password: (the one you set during install) → Save
2. Once connected, expand `Local Postgres → Databases`.
3. Right-click **Databases → Create → Database**.
   - Database name: `football_predictor`
   - Owner: `postgres` (fine for local dev)
   - Save
4. Create a dedicated app user instead of using the `postgres` superuser for
   your app (good practice, not strictly required for local dev):
   - Right-click **Login/Group Roles → Create → Login/Group Role**
   - General tab → Name: `football_user`
   - Definition tab → Password: choose one (this is what goes in your `.env`)
   - Privileges tab → toggle **Can login?** to Yes
   - Save
5. Grant that user access to the database:
   - Right-click the `football_predictor` database → **Properties → Security tab**
   - Add `football_user` with `ALL` privileges → Save
   - Alternatively, run this in the **Query Tool** (right-click the database → Query Tool):
     ```sql
     GRANT ALL PRIVILEGES ON DATABASE football_predictor TO football_user;
     ```

You now have an empty `football_predictor` database with a user that can
access it — but no tables yet. Those get created from code in Part 4.

---

## Part 3 — VS Code setup

1. Install VS Code: https://code.visualstudio.com/
2. Install the **Python** extension (Microsoft) from the Extensions panel.
3. Optional but useful: install the **PostgreSQL** extension (by Chris Kolkman
   or similar) if you want to browse tables inside VS Code too, not just pgAdmin.
4. Open the `football_predictor/` folder in VS Code (`File → Open Folder`).
5. Open a terminal inside VS Code (`` Ctrl+` ``) and create a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate      # on Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```
6. In VS Code, press `Ctrl+Shift+P` → **Python: Select Interpreter** → choose
   the one inside `.venv` so VS Code uses the same environment as your terminal.
7. Copy the environment file and fill in the password you set in Part 2:
   ```bash
   cp .env.example .env
   ```
   Edit `.env`:
   ```
   POSTGRES_USER=football_user
   POSTGRES_PASSWORD=<the password you set in pgAdmin>
   POSTGRES_HOST=localhost
   POSTGRES_PORT=5432
   POSTGRES_DB=football_predictor
   ```

---

## Part 4 — Create the tables

With your `.env` filled in and the virtual environment active:

```bash
python -m app.init_db
```

Expected output:
```
Creating tables (if not present) at: postgresql+psycopg2://football_user:***@localhost:5432/football_predictor
Tables created:
  - teams
  - matches
  - players
  - injuries
  - elo_history
```

**Verify in pgAdmin:** expand `football_predictor → Schemas → public → Tables`
in the sidebar (right-click **Tables → Refresh** if they don't show up
immediately) — you should see all five tables listed.

---

## Part 5 — Verify each section works

There are two complementary ways to verify things, use both:

### A. Automated tests (fast, run every time you change something)

```bash
pytest -v
```

This runs 13 tests against a throwaway **in-memory SQLite** database (not
your real Postgres — that's intentional, so tests are instant and never
touch real data). Expected output ends with:

```
tests/test_db.py::test_create_team PASSED
tests/test_db.py::test_team_name_must_be_unique PASSED
tests/test_db.py::test_team_pydantic_rejects_bad_latitude PASSED
tests/test_db.py::test_create_match_with_team_relationship PASSED
tests/test_db.py::test_duplicate_fixture_same_date_rejected PASSED
tests/test_db.py::test_create_player_linked_to_team PASSED
tests/test_db.py::test_importance_weight_bounds_enforced_by_pydantic PASSED
tests/test_db.py::test_create_injury_linked_to_player_and_match PASSED
tests/test_db.py::test_injury_without_match_missed_is_allowed PASSED
tests/test_db.py::test_create_elo_history_entry PASSED
tests/test_db.py::test_elo_rating_must_be_positive_pydantic PASSED
tests/test_db.py::test_one_elo_entry_per_team_per_date PASSED
tests/test_db.py::test_full_workflow_across_all_tables PASSED

13 passed
```

What each group actually proves:
- **Teams**: insert works, `name` uniqueness constraint is enforced, Pydantic rejects an invalid latitude (e.g. 999).
- **Matches**: foreign keys to `teams` resolve correctly in both directions (`match.home_team.name` and `team.home_matches`), and the same fixture can't be inserted twice on the same date.
- **Players**: linked correctly to a team, and Pydantic rejects an `importance_weight` outside 0–1.
- **Injuries**: link to both a player and (optionally) a match; confirms `match_missed_id` can be null (long-term injury not tied to one specific missed game).
- **Elo history**: one rating per team per date is enforced, and negative ratings are rejected by Pydantic.
- **Full workflow test**: builds a realistic mini-scenario touching all five tables at once and confirms every join works together.

### B. Manual verification against real Postgres (seeing actual rows)

```bash
python -m app.seed_sample_data
```

This inserts a small real example (2 teams, 1 match, 1 player, 1 injury, 2
elo entries) into your actual Postgres database.

**Then check in pgAdmin:**
1. Right-click the `football_predictor` database → **Query Tool**
2. Run:
   ```sql
   SELECT * FROM teams;
   SELECT * FROM matches;
   SELECT * FROM players;
   SELECT * FROM injuries;
   SELECT * FROM elo_history;
   ```
3. Confirm the row counts and values match what the script printed to your terminal.
4. Try a join to confirm relationships work at the SQL level too:
   ```sql
   SELECT m.date, ht.name AS home_team, at.name AS away_team, m.home_score, m.away_score
   FROM matches m
   JOIN teams ht ON m.home_team_id = ht.id
   JOIN teams at ON m.away_team_id = at.id;
   ```

If that join returns a readable row (e.g. `2025-10-04 | Arsenal | Chelsea | 2 | 2`),
your schema and foreign keys are working correctly end to end.

### Resetting the database

If you want to wipe and re-seed:
```sql
TRUNCATE teams, matches, players, injuries, elo_history RESTART IDENTITY CASCADE;
```
Run this in the pgAdmin Query Tool, then re-run `python -m app.seed_sample_data`.

---

## What's next

This gives you a full pipeline through a trained, evaluated XGBoost model,
plus now a working frontend (see "Running the Frontend" below) that lets
you pick two teams and get a live prediction with plain-language reasoning.

Next steps from here:
1. Get more real historical data ingested — model quality is still capped
   by data volume.
2. SHAP-based explanations (a more rigorous alternative to the current
   heuristic "key factors") — see Phase 6 of the original roadmap.

---

## Ingesting Data

`app/ingest_csv.py` loads historical results from a football-data.co.uk-format
CSV (`Date, HomeTeam, AwayTeam, FTHG, FTAG` columns required) and inserts
them through Pydantic validation.

**Get real data:** https://football-data.co.uk/data.php — pick a league,
download the CSV for one or more seasons.

**Team name handling:** different sources spell team names differently
("Man Utd" vs "Manchester United"). `app/team_aliases.py` holds a manually
curated mapping — every raw name gets normalized through it before a `Team`
row is looked up or created, so aliases collapse into one team instead of
silently creating duplicates. Extend `TEAM_NAME_ALIASES` as you encounter
new variants; the ingestion script prints `[new team]` for every team it
creates so you can spot an unmapped alias immediately.

**Run it:**
```bash
python -m app.ingest_csv path/to/E0.csv --competition "Premier League"
```

Use `--dry-run` first on a new file to validate without writing anything:
```bash
python -m app.ingest_csv path/to/E0.csv --competition "Premier League" --dry-run
```

Safe to re-run on the same file — duplicate fixtures (same date + same two
teams) are detected and skipped, not double-inserted.

---

## Building Elo Ratings

`app/elo.py` reads every scored match from `matches` in chronological
order and writes the **pre-match** rating for each team into
`elo_history` — i.e. the rating as it stood before kickoff, since that's
what's actually usable as a prediction feature.

**Run it:**
```bash
python -m app.elo
```

**Options:**
```bash
python -m app.elo --k 32                    # more reactive to recent results
python -m app.elo --home-advantage 100      # stronger home-field bonus
python -m app.elo --reset                   # wipe elo_history and rebuild from scratch
```

Safe to re-run without `--reset` — it upserts existing rows in place. A full
`--reset` rebuild is more correct after ingesting a large new backfill,
since Elo is sequential.

---

## Geocoding Stadiums (for travel distance)

`app/geocode_teams.py` fills in `stadium_lat` / `stadium_lon` for every team
that doesn't have coordinates yet, using OpenStreetMap Nominatim (free, no
API key), with an automatic fallback to Photon (komoot's free OSM-based
geocoder) if Nominatim is unavailable, rate-limited, or blocked.

**Before running — REQUIRED:** open `app/venue.py` and change `USER_AGENT`
away from its placeholder value to something that identifies your app, e.g.:
```python
USER_AGENT = "my-football-predictor/1.0"
```
**This step isn't optional.** Nominatim actively rejects requests with a
generic/unfilled User-Agent — it returns an HTTP 403 error
(`GeocoderInsufficientPrivileges`), and the script will now refuse to run
at all until this is fixed, failing immediately with a clear message rather
than burning through your whole team list hitting the same rejection
repeatedly. No need for a real personal email — Nominatim's policy just
wants something non-generic, not personal contact details specifically.
Policy: https://operations.osmfoundation.org/policies/nominatim/

**Run it:**
```bash
python -m app.geocode_teams              # geocode every team missing coordinates
python -m app.geocode_teams --team "Arsenal"   # just one team
```

It rate-limits itself to 1 request/second automatically (required by
Nominatim's policy) so this will take a few seconds per team — expect it to
take a couple of minutes for a full league.

**If a specific team still fails after fixing USER_AGENT:** it'll print
which geocoder failed and why, then try the other one automatically. If
both fail for a given team, that one prints `[NOT FOUND]` and gets skipped
— set it manually afterward:
```sql
UPDATE teams SET stadium_lat = 51.5549, stadium_lon = -0.1084 WHERE name = 'Arsenal';
```

**Important either way:** stadium name searches occasionally return the
wrong venue (a training ground, a redirect, a similarly-named place
elsewhere). Spot-check a handful of results against a map before trusting
the rest:
```sql
SELECT name, stadium_lat, stadium_lon FROM teams WHERE stadium_lat IS NOT NULL;
```

---

## Important: Database Migration Needed

This update adds three new columns to `matches` (`venue_lat`, `venue_lon`,
`venue_name`) for the local-match venue feature below. `Base.metadata.create_all()`
(used by `init_db.py`) only creates tables that don't exist yet — it will
**not** add new columns to a `matches` table you already created in earlier
steps. Run this once against your existing database before using the venue
features:

```sql
ALTER TABLE matches ADD COLUMN venue_lat FLOAT;
ALTER TABLE matches ADD COLUMN venue_lon FLOAT;
ALTER TABLE matches ADD COLUMN venue_name VARCHAR(200);
```

Run that in pgAdmin's Query Tool. (If you'd rather not hand-write `ALTER TABLE`
statements every time the schema changes going forward, this is exactly what
Alembic migrations are for — worth adopting once the schema stabilizes.)

---

## Setting Match Venues (for local matches)

By default, travel distance assumes a match is played at the home team's
registered stadium. For local matches, the actual pitch may not be the
team's usual ground — or may not be geocodable by team name at all —
so you can override the venue per match, three ways:

### 1. Type in an address (most common case)

```bash
python -m app.set_match_venue --match-id 42 --address "Kasarani Stadium, Nairobi"
```
Geocoded via Nominatim, free, same service used for team stadiums.

### 2. Enter exact coordinates directly

```bash
python -m app.set_match_venue --match-id 42 --lat -1.2921 --lon 36.8219 --name "Kasarani Stadium"
```
Useful if you already have coordinates from Google Maps ("What's here?" gives
you lat/lon directly — right-click a location on the map).

### 3. Live GPS detection (genuinely "detect current location")

A Python script **cannot** access a phone's GPS directly — only a browser
or app running on the device can. `static/detect_location.html` is a
standalone page (no server needed) that uses the browser's own Geolocation
API:

1. Open `static/detect_location.html` on your phone or laptop **while
   physically at (or near) the venue**
2. Tap "Detect My Location" and allow the location permission prompt
3. It shows you the exact lat/lon **and a ready-to-run command** you can
   copy straight into your terminal:
   ```bash
   python -m app.set_match_venue --match-id 42 --lat <detected> --lon <detected>
   ```

This gives you real GPS accuracy (the page shows an accuracy estimate in
meters) — the most reliable of the three options when you can physically
be at the venue.

### 4. Coarse auto-detect fallback (approximate — use with caution)

```bash
python -m app.set_match_venue --match-id 42 --auto-detect
```
Uses IP-based geolocation (free, no setup) to guess a location and asks you
to confirm before saving. **Be aware:** this reflects your network's
approximate location — often only accurate to city level, and can be
noticeably wrong on mobile data or VPNs. Treat it as a quick rough default
you can correct, not as reliable venue detection. Prefer options 1–3
whenever you can.

**Verify in pgAdmin:**
```sql
SELECT id, date, venue_name, venue_lat, venue_lon FROM matches WHERE venue_lat IS NOT NULL;
```

**Verify with tests:**
```bash
pytest tests/test_venue.py tests/test_features.py tests/test_venue_timeout.py -v
```
Covers address geocoding (including the Nominatim -> Photon fallback path),
IP-location parsing/failure handling, the placeholder-User-Agent rejection,
and the venue-override vs. fallback logic in travel distance — 45 tests
across the three files.

---

## Building the Feature Table

Once you have ingested matches, built Elo ratings, and geocoded stadiums,
`app/build_feature_table.py` assembles everything into one row per match —
ready for the modeling phase.

**Run it:**
```bash
python -m app.build_feature_table --output features.csv
```

**What's in each row:** Elo ratings for both teams, rest days, rolling form
(win rate + goals for/against over the last 5 matches), head-to-head record,
travel distance, derby flag, injury impact for both teams, and the target
column (`result`: H/D/A).

**Two things worth understanding before you move to modeling:**

1. **Every feature function only looks at matches strictly *before* the
   match being featurized** (see `app/features.py`) — this prevents
   leaking future information into training. This is tested explicitly in
   `tests/test_features.py` (search for "leakage" in that file).
2. **Missing values are left as-is, not silently imputed** — a team's very
   first match will have `None` for rest days and neutral defaults for
   form (0.5 win rate, 0 matches played), and any match without geocoded
   stadiums will have `None` for `travel_distance_km`. Decide how to
   handle these (drop, impute league-average, etc.) explicitly in your
   modeling script — that decision belongs there, not hidden in this one.

**Derby pairs:** `app/derbies.py` is a manually curated set, same pattern
as team aliases — extend `DERBY_PAIRS` for your leagues.

**Verify with tests:**
```bash
pytest tests/test_distance.py tests/test_geocode.py tests/test_features.py -v
```
This covers: haversine distance against known real-world distances, geocoding
logic against a mocked service, and — most importantly — the leakage-boundary
behavior of rest days, rolling form, and head-to-head (confirming a match
exactly on the cutoff date is correctly excluded from "history").

---

## Training the Model

`app/train_model.py` trains an XGBoost model on the feature table, using a
**time-based** train/validation/test split (never random — see
`app/data_split.py`), and evaluates it against two baselines so you can
tell whether the model is actually adding value:

- **Naive baseline** — predicts the training set's average outcome
  distribution (e.g. "45% home win, 25% draw, 30% away win") for every
  match, regardless of who's playing. If your model can't beat this,
  something is fundamentally wrong.
- **Elo baseline** — turns the Elo rating gap into a probability using a
  simple heuristic (`app/baselines.py`). A reasonable model should
  meaningfully beat this once you have real data with real signal in the
  other features (rest days, injuries, derbies) — Elo alone doesn't see
  any of that.

**Run it:**
```bash
python -m app.train_model --input features.csv --model-out model.joblib
```
Or, to build the feature table fresh from the database and train in one step:
```bash
python -m app.train_model --model-out model.joblib
```

**What the output tells you:**
- **Split summary** — confirms the chronological boundaries; sanity-check
  that train/val/test dates look right for your data.
- **Early stopping iteration** — training stops once validation performance
  stops improving. If this number is much lower than the max (200 by
  default), the extra trees weren't helping — normal, not a bug.
- **Overfitting check (train vs test)** — if train log loss is dramatically
  better than test log loss, the model has memorized noise rather than
  learned generalizable patterns. Some gap is normal; a huge one means you
  likely need more data, fewer features, or stronger regularization
  (`reg_lambda`, lower `max_depth`) before trusting the model.
- **Test set evaluation** — the real, honest comparison: naive baseline vs
  Elo baseline vs your model, on data the model never saw during training
  or early stopping.

**Important — a note on how much data you actually need:** with only your
current 13-match sample, the script runs correctly (proving the pipeline
works end-to-end) but prints a loud warning and the numbers are meaningless
— 2-3 test matches tell you nothing. You need real historical data (several
seasons, hundreds+ of matches) before any of these metrics are worth
looking at. Run `app.ingest_csv` against real football-data.co.uk seasons
first.

**Tuning options:**
```bash
python -m app.train_model --input features.csv --train-frac 0.6 --val-frac 0.2
```

**Verify with tests:**
```bash
pytest tests/test_baselines.py tests/test_evaluation.py tests/test_data_split.py tests/test_train_model.py -v
```
34 tests covering: baseline probability math, log loss / Brier score against
hand-computed values, chronological split correctness (including a test that
a shuffled input still produces a correctly time-ordered split), and the
full training pipeline against synthetic data (probabilities sum to 1, beats
the naive baseline, early stopping actually engages, same-seed
reproducibility, and a save/load round-trip that confirms a loaded model
produces identical predictions to the original).

**One honest caveat about the synthetic test data** (`app/synthetic_data.py`):
its outcomes are generated using the exact same formula as the Elo baseline,
which makes Elo close to mathematically optimal *for that specific synthetic
set* — so the automated tests check that the model beats the *naive*
baseline (an honest, achievable bar), not that it beats Elo (which isn't a
fair test on data generated that way). Whether your real model beats Elo
depends entirely on how much genuine signal is in your real rest-days,
injuries, and derby features once you're training on real matches — that's
for you to find out once real data is flowing through this pipeline, not
something synthetic tests can honestly promise either way.

---

## Ingesting Local/Non-Standard League Data

`app/ingest_local_csv.py` handles a different CSV format from
football-data.co.uk — column names like `Home Team`/`Away Team`/`Home
Goals`, and `yyyy-mm-dd` dates instead of `dd/mm/yy`. Use this for
datasets like local/Kenyan football data that don't follow the
football-data.co.uk convention.

**Required columns:** `Date, League, Home Team, Away Team, Home Goals, Away Goals`

**Run it:**
```bash
python -m app.ingest_local_csv path/to/local_matches.csv --league-group KPL
```

Every team gets tagged with `league="KPL"` (or whatever `--league-group`
you pass) — this is the broad UI bucket used by the league picker in the
frontend, not the specific competition. The specific competition name
(e.g. "Kenyan Premier League", "FKF Division One") is preserved per-match
in `Match.competition` regardless — nothing is lost, it's just grouped for
the UI. See `LOCAL_MATCH_DATA_GUIDE.md` for the full reasoning and for how
to get local data when no ready-made CSV exists for your league.

**If most/all rows have no venue data** (common for local leagues —
checked directly against a real uploaded Kenyan dataset where 100% of rows
had `Venue = "Not listed"`): `app/geocode_teams.py` now falls back to a
league-level approximate location (`LEAGUE_FALLBACK_LOCATIONS` in
`app/venue.py`) when a team's exact stadium can't be geocoded by name.
Every fallback use is logged as `[APPROXIMATE]` (vs `[OK]` for exact
matches) so you always know which travel distances are real vs estimated.
Extend `LEAGUE_FALLBACK_LOCATIONS` for other leagues/countries as needed.

---

## The `league` Model Feature

Once you're mixing data from multiple leagues (e.g. EPL + KPL), the model
needs a way to tell them apart — a high-scoring local league and a
defensive top-flight league don't share the same scoring baseline. `league`
(the team's UI-bucket league, e.g. "EPL"/"KPL") is now a genuine
**categorical** feature in the model, not just metadata.

**Why this needed real handling, not just adding a column:** XGBoost needs
the *exact same* category encoding across training, validation, test, and
live predictions — inferring categories fresh from each split risks
inconsistent encoding if one split doesn't contain every category. This is
handled via `get_category_dtypes()` (captures categories from the training
set only) and `category_dtypes` saved alongside the model file, reused by
`app/api.py` for live predictions.

**Verify with tests:**
```bash
pytest tests/test_categorical_features.py tests/test_ingest_local.py -v
```
11 tests covering: correct categorical dtype assignment, consistent
category encoding across splits, training actually succeeding on
mixed-league data, and the single-row live-prediction path using the saved
training-time categories (not inferring fresh from one row, which would be
wrong).

---

## A Real Trap: A League Can Get Silently Excluded From Training

**This happened in practice, not just in theory** — worth reading even if
you think your data is fine.

If one league's matches cluster in a different time window than another's
(e.g. all your KPL data is from 2024-2026, while EPL spans back to 2015),
a single global chronological 70/15/15 split can put **100% of that
league's matches into the test set and 0% into training**. Confirmed on
a real run: `league` feature importance came out as exactly `0.0000`, and
a per-league breakdown of the test set showed one league had never
appeared in training at all. The model wasn't "bad" at that league — it
had literally never seen an example of it.

**Detection is now automatic.** Every training run prints the league
distribution per split, and warns you explicitly if a league is present in
your data but missing from training:
```
League distribution per split:
  train: {'Premier League': 2839}
  val:   {'Premier League': 609}
  test:  {'Premier League': 363, 'KPL': 246}
  WARNING: {'KPL'} present in the data but MISSING from training entirely.
```

**The fix:** `--split-by-league` splits each league on its **own**
timeline (its own 70/15/15), then combines them — so every league gets
fair representation in train, val, and test, instead of one global cut
that can accidentally exclude a whole league.
```bash
python -m app.train_model --input features.csv --model-out model.joblib --split-by-league
```

This is opt-in, not the default — a global split is still the right choice
if you're only ever training on one league, or if your leagues' time
ranges genuinely overlap and this isn't an issue for your data. Check the
printed warning first; only reach for this flag if it actually fires.

**Verify with tests:**
```bash
pytest tests/test_data_split.py -v
```
10 tests, including one that first reproduces the bug with the plain split
(confirming it's real, not hypothetical) and then confirms
`time_based_split_grouped` fixes it — every group present in every split,
chronological integrity preserved within each group, no rows lost or
duplicated, and correct handling of a group too small to split three ways.

---

## Running the Frontend

A football-themed, installable PWA (`frontend/index.html`) with a
three-screen flow:

1. **League picker** — choose EPL, KPL, or whatever leagues exist in your
   database (populated dynamically from `GET /api/leagues`, with live team
   counts per league).
2. **Matchup screen** — pick home/away teams (filtered to the selected
   league), with a **"+ Team not listed?"** option per side to create a
   brand-new team on the fly (`POST /api/teams`), and an optional **"Enter
   match details manually"** panel for overriding computed features — most
   useful right after adding a new team, since it has no real history yet
   for the normal features to compute from.
3. **Result screen** — probabilities as animated bars, plain-language key
   factors, "Predict Another Match" to go again.

**The predict button plays a short animation** — a football flies from the
button into a net graphic (SVG, drawn in CSS) with a brief flash on
impact — synced with the actual API call via `Promise.all`, so the result
only reveals once both the animation and the real prediction have finished
(not a fixed fake delay).

**Requires a trained model first** (see "Training the Model" above):
```bash
python -m app.train_model --model-out model.joblib
```

**Run the server:**
```bash
uvicorn app.api:app --reload
```
Then open **http://localhost:8000** in your browser. The backend
(`app/api.py`) serves the API and the frontend from the same origin — one
command runs everything, no CORS setup needed.

**API endpoints:**
- `GET /api/leagues` — distinct league values, for the league picker
- `GET /api/teams?league=EPL` — teams filtered by league (omit the query param for all teams)
- `POST /api/teams` — body: `{"name": "New FC", "league": "KPL"}` → creates a team, returns its id
- `POST /api/predict` — body: `{"home_team_id": 1, "away_team_id": 2}`, plus optional:
  - `"match_date"`: defaults to today
  - `"is_derby"`: override the auto-detected derby flag
  - `"travel_distance_km"`: override computed travel distance
  - `"home_overrides"` / `"away_overrides"`: `{"elo": 1500, "rest_days": 7, "win_rate_last5": 0.6, "avg_goals_scored_last5": 1.5, "avg_goals_conceded_last5": 1.0, "injury_count": 1, "injury_importance_sum": 0.5}` — any field omitted keeps the computed value

**Using a different model file:**
```bash
MODEL_PATH=other_model.joblib uvicorn app.api:app --reload
```

**On the football image:** `frontend/football-hero.png` is an **original
image I generated**, not a hot-linked third-party photo. Two deliberate
reasons: (1) most real football photos on the web require attribution or
have unclear licensing for embedding in a redistributed codebase, and (2)
a hot-linked external image would break the PWA's offline caching
guarantee — the whole point of the service worker is that the app shell
works without a network connection, which doesn't hold if a critical image
lives on someone else's server. Regenerate or replace it anytime — it's
just a PNG in `frontend/`.

**This is a PWA (installable app), not just a webpage.** See
`HOSTING_GUIDE.md` for testing it on your phone (same-WiFi today, or
properly hosted with real HTTPS for reliable "Add to Home Screen" install).

**Want more data, including local/lower leagues?** See
`LOCAL_MATCH_DATA_GUIDE.md`.

**How "live" predictions differ from training data:** an upcoming match
has no historical row yet, so a few features are computed differently for
predictions than for training:
- **Elo** uses `get_current_elo_rating` (most recent rating on or before
  today) instead of an exact-date lookup, since there's no `elo_history`
  entry for a date that hasn't happened yet.
- **Injuries** use `get_current_injury_impact` — players currently in
  their injury window as of today — instead of the match-specific lookup
  used for historical training rows (which relies on a `match_missed_id`
  that doesn't exist for a hypothetical future match).

Both of these were caught and fixed during testing, not assumed correct —
worth knowing if you ever extend the feature set further, since it's an
easy category of bug to reintroduce.

**On the "key factors" explanations:** these are a heuristic — comparing
this match's actual feature values against reasonable thresholds — not a
SHAP explanation. Grounded in real data for the specific matchup, but not
as rigorous as SHAP's exact per-prediction attribution. Revisit once SHAP
is added (see "What's next" above).

**Verify with tests:**
```bash
pytest tests/test_predict.py tests/test_api.py -v
```
27 tests covering: the live-feature builder, and the full API stack via
FastAPI's TestClient — including league filtering, team creation, manual
overrides actually changing predictions (not just accepted without effect),
and a regression test for a real bug caught during manual testing (a team
with no geocoded stadium coordinates crashing prediction via a pandas
dtype issue — fixed and locked in by the test).
