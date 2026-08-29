# Getting More Data — Including Local Matches

## Can the model predict local matches? Yes — with caveats.

Nothing in the pipeline requires a "major league." The moment two teams
exist in your `teams` table and have even a few matches ingested, the
model can compute features and produce a prediction for them, using the
exact same code path as any other match.

**But be realistic about accuracy at low data volumes:**
- **Elo starts neutral (1500)** for any brand-new team, and only becomes
  meaningful after several results. A local team with 3 matches in the
  database will have a very noisy Elo rating.
- **Rolling form, head-to-head, and injury features** need history to mean
  anything — a team's first few matches will show neutral defaults (see
  `app/features.py`), not real signal.
- **Update:** the model now includes `league` as a genuine categorical
  feature (previously flagged here as a limitation — it's fixed). This
  lets it distinguish a high-scoring local league from a defensive
  top-flight one, rather than treating all matches as one scoring
  distribution. See `README.md`'s "The `league` Model Feature" section.

**Bottom line:** the model *will* produce a prediction for any two teams
in your database, on day one. Whether that prediction is any good depends
entirely on how much real history you've fed it — same as any league.

---

## Getting more data for major leagues (recap)

See `ADDING_MORE_DATA.md` for the full walkthrough — download seasons from
https://football-data.co.uk/data.php and run `app.ingest_csv` on each one.

---

## Getting data for local/lower leagues (no CSV source exists)

There's no equivalent of football-data.co.uk for most local leagues —
this genuinely requires more manual work. Checked directly for the Kenyan
Premier League as an example: no downloadable historical CSV/API exists,
but there ARE structured sources you can transcribe from:

- **Wikipedia season pages** — surprisingly complete. E.g. search
  "20XX Kenyan Premier League" (or your league + year) — these often have
  full standings and sometimes match-by-match results in a structured
  table you can copy by hand.
- **Live-score sites with historical tables** — Sofascore, Flashscore, and
  Soccerway all have pages for many local leagues (search the league name
  on any of these) with fixtures/results tables you can view and manually
  transcribe. These aren't downloadable as CSV, but the data is there to
  read and copy.
- **Your league's official association website**, if one exists — often
  publishes fixtures and results as HTML tables, sometimes with a
  downloadable PDF/fixture list.

### The practical path: use the built-in local ingestion script

If your data comes as `Date, League, Home Team, Away Team, Home Goals, Away
Goals` (yyyy-mm-dd dates) — the format of a real Kenyan football dataset
this was tested against — just run:
```bash
python -m app.ingest_local_csv your_file.csv --league-group KPL
```
No manual spreadsheet work needed. If your source uses different column
names, either rename the columns to match, or adjust the small mapping in
`app/ingest_local_csv.py` — it's a short, readable script.

### If your local data has no venue/location info at all

Common for local leagues (confirmed directly: a real 246-row Kenyan
dataset had 100% of rows with no venue listed). `app/geocode_teams.py`
automatically falls back to a league-level approximate location when a
team's exact stadium can't be found by name — logged as `[APPROXIMATE]`
so you always know which travel distances are estimated vs exact. See
`LEAGUE_FALLBACK_LOCATIONS` in `app/venue.py` to add a fallback for
another league/country.

### The manual path — ONLY if your CSV doesn't match `app.ingest_local_csv`'s columns

**Skip this section if your file has `Date, League, Home Team, Away Team,
Home Goals, Away Goals` columns** — that's exactly what
`app.ingest_local_csv` (above) already handles directly, no manual work
needed. This section is only for a CSV in some other, different shape.

If your data isn't already in a clean CSV, or uses a completely different
structure than `app.ingest_local_csv` expects:

Since `app.ingest_csv` already expects a simple format
(`Date,HomeTeam,AwayTeam,FTHG,FTAG`), you don't need any new code — just
fill in a spreadsheet by hand and export it as CSV.

1. Open a spreadsheet (Excel, Google Sheets, LibreOffice) and create these
   exact column headers in row 1:
   ```
   Date,HomeTeam,AwayTeam,FTHG,FTAG
   ```
2. Fill in one row per match, using `dd/mm/yyyy` for the date, e.g.:
   ```
   Date,HomeTeam,AwayTeam,FTHG,FTAG
   15/03/2024,Gor Mahia,AFC Leopards,2,1
   16/03/2024,Tusker,Bandari,0,0
   ```
3. Export as CSV (File → Download → CSV, or Save As → CSV)
4. Ingest it exactly like any other file:
   ```bash
   python -m app.ingest_csv local_matches.csv --competition "Kenyan Premier League"
   ```

This is slower than downloading a ready-made file, but it's the same
ingestion pipeline either way — no special-casing needed for "local" data.

### Speeding up manual entry

- Start with just the **current season** for your league(s) of interest —
  30-40 matches gets you something usable faster than trying to backfill
  years of history first.
- Prioritize the teams you actually care about predicting — you don't need
  every team in the league on day one, just the ones you'll query.
- Keep adding a few matches at a time as new results come in — the
  ingestion script is idempotent (safe to re-run), so little-and-often
  works fine.

### After adding local match data

Same as adding any new season — re-run in this order:
```bash
python -m app.elo --reset
python -m app.geocode_teams
python -m app.build_feature_table --output features.csv
python -m app.train_model --input features.csv --model-out model.joblib
```
