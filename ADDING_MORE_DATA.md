# Adding More Data — Quick Guide

You currently have 11 scored matches in the database. That's nowhere near
enough to train on (see the guidelines below) — this guide walks through
getting several real seasons in.

---

## 1. Download more seasons

Go to: https://football-data.co.uk/data.php

Pick your league (e.g. Premier League = "E0"), and download the CSV for
as many past seasons as you can — aim for at least 5 seasons total.

Each season is a separate CSV download on that page (e.g. "2023/24",
"2022/23", "2021/22", etc.). Save them somewhere easy to find, e.g. a
`downloads/` folder inside your project.

---

## 2. Ingest each season

Run the ingestion script once per CSV file you downloaded:

```bash
python -m app.ingest_csv downloads/E0_2023_24.csv --competition "Premier League"
python -m app.ingest_csv downloads/E0_2022_23.csv --competition "Premier League"
python -m app.ingest_csv downloads/E0_2021_22.csv --competition "Premier League"
python -m app.ingest_csv downloads/E0_2020_21.csv --competition "Premier League"
python -m app.ingest_csv downloads/E0_2019_20.csv --competition "Premier League"
```

(Adjust filenames to whatever you actually saved them as.)

Each run prints a summary — check "Matches inserted" each time to confirm
it worked. It's safe to re-run any file twice; duplicates get skipped
automatically, not double-inserted.

---

## 3. Confirm the count went up

```bash
python3 -c "
from app.database import SessionLocal
from app.models import Match
db = SessionLocal()
total = db.query(Match).filter(Match.home_score.isnot(None)).count()
print(f'Scored matches in database: {total}')
db.close()
"
```

You're aiming for several hundred at minimum, ideally 1,000+.

---

## 4. Rebuild Elo ratings (must be re-run after adding matches)

```bash
python -m app.elo --reset
```

`--reset` matters here — you're adding matches that happened *before* your
existing ones chronologically, so Elo needs to rebuild from scratch rather
than just appending.

---

## 5. Geocode any new teams

New seasons likely introduce teams you haven't geocoded yet (promoted/relegated clubs):

```bash
python -m app.geocode_teams
```

This only geocodes teams still missing coordinates, so it's quick and safe
to re-run anytime.

---

## 6. Rebuild the feature table and retrain

```bash
python -m app.build_feature_table --output features.csv
python -m app.train_model --input features.csv --model-out model.joblib
```

Check the "Split summary" at the top of the output — that's your real
train/val/test counts. Once those numbers are in the hundreds, the
evaluation results become worth actually trusting.

---

## Rough guide: how much is "enough"?

| Scored matches | Verdict |
|---|---|
| Under ~50 | Not enough — don't trust any results yet |
| ~50–300 | Marginal — one league season is only ~380 matches total |
| ~500–1,500 | Reasonable — a few seasons, patterns start being real |
| 2,000+ | Solid — multiple seasons/leagues, trust the comparisons |
