"""
Adds the new columns to an EXISTING database (Postgres or SQLite) without
touching your data. `Base.metadata.create_all` creates missing TABLES but never
adds columns to tables that already exist, so run this once after upgrading:

    python -m app.migrate

It is idempotent — safe to run any number of times.
"""

from __future__ import annotations

from sqlalchemy import inspect, text

from app.database import Base, engine
import app.models  # noqa: F401  (registers all tables on Base.metadata)

# (table, column, SQL type, default clause)
NEW_COLUMNS = [
    ("teams", "is_active", "BOOLEAN", "DEFAULT TRUE"),
    ("teams", "fdorg_id", "INTEGER", ""),
    ("teams", "fpl_id", "INTEGER", ""),
    ("matches", "season", "INTEGER", ""),
    *[("matches", c, "INTEGER", "") for c in (
        "home_shots", "away_shots", "home_sot", "away_sot", "home_corners", "away_corners",
        "home_fouls", "away_fouls", "home_yellow", "away_yellow", "home_red", "away_red")],
    *[("matches", c, "FLOAT", "") for c in ("odds_home", "odds_draw", "odds_away", "temp_c", "wind_kph", "precip_mm")],
    ("players", "fpl_id", "INTEGER", ""),
    ("players", "minutes", "INTEGER", ""),
    ("players", "goals", "INTEGER", ""),
    ("players", "assists", "INTEGER", ""),
    ("players", "xg", "FLOAT", ""),
    ("players", "xa", "FLOAT", ""),
    ("players", "status", "VARCHAR(20)", ""),
    ("players", "chance_of_playing", "INTEGER", ""),
    ("players", "news", "VARCHAR(300)", ""),
    ("injuries", "reason", "VARCHAR(200)", ""),
    ("injuries", "source", "VARCHAR(30)", ""),
]


def migrate() -> int:
    Base.metadata.create_all(bind=engine)  # creates the new `predictions` table (and anything else missing)
    insp = inspect(engine)
    added = 0
    with engine.begin() as conn:
        for table, col, sql_type, default in NEW_COLUMNS:
            existing = {c["name"] for c in insp.get_columns(table)}
            if col in existing:
                continue
            default_sql = default
            if engine.dialect.name == "sqlite" and "TRUE" in default:
                default_sql = "DEFAULT 1"
            conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {col} {sql_type} {default_sql}".strip()))
            print(f"  added {table}.{col}")
            added += 1
        # Existing teams predate is_active: make sure NULLs become TRUE (sync_data refines this later).
        conn.execute(text("UPDATE teams SET is_active = :t WHERE is_active IS NULL"), {"t": True})
    print(f"Migration complete. {added} column(s) added.")
    return added


if __name__ == "__main__":
    migrate()
