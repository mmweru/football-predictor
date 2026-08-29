"""
Creates all tables defined in app.models against whatever database
app.config.settings currently points to.

Run directly:
    python -m app.init_db

Safe to re-run — `create_all` only creates tables that don't already exist,
it will NOT drop or alter existing ones. For real schema changes later,
switch to Alembic migrations instead of editing tables by hand.
"""

from app.database import Base, engine

# Importing app.models registers all the model classes onto Base.metadata —
# this import is required even though nothing below references it directly.
from app import models  # noqa: F401


def init_db() -> None:
    print(f"Creating tables (if not present) at: {engine.url}")
    Base.metadata.create_all(bind=engine)
    print("Tables created:")
    for table_name in Base.metadata.tables:
        print(f"  - {table_name}")


if __name__ == "__main__":
    init_db()
