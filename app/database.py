"""
SQLAlchemy engine + session setup.

Everything else in the app imports `Base` (to define models) and
`get_db` / `SessionLocal` (to talk to the database) from this module.
"""

from typing import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.config import settings

engine = create_engine(settings.sqlalchemy_database_url, echo=settings.echo_sql, future=True)

SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)


class Base(DeclarativeBase):
    """Base class every ORM model inherits from."""

    pass


def get_db() -> Generator[Session, None, None]:
    """
    Dependency-style generator that yields a session and guarantees it's
    closed afterwards. Use like:

        with next(get_db()) as db:   # or via FastAPI's Depends(get_db) later
            ...
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
