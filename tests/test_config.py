import pytest
from sqlalchemy.engine import make_url

from app.config import Settings


@pytest.mark.parametrize("scheme", ["postgresql", "postgresql+psycopg"])
def test_database_url_uses_installed_psycopg2_driver(scheme):
    settings = Settings(database_url=f"{scheme}://user:p%40ss@db.example/test_db?sslmode=require")

    url = make_url(settings.sqlalchemy_database_url)

    assert url.drivername == "postgresql+psycopg2"
    assert url.username == "user"
    assert url.password == "p@ss"
    assert url.host == "db.example"
    assert url.database == "test_db"
    assert url.query["sslmode"] == "require"