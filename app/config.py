"""
Application configuration.

Reads settings from environment variables / a .env file using pydantic-settings.
This is the ONLY place that should know about connection strings — every other
module imports `settings` from here instead of reading os.environ directly.
"""

from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Individual Postgres connection parts (used to build DATABASE_URL below).
    postgres_user: str = "football_user"
    postgres_password: str = "changeme"
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "football_predictor"

    # If DATABASE_URL is set explicitly in the environment, it overrides the
    # individual parts above. This is handy for testing (e.g. pointing at
    # SQLite) without touching the Postgres settings.
    database_url: Optional[str] = None

    # --- external data sources (all optional; football-data.co.uk, FPL and Open-Meteo need no key) ---
    football_data_org_token: Optional[str] = None   # free token: https://www.football-data.org/client/register
    bigballs_api_key: Optional[str] = None          # free key: https://bigballsdata.com/signup
    football_api_key: Optional[str] = None           # API-Football key: https://www.api-football.com/
    market_weight: float = 0.8                      # weight on bookmaker odds when blending (see app/modeling.py)

    echo_sql: bool = False  # set True to log every SQL statement (useful for debugging)

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    @property
    def sqlalchemy_database_url(self) -> str:
        if self.database_url:
            return self.database_url
        return (
            f"postgresql+psycopg2://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )


settings = Settings()
