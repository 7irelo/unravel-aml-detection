"""Runtime settings, read from the environment."""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="UNRAVEL_", env_file=".env", extra="ignore")

    database_url: str = "postgresql://unravel:unravel_local@localhost:55432/unravel"
    agent_database_url: str = (
        "postgresql://unravel_agent:agent_read_only_local@localhost:55432/unravel"
    )
    kafka_bootstrap: str = "localhost:19092"
    llm: str = "scripted"  # "scripted" needs no key; "anthropic" spends money
    llm_budget_usd: float = 25.0
    capacity_per_day: int = 100
    dataset: str = "fixture"
