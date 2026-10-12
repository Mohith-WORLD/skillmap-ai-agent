import os
from dataclasses import dataclass
from typing import List, Optional
from dotenv import load_dotenv


@dataclass
class Config:
    """Application configuration and credentials."""
    google_api_key: Optional[str] = None
    tavily_api_key: Optional[str] = None
    rapidapi_key: Optional[str] = None
    model_name: str = "gemini-2.5-flash"
    sqlite_db_path: str = "data/checkpoints.db"

    @property
    def has_google_key(self) -> bool:
        return bool(self.google_api_key and self.google_api_key.strip())

    @property
    def has_tavily_key(self) -> bool:
        return bool(self.tavily_api_key and self.tavily_api_key.strip())

    @property
    def has_rapidapi_key(self) -> bool:
        return bool(self.rapidapi_key and self.rapidapi_key.strip())

    def missing_keys(self) -> List[str]:
        missing = []
        if not self.has_google_key:
            missing.append("GOOGLE_API_KEY")
        if not self.has_tavily_key:
            missing.append("TAVILY_API_KEY")
        if not self.has_rapidapi_key:
            missing.append("RAPIDAPI_KEY")
        return missing

    def is_fully_configured(self) -> bool:
        return len(self.missing_keys()) == 0

    def get_status_report(self) -> dict:
        def _mask(k: Optional[str]) -> str:
            if not k:
                return "Not Configured"
            k = k.strip()
            if len(k) <= 6:
                return "***"
            return f"{k[:3]}...{k[-3:]}"

        return {
            "GOOGLE_API_KEY": _mask(self.google_api_key),
            "TAVILY_API_KEY": _mask(self.tavily_api_key),
            "RAPIDAPI_KEY": _mask(self.rapidapi_key),
            "GEMINI_MODEL": self.model_name,
            "SQLITE_DB_PATH": self.sqlite_db_path,
            "is_ready": self.is_fully_configured(),
        }


def load_config(env_file: Optional[str] = None) -> Config:
    """Load configuration from environment variables and .env file."""
    if env_file and os.path.exists(env_file):
        load_dotenv(dotenv_path=env_file, override=True)
    else:
        load_dotenv(override=True)

    # Support common aliases
    google_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    tavily_key = os.getenv("TAVILY_API_KEY")
    rapidapi_key = os.getenv("RAPIDAPI_KEY") or os.getenv("JSEARCH_API_KEY")
    model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    sqlite_db_path = os.getenv("SQLITE_DB_PATH", "data/checkpoints.db")

    return Config(
        google_api_key=google_key,
        tavily_api_key=tavily_key,
        rapidapi_key=rapidapi_key,
        model_name=model_name,
        sqlite_db_path=sqlite_db_path,
    )
