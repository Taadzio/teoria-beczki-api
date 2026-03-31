from os import getenv
from pathlib import Path


def _parse_allowed_origins() -> list[str]:
    raw = getenv("ALLOWED_ORIGINS", "").strip()
    if not raw:
        return [
            "http://localhost:5173",
            "http://127.0.0.1:5173",
            "http://localhost:5174",
            "http://127.0.0.1:5174",
            "https://zdankiewicz.pl",
            "http://zdankiewicz.pl",
            "https://www.zdankiewicz.pl",
            "http://www.zdankiewicz.pl",
        ]

    return [origin.strip() for origin in raw.split(",") if origin.strip()]


ALLOWED_ORIGINS = _parse_allowed_origins()

DATA_PATH = Path("data/base/wroclaw_synop_full_1961_2025.parquet")

DEFAULT_CITY = "Wrocław"
DEFAULT_YEARS_COUNT = 65
