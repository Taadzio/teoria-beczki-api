from __future__ import annotations

from functools import lru_cache

import pandas as pd

from .config import DATA_PATH

REQUIRED_COLUMNS = ["ROK", "MC", "DZ", "SMDB", "WSMDB", "TMAX", "TMIN", "STD"]


@lru_cache(maxsize=1)
def load_wroclaw() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Brak pliku danych: {DATA_PATH.resolve()}")

    df = pd.read_parquet(DATA_PATH)

    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Brak kolumn: {missing}. Dostępne: {list(df.columns)}")

    df["SMDB"] = pd.to_numeric(df["SMDB"], errors="coerce")
    df["WSMDB"] = pd.to_numeric(df["WSMDB"], errors="coerce")

    # 9 = brak zjawiska, 8 = brak pomiaru; na start oba traktujemy jako 0 mm.
    df.loc[df["WSMDB"] == 9, "SMDB"] = 0.0
    df.loc[df["WSMDB"] == 8, "SMDB"] = 0.0
    df["SMDB"] = df["SMDB"].fillna(0.0).clip(lower=0.0)

    for column in ["TMAX", "TMIN", "STD"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    return df
