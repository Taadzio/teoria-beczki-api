from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException

app = FastAPI()

DATA_PATH = Path("data/base/wroclaw_synop_full_1961_2025.parquet")

@app.get("/")
def read_root():
    return {"message": "Teoria Beczki działa ??"}

@app.get("/ping")
def ping():
    return {"status": "ok"}

@lru_cache(maxsize=1)
def load_wroclaw() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Brak pliku danych: {DATA_PATH.resolve()}")

    df = pd.read_parquet(DATA_PATH)

    required = ["ROK", "MC", "DZ", "SMDB", "WSMDB", "TMAX", "TMIN", "STD"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Brak kolumn: {missing}. Dostępne: {list(df.columns)}")

    df["SMDB"] = pd.to_numeric(df["SMDB"], errors="coerce")
    df["WSMDB"] = pd.to_numeric(df["WSMDB"], errors="coerce")

    # 9 = brak zjawiska -> 0 mm; 8 = brak pomiaru -> na start 0 mm
    df.loc[df["WSMDB"] == 9, "SMDB"] = 0.0
    df.loc[df["WSMDB"] == 8, "SMDB"] = 0.0

    df["SMDB"] = df["SMDB"].fillna(0.0).clip(lower=0.0)

    for c in ["TMAX", "TMIN", "STD"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return df

@app.get("/columns")
def columns():
    df = load_wroclaw()
    return {"columns": list(df.columns)}

def simulate_constant(df: pd.DataFrame, area_m2: float, daily_use_mm: float, capacity_l: float) -> dict:
    storage = 0.0
    overflow_total = 0.0
    used_total = 0.0

    daily_use_l = max(0.0, daily_use_mm) * area_m2

    for precip_mm in df["SMDB"].to_numpy():
        inflow_l = float(precip_mm) * area_m2
        storage += inflow_l

        if storage > capacity_l:
            overflow_total += (storage - capacity_l)
            storage = capacity_l

        take = min(storage, daily_use_l)
        storage -= take
        used_total += take

    return {"overflow_l": overflow_total, "used_l": used_total}

def find_best_capacity(df: pd.DataFrame, area_m2: float, daily_use_mm: float) -> dict:
    alpha = 1.0
    beta = 0.002  # kara za wielkość zbiornika (do strojenia)

    capacities_l = [x * 100.0 for x in range(2, 201)]  # 200..20000 L

    best = None
    for cap in capacities_l:
        res = simulate_constant(df, area_m2, daily_use_mm, cap)
        loss = alpha * res["overflow_l"] + beta * cap
        if best is None or loss < best["loss"]:
            best = {"capacity_l": cap, "loss": loss, **res}

    return best

@app.get("/simulate")
def simulate(area_m2: float, daily_use_mm: float = 2.0):
    if area_m2 <= 0:
        raise HTTPException(status_code=400, detail="area_m2 musi być > 0")

    try:
        df = load_wroclaw()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    best = find_best_capacity(df, area_m2, daily_use_mm)

    return {
        "city": "Wrocław",
        "strategy": "constant",
        "inputs": {"area_m2": area_m2, "daily_use_mm": daily_use_mm},
        "best": {
            "capacity_l": best["capacity_l"],
            "capacity_m3": round(best["capacity_l"] / 1000.0, 3),
            "overflow_l": round(best["overflow_l"], 2),
            "used_l": round(best["used_l"], 2),
        },
    }