from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import List, Dict, Any

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

    # opad
    df["SMDB"] = pd.to_numeric(df["SMDB"], errors="coerce")
    df["WSMDB"] = pd.to_numeric(df["WSMDB"], errors="coerce")

    # statusy:
    # 9 = brak zjawiska -> 0 mm
    # 8 = brak pomiaru -> na start 0 mm (żeby symulacja nie padała)
    df.loc[df["WSMDB"] == 9, "SMDB"] = 0.0
    df.loc[df["WSMDB"] == 8, "SMDB"] = 0.0

    df["SMDB"] = df["SMDB"].fillna(0.0).clip(lower=0.0)

    # temperatury (na przyszłość)
    for c in ["TMAX", "TMIN", "STD"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


@app.get("/columns")
def columns():
    df = load_wroclaw()
    return {"columns": list(df.columns)}


def simulate_constant(df: pd.DataFrame, area_m2: float, daily_use_mm: float, capacity_l: float) -> dict:
    """
    Symulacja zbiornika dzień-po-dniu.
    Dopływ [L] = SMDB[mm] * area[m2]
    Pobór [L]  = daily_use_mm[mm] * area[m2]
    """
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


def make_curve(
    df: pd.DataFrame,
    area_m2: float,
    daily_use_mm: float,
    alpha: float,
    beta: float,
    max_capacity_l: float,
    step_l: float,
) -> List[Dict[str, Any]]:
    """
    Zwraca listę punktów krzywej:
    capacity_l, capacity_m3, overflow_l, used_l, loss
    """
    if step_l <= 0 or max_capacity_l <= 0:
        raise ValueError("step_l i max_capacity_l muszą być > 0")

    step_l_i = int(step_l)
    max_cap_i = int(max_capacity_l)

    points: List[Dict[str, Any]] = []

    for cap in range(step_l_i, max_cap_i + 1, step_l_i):
        res = simulate_constant(df, area_m2, daily_use_mm, float(cap))
        loss = alpha * res["overflow_l"] + beta * float(cap)

        points.append(
            {
                "capacity_l": float(cap),
                "capacity_m3": round(float(cap) / 1000.0, 3),
                "overflow_l": round(res["overflow_l"], 2),
                "used_l": round(res["used_l"], 2),
                "loss": round(loss, 2),
            }
        )

    return points


def pick_best(points: List[Dict[str, Any]]) -> Dict[str, Any]:
    return min(points, key=lambda p: p["loss"])


@app.get("/curve")
def curve(
    area_m2: float,
    daily_use_mm: float = 2.0,
    alpha: float = 1.0,
    beta: float = 0.05,
    max_capacity_l: float = 20000.0,
    step_l: float = 200.0,
    limit_points: int = 120,
):
    """
    Zwraca krzywą (capacity vs overflow vs loss) + najlepszy punkt.
    limit_points ogranicza liczbę punktów w odpowiedzi (żeby JSON nie był gigantyczny).
    """
    if area_m2 <= 0:
        raise HTTPException(status_code=400, detail="area_m2 musi być > 0")
    if max_capacity_l <= 0 or step_l <= 0:
        raise HTTPException(status_code=400, detail="max_capacity_l i step_l muszą być > 0")

    try:
        df = load_wroclaw()
        points = make_curve(df, area_m2, daily_use_mm, alpha, beta, max_capacity_l, step_l)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    best = pick_best(points)

    # limitowanie punktów do przeglądania na WWW
    if limit_points and len(points) > limit_points:
        # proste "próbkowanie" co n-ty punkt + dopisanie najlepszego (żeby nie zniknął)
        n = max(1, len(points) // limit_points)
        sampled = points[::n]
        if best not in sampled:
            sampled.append(best)
        points_out = sorted(sampled, key=lambda p: p["capacity_l"])
    else:
        points_out = points

    return {
        "city": "Wrocław",
        "strategy": "constant",
        "inputs": {
            "area_m2": area_m2,
            "daily_use_mm": daily_use_mm,
            "alpha": alpha,
            "beta": beta,
            "max_capacity_l": max_capacity_l,
            "step_l": step_l,
        },
        "best": best,
        "points": points_out,
        "points_total": len(points),
        "points_returned": len(points_out),
    }


@app.get("/simulate")
def simulate(
    area_m2: float,
    daily_use_mm: float = 2.0,
    alpha: float = 1.0,
    beta: float = 0.05,
    max_capacity_l: float = 20000.0,
    step_l: float = 200.0,
):
    """
    Krótsza odpowiedź: zwraca tylko najlepszą pojemność (bez krzywej).
    """
    if area_m2 <= 0:
        raise HTTPException(status_code=400, detail="area_m2 musi być > 0")

    try:
        df = load_wroclaw()
        points = make_curve(df, area_m2, daily_use_mm, alpha, beta, max_capacity_l, step_l)
        best = pick_best(points)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {
        "city": "Wrocław",
        "strategy": "constant",
        "inputs": {
            "area_m2": area_m2,
            "daily_use_mm": daily_use_mm,
            "alpha": alpha,
            "beta": beta,
            "max_capacity_l": max_capacity_l,
            "step_l": step_l,
        },
        "best": best,
    }