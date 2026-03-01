from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import List, Dict, Any, Tuple

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


def total_rain_l(df: pd.DataFrame, area_m2: float) -> float:
    # suma opadu [mm] * area[m2] => litry
    return float(df["SMDB"].sum()) * area_m2


def make_curve(
    df: pd.DataFrame,
    area_m2: float,
    daily_use_mm: float,
    alpha: float,
    beta: float,
    max_capacity_l: float,
    step_l: float,
    mode: str,
    cap_mm_ref: float,
) -> List[Dict[str, Any]]:
    """
    mode:
      - "raw":  loss = alpha*overflow_l + beta*capacity_l
      - "norm": loss = alpha*overflow_ratio + beta*capacity_ratio
               overflow_ratio = overflow_l / total_rain_l
               capacity_mm = capacity_l / area_m2
               capacity_ratio = capacity_mm / cap_mm_ref
    """
    if step_l <= 0 or max_capacity_l <= 0:
        raise ValueError("step_l i max_capacity_l muszą być > 0")
    if mode not in ("raw", "norm"):
        raise ValueError("mode musi być 'raw' albo 'norm'")
    if cap_mm_ref <= 0:
        raise ValueError("cap_mm_ref musi być > 0")

    step_l_i = int(step_l)
    max_cap_i = int(max_capacity_l)

    rain_total = total_rain_l(df, area_m2)
    # zabezpieczenie: gdyby seria miała same zera
    if rain_total <= 0:
        rain_total = 1.0

    points: List[Dict[str, Any]] = []

    for cap in range(step_l_i, max_cap_i + 1, step_l_i):
        cap_l = float(cap)
        res = simulate_constant(df, area_m2, daily_use_mm, cap_l)

        overflow_l = float(res["overflow_l"])
        used_l = float(res["used_l"])

        # normalizacje
        overflow_ratio = overflow_l / rain_total
        capacity_mm = cap_l / area_m2  # bo 1 mm na 1 m2 = 1 L
        capacity_ratio = capacity_mm / cap_mm_ref

        if mode == "raw":
            loss = alpha * overflow_l + beta * cap_l
        else:
            loss = alpha * overflow_ratio + beta * capacity_ratio

        points.append(
            {
                "capacity_l": cap_l,
                "capacity_m3": round(cap_l / 1000.0, 3),
                "overflow_l": round(overflow_l, 2),
                "used_l": round(used_l, 2),

                # pola z normalizacji zawsze zwracamy (żeby WWW mogła je rysować)
                "total_rain_l": round(rain_total, 2),
                "overflow_ratio": round(overflow_ratio, 6),
                "capacity_mm": round(capacity_mm, 3),
                "capacity_ratio": round(capacity_ratio, 6),

                "loss": round(float(loss), 6 if mode == "norm" else 2),
            }
        )

    return points


def pick_best(points: List[Dict[str, Any]]) -> Dict[str, Any]:
    return min(points, key=lambda p: p["loss"])


def elbow_point(points: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Wybór "kolana" krzywej metodą maks. odległości od prostej łączącej skrajne punkty.
    Używamy osi:
      x = capacity_l
      y = overflow_l
    Najpierw normalizacja x i y do [0,1], potem odległość punktu od prostej.
    """
    if len(points) < 3:
        return points[0]

    xs = [p["capacity_l"] for p in points]
    ys = [p["overflow_l"] for p in points]

    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    # unikanie dzielenia przez zero
    dx = (x_max - x_min) or 1.0
    dy = (y_max - y_min) or 1.0

    norm = []
    for p in points:
        x = (p["capacity_l"] - x_min) / dx
        y = (p["overflow_l"] - y_min) / dy
        norm.append((x, y))

    # prosta od pierwszego do ostatniego punktu
    x1, y1 = norm[0]
    x2, y2 = norm[-1]

    # odległość punktu od prostej (w 2D) – wersja z iloczynem wektorowym
    def dist_to_line(x0, y0) -> float:
        return abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1) / (
            ((y2 - y1) ** 2 + (x2 - x1) ** 2) ** 0.5 or 1.0
        )

    best_i = 0
    best_d = -1.0
    for i, (x0, y0) in enumerate(norm):
        d = dist_to_line(x0, y0)
        if d > best_d:
            best_d = d
            best_i = i

    out = dict(points[best_i])
    out["elbow_distance_norm"] = round(best_d, 6)
    return out


@app.get("/curve")
def curve(
    area_m2: float,
    daily_use_mm: float = 2.0,
    alpha: float = 1.0,
    beta: float = 0.05,
    max_capacity_l: float = 20000.0,
    step_l: float = 200.0,
    mode: str = "raw",
    cap_mm_ref: float = 100.0,
    limit_points: int = 120,
):
    if area_m2 <= 0:
        raise HTTPException(status_code=400, detail="area_m2 musi być > 0")

    try:
        df = load_wroclaw()
        points = make_curve(df, area_m2, daily_use_mm, alpha, beta, max_capacity_l, step_l, mode, cap_mm_ref)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    best = pick_best(points)

    # limit punktów do WWW
    if limit_points and len(points) > limit_points:
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
        "mode": mode,
        "inputs": {
            "area_m2": area_m2,
            "daily_use_mm": daily_use_mm,
            "alpha": alpha,
            "beta": beta,
            "max_capacity_l": max_capacity_l,
            "step_l": step_l,
            "cap_mm_ref": cap_mm_ref,
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
    mode: str = "raw",
    cap_mm_ref: float = 100.0,
):
    if area_m2 <= 0:
        raise HTTPException(status_code=400, detail="area_m2 musi być > 0")

    try:
        df = load_wroclaw()
        points = make_curve(df, area_m2, daily_use_mm, alpha, beta, max_capacity_l, step_l, mode, cap_mm_ref)
        best = pick_best(points)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {
        "city": "Wrocław",
        "strategy": "constant",
        "mode": mode,
        "inputs": {
            "area_m2": area_m2,
            "daily_use_mm": daily_use_mm,
            "alpha": alpha,
            "beta": beta,
            "max_capacity_l": max_capacity_l,
            "step_l": step_l,
            "cap_mm_ref": cap_mm_ref,
        },
        "best": best,
    }


@app.get("/elbow")
def elbow(
    area_m2: float,
    daily_use_mm: float = 2.0,
    max_capacity_l: float = 20000.0,
    step_l: float = 200.0,
):
    """
    Zwraca "kolano" krzywej capacity vs overflow (bez wag).
    """
    if area_m2 <= 0:
        raise HTTPException(status_code=400, detail="area_m2 musi być > 0")

    try:
        df = load_wroclaw()
        # tu tryb loss nie ma znaczenia, bo elbow liczymy z capacity/overflow
        points = make_curve(df, area_m2, daily_use_mm, alpha=1.0, beta=0.0,
                            max_capacity_l=max_capacity_l, step_l=step_l,
                            mode="raw", cap_mm_ref=100.0)
        e = elbow_point(points)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {
        "city": "Wrocław",
        "strategy": "constant",
        "inputs": {"area_m2": area_m2, "daily_use_mm": daily_use_mm, "max_capacity_l": max_capacity_l, "step_l": step_l},
        "elbow": e,
    }