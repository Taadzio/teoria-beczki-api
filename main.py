from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import pandas as pd
import numpy as np

from fastapi import FastAPI, HTTPException

app = FastAPI()

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        # docelowo adres z Vercel, np. https://teoria-beczki-web.vercel.app
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_PATH = Path("data/base/wroclaw_synop_full_1961_2025.parquet")

StrategyType = Literal["constant", "no_rain", "seasonal", "temp_seasonal"]


@app.get("/")
def read_root():
    return {"message": "Teoria Beczki działa ?"}


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


# -------------------------
# Helpers (strategy)
# -------------------------
def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def day_of_year(row: pd.Series) -> int:
    dt = pd.Timestamp(int(row["ROK"]), int(row["MC"]), int(row["DZ"]))
    return int(dt.dayofyear)


def temp_value(row: pd.Series, source: str) -> Optional[float]:
    """
    source: "TMAX" | "TMIN" | "STD" | "TAVG"
    """
    if source == "TAVG":
        tmax = row.get("TMAX")
        tmin = row.get("TMIN")
        if pd.notna(tmax) and pd.notna(tmin):
            return float(tmax + tmin) / 2.0
        return None

    v = row.get(source)
    if pd.notna(v):
        return float(v)
    return None


# -------------------------
# Simulation
# -------------------------
def simulate_strategy(
    df: pd.DataFrame,
    area_m2: float,
    capacity_l: float,
    *,
    strategy: StrategyType,
    base_mm: float,
    rain_threshold_mm: float = 0.1,
    season_start_doy: int = 100,
    season_end_doy: int = 280,
    temp_source: str = "TMAX",
    t0: float = 20.0,
    k: float = 0.15,
    min_mm: float = 0.0,
    max_mm: float = 8.0,
) -> dict:
    """
    Szybka symulacja (numpy): nadal dzień-po-dniu, ale bez iterrows().
    """

    # --- tablice ---
    rain_mm = df["SMDB"].to_numpy(dtype=np.float64)

    # day-of-year (liczymy raz)
    dt = pd.to_datetime(
        dict(year=df["ROK"].astype(int), month=df["MC"].astype(int), day=df["DZ"].astype(int)),
        errors="coerce",
    )
    doy = dt.dt.dayofyear.to_numpy(dtype=np.int32)
    in_season = (doy >= season_start_doy) & (doy <= season_end_doy)

    # temperatury
    if temp_source == "TAVG":
        tmax = df["TMAX"].to_numpy(dtype=np.float64)
        tmin = df["TMIN"].to_numpy(dtype=np.float64)
        temp = (tmax + tmin) / 2.0
        temp[np.isnan(tmax) | np.isnan(tmin)] = np.nan
    else:
        # "TMAX" | "TMIN" | "STD"
        temp = df[temp_source].to_numpy(dtype=np.float64)

    n = rain_mm.shape[0]

    storage = 0.0
    overflow_total = 0.0
    used_total = 0.0
    demand_total = 0.0
    deficit_total = 0.0
    spill_days = 0
    empty_days = 0

    area = float(area_m2)
    cap = float(capacity_l)
    base = max(0.0, float(base_mm))

    for i in range(n):
        # dopływ
        inflow_l = rain_mm[i] * area
        storage += inflow_l

        if storage > cap:
            overflow_total += storage - cap
            storage = cap
            spill_days += 1

        # strategia: use_mm
        use_mm = base

        if strategy == "no_rain":
            if rain_mm[i] >= rain_threshold_mm:
                use_mm = 0.0

        elif strategy == "seasonal":
            if not in_season[i]:
                use_mm = 0.0
            elif rain_mm[i] >= rain_threshold_mm:
                use_mm = 0.0

        elif strategy == "temp_seasonal":
            if (not in_season[i]) or (rain_mm[i] >= rain_threshold_mm):
                use_mm = 0.0
            else:
                t = temp[i]
                if np.isnan(t):
                    use_mm = base
                else:
                    use_mm = base + k * (float(t) - t0)
                    if use_mm < min_mm:
                        use_mm = min_mm
                    elif use_mm > max_mm:
                        use_mm = max_mm

        # pobór
        daily_demand_l = max(0.0, use_mm) * area
        demand_total += daily_demand_l

        take = storage if storage < daily_demand_l else daily_demand_l
        used_total += take
        storage -= take

        if daily_demand_l > take:
            deficit_total += (daily_demand_l - take)
            if storage <= 1e-9:
                empty_days += 1

    coverage = used_total / demand_total if demand_total > 0 else 1.0

    return {
        "overflow_l": overflow_total,
        "used_l": used_total,
        "demand_l": demand_total,
        "deficit_l": deficit_total,
        "coverage_ratio": coverage,
        "spill_days": spill_days,
        "empty_days": empty_days,
    }


# kompatybilność wsteczna (Twoje stare API logicznie nadal działa)
def simulate_constant(df: pd.DataFrame, area_m2: float, daily_use_mm: float, capacity_l: float) -> dict:
    return simulate_strategy(
        df,
        area_m2,
        capacity_l,
        strategy="constant",
        base_mm=daily_use_mm,
    )


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
    *,
    # strategy
    strategy: StrategyType = "constant",
    rain_threshold_mm: float = 0.1,
    season_start_doy: int = 100,
    season_end_doy: int = 280,
    temp_source: str = "TMAX",
    t0: float = 20.0,
    k: float = 0.15,
    min_mm: float = 0.0,
    max_mm: float = 8.0,
    # cost mode
    tank_cost_per_m3: float = 800.0,
    overflow_cost_per_m3: float = 0.0,
    deficit_cost_per_m3: float = 0.0,
) -> List[Dict[str, Any]]:
    """
    mode:
    - "raw":  loss = alpha*overflow_l + beta*capacity_l
    - "norm": loss = alpha*overflow_ratio + beta*capacity_ratio
             overflow_ratio = overflow_l / total_rain_l
             capacity_mm = capacity_l / area_m2
             capacity_ratio = capacity_mm / cap_mm_ref
    - "cost": loss = tank_cost_per_m3*cap_m3 + overflow_cost_per_m3*overflow_m3 + deficit_cost_per_m3*deficit_m3
    """
    if step_l <= 0 or max_capacity_l <= 0:
        raise ValueError("step_l i max_capacity_l muszą być > 0")

    if mode not in ("raw", "norm", "cost"):
        raise ValueError("mode musi być 'raw' albo 'norm' albo 'cost'")

    if mode == "norm" and cap_mm_ref <= 0:
        raise ValueError("cap_mm_ref musi być > 0 dla trybu norm")

    step_l_i = int(step_l)
    max_cap_i = int(max_capacity_l)

    rain_total = total_rain_l(df, area_m2)
    if rain_total <= 0:
        rain_total = 1.0

    points: List[Dict[str, Any]] = []

    for cap in range(step_l_i, max_cap_i + 1, step_l_i):
        cap_l = float(cap)

        res = simulate_strategy(
            df,
            area_m2,
            cap_l,
            strategy=strategy,
            base_mm=daily_use_mm,
            rain_threshold_mm=rain_threshold_mm,
            season_start_doy=season_start_doy,
            season_end_doy=season_end_doy,
            temp_source=temp_source,
            t0=t0,
            k=k,
            min_mm=min_mm,
            max_mm=max_mm,
        )

        overflow_l = float(res["overflow_l"])
        used_l = float(res["used_l"])
        demand_l = float(res["demand_l"])
        deficit_l = float(res["deficit_l"])
        coverage_ratio = float(res["coverage_ratio"])

        overflow_ratio = overflow_l / rain_total
        capacity_mm = cap_l / area_m2
        capacity_ratio = (capacity_mm / cap_mm_ref) if cap_mm_ref > 0 else 0.0

        if mode == "raw":
            loss = alpha * overflow_l + beta * cap_l
        elif mode == "norm":
            loss = alpha * overflow_ratio + beta * capacity_ratio
        else:  # cost
            cap_m3 = cap_l / 1000.0
            overflow_m3 = overflow_l / 1000.0
            deficit_m3 = deficit_l / 1000.0
            loss = (
                tank_cost_per_m3 * cap_m3
                + overflow_cost_per_m3 * overflow_m3
                + deficit_cost_per_m3 * deficit_m3
            )

        points.append(
            {
                "capacity_l": cap_l,
                "capacity_m3": round(cap_l / 1000.0, 3),
                "overflow_l": round(overflow_l, 2),
                "used_l": round(used_l, 2),
                "demand_l": round(demand_l, 2),
                "deficit_l": round(deficit_l, 2),
                "coverage_ratio": round(coverage_ratio, 6),
                "spill_days": int(res["spill_days"]),
                "empty_days": int(res["empty_days"]),
                # normalizacje zawsze zwracamy (pod wykresy WWW)
                "total_rain_l": round(rain_total, 2),
                "overflow_ratio": round(overflow_ratio, 6),
                "capacity_mm": round(capacity_mm, 3),
                "capacity_ratio": round(capacity_ratio, 6),
                "loss": round(float(loss), 6 if mode in ("norm", "cost") else 2),
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

    dx = (x_max - x_min) or 1.0
    dy = (y_max - y_min) or 1.0

    norm = []
    for p in points:
        x = (p["capacity_l"] - x_min) / dx
        y = (p["overflow_l"] - y_min) / dy
        norm.append((x, y))

    x1, y1 = norm[0]
    x2, y2 = norm[-1]

    def dist_to_line(x0: float, y0: float) -> float:
        # odległość punktu od prostej w 2D (z iloczynem wektorowym)
        num = abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1)
        den = ((y2 - y1) ** 2 + (x2 - x1) ** 2) ** 0.5 or 1.0
        return num / den

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


# -------------------------
# Endpoints
# -------------------------
@app.get("/curve")
def curve(
    area_m2: float,
    daily_use_mm: float = 2.0,
    alpha: float = 1.0,
    beta: float = 0.05,
    max_capacity_l: float = 10000.0,
    step_l: float = 100.0,
    mode: str = "raw",
    cap_mm_ref: float = 100.0,
    limit_points: int = 120,
    # strategy params
    strategy: StrategyType = "constant",
    rain_threshold_mm: float = 0.1,
    season_start_doy: int = 100,
    season_end_doy: int = 280,
    temp_source: str = "TMAX",
    t0: float = 20.0,
    k: float = 0.15,
    min_mm: float = 0.0,
    max_mm: float = 8.0,
    # cost params
    tank_cost_per_m3: float = 800.0,
    overflow_cost_per_m3: float = 0.0,
    deficit_cost_per_m3: float = 0.0,
):
    if area_m2 <= 0:
        raise HTTPException(status_code=400, detail="area_m2 musi być > 0")

    try:
        df = load_wroclaw()
        points = make_curve(
            df,
            area_m2,
            daily_use_mm,
            alpha,
            beta,
            max_capacity_l,
            step_l,
            mode,
            cap_mm_ref,
            strategy=strategy,
            rain_threshold_mm=rain_threshold_mm,
            season_start_doy=season_start_doy,
            season_end_doy=season_end_doy,
            temp_source=temp_source,
            t0=t0,
            k=k,
            min_mm=min_mm,
            max_mm=max_mm,
            tank_cost_per_m3=tank_cost_per_m3,
            overflow_cost_per_m3=overflow_cost_per_m3,
            deficit_cost_per_m3=deficit_cost_per_m3,
        )
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
        "strategy": strategy,
        "mode": mode,
        "inputs": {
            "area_m2": area_m2,
            "daily_use_mm": daily_use_mm,
            "alpha": alpha,
            "beta": beta,
            "max_capacity_l": max_capacity_l,
            "step_l": step_l,
            "cap_mm_ref": cap_mm_ref,
            "rain_threshold_mm": rain_threshold_mm,
            "season_start_doy": season_start_doy,
            "season_end_doy": season_end_doy,
            "temp_source": temp_source,
            "t0": t0,
            "k": k,
            "min_mm": min_mm,
            "max_mm": max_mm,
            "tank_cost_per_m3": tank_cost_per_m3,
            "overflow_cost_per_m3": overflow_cost_per_m3,
            "deficit_cost_per_m3": deficit_cost_per_m3,
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
    # strategy params
    strategy: StrategyType = "constant",
    rain_threshold_mm: float = 0.1,
    season_start_doy: int = 100,
    season_end_doy: int = 280,
    temp_source: str = "TMAX",
    t0: float = 20.0,
    k: float = 0.15,
    min_mm: float = 0.0,
    max_mm: float = 8.0,
    # cost params
    tank_cost_per_m3: float = 800.0,
    overflow_cost_per_m3: float = 0.0,
    deficit_cost_per_m3: float = 0.0,
):
    if area_m2 <= 0:
        raise HTTPException(status_code=400, detail="area_m2 musi być > 0")

    try:
        df = load_wroclaw()
        points = make_curve(
            df,
            area_m2,
            daily_use_mm,
            alpha,
            beta,
            max_capacity_l,
            step_l,
            mode,
            cap_mm_ref,
            strategy=strategy,
            rain_threshold_mm=rain_threshold_mm,
            season_start_doy=season_start_doy,
            season_end_doy=season_end_doy,
            temp_source=temp_source,
            t0=t0,
            k=k,
            min_mm=min_mm,
            max_mm=max_mm,
            tank_cost_per_m3=tank_cost_per_m3,
            overflow_cost_per_m3=overflow_cost_per_m3,
            deficit_cost_per_m3=deficit_cost_per_m3,
        )
        best = pick_best(points)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {
        "city": "Wrocław",
        "strategy": strategy,
        "mode": mode,
        "inputs": {
            "area_m2": area_m2,
            "daily_use_mm": daily_use_mm,
            "alpha": alpha,
            "beta": beta,
            "max_capacity_l": max_capacity_l,
            "step_l": step_l,
            "cap_mm_ref": cap_mm_ref,
            "rain_threshold_mm": rain_threshold_mm,
            "season_start_doy": season_start_doy,
            "season_end_doy": season_end_doy,
            "temp_source": temp_source,
            "t0": t0,
            "k": k,
            "min_mm": min_mm,
            "max_mm": max_mm,
            "tank_cost_per_m3": tank_cost_per_m3,
            "overflow_cost_per_m3": overflow_cost_per_m3,
            "deficit_cost_per_m3": deficit_cost_per_m3,
        },
        "best": best,
    }


@app.get("/elbow")
def elbow(
    area_m2: float,
    daily_use_mm: float = 2.0,
    max_capacity_l: float = 20000.0,
    step_l: float = 200.0,
    # strategy params (żeby „kolano” działało też dla strategii wegetacyjnych)
    strategy: StrategyType = "constant",
    rain_threshold_mm: float = 0.1,
    season_start_doy: int = 100,
    season_end_doy: int = 280,
    temp_source: str = "TMAX",
    t0: float = 20.0,
    k: float = 0.15,
    min_mm: float = 0.0,
    max_mm: float = 8.0,
):
    """
    Zwraca "kolano" krzywej capacity vs overflow (bez wag).
    """
    if area_m2 <= 0:
        raise HTTPException(status_code=400, detail="area_m2 musi być > 0")

    try:
        df = load_wroclaw()
        points = make_curve(
            df,
            area_m2,
            daily_use_mm,
            alpha=1.0,
            beta=0.0,
            max_capacity_l=max_capacity_l,
            step_l=step_l,
            mode="raw",  # elbow bazuje na capacity/overflow, nie na loss
            cap_mm_ref=100.0,
            strategy=strategy,
            rain_threshold_mm=rain_threshold_mm,
            season_start_doy=season_start_doy,
            season_end_doy=season_end_doy,
            temp_source=temp_source,
            t0=t0,
            k=k,
            min_mm=min_mm,
            max_mm=max_mm,
        )
        e = elbow_point(points)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {
        "city": "Wrocław",
        "strategy": strategy,
        "inputs": {
            "area_m2": area_m2,
            "daily_use_mm": daily_use_mm,
            "max_capacity_l": max_capacity_l,
            "step_l": step_l,
            "rain_threshold_mm": rain_threshold_mm,
            "season_start_doy": season_start_doy,
            "season_end_doy": season_end_doy,
            "temp_source": temp_source,
            "t0": t0,
            "k": k,
            "min_mm": min_mm,
            "max_mm": max_mm,
        },
        "elbow": e,
    }