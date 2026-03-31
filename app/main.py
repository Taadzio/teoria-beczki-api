from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .config import ALLOWED_ORIGINS, DEFAULT_CITY, DEFAULT_YEARS_COUNT
from .data import load_wroclaw
from .schemas import curve_response, recommendation_response
from .simulation import (
    StrategyType,
    dataset_stats,
    elbow_point,
    make_curve,
    pick_best,
)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _curve_inputs(**kwargs):
    return kwargs


@app.get("/")
def read_root():
    return {"message": "Teoria Beczki dziala", "years_count": DEFAULT_YEARS_COUNT}


@app.get("/ping")
def ping():
    return {"status": "ok"}


@app.get("/columns")
def columns():
    df = load_wroclaw()
    return {"columns": list(df.columns)}


@app.get("/stats")
def stats(
    rain_threshold_mm: float = 0.1,
    season_start_doy: int = 100,
    season_end_doy: int = 280,
    min_temp_block_c: float = 4.0,
):
    try:
        df = load_wroclaw()
        summary = dataset_stats(
            df,
            rain_threshold_mm=rain_threshold_mm,
            season_start_doy=season_start_doy,
            season_end_doy=season_end_doy,
            min_temp_block_c=min_temp_block_c,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {
        "city": DEFAULT_CITY,
        "inputs": {
            "rain_threshold_mm": rain_threshold_mm,
            "season_start_doy": season_start_doy,
            "season_end_doy": season_end_doy,
            "min_temp_block_c": min_temp_block_c,
        },
        "stats": summary,
    }


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
    strategy: StrategyType = "constant",
    rain_threshold_mm: float = 0.1,
    season_start_doy: int = 100,
    season_end_doy: int = 280,
    temp_source: str = "TMAX",
    t0: float = 20.0,
    k: float = 0.15,
    min_mm: float = 0.0,
    max_mm: float = 8.0,
    tank_cost_per_m3: float = 800.0,
    overflow_cost_per_m3: float = 0.0,
    deficit_cost_per_m3: float = 0.0,
    normalize_annual_demand: bool = False,
    block_below_min_temp: bool = False,
    min_temp_block_c: float = 4.0,
):
    if area_m2 <= 0:
        raise HTTPException(status_code=400, detail="area_m2 musi byc > 0")

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
            normalize_annual_demand=normalize_annual_demand,
            block_below_min_temp=block_below_min_temp,
            min_temp_block_c=min_temp_block_c,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    best = pick_best(points)

    if limit_points and len(points) > limit_points:
        stride = max(1, len(points) // limit_points)
        sampled_points = points[::stride]
        if best not in sampled_points:
            sampled_points.append(best)
        points_out = sorted(sampled_points, key=lambda point: point["capacity_l"])
    else:
        points_out = points

    inputs = _curve_inputs(
        area_m2=area_m2,
        daily_use_mm=daily_use_mm,
        alpha=alpha,
        beta=beta,
        max_capacity_l=max_capacity_l,
        step_l=step_l,
        cap_mm_ref=cap_mm_ref,
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
        normalize_annual_demand=normalize_annual_demand,
        block_below_min_temp=block_below_min_temp,
        min_temp_block_c=min_temp_block_c,
    )

    return curve_response(
        city=DEFAULT_CITY,
        strategy=strategy,
        mode=mode,
        inputs=inputs,
        best=best,
        points=points_out,
        points_total=len(points),
        points_returned=len(points_out),
    )


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
    strategy: StrategyType = "constant",
    rain_threshold_mm: float = 0.1,
    season_start_doy: int = 100,
    season_end_doy: int = 280,
    temp_source: str = "TMAX",
    t0: float = 20.0,
    k: float = 0.15,
    min_mm: float = 0.0,
    max_mm: float = 8.0,
    tank_cost_per_m3: float = 800.0,
    overflow_cost_per_m3: float = 0.0,
    deficit_cost_per_m3: float = 0.0,
    normalize_annual_demand: bool = False,
    block_below_min_temp: bool = False,
    min_temp_block_c: float = 4.0,
):
    if area_m2 <= 0:
        raise HTTPException(status_code=400, detail="area_m2 musi byc > 0")

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
            normalize_annual_demand=normalize_annual_demand,
            block_below_min_temp=block_below_min_temp,
            min_temp_block_c=min_temp_block_c,
        )
        best = pick_best(points)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    inputs = _curve_inputs(
        area_m2=area_m2,
        daily_use_mm=daily_use_mm,
        alpha=alpha,
        beta=beta,
        max_capacity_l=max_capacity_l,
        step_l=step_l,
        cap_mm_ref=cap_mm_ref,
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
        normalize_annual_demand=normalize_annual_demand,
        block_below_min_temp=block_below_min_temp,
        min_temp_block_c=min_temp_block_c,
    )

    return recommendation_response(
        city=DEFAULT_CITY,
        strategy=strategy,
        mode=mode,
        inputs=inputs,
        best_key="best",
        best_value=best,
    )


@app.get("/elbow")
def elbow(
    area_m2: float,
    daily_use_mm: float = 2.0,
    max_capacity_l: float = 20000.0,
    step_l: float = 200.0,
    strategy: StrategyType = "constant",
    rain_threshold_mm: float = 0.1,
    season_start_doy: int = 100,
    season_end_doy: int = 280,
    temp_source: str = "TMAX",
    t0: float = 20.0,
    k: float = 0.15,
    min_mm: float = 0.0,
    max_mm: float = 8.0,
    normalize_annual_demand: bool = False,
    block_below_min_temp: bool = False,
    min_temp_block_c: float = 4.0,
):
    if area_m2 <= 0:
        raise HTTPException(status_code=400, detail="area_m2 musi byc > 0")

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
            mode="raw",
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
            normalize_annual_demand=normalize_annual_demand,
            block_below_min_temp=block_below_min_temp,
            min_temp_block_c=min_temp_block_c,
        )
        elbow_result = elbow_point(points)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    inputs = _curve_inputs(
        area_m2=area_m2,
        daily_use_mm=daily_use_mm,
        max_capacity_l=max_capacity_l,
        step_l=step_l,
        rain_threshold_mm=rain_threshold_mm,
        season_start_doy=season_start_doy,
        season_end_doy=season_end_doy,
        temp_source=temp_source,
        t0=t0,
        k=k,
        min_mm=min_mm,
        max_mm=max_mm,
        normalize_annual_demand=normalize_annual_demand,
        block_below_min_temp=block_below_min_temp,
        min_temp_block_c=min_temp_block_c,
    )

    return recommendation_response(
        city=DEFAULT_CITY,
        strategy=strategy,
        mode=None,
        inputs=inputs,
        best_key="elbow",
        best_value=elbow_result,
    )
