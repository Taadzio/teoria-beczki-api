from __future__ import annotations

from typing import Any, Dict, List, Literal

import numpy as np
import pandas as pd

StrategyType = Literal["constant", "no_rain", "seasonal", "temp_seasonal"]


def _calendar_arrays(
    df: pd.DataFrame,
    *,
    rain_threshold_mm: float,
    season_start_doy: int,
    season_end_doy: int,
    temp_source: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rain_mm = df["SMDB"].to_numpy(dtype=np.float64)

    dt = pd.to_datetime(
        dict(year=df["ROK"].astype(int), month=df["MC"].astype(int), day=df["DZ"].astype(int)),
        errors="coerce",
    )
    doy = dt.dt.dayofyear.to_numpy(dtype=np.int32)
    years = dt.dt.year.to_numpy(dtype=np.int32)
    in_season = (doy >= season_start_doy) & (doy <= season_end_doy)
    rainy_days = rain_mm >= rain_threshold_mm
    tmin = df["TMIN"].to_numpy(dtype=np.float64)

    if temp_source == "TAVG":
        tmax = df["TMAX"].to_numpy(dtype=np.float64)
        temp = (tmax + tmin) / 2.0
        temp[np.isnan(tmax) | np.isnan(tmin)] = np.nan
    else:
        temp = df[temp_source].to_numpy(dtype=np.float64)

    return rain_mm, temp, in_season, rainy_days, years, tmin


def dataset_stats(
    df: pd.DataFrame,
    *,
    rain_threshold_mm: float = 0.1,
    season_start_doy: int = 100,
    season_end_doy: int = 280,
    min_temp_block_c: float = 4.0,
) -> Dict[str, Any]:
    rain_mm, _temp, in_season, rainy_days, years, tmin = _calendar_arrays(
        df,
        rain_threshold_mm=rain_threshold_mm,
        season_start_doy=season_start_doy,
        season_end_doy=season_end_doy,
        temp_source="TMAX",
    )

    years_count = int(len(np.unique(years))) or 1
    days_total = int(len(rain_mm))

    avg_days_per_year = days_total / years_count
    avg_annual_rain_mm = float(rain_mm.sum()) / years_count
    avg_rainy_days_per_year = float(rainy_days.sum()) / years_count
    avg_non_rain_days_per_year = float((~rainy_days).sum()) / years_count
    avg_season_days_per_year = float(in_season.sum()) / years_count
    avg_rainy_days_in_season_per_year = float((rainy_days & in_season).sum()) / years_count
    avg_eligible_days_seasonal_per_year = float(((~rainy_days) & in_season).sum()) / years_count
    cold_days = tmin < min_temp_block_c
    avg_cold_days_per_year = float(cold_days.sum()) / years_count
    avg_cold_days_in_season_per_year = float((cold_days & in_season).sum()) / years_count
    avg_non_rain_warm_days_per_year = float((~rainy_days & ~cold_days).sum()) / years_count
    avg_non_rain_warm_days_in_season_per_year = float((~rainy_days & in_season & ~cold_days).sum()) / years_count

    def safe_div(numerator: float, denominator: float) -> float:
        return numerator / denominator if denominator > 0 else 0.0

    annual_target_mm = avg_annual_rain_mm

    strategy_defaults = {
        "constant": {
            "eligible_days_per_year": avg_days_per_year,
            "suggested_daily_use_mm": safe_div(annual_target_mm, avg_days_per_year),
        },
        "no_rain": {
            "eligible_days_per_year": avg_non_rain_days_per_year,
            "suggested_daily_use_mm": safe_div(annual_target_mm, avg_non_rain_days_per_year),
        },
        "seasonal": {
            "eligible_days_per_year": avg_eligible_days_seasonal_per_year,
            "suggested_daily_use_mm": safe_div(annual_target_mm, avg_eligible_days_seasonal_per_year),
        },
        "temp_seasonal": {
            "eligible_days_per_year": avg_eligible_days_seasonal_per_year,
            "suggested_daily_use_mm": safe_div(annual_target_mm, avg_eligible_days_seasonal_per_year),
        },
    }

    return {
        "years_count": years_count,
        "avg_annual_rain_mm": round(avg_annual_rain_mm, 4),
        "avg_days_per_year": round(avg_days_per_year, 4),
        "avg_rainy_days_per_year": round(avg_rainy_days_per_year, 4),
        "avg_non_rain_days_per_year": round(avg_non_rain_days_per_year, 4),
        "avg_cold_days_per_year": round(avg_cold_days_per_year, 4),
        "avg_non_rain_warm_days_per_year": round(avg_non_rain_warm_days_per_year, 4),
        "avg_season_days_per_year": round(avg_season_days_per_year, 4),
        "avg_rainy_days_in_season_per_year": round(avg_rainy_days_in_season_per_year, 4),
        "avg_eligible_days_seasonal_per_year": round(avg_eligible_days_seasonal_per_year, 4),
        "avg_cold_days_in_season_per_year": round(avg_cold_days_in_season_per_year, 4),
        "avg_non_rain_warm_days_in_season_per_year": round(avg_non_rain_warm_days_in_season_per_year, 4),
        "min_temp_block_c": min_temp_block_c,
        "strategy_defaults": {
            key: {
                "eligible_days_per_year": round(value["eligible_days_per_year"], 4),
                "suggested_daily_use_mm": round(value["suggested_daily_use_mm"], 4),
            }
            for key, value in strategy_defaults.items()
        },
    }


def effective_daily_use_mm(
    df: pd.DataFrame,
    *,
    daily_use_mm: float,
    strategy: StrategyType,
    normalize_annual_demand: bool,
    rain_threshold_mm: float = 0.1,
    season_start_doy: int = 100,
    season_end_doy: int = 280,
    block_below_min_temp: bool = False,
    min_temp_block_c: float = 4.0,
) -> float:
    base_mm = max(0.0, float(daily_use_mm))
    if not normalize_annual_demand or strategy == "constant":
        return base_mm

    stats = dataset_stats(
        df,
        rain_threshold_mm=rain_threshold_mm,
        season_start_doy=season_start_doy,
        season_end_doy=season_end_doy,
        min_temp_block_c=min_temp_block_c,
    )

    annual_target_mm = base_mm * stats["avg_days_per_year"]
    eligible_days = stats["strategy_defaults"][strategy]["eligible_days_per_year"]
    if block_below_min_temp and strategy == "no_rain":
        eligible_days = stats["avg_non_rain_warm_days_per_year"]
    elif block_below_min_temp and strategy in ("seasonal", "temp_seasonal"):
        eligible_days = stats["avg_non_rain_warm_days_in_season_per_year"]
    if eligible_days <= 0:
        return 0.0

    return annual_target_mm / eligible_days


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
    block_below_min_temp: bool = False,
    min_temp_block_c: float = 4.0,
) -> dict:
    rain_mm, temp, in_season, rainy_days, _years, tmin = _calendar_arrays(
        df,
        rain_threshold_mm=rain_threshold_mm,
        season_start_doy=season_start_doy,
        season_end_doy=season_end_doy,
        temp_source=temp_source,
    )
    cold_days = tmin < min_temp_block_c

    storage = 0.0
    overflow_total = 0.0
    used_total = 0.0
    demand_total = 0.0
    deficit_total = 0.0
    spill_days = 0
    empty_days = 0

    area = float(area_m2)
    capacity = float(capacity_l)
    base = max(0.0, float(base_mm))

    for index in range(rain_mm.shape[0]):
        inflow_l = rain_mm[index] * area
        storage += inflow_l

        if storage > capacity:
            overflow_total += storage - capacity
            storage = capacity
            spill_days += 1

        use_mm = base

        if strategy == "no_rain":
            if rainy_days[index] or (block_below_min_temp and cold_days[index]):
                use_mm = 0.0
        elif strategy == "seasonal":
            if (not in_season[index]) or rainy_days[index] or (block_below_min_temp and cold_days[index]):
                use_mm = 0.0
        elif strategy == "temp_seasonal":
            if (not in_season[index]) or rainy_days[index] or (block_below_min_temp and cold_days[index]):
                use_mm = 0.0
            else:
                temperature = temp[index]
                if np.isnan(temperature):
                    use_mm = base
                else:
                    use_mm = base + k * (float(temperature) - t0)
                    if use_mm < min_mm:
                        use_mm = min_mm
                    elif use_mm > max_mm:
                        use_mm = max_mm

        daily_demand_l = max(0.0, use_mm) * area
        demand_total += daily_demand_l

        taken = storage if storage < daily_demand_l else daily_demand_l
        used_total += taken
        storage -= taken

        if daily_demand_l > taken:
            deficit_total += daily_demand_l - taken
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


def total_rain_l(df: pd.DataFrame, area_m2: float) -> float:
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
) -> List[Dict[str, Any]]:
    if step_l <= 0 or max_capacity_l <= 0:
        raise ValueError("step_l i max_capacity_l musza byc > 0")

    if mode not in ("raw", "norm", "cost"):
        raise ValueError("mode musi byc 'raw', 'norm' albo 'cost'")

    if mode == "norm" and cap_mm_ref <= 0:
        raise ValueError("cap_mm_ref musi byc > 0 dla trybu norm")

    step_l_i = int(step_l)
    max_capacity_i = int(max_capacity_l)
    rain_total = total_rain_l(df, area_m2) or 1.0

    effective_use_mm = effective_daily_use_mm(
        df,
        daily_use_mm=daily_use_mm,
        strategy=strategy,
        normalize_annual_demand=normalize_annual_demand,
        block_below_min_temp=block_below_min_temp,
        min_temp_block_c=min_temp_block_c,
        rain_threshold_mm=rain_threshold_mm,
        season_start_doy=season_start_doy,
        season_end_doy=season_end_doy,
    )

    points: List[Dict[str, Any]] = []

    for cap in range(step_l_i, max_capacity_i + 1, step_l_i):
        capacity_l = float(cap)

        result = simulate_strategy(
            df,
            area_m2,
            capacity_l,
            strategy=strategy,
            base_mm=effective_use_mm,
            rain_threshold_mm=rain_threshold_mm,
            season_start_doy=season_start_doy,
            season_end_doy=season_end_doy,
            temp_source=temp_source,
            t0=t0,
            k=k,
            min_mm=min_mm,
            max_mm=max_mm,
            block_below_min_temp=block_below_min_temp,
            min_temp_block_c=min_temp_block_c,
        )

        overflow_l = float(result["overflow_l"])
        used_l = float(result["used_l"])
        demand_l = float(result["demand_l"])
        deficit_l = float(result["deficit_l"])
        coverage_ratio = float(result["coverage_ratio"])

        overflow_ratio = overflow_l / rain_total
        capacity_mm = capacity_l / area_m2
        capacity_ratio = (capacity_mm / cap_mm_ref) if cap_mm_ref > 0 else 0.0

        if mode == "raw":
            loss = alpha * overflow_l + beta * capacity_l
        elif mode == "norm":
            loss = alpha * overflow_ratio + beta * capacity_ratio
        else:
            capacity_m3 = capacity_l / 1000.0
            overflow_m3 = overflow_l / 1000.0
            deficit_m3 = deficit_l / 1000.0
            loss = (
                tank_cost_per_m3 * capacity_m3
                + overflow_cost_per_m3 * overflow_m3
                + deficit_cost_per_m3 * deficit_m3
            )

        points.append(
            {
                "capacity_l": capacity_l,
                "capacity_m3": round(capacity_l / 1000.0, 3),
                "overflow_l": round(overflow_l, 2),
                "used_l": round(used_l, 2),
                "demand_l": round(demand_l, 2),
                "deficit_l": round(deficit_l, 2),
                "coverage_ratio": round(coverage_ratio, 6),
                "spill_days": int(result["spill_days"]),
                "empty_days": int(result["empty_days"]),
                "total_rain_l": round(rain_total, 2),
                "overflow_ratio": round(overflow_ratio, 6),
                "capacity_mm": round(capacity_mm, 3),
                "capacity_ratio": round(capacity_ratio, 6),
                "loss": round(float(loss), 6 if mode in ("norm", "cost") else 2),
                "effective_daily_use_mm": round(effective_use_mm, 6),
                "block_below_min_temp": block_below_min_temp,
                "min_temp_block_c": min_temp_block_c,
            }
        )

    return points


def pick_best(points: List[Dict[str, Any]]) -> Dict[str, Any]:
    return min(points, key=lambda point: point["loss"])


def elbow_point(points: List[Dict[str, Any]]) -> Dict[str, Any]:
    if len(points) < 3:
        return points[0]

    xs = [point["capacity_l"] for point in points]
    ys = [point["overflow_l"] for point in points]

    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    dx = (x_max - x_min) or 1.0
    dy = (y_max - y_min) or 1.0

    normalized = []
    for point in points:
        x = (point["capacity_l"] - x_min) / dx
        y = (point["overflow_l"] - y_min) / dy
        normalized.append((x, y))

    x1, y1 = normalized[0]
    x2, y2 = normalized[-1]

    def dist_to_line(x0: float, y0: float) -> float:
        numerator = abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1)
        denominator = ((y2 - y1) ** 2 + (x2 - x1) ** 2) ** 0.5 or 1.0
        return numerator / denominator

    best_index = 0
    best_distance = -1.0

    for index, (x0, y0) in enumerate(normalized):
        distance = dist_to_line(x0, y0)
        if distance > best_distance:
            best_distance = distance
            best_index = index

    output = dict(points[best_index])
    output["elbow_distance_norm"] = round(best_distance, 6)
    return output
