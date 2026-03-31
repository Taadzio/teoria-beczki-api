from __future__ import annotations

from typing import Any, Dict, Optional


def curve_response(
    *,
    city: str,
    strategy: str,
    mode: str,
    inputs: Dict[str, Any],
    best: Dict[str, Any],
    points: list[Dict[str, Any]],
    points_total: int,
    points_returned: int,
) -> Dict[str, Any]:
    return {
        "city": city,
        "strategy": strategy,
        "mode": mode,
        "inputs": inputs,
        "best": best,
        "points": points,
        "points_total": points_total,
        "points_returned": points_returned,
    }


def recommendation_response(
    *,
    city: str,
    strategy: str,
    mode: Optional[str],
    inputs: Dict[str, Any],
    best_key: str,
    best_value: Dict[str, Any],
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "city": city,
        "strategy": strategy,
        "inputs": inputs,
        best_key: best_value,
    }
    if mode is not None:
        payload["mode"] = mode
    return payload
