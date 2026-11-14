#!/usr/bin/env python3
"""polymarket-faa-attribution-01: persistence-median forecaster with
level-dependent volatility and fat outer quantiles.

For each market, the median forecast is the last finite price at the
checkpoint (persistence). Around it, the spread widens like a random
walk, with a per-market volatility that scales with the Bernoulli
variance of the price:

    sigma = 0.0012 * sqrt(p * (1 - p) / 0.0475)   [price units per sqrt-minute]

where p is the checkpoint price. The outer quantiles (5% and 95%) are
widened by a factor of 1.5 to cover the occasional jumps these markets
make. Every quantile is clamped to [0, 1] and rounded to 6 decimals.

Non-finite prices in a market's history are ignored; if the price at
the checkpoint itself is non-finite, the last finite price before it
is used instead.

Reads every *.in case in the input directory (MAKEFUN_INPUT_DIR, else
/input) and writes one <id>.out per case to MAKEFUN_OUTPUT_DIR (else
/output). Standard library only, deterministic, no network.
"""

import json
import math
import os
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# Quantile levels, in the order each output row lists them.
LEVELS = (0.05, 0.25, 0.5, 0.75, 0.95)
# Standard normal quantiles at those levels.
Z_SCORES = (-1.644854, -0.674490, 0.0, 0.674490, 1.644854)
# Widening applied to the two outer quantiles (jump insurance).
OUTER_WIDEN = 1.5
# Base volatility scale, price units per square-root minute.
BASE_SIGMA = 0.0012
# Reference Bernoulli variance the volatility is normalised against.
REF_VAR = 0.0475
# Floor on sigma: a market parked at 0 or 1 still has a non-zero chance
# of jumping (the FAA family saw 20-point jumps in its first minutes),
# so a zero-width forecast would pay for every such move.
SIGMA_FLOOR = 0.0004


def clamp6(value: float) -> float:
    """Clamp to [0, 1] and round to 6 decimals (0, never -0)."""
    rounded = round(min(1.0, max(0.0, value)) * 1_000_000) / 1_000_000
    return rounded + 0.0


def checkpoint_price(prices):
    """Last finite price in the series, or 0.5 if none is finite."""
    for price in reversed(prices):
        if isinstance(price, (int, float)) and math.isfinite(price):
            return float(price)
    return 0.5


def market_volatility(p: float) -> float:
    """Per-market sigma: grows with p(1-p), the price's own variance,
    floored so a market parked at 0 or 1 still carries some spread."""
    variance = p * (1.0 - p)
    if variance <= 0.0:
        return SIGMA_FLOOR
    return max(SIGMA_FLOOR, BASE_SIGMA * math.sqrt(variance / REF_VAR))


def forecast_rows(p: float, horizon: int):
    """Horizon rows of five quantiles around the checkpoint price p."""
    sigma = market_volatility(p)
    rows = []
    for minute in range(1, horizon + 1):
        spread = sigma * math.sqrt(minute)
        row = []
        for level, z in zip(LEVELS, Z_SCORES):
            widen = OUTER_WIDEN if level in (0.05, 0.95) else 1.0
            row.append(clamp6(p + z * widen * spread))
        # Guard monotonicity after rounding (clamping can only help,
        # but keep the invariant explicit).
        for j in range(1, len(row)):
            if row[j] < row[j - 1]:
                row[j] = row[j - 1]
        rows.append(row)
    return rows


def forecast_case(case: dict) -> dict:
    horizon = int(case["horizon_minutes"])
    forecasts = []
    for market in case["markets"]:
        anchor = checkpoint_price(market["prices"])
        forecasts.append({"id": market["id"], "q": forecast_rows(anchor, horizon)})
    return {"forecasts": forecasts}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    case_files = sorted(INPUT_DIR.glob("*.in"))
    for case_file in case_files:
        case = json.loads(case_file.read_text(encoding="utf-8"))
        out_path = OUTPUT_DIR / f"{case_file.stem}.out"
        out_path.write_text(
            json.dumps(forecast_case(case)) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
