"""Build the JSON data blob for forecast-prototype.qmd.

Reads observed target data from ../flu-metrocast and SIMULATES forecast
trajectories from a hypothetical model (no real hub forecasts are used).
Each (location, model) gets N_SAMPLES sample paths over N_HORIZONS weeks.

Output: pages/data/forecast_prototype_data.json
"""

import csv
import json
import math
import random
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path

HUB = Path("/Users/nick/Documents/research-versioned/flu-metrocast")
OUT = Path("/Users/nick/Documents/research-versioned/metrocast-dashboard/pages/data/forecast_prototype_data.json")

# Early-season reference date. Matches the predevals-config min eval round.
REFERENCE_DATE = "2025-11-22"
SEASON_START = "2025-08-01"
PRIOR_SEASON_START = "2024-08-01"
PRIOR_SEASON_END = "2025-07-31"
HISTORY_START = "2015-01-01"

INCLUDE_LOCATIONS = {"texas", "houston", "massachusetts", "denver"}

N_SAMPLES = 100
# Simulate through end of June 2026 (~31 weeks from Nov 22).
HORIZON_END = "2026-06-27"
QUANTILES = [0.025, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.975]
RNG_SEED = 20251122

# Damping applied to recent observed log-growth before it becomes the model's
# initial drift. Recent growth alone is too aggressive as a multi-week drift
# (a one-off acceleration shouldn't extrapolate exponentially), so we shrink it
# and cap the magnitude.
GROWTH_DAMPING = 0.55
GROWTH_CAP = 0.22  # max |initial drift|, roughly ±25% week-over-week

# Per-model behavior controlling the damped-growth trajectory simulator.
#   drift_init_offset: added to the damped+capped observed growth at week 1
#                      ("flat" means ignore observed growth and start at 0)
#   peak_week:         week at which drift crosses zero (= deterministic peak time)
#   decline_slope:     how steeply drift turns negative after the peak
#   noise_sigma:       per-step Gaussian σ in log-space
MODEL_PARAMS = {
    "Ensemble": {"drift_init_offset": -0.02, "peak_week": 10, "decline_slope": 0.012, "noise_sigma": 0.10},
    "GAM":      {"drift_init_offset":  0.03, "peak_week": 12, "decline_slope": 0.010, "noise_sigma": 0.12},
    "Baseline": {"drift_init_offset": "flat", "peak_week": 8,  "decline_slope": 0.008, "noise_sigma": 0.18},
}

# Hard cap to prevent runaway exponential trajectories (in percent units).
VALUE_CEILING = 30.0
VALUE_FLOOR = 0.001

REF = date.fromisoformat(REFERENCE_DATE)
END = date.fromisoformat(HORIZON_END)
N_HORIZONS = (END - REF).days // 7


def load_locations():
    locs = []
    with open(HUB / "auxiliary-data" / "locations.csv") as f:
        for row in csv.DictReader(f):
            locs.append({
                "id": row["location"],
                "name": row["location_name"],
                "state": row["state"],
                "state_abb": row["state_abb"],
                "population": int(row["population"]) if row["population"] else None,
            })
    return locs


def load_observed():
    """Return {location_id: {target: [{date, value}, ...]}}."""
    by_loc = defaultdict(lambda: defaultdict(list))
    with open(HUB / "target-data" / "latest-data.csv") as f:
        for row in csv.DictReader(f):
            d = row["target_end_date"]
            if d < HISTORY_START:
                continue
            by_loc[row["location"]][row["target"]].append({
                "date": d,
                "value": float(row["observation"]) if row["observation"] else None,
            })
    for loc in by_loc:
        for tgt in by_loc[loc]:
            by_loc[loc][tgt].sort(key=lambda r: r["date"])
    return by_loc


def recent_log_growth(rows_through_ref, lookback_weeks=4):
    """Mean per-week log-growth over the last `lookback_weeks` observations."""
    pts = [r for r in rows_through_ref if r["value"] is not None and r["value"] > 0]
    if len(pts) < 2:
        return 0.0
    pts = pts[-(lookback_weeks + 1):]
    diffs = []
    for i in range(1, len(pts)):
        diffs.append(math.log(pts[i]["value"]) - math.log(pts[i - 1]["value"]))
    return sum(diffs) / len(diffs)


def simulate(seed_value, base_growth, params, rng):
    """Return one full-season trajectory of length N_HORIZONS (percent values).

    Damped-growth model: drift starts at (base_growth + drift_init_offset),
    decays linearly so it crosses zero at peak_week, then turns negative
    at -decline_slope per week. Gaussian noise in log-space at each step.
    """
    if params["drift_init_offset"] == "flat":
        drift0 = 0.0
    else:
        damped = base_growth * GROWTH_DAMPING
        damped = max(-GROWTH_CAP, min(GROWTH_CAP, damped))
        drift0 = damped + params["drift_init_offset"]
    peak_week = params["peak_week"]
    decline_slope = params["decline_slope"]
    sigma = params["noise_sigma"]

    log_y = math.log(seed_value)
    out = []
    for t in range(1, N_HORIZONS + 1):
        if t <= peak_week:
            drift_t = drift0 * (1.0 - t / peak_week)
        else:
            drift_t = -decline_slope * (t - peak_week)
        log_y += drift_t + rng.gauss(0.0, sigma)
        y = math.exp(log_y)
        y = max(VALUE_FLOOR, min(VALUE_CEILING, y))
        log_y = math.log(y)  # clamp the latent state too so we don't drift past ceiling forever
        out.append(y)
    return out


def horizon_dates():
    return [(REF + timedelta(days=7 * h)).isoformat() for h in range(1, N_HORIZONS + 1)]


def simulate_all_models(seed_value, base_growth, loc_id):
    """Return {model: {dates: [...], samples: [[w1, w2, ...wN], ...]}}."""
    out = {}
    for model, params in MODEL_PARAMS.items():
        rng = random.Random(hash((RNG_SEED, loc_id, model)) & 0xFFFFFFFF)
        samples = [simulate(seed_value, base_growth, params, rng) for _ in range(N_SAMPLES)]
        out[model] = {
            "target": "Flu ED visits pct",
            "dates": horizon_dates(),
            "samples": samples,
            "peak_week": params["peak_week"],
            "noise_sigma": params["noise_sigma"],
        }
    return out


def main():
    locations = load_locations()
    observed_all = load_observed()

    keep_ids = set(INCLUDE_LOCATIONS) & set(observed_all.keys())

    observed_current = {}
    observed_prior = {}
    history_for_percentiles = {}
    forecasts = {}

    for loc_id in keep_ids:
        target = "Flu ED visits pct"
        rows = observed_all[loc_id].get(target, [])
        if not rows:
            continue

        rows_through_ref = [r for r in rows if r["date"] <= REFERENCE_DATE]
        if not rows_through_ref:
            continue
        seed_row = next((r for r in reversed(rows_through_ref) if r["value"] is not None and r["value"] > 0), None)
        if seed_row is None:
            continue

        observed_current[loc_id] = [r for r in rows if SEASON_START <= r["date"] <= REFERENCE_DATE]
        observed_prior[loc_id] = [r for r in rows if PRIOR_SEASON_START <= r["date"] <= PRIOR_SEASON_END]
        history_for_percentiles[loc_id] = rows

        base_growth = recent_log_growth(rows_through_ref, lookback_weeks=4)
        forecasts[loc_id] = simulate_all_models(seed_row["value"], base_growth, loc_id)

    keep_final = set(forecasts.keys())
    locations = [l for l in locations if l["id"] in keep_final]
    for l in locations:
        l["target"] = "Flu ED visits pct"

    out_blob = {
        "reference_date": REFERENCE_DATE,
        "season_label": "2025-2026",
        "quantile_levels": QUANTILES,
        "models": list(MODEL_PARAMS.keys()),
        "horizons": list(range(1, N_HORIZONS + 1)),
        "n_samples": N_SAMPLES,
        "simulated": True,
        "locations": sorted(locations, key=lambda l: l["name"]),
        "observed_current": observed_current,
        "observed_prior": observed_prior,
        "history": history_for_percentiles,
        "forecasts": forecasts,
    }
    OUT.write_text(json.dumps(out_blob, separators=(",", ":")))
    print(f"Wrote {OUT} ({OUT.stat().st_size / 1024:.1f} KB)")
    print(f"  reference_date: {REFERENCE_DATE}")
    print(f"  locations: {[l['name'] for l in locations]}")
    for loc_id in keep_final:
        seed = forecasts[loc_id]["Ensemble"]["samples"][0]
        print(f"  {loc_id}: ensemble first sample = {[round(x, 2) for x in seed]}")


if __name__ == "__main__":
    main()
