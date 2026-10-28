# /// script
# requires-python = ">=3.10"
# dependencies = ["pandas", "pyarrow"]
# ///
"""Build the model-card data for forecast-prototype.qmd from flu-metrocast hub output.

Reads observed target data and the sample trajectories submitted by the models in MODELS for
one round (by default the most recent round with a file from the first model), and writes:

  <out>/options.json                 reference date, season, models, and location list
  <out>/locations/<location>.json    observations and forecast trajectories for one location

Run with `uv run scripts/build_card_data.py`; see --help for options.
"""

import argparse
import json
import shutil
from datetime import date
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]

TARGET = "Flu ED visits pct"
# Models to include, as {model_id: label shown on the page}. The cards summarize the first one.
MODELS = {"epiENGAGE-ensemble_long": "Ensemble"}
# Start of a season (Aug 1), so the oldest season shown on the chart is complete.
HISTORY_START = "2014-08-01"


def season_of(d):
    """Flu season (Aug 1 -> Jul 31), identified by the year it starts in."""
    d = date.fromisoformat(d)
    return d.year if d.month >= 8 else d.year - 1


def latest_round(hub, model_id):
    files = sorted((hub / "model-output" / model_id).glob(f"????-??-??-{model_id}.*"))
    if not files:
        raise SystemExit(f"No model output found for {model_id}")
    return files[-1].name[:10]


def read_model_output(hub, model_id, reference_date):
    files = list((hub / "model-output" / model_id).glob(f"{reference_date}-{model_id}.*"))
    if not files:
        return None
    f = files[0]
    df = pd.read_parquet(f) if f.suffix == ".parquet" else pd.read_csv(f, dtype={"output_type_id": str})
    df = df[(df["output_type"] == "sample") & (df["target"] == TARGET)].copy()
    df["target_end_date"] = df["target_end_date"].astype(str).str[:10]
    return df


def trajectories(df):
    """Return {location: {dates, samples}} with one complete trajectory per sample id."""
    out = {}
    for loc, g in df.groupby("location"):
        wide = g.pivot(index="output_type_id", columns="target_end_date", values="value").dropna()
        if wide.empty:
            continue
        wide = wide[sorted(wide.columns)]
        out[loc] = {
            "dates": list(wide.columns),
            "samples": [[round(float(v), 4) for v in row] for row in wide.to_numpy()],
        }
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--hub", type=Path, default=REPO.parent / "flu-metrocast", help="path to the hub repo")
    parser.add_argument("--out", type=Path, default=REPO / "pages" / "data" / "cards", help="output directory")
    parser.add_argument("--reference-date", help="round to use (default: latest round of the first model)")
    args = parser.parse_args()

    primary = next(iter(MODELS))
    ref = args.reference_date or latest_round(args.hub, primary)
    ref_season = season_of(ref)

    locs = pd.read_csv(args.hub / "auxiliary-data" / "locations.csv").set_index("location")
    obs = pd.read_csv(args.hub / "target-data" / "latest-data.csv")
    obs = obs[(obs["target"] == TARGET) & obs["observation"].notna()
              & (obs["target_end_date"] >= HISTORY_START) & (obs["target_end_date"] <= ref)]

    forecasts = {}
    for model_id, label in MODELS.items():
        df = read_model_output(args.hub, model_id, ref)
        if df is None:
            print(f"  {model_id}: no output for {ref}, skipping")
            continue
        for loc, traj in trajectories(df).items():
            forecasts.setdefault(loc, {})[label] = {"model_id": model_id, **traj}
    if not forecasts:
        raise SystemExit(f"No forecasts found for {ref}")

    # Only locations the cards can describe: forecasts from the primary model plus observations.
    keep = sorted(loc for loc, f in forecasts.items()
                  if MODELS[primary] in f and loc in locs.index and (obs["location"] == loc).any())

    # Clear only what this script writes, so locations dropped from a round don't linger.
    shutil.rmtree(args.out / "locations", ignore_errors=True)
    (args.out / "locations").mkdir(parents=True)
    for loc in keep:
        rows = obs[obs["location"] == loc].sort_values("target_end_date")
        blob = {
            "location": loc,
            "observed": [{"date": d, "value": float(v)} for d, v in zip(rows["target_end_date"], rows["observation"])],
            "forecasts": forecasts[loc],
        }
        (args.out / "locations" / f"{loc}.json").write_text(json.dumps(blob, separators=(",", ":")))

    options = {
        "reference_date": ref,
        "season": ref_season,
        "season_label": f"{ref_season}-{ref_season + 1}",
        "target": TARGET,
        "models": [label for label in MODELS.values()],
        "locations": sorted(
            ({"id": loc, "name": locs.at[loc, "location_name"]} for loc in keep), key=lambda l: l["name"]
        ),
    }
    (args.out / "options.json").write_text(json.dumps(options, indent=2))

    size = sum(f.stat().st_size for f in args.out.rglob("*.json")) / 1024
    print(f"Wrote {args.out}: {len(keep)} locations, reference date {ref} ({size:.0f} KB)")


if __name__ == "__main__":
    main()
