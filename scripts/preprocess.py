#!/usr/bin/env python3
"""
Step 4 — Build the preprocessing pipeline and feature matrix.

Reads farms + weather_history + factory_metadata from MongoDB.
Outputs a feature matrix (X) and target vector (y) as parquet files,
plus a fitted sklearn ColumnTransformer saved to models/preprocessor.pkl.

Run inside ml_engine container:
    docker compose exec ml_engine python scripts/preprocess.py

Options:
    --output-dir   Where to write parquet + pkl  (default: /app/models)
    --validate     Print feature matrix sample and stats after building
"""

import os
import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from pymongo import MongoClient
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017/chaimterics")
DATA_DIR    = Path(os.getenv("DATA_DIR",    str(Path(__file__).resolve().parent.parent / "data")))
MODELS_DIR  = Path(os.getenv("MODELS_DIR", str(Path(__file__).resolve().parent.parent / "models")))

# Agronomic constants from synthetic_metadata.json
PRUNING_SUPPRESSION   = 0.45   # yield in pruning month is ~45% of normal
PRUNING_RECOVERY_1    = 0.78   # month+1 after pruning
PRUNING_RECOVERY_2    = 1.08   # month+2 (flush)
MINIBONUS_MONTHS      = {0, 1, 2, 4, 5}   # Jul=0, Aug=1, Sep=2, Nov=4, Dec=5
SEASON_MONTH_NAMES    = ["Jul","Aug","Sep","Oct","Nov","Dec","Jan","Feb","Mar","Apr","May","Jun"]

# Pruning months by season_month_idx: Jun(11) and Aug(1)
PRUNING_MONTH_IDXS    = {1, 11}


def load_metadata(db) -> dict:
    meta = db.factory_metadata.find_one({"_type": "factory_metadata"})
    if not meta:
        print("  [WARN] factory_metadata not found — using hardcoded agronomic params.")
        return {}
    return meta


def load_weather_lookup(db) -> dict:
    """
    Returns dict: {(collection_centre, season_year, month_idx): {rainfall_mm, temp_c}}
    month_idx is 0=Jul ... 11=Jun (already season-anchored in weather_history docs).
    """
    lookup = {}
    for doc in db.weather_history.find({}, {"_id": 0}):
        centre     = doc["collection_centre"]
        year       = doc["year"]
        r_offset   = doc.get("rainfall_offset", 1.0)
        rainfall   = doc.get("monthly_rainfall_mm", [None] * 12)
        temp       = doc.get("monthly_temp_c",      [None] * 12)
        for idx in range(12):
            lookup[(centre, year, idx)] = {
                "rainfall_mm": (rainfall[idx] * r_offset) if rainfall[idx] is not None else None,
                "temp_c":      temp[idx],
            }
    return lookup


def fertiliser_effect(applications: list, month_idx: int) -> tuple[float, float]:
    """
    Return (fertiliser_kg_this_month, manure_kg_this_month) applied in month_idx.
    Separate by type: CAN/NPK/DAP/etc → fertiliser; Manure/FYM → manure.
    """
    fert_kg   = 0.0
    manure_kg = 0.0
    for app in applications:
        if app.get("season_month_idx") == month_idx:
            qty  = float(app.get("quantity_kg", 0))
            kind = str(app.get("input_type", "")).upper()
            if any(k in kind for k in ("MANURE", "FYM", "COMPOST")):
                manure_kg += qty
            else:
                fert_kg   += qty
    return fert_kg, manure_kg


def detect_pruning_month(monthly_kg: list, base_yield_kgha: float, hectares: float) -> set:
    """
    Infer pruning months heuristically: any month where yield drops to
    <= 50% of expected (base_yield * ha / 12) is flagged as a pruning month.
    Returns set of season_month_idx values.
    """
    expected_monthly = (base_yield_kgha * hectares) / 12.0
    if expected_monthly <= 0:
        return set()
    pruning = set()
    for idx, kg in enumerate(monthly_kg):
        if kg is not None and kg <= expected_monthly * 0.50:
            pruning.add(idx)
    return pruning


def build_rows(db, weather_lookup: dict, meta: dict) -> list[dict]:
    """
    Flatten every farm × season × month into one row per observation.
    Returns list of dicts — one per (farm, season, month).
    """
    rows = []

    # Load base yields per factory from metadata
    factory_base_yields = {
        f["factory_code"]: f["base_yield_kg_per_ha"]
        for f in meta.get("factories", [])
    }

    farms = list(db.farms.find({}, {"_id": 0}))
    print(f"  Loaded {len(farms)} farms from MongoDB.")

    for farm in farms:
        member_no   = farm["ktda_member_no"]
        factory     = farm["factory_code"]
        centre      = farm["collection_centre"]
        hectares    = float(farm.get("hectares", 1.0))
        altitude    = float(farm.get("altitude_m", 1700))
        reg_year    = int(farm.get("registered_year", 2010))
        fairtrade   = int(farm.get("fairtrade_certified", False))
        base_yield  = factory_base_yields.get(factory, 375.0)

        seasons = farm.get("historical_seasons", [])

        for season in seasons:
            season_year  = int(season["season_year"])
            monthly_kg   = season.get("monthly_kg",   [None] * 12)
            rainfall_syn = season.get("season_rainfall_mm", [None] * 12)  # synthetic rainfall in farm doc
            fert_apps    = season.get("fertiliser_applications", [])
            years_active = max(1, season_year - reg_year)

            # Detect pruning months for this season from yield suppression
            pruned_months = detect_pruning_month(monthly_kg, base_yield, hectares)

            # Rolling 3-month yield (previous months in same season)
            kg_series = [kg if kg is not None else 0.0 for kg in monthly_kg]

            for month_idx in range(12):
                target_kg = monthly_kg[month_idx]
                if target_kg is None:
                    continue   # skip missing observations

                # ── Weather features ─────────────────────────────────────────
                wx = weather_lookup.get((centre, season_year, month_idx), {})
                # Fall back to synthetic rainfall in farm doc if Open-Meteo missing
                rain_mm = wx.get("rainfall_mm") or (
                    rainfall_syn[month_idx] if rainfall_syn and month_idx < len(rainfall_syn) else None
                )
                temp_c  = wx.get("temp_c")

                # Optimal rainfall from metadata (centre-level offset applied in weather_fetch)
                # Rainfall deficit: how far from optimum (~115mm)?
                rain_deficit = (rain_mm - 115.0) if rain_mm is not None else 0.0

                # Rolling 3-month rainfall sum (months 0..month_idx-1, capped at season start)
                rain_3mo = None
                if rain_mm is not None:
                    # Use synthetic series if open-meteo missing for prior months
                    prior_rains = []
                    for pm in range(max(0, month_idx - 2), month_idx + 1):
                        pw = weather_lookup.get((centre, season_year, pm), {})
                        pr = pw.get("rainfall_mm") or (
                            rainfall_syn[pm] if rainfall_syn and pm < len(rainfall_syn) else None
                        )
                        if pr is not None:
                            prior_rains.append(pr)
                    rain_3mo = sum(prior_rains) if prior_rains else rain_mm

                # ── Fertiliser features ───────────────────────────────────────
                fert_kg, manure_kg = fertiliser_effect(fert_apps, month_idx)
                # Lag effect: fertiliser applied in month-1 and month-2
                fert_lag1, _  = fertiliser_effect(fert_apps, month_idx - 1) if month_idx > 0 else (0, 0)
                fert_lag2, _  = fertiliser_effect(fert_apps, month_idx - 2) if month_idx > 1 else (0, 0)

                # ── Pruning features ──────────────────────────────────────────
                is_pruning_month = int(month_idx in pruned_months)
                # months_since_pruning: look back within season
                months_since_pruning = 12  # default: no pruning this season
                for pm in range(month_idx - 1, -1, -1):
                    if pm in pruned_months:
                        months_since_pruning = month_idx - pm
                        break

                # ── Rolling yield features ────────────────────────────────────
                rolling_yield_3mo = float(np.mean(kg_series[max(0, month_idx-2): month_idx+1]))
                prior_season_avg  = None
                # We'll fill this in a second pass below (needs all seasons loaded first)

                # ── Calendar features ─────────────────────────────────────────
                is_minibonus_month = int(month_idx in MINIBONUS_MONTHS)
                is_annual_bonus    = int(month_idx == 11)   # June = season close
                # Cyclic month encoding (period=12)
                season_month_sin = float(np.sin(2 * np.pi * month_idx / 12))
                season_month_cos = float(np.cos(2 * np.pi * month_idx / 12))

                rows.append({
                    # Identifiers (not features — dropped before training)
                    "ktda_member_no":       member_no,
                    "factory_code":         factory,
                    "collection_centre":    centre,
                    "season_year":          season_year,
                    "season_month_idx":     month_idx,
                    "season_month_name":    SEASON_MONTH_NAMES[month_idx],

                    # Farm-level static features
                    "hectares":             hectares,
                    "altitude_m":           altitude,
                    "years_active":         years_active,
                    "fairtrade":            fairtrade,

                    # Weather features
                    "rainfall_mm":          rain_mm if rain_mm is not None else 0.0,
                    "temp_c":               temp_c  if temp_c  is not None else 18.0,
                    "rain_deficit_mm":      rain_deficit,
                    "rain_3mo_sum":         rain_3mo if rain_3mo is not None else 0.0,

                    # Fertiliser features
                    "fert_kg":              fert_kg,
                    "manure_kg":            manure_kg,
                    "fert_lag1_kg":         fert_lag1,
                    "fert_lag2_kg":         fert_lag2,

                    # Pruning features
                    "is_pruning_month":     is_pruning_month,
                    "months_since_pruning": months_since_pruning,

                    # Season structure features
                    "is_minibonus_month":   is_minibonus_month,
                    "is_annual_bonus":      is_annual_bonus,
                    "season_month_sin":     season_month_sin,
                    "season_month_cos":     season_month_cos,

                    # Rolling yield
                    "rolling_yield_3mo":    rolling_yield_3mo,

                    # Target
                    "monthly_kg":           float(target_kg),
                })

    return rows


def add_prior_season_avg(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each (farm, season_month_idx), compute the mean yield in that same
    month across all prior seasons for that farm. Fills prior_season_avg.
    This requires all rows to be present first, hence the second pass.
    """
    df = df.copy()
    df["prior_season_avg"] = np.nan

    for (member, month_idx), grp in df.groupby(["ktda_member_no", "season_month_idx"]):
        grp_sorted = grp.sort_values("season_year")
        expanding  = grp_sorted["monthly_kg"].expanding().mean().shift(1)
        df.loc[grp_sorted.index, "prior_season_avg"] = expanding.values

    # Fill NaN for first season of each farm with the farm's overall mean
    farm_means = df.groupby("ktda_member_no")["monthly_kg"].transform("mean")
    df["prior_season_avg"] = df["prior_season_avg"].fillna(farm_means)
    return df


def build_preprocessor(numeric_cols: list, cat_cols: list) -> ColumnTransformer:
    return ColumnTransformer(transformers=[
        ("num", StandardScaler(), numeric_cols),
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), cat_cols),
    ], remainder="drop")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=str(MODELS_DIR))
    parser.add_argument("--validate",   action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Connecting to {MONGODB_URI} ...")
    client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=5000)
    db = client.get_default_database()
    db.command("ping")
    print("Connected.\n")

    print("Loading factory metadata ...")
    meta = load_metadata(db)

    print("Loading weather history ...")
    weather_lookup = load_weather_lookup(db)
    print(f"  {len(weather_lookup)} (centre, year, month) weather records loaded.")

    print("\nBuilding feature rows ...")
    rows = build_rows(db, weather_lookup, meta)
    print(f"  {len(rows)} rows built.")

    df = pd.DataFrame(rows)
    print("\nAdding prior season averages ...")
    df = add_prior_season_avg(df)

    # ── Feature columns ───────────────────────────────────────────────────────
    numeric_features = [
        "hectares", "altitude_m", "years_active", "fairtrade",
        "rainfall_mm", "temp_c", "rain_deficit_mm", "rain_3mo_sum",
        "fert_kg", "manure_kg", "fert_lag1_kg", "fert_lag2_kg",
        "is_pruning_month", "months_since_pruning",
        "is_minibonus_month", "is_annual_bonus",
        "rolling_yield_3mo", "prior_season_avg",
        "season_month_idx",
        "season_month_sin", "season_month_cos",
    ]
    categorical_features = [
        "factory_code", "collection_centre",
    ]

    X = df[numeric_features + categorical_features]
    y = df["monthly_kg"]

    # ── Fit preprocessor ──────────────────────────────────────────────────────
    print("\nFitting sklearn ColumnTransformer ...")
    preprocessor = build_preprocessor(numeric_features, categorical_features)
    preprocessor.fit(X)

    # ── Save artifacts ────────────────────────────────────────────────────────
    X_path    = out_dir / "X_features.parquet"
    y_path    = out_dir / "y_target.parquet"
    meta_path = out_dir / "feature_matrix_meta.parquet"
    pre_path  = out_dir / "preprocessor.pkl"
    id_path   = out_dir / "row_ids.parquet"

    id_cols = ["ktda_member_no", "factory_code", "collection_centre",
               "season_year", "season_month_idx", "season_month_name"]

    df[id_cols].to_parquet(id_path,   index=False)
    X.to_parquet(X_path,              index=False)
    y.to_frame().to_parquet(y_path,   index=False)
    joblib.dump(preprocessor, pre_path)

    print(f"\n-- Saved --")
    print(f"  {X_path}      ({X.shape[0]} rows x {X.shape[1]} features)")
    print(f"  {y_path}")
    print(f"  {id_path}")
    print(f"  {pre_path}")

    if args.validate:
        print(f"\n-- Feature matrix sample (5 rows) --")
        print(df[id_cols[:3] + ["season_year","season_month_name"] + numeric_features[:6] + ["monthly_kg"]].head().to_string())
        print(f"\n-- Target stats --")
        print(y.describe().to_string())
        print(f"\n-- Missing values --")
        missing = X.isnull().sum()
        missing = missing[missing > 0]
        if missing.empty:
            print("  None.")
        else:
            print(missing.to_string())

    print("\nStep 4 done. Run train_xgboost.py next.")


if __name__ == "__main__":
    main()