#!/usr/bin/env python3
"""
Step 6a — Rule engine: converts ML model outputs into structured JSON recommendations.

Takes XGBoost prediction + feature importances + SARIMA forecast + pricing forecast
and produces a structured recommendation object ready for the Ollama LLM narrative layer.

This is pure Python — no ML inference happens here. The rule engine is deterministic
and auditable. The LLM is only called in the next step (test_ollama.py).

Can be imported as a module by the Flask API, or run standalone for testing:
    docker compose exec ml_engine python scripts/rule_engine.py --member KTD-13033
"""

import os
import json
import argparse
from pathlib import Path
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
import joblib
from pymongo import MongoClient

MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017/chaimterics")
MODELS_DIR  = Path(os.getenv("MODELS_DIR", str(Path(__file__).resolve().parent.parent / "models")))
SARIMA_DIR  = MODELS_DIR / "sarima"
PRICING_DIR = SARIMA_DIR / "pricing"

SEASON_MONTH_NAMES = ["Jul","Aug","Sep","Oct","Nov","Dec","Jan","Feb","Mar","Apr","May","Jun"]
MINIBONUS_MONTHS   = {0, 1, 2, 4, 5}

# Default thresholds (used if factory_metadata not available)
DEFAULT_THRESHOLDS = {
    "yield_gap_warn_pct":   15.0,
    "yield_gap_alert_pct":  30.0,
    "forecast_drop_pct":    20.0,
    "pricing_rise_pct":     5.0,
    "pricing_drop_pct":     5.0,
    "gnn_anomaly_high":     0.75,
    "gnn_anomaly_medium":   0.50,
}


def load_thresholds(factory_code: str) -> dict:
    """Load rule thresholds from factory_metadata, with defaults as fallback."""
    try:
        from pymongo import MongoClient
        client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=3000)
        db = client.get_default_database()
        meta = db.factory_metadata.find_one({"_type": "factory_metadata"}, {"_id": 0})
        if meta:
            # Check for factory-specific thresholds
            for f in meta.get("factories", []):
                if f["factory_code"] == factory_code:
                    thresholds = f.get("rule_thresholds", {})
                    return {**DEFAULT_THRESHOLDS, **thresholds}
    except Exception:
        pass
    return DEFAULT_THRESHOLDS


@dataclass
class Recommendation:
    priority:    str    # "high" | "medium" | "low"
    category:   str    # "yield" | "input" | "pricing" | "anomaly"
    title:      str
    detail:     str
    action:     str


def load_xgb_artifacts() -> tuple:
    """Load XGBoost model, preprocessor, and feature importances."""
    model       = joblib.load(MODELS_DIR / "xgb_yield.pkl")
    preprocessor = joblib.load(MODELS_DIR / "preprocessor.pkl")
    importances = pd.read_parquet(MODELS_DIR / "xgb_feature_importances.parquet")
    return model, preprocessor, importances


def load_farm_sarima_meta(member_no: str, factory: str) -> dict | None:
    meta_path = SARIMA_DIR / factory / f"{member_no}_meta.json"
    if not meta_path.exists():
        return None
    with open(meta_path) as f:
        return json.load(f)


def load_pricing_meta(factory: str) -> dict:
    result = {}
    for label in ("monthly_rate", "minibonus_rate", "annual_bonus"):
        path = PRICING_DIR / f"{factory}_{label}_meta.json"
        if path.exists():
            with open(path) as f:
                result[label] = json.load(f)
    return result


def get_current_season_summary(farm: dict) -> dict:
    """
    Returns the most recent season's data as a summary dict.
    For Phase 1 (no live daily data yet), this is the last historical season.
    """
    seasons = sorted(farm.get("historical_seasons", []), key=lambda s: s["season_year"])
    if not seasons:
        return {}
    latest = seasons[-1]
    kg     = latest.get("monthly_kg", [])
    valid  = [k for k in kg if k is not None]
    return {
        "season_year":     latest["season_year"],
        "total_kg":        sum(valid),
        "months_complete": len(valid),
        "monthly_kg":      kg,
        "last_month_kg":   valid[-1] if valid else None,
        "season_avg_kg":   float(np.mean(valid)) if valid else None,
        "yearly_bonus":    latest.get("yearly_bonus", 0),
    }


def compute_xgb_prediction(farm: dict, model, preprocessor,
                             importances: pd.DataFrame, current_month_idx: int = 6) -> dict:
    """
    Run XGBoost inference for a single farm at a given month.
    Returns prediction + top contributing features.
    In Phase 1 we predict for the next upcoming month (current_month_idx).
    """
    try:
        X = pd.read_parquet(MODELS_DIR / "X_features.parquet")
        ids = pd.read_parquet(MODELS_DIR / "row_ids.parquet")

        # Filter to this farm's rows
        mask = ids["ktda_member_no"] == farm["ktda_member_no"]
        if not mask.any():
            return {"error": "farm not in feature matrix"}

        farm_X   = X[mask]
        farm_ids = ids[mask]

        # Get the row closest to current_month_idx in the latest season
        latest_season = farm_ids["season_year"].max()
        month_mask = (farm_ids["season_year"] == latest_season) & \
                     (farm_ids["season_month_idx"] == current_month_idx)

        if not month_mask.any():
            # Fall back to any row from latest season
            month_mask = farm_ids["season_year"] == latest_season

        X_row     = farm_X[month_mask].iloc[[-1]]
        X_scaled  = preprocessor.transform(X_row)
        pred_kg   = float(model.predict(X_scaled)[0])

        # Top 5 feature importances for this farm
        top_features = importances.head(5)[["feature", "importance"]].to_dict("records")

        return {
            "predicted_kg":  max(0.0, pred_kg),
            "month_idx":     current_month_idx,
            "month_name":    SEASON_MONTH_NAMES[current_month_idx],
            "top_features":  top_features,
        }
    except Exception as e:
        return {"error": str(e)[:200]}


def build_recommendations(farm: dict, xgb_result: dict, sarima_meta: dict | None,
                           pricing_meta: dict, importances: pd.DataFrame) -> list[Recommendation]:
    recs = []
    current = get_current_season_summary(farm)
    member_no = farm["ktda_member_no"]
    ha        = float(farm.get("hectares", 1.0))
    factory   = farm["factory_code"]

    # Load configurable thresholds
    thresholds = load_thresholds(factory)
    YIELD_GAP_WARN_PCT   = thresholds["yield_gap_warn_pct"]
    YIELD_GAP_ALERT_PCT  = thresholds["yield_gap_alert_pct"]
    FORECAST_DROP_PCT    = thresholds["forecast_drop_pct"]
    GNN_ANOMALY_HIGH     = thresholds["gnn_anomaly_high"]
    GNN_ANOMALY_MEDIUM   = thresholds["gnn_anomaly_medium"]

    #  1. Yield gap check
    if "predicted_kg" in xgb_result and current.get("season_avg_kg"):
        predicted = xgb_result["predicted_kg"]
        actual    = current["season_avg_kg"]
        gap_pct   = (predicted - actual) / predicted * 100 if predicted > 0 else 0

        if gap_pct > YIELD_GAP_ALERT_PCT:
            recs.append(Recommendation(
                priority = "high",
                category = "yield",
                title    = "Yield significantly below model expectation",
                detail   = f"Current season average is {actual:.0f}kg/month against "
                           f"model expectation of {predicted:.0f}kg/month — "
                           f"a gap of {gap_pct:.0f}%.",
                action   = "Review fertiliser schedule, check for pest pressure or soil pH issues. "
                           "Compare with neighbouring farms in the same collection centre.",
            ))
        elif gap_pct > YIELD_GAP_WARN_PCT:
            recs.append(Recommendation(
                priority = "medium",
                category = "yield",
                title    = "Yield slightly below model expectation",
                detail   = f"Current average {actual:.0f}kg/month vs expected {predicted:.0f}kg — "
                           f"gap of {gap_pct:.0f}%.",
                action   = "Monitor closely. Consider soil test before next fertiliser application.",
            ))

    #  2. SARIMA forecast check
    if sarima_meta and sarima_meta.get("status") == "ok":
        forecast = sarima_meta.get("forecast_6mo", [])
        if forecast and current.get("last_month_kg"):
            baseline = current["last_month_kg"]
            forecast_avg = np.mean(forecast)
            drop_pct = (baseline - forecast_avg) / baseline * 100 if baseline > 0 else 0

            if drop_pct > FORECAST_DROP_PCT:
                recs.append(Recommendation(
                    priority = "medium",
                    category = "yield",
                    title    = "Forecast shows expected yield decline",
                    detail   = f"SARIMA model forecasts an average of {forecast_avg:.0f}kg/month "
                               f"over the next 6 months — {drop_pct:.0f}% below current output.",
                    action   = "Check seasonal pattern — if this is a known lean season, no action needed. "
                               "If unexpected, review pruning schedule and input plan.",
                ))
            elif drop_pct < -FORECAST_DROP_PCT:  # rising forecast
                recs.append(Recommendation(
                    priority = "low",
                    category = "yield",
                    title    = "Forecast shows expected yield improvement",
                    detail   = f"Output projected to rise to ~{forecast_avg:.0f}kg/month "
                               f"over the next 6 months.",
                    action   = "Ensure collection logistics are ready for higher volumes.",
                ))

    # ── 3. Pricing intelligence ───────────────────────────────────────────────
    if "monthly_rate" in pricing_meta:
        pfc = pricing_meta["monthly_rate"].get("forecast", [])
        if len(pfc) >= 3:
            near_term_avg = np.mean(pfc[:3])
            # Compare to last known rate — use median of historical as proxy
            # (actual last rate would come from ktda_pricing in production)
            if near_term_avg > 0:
                recs.append(Recommendation(
                    priority = "low",
                    category = "pricing",
                    title    = "Pricing outlook available",
                    detail   = f"SARIMA pricing model projects a 3-month average rate of "
                               f"KES {near_term_avg:.2f}/kg for {farm['factory_code']}.",
                    action   = "No action required — informational. Rate changes are declared by KTDA.",
                ))

    # 4. Minibonus months approaching
    current_month_idx = xgb_result.get("month_idx", 6)
    upcoming = [(current_month_idx + i) % 12 for i in range(1, 4)]
    upcoming_minibonus = [m for m in upcoming if m in MINIBONUS_MONTHS]
    if upcoming_minibonus:
        month_names = [SEASON_MONTH_NAMES[m] for m in upcoming_minibonus]
        recs.append(Recommendation(
            priority = "low",
            category = "pricing",
            title    = f"Minibonus month(s) approaching: {', '.join(month_names)}",
            detail   = f"KTDA minibonus payments are made in Jul, Aug, Sep, Nov, Dec. "
                       f"Maximising leaf delivery in these months increases earnings per kg.",
            action   = "Prioritise plucking rounds in minibonus months for maximum income.",
        ))

    #  5. Feature-driven input recommendation
    top_features = xgb_result.get("top_features", [])
    feat_names   = [f["feature"] for f in top_features]
    if "fert_kg" in feat_names or "fert_lag1_kg" in feat_names:
        recs.append(Recommendation(
            priority = "medium",
            category = "input",
            title    = "Fertiliser is a top yield driver for this farm",
            detail   = "XGBoost feature importance shows fertiliser application ranks "
                       "among the top predictors of yield variance on this farm.",
            action   = f"Ensure NPK/CAN application is on schedule. "
                       f"Recommended: 50kg/ha every 6 weeks during growing season "
                       f"(~{50 * ha:.0f}kg total for your {ha:.1f}ha).",
        ))

    # 6. GNN anomaly detection
    gnn = farm.get("_gnn")
    if gnn:
        anomaly_score = gnn.get("anomaly_score", 0)
        cluster_id    = gnn.get("cluster_id")
        peers         = gnn.get("peers", [])

        if anomaly_score > GNN_ANOMALY_HIGH:
            peer_names = ", ".join(p["ktda_member_no"] for p in peers[:3])
            recs.append(Recommendation(
                priority = "high",
                category = "anomaly",
                title    = "Farm performance is a significant outlier in its cluster",
                detail   = f"GNN anomaly score {anomaly_score:.2f}/1.0 — this farm deviates "
                         f"substantially from its {len(peers)} structural peers in cluster {cluster_id}.",
                action   = f"Compare management practices with nearby farms: {peer_names}. "
                         f"Investigate whether the gap is due to inputs, pruning timing, or soil."))
        elif anomaly_score > GNN_ANOMALY_MEDIUM:
            recs.append(Recommendation(
                priority = "medium",
                category = "anomaly",
                title    = "Farm shows moderate deviation from cluster peers",
                detail   = f"GNN anomaly score {anomaly_score:.2f}/1.0 — performance diverges "
                         f"from structurally similar farms in the same zone.",
                action   = "Review fertiliser and pruning schedule against collection centre average."))

    if not recs:
        recs.append(Recommendation(
            priority = "low",
            category = "yield",
            title    = "Farm performing within expected range",
            detail   = "No significant anomalies detected. Yield and forecast are aligned "
                       "with historical patterns.",
            action   = "Maintain current management practices.",
        ))

    return recs


def run_pipeline(member_no: str, current_month_idx: int = 6) -> dict:
    """
    Full pipeline for one farm: load → predict → forecast → recommend.
    Returns a structured dict suitable for Ollama and API serialisation.
    """
    client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=5000)
    db     = client.get_default_database()

    farm = db.farms.find_one({"ktda_member_no": member_no}, {"_id": 0})
    if not farm:
        return {"error": f"Farm {member_no} not found"}

    model, preprocessor, importances = load_xgb_artifacts()

    xgb_result   = compute_xgb_prediction(farm, model, preprocessor,
                                           importances, current_month_idx)
    sarima_meta  = load_farm_sarima_meta(member_no, farm["factory_code"])
    pricing_meta = load_pricing_meta(farm["factory_code"])
    current      = get_current_season_summary(farm)
    recs         = build_recommendations(farm, xgb_result, sarima_meta,
                                         pricing_meta, importances)

    forecast_summary = None
    if sarima_meta and sarima_meta.get("status") == "ok":
        forecast_summary = {
            "forecast_6mo":  sarima_meta["forecast_6mo"],
            "ci_80_lower":   sarima_meta["ci_80_lower"],
            "ci_80_upper":   sarima_meta["ci_80_upper"],
        }

    output = {
        "farm": {
            "ktda_member_no":    farm["ktda_member_no"],
            "name":              farm.get("name"),
            "owner_name":        farm.get("owner_name"),
            "factory_code":      farm["factory_code"],
            "collection_centre": farm["collection_centre"],
            "hectares":          farm.get("hectares"),
            "altitude_m":        farm.get("altitude_m"),
            "fairtrade":         farm.get("fairtrade_certified"),
        },
        "current_season":   current,
        "xgb_prediction":   xgb_result,
        "sarima_forecast":  forecast_summary,
        "pricing_forecast": {k: v.get("forecast", [])[:6] for k, v in pricing_meta.items()},
        "recommendations":  [asdict(r) for r in recs],
        "performance_score": _score(xgb_result, sarima_meta, recs),
    }
    return output


def _score(xgb_result: dict, sarima_meta: dict | None, recs: list) -> int:
    """
    Simple 0-100 performance score.
    High-priority recommendations reduce the score.
    """
    score = 80
    for r in recs:
        if r.priority == "high":
            score -= 20
        elif r.priority == "medium":
            score -= 8
    return max(0, min(100, score))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--member", type=str, required=True,
                        help="ktda_member_no to run pipeline for, e.g. KTD-13033")
    parser.add_argument("--month-idx", type=int, default=6,
                        help="season_month_idx to predict for (0=Jul..11=Jun, default=6=Jan)")
    args = parser.parse_args()

    print(f"Running rule engine for {args.member} ...")
    result = run_pipeline(args.member, args.month_idx)

    if "error" in result:
        print(f"[ERROR] {result['error']}")
        return

    print(f"\n-- Farm: {result['farm']['name']} ({result['farm']['ktda_member_no']}) --")
    print(f"   Factory: {result['farm']['factory_code']}  |  "
          f"Centre: {result['farm']['collection_centre']}  |  "
          f"{result['farm']['hectares']}ha")
    print(f"   Performance score: {result['performance_score']}/100")

    print(f"\n-- Current season ({result['current_season'].get('season_year')}) --")
    print(f"   Total so far: {result['current_season'].get('total_kg',0):.0f}kg  "
          f"({result['current_season'].get('months_complete',0)} months)")

    if result["xgb_prediction"].get("predicted_kg"):
        print(f"\n-- XGBoost prediction for month {result['xgb_prediction']['month_name']} --")
        print(f"   Predicted: {result['xgb_prediction']['predicted_kg']:.1f}kg")

    if result["sarima_forecast"]:
        fc = result["sarima_forecast"]["forecast_6mo"]
        print(f"\n-- SARIMA 6-month forecast --")
        for i, kg in enumerate(fc):
            print(f"   Month+{i+1}: {kg:.1f}kg")

    print(f"\n-- Recommendations ({len(result['recommendations'])}) --")
    for rec in result["recommendations"]:
        print(f"   [{rec['priority'].upper()}] {rec['title']}")
        print(f"     → {rec['action']}")

    print(f"\n-- Full JSON output --")
    # Print without model objects
    safe = {k: v for k, v in result.items()}
    print(json.dumps(safe, indent=2, default=str))


if __name__ == "__main__":
    main()