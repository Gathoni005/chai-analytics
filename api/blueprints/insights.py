"""
api/blueprints/insights.py

GET /farms/<member_no>/insights
    Runs the full pipeline: XGBoost prediction + SARIMA forecast +
    rule engine + optional Ollama narrative.
    Results cached in model_outputs collection.

Query params:
    refresh     true  — force recompute even if cache is fresh
    narrative   true  — include Ollama LLM narrative (slower, ~5-10s)
    month_idx   0-11  — season month to predict for (default: current)
"""

import sys
import json
import time
from datetime import datetime
from pathlib import Path
from dataclasses import asdict

import numpy as np
import requests as http_requests
from flask import Blueprint, request, jsonify, current_app
from flask_jwt_extended import jwt_required, get_jwt_identity

from utils import get_db, get_models, get_sarima_meta, get_pricing_meta, get_gnn_output, get_cluster_peers

insights_bp = Blueprint("insights", __name__)

SEASON_MONTH_NAMES = ["Jul","Aug","Sep","Oct","Nov","Dec","Jan","Feb","Mar","Apr","May","Jun"]
MINIBONUS_MONTHS   = {0, 1, 2, 4, 5}

# Default thresholds (matching rule_engine.py)
DEFAULT_THRESHOLDS = {
    "yield_gap_warn_pct":   15.0,
    "yield_gap_alert_pct":  30.0,
    "forecast_drop_pct":    20.0,
    "gnn_anomaly_high":     0.75,
    "gnn_anomaly_medium":   0.50,
}


def _load_thresholds(db, factory_code: str) -> dict:
    """Load rule thresholds from factory_metadata, with defaults as fallback."""
    try:
        meta = db.factory_metadata.find_one({"_type": "factory_metadata"}, {"_id": 0})
        if meta:
            for f in meta.get("factories", []):
                if f["factory_code"] == factory_code:
                    thresholds = f.get("rule_thresholds", {})
                    return {**DEFAULT_THRESHOLDS, **thresholds}
    except Exception:
        pass
    return DEFAULT_THRESHOLDS


def _current_month_idx() -> int:
    """Return the current season_month_idx based on today's calendar month."""
    month_to_idx = {7:0,8:1,9:2,10:3,11:4,12:5,1:6,2:7,3:8,4:9,5:10,6:11}
    return month_to_idx.get(datetime.utcnow().month, 6)


def _get_current_season_summary(farm: dict) -> dict:
    seasons = sorted(farm.get("historical_seasons", []), key=lambda s: s["season_year"])
    if not seasons:
        return {}
    latest = seasons[-1]
    kg     = [k for k in latest.get("monthly_kg", []) if k is not None]
    earn   = [e for e in latest.get("monthly_earn", []) if e is not None]
    return {
        "season_year":     latest["season_year"],
        "monthly_kg":      latest.get("monthly_kg", []),
        "monthly_earn":    latest.get("monthly_earn", []),
        "total_kg":        round(sum(kg), 1),
        "total_earnings":  round(sum(earn), 2),
        "months_complete": len(kg),
        "last_month_kg":   kg[-1] if kg else None,
        "season_avg_kg":   round(float(np.mean(kg)), 1) if kg else None,
        "yearly_bonus":    latest.get("yearly_bonus", 0),
    }


def _run_xgb_prediction(farm: dict, models: dict, month_idx: int) -> dict:
    try:
        X    = models["X"]
        ids  = models["row_ids"]
        mask = ids["ktda_member_no"] == farm["ktda_member_no"]
        if not mask.any():
            return {"error": "farm not in feature matrix"}

        farm_ids = ids[mask]
        latest_season = farm_ids["season_year"].max()
        month_mask = (
            (farm_ids["season_year"] == latest_season) &
            (farm_ids["season_month_idx"] == month_idx)
        )
        if not month_mask.any():
            month_mask = farm_ids["season_year"] == latest_season

        X_row    = X[mask][month_mask].iloc[[-1]]
        X_scaled = models["preprocessor"].transform(X_row)
        pred_kg  = float(models["xgb"].predict(X_scaled)[0])

        top_features = models["importances"].head(5)[["feature","importance"]].to_dict("records")

        return {
            "predicted_kg": round(max(0.0, pred_kg), 1),
            "month_idx":    month_idx,
            "month_name":   SEASON_MONTH_NAMES[month_idx],
            "top_features": top_features,
        }
    except Exception as e:
        return {"error": str(e)[:200]}


def _build_recommendations(farm: dict, xgb: dict, sarima_meta: dict | None,
                            pricing_meta: dict, db) -> list[dict]:
    recs    = []
    current = _get_current_season_summary(farm)
    ha      = float(farm.get("hectares", 1.0))
    factory = farm["factory_code"]

    # Load configurable thresholds
    thresholds = _load_thresholds(db, factory)
    YIELD_GAP_WARN_PCT  = thresholds["yield_gap_warn_pct"]
    YIELD_GAP_ALERT_PCT = thresholds["yield_gap_alert_pct"]
    FORECAST_DROP_PCT   = thresholds["forecast_drop_pct"]
    GNN_ANOMALY_HIGH    = thresholds["gnn_anomaly_high"]
    GNN_ANOMALY_MEDIUM  = thresholds["gnn_anomaly_medium"]

    # Yield gap
    if xgb.get("predicted_kg") and current.get("season_avg_kg"):
        pred   = xgb["predicted_kg"]
        actual = current["season_avg_kg"]
        gap    = (pred - actual) / pred * 100 if pred > 0 else 0
        if gap > YIELD_GAP_ALERT_PCT:
            recs.append({"priority":"high","category":"yield",
                "title":"Yield significantly below model expectation",
                "detail":f"Average {actual}kg/mo vs expected {pred}kg/mo — {gap:.0f}% gap.",
                "action":"Review fertiliser schedule and check for pest or soil pH issues."})
        elif gap > YIELD_GAP_WARN_PCT:
            recs.append({"priority":"medium","category":"yield",
                "title":"Yield slightly below model expectation",
                "detail":f"Average {actual}kg/mo vs expected {pred}kg/mo — {gap:.0f}% gap.",
                "action":"Monitor closely. Consider soil test before next fertiliser application."})

    # Forecast drop
    if sarima_meta and sarima_meta.get("status") == "ok":
        fc  = sarima_meta.get("forecast_6mo", [])
        if fc and current.get("last_month_kg"):
            drop = (current["last_month_kg"] - np.mean(fc)) / current["last_month_kg"] * 100
            if drop > FORECAST_DROP_PCT:
                recs.append({"priority":"medium","category":"yield",
                    "title":"Forecast shows expected yield decline",
                    "detail":f"SARIMA projects avg {np.mean(fc):.0f}kg/mo over next 6 months.",
                    "action":"Check seasonal pattern. If unexpected, review pruning and input plan."})

    # Minibonus approaching
    mi = xgb.get("month_idx", 6)
    upcoming_mini = [SEASON_MONTH_NAMES[(mi+i) % 12] for i in range(1,4)
                     if (mi+i) % 12 in MINIBONUS_MONTHS]
    if upcoming_mini:
        recs.append({"priority":"low","category":"pricing",
            "title":f"Minibonus month(s) approaching: {', '.join(upcoming_mini)}",
            "detail":"KTDA minibonus paid in Jul, Aug, Sep, Nov, Dec.",
            "action":"Prioritise plucking rounds in minibonus months for maximum income."})

    # Fertiliser driver
    top_feat_names = [f["feature"] for f in xgb.get("top_features",[])]
    if any(f in top_feat_names for f in ("fert_kg","fert_lag1_kg")):
        recs.append({"priority":"medium","category":"input",
            "title":"Fertiliser is a top yield driver for this farm",
            "detail":"XGBoost ranks fertiliser among the top predictors of yield variance.",
            "action":f"Ensure NPK/CAN on schedule (~{50*ha:.0f}kg total for your {ha:.1f}ha)."})

    # ── GNN relational anomaly ────────────────────────────────────────
    gnn = farm.get("_gnn")   # injected by get_insights before calling this
    if gnn:
        anomaly_score = gnn.get("anomaly_score", 0)
        cluster_id    = gnn.get("cluster_id")
        peers         = gnn.get("peers", [])

        if anomaly_score > GNN_ANOMALY_HIGH:
            peer_names = ", ".join(p["ktda_member_no"] for p in peers[:3])
            recs.append({"priority":"high","category":"anomaly",
                "title":"Farm performance is a significant outlier in its cluster",
                "detail":f"GNN anomaly score {anomaly_score:.2f}/1.0 — this farm deviates "
                         f"substantially from its {len(peers)} structural peers in cluster {cluster_id}.",
                "action":f"Compare management practices with nearby farms: {peer_names}. "
                         f"Investigate whether the gap is due to inputs, pruning timing, or soil."})
        elif anomaly_score > GNN_ANOMALY_MEDIUM:
            recs.append({"priority":"medium","category":"anomaly",
                "title":"Farm shows moderate deviation from cluster peers",
                "detail":f"GNN anomaly score {anomaly_score:.2f}/1.0 — performance diverges "
                         f"from structurally similar farms in the same zone.",
                "action":"Review fertiliser and pruning schedule against collection centre average."})

    if not recs:
        recs.append({"priority":"low","category":"yield",
            "title":"Farm performing within expected range",
            "detail":"No significant anomalies detected.",
            "action":"Maintain current management practices."})

    return recs


def _call_groq(pipeline_result: dict, api_key: str, model: str) -> str | None:
    farm    = pipeline_result["farm"]
    current = pipeline_result["current_season"]
    xgb     = pipeline_result["xgb_prediction"]
    sarima  = pipeline_result.get("sarima_forecast")
    recs    = pipeline_result["recommendations"]
    score   = pipeline_result["performance_score"]

    top_recs = sorted(recs, key=lambda r: {"high":0,"medium":1,"low":2}[r["priority"]])[:2]
    fc_str   = None
    if sarima:
        fc = sarima["forecast_6mo"]
        fc_str = f"{round(np.mean(fc[:3])):.0f}kg avg over next 3 months"

    compact = {
        "farm":           f"{farm.get('name')} ({farm.get('owner_name')})",
        "factory":        farm.get("factory_code"),
        "centre":         farm.get("collection_centre"),
        "hectares":       farm.get("hectares"),
        "season":         current.get("season_year"),
        "total_kg":       round(current.get("total_kg", 0)),
        "avg_monthly_kg": round(current.get("season_avg_kg", 0)) if current.get("season_avg_kg") else None,
        "xgb_next_month": f"{round(xgb.get('predicted_kg',0))}kg in {xgb.get('month_name','')}",
        "forecast":       fc_str,
        "score":          score,
        "top_recommendations": [
            f"[{r['priority'].upper()}] {r['title']}: {r['action']}"
            for r in top_recs
        ],
    }

    system = "You are a KTDA tea farm advisor. Write a 3-paragraph plain-language summary for the farmer using only the data provided. No bullet points. Under 150 words. Warm, direct tone. End with one clear action."
    prompt = system + "\n\nFarm data:\n" + json.dumps(compact, indent=2)

    try:
        resp = http_requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type":  "application/json",
            },
            json={
                "model":       model,
                "messages":    [{"role": "user", "content": prompt}],
                "temperature": 0.3,
                "max_tokens":  500,
            },
            timeout=30,
        )
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"].strip()
        import re
        content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
        return content

    except Exception as e:
        return f"[Narrative unavailable: {e}]"


@insights_bp.route("/<member_no>/insights", methods=["GET"])
@jwt_required()
def get_insights(member_no: str):
   # current_user = get_jwt_identity()
    
    db = get_db()

    farm = db.farms.find_one({"ktda_member_no": member_no}, {"_id": 0})
    if not farm:
        return jsonify({"error": f"Farm {member_no} not found"}), 404

    refresh   = request.args.get("refresh",   "false").lower() == "true"
    narrative = request.args.get("narrative", "false").lower() == "true"
    month_idx = int(request.args.get("month_idx", _current_month_idx()))

    #  Cache check 
    if not refresh:
        cached = db.model_outputs.find_one({"ktda_member_no": member_no}, {"_id": 0})
        if cached and not narrative:
            cached["from_cache"] = True
            return jsonify(cached)

    #  Run pipeline 
    t0      = time.time()
    models  = get_models()
    current = _get_current_season_summary(farm)
    xgb     = _run_xgb_prediction(farm, models, month_idx)
    sarima  = get_sarima_meta(member_no, farm["factory_code"])
    pricing = get_pricing_meta(farm["factory_code"])
    gnn     = get_gnn_output(member_no)
    peers   = get_cluster_peers(member_no, gnn["cluster_id"]) if gnn else []
    if gnn:
        gnn["peers"] = peers
    farm["_gnn"] = gnn   # inject for recommendation builder
    recs    = _build_recommendations(farm, xgb, sarima, pricing, db)

    # Score
    score = 80
    for r in recs:
        score -= {"high":20,"medium":8,"low":0}[r["priority"]]
    score = max(0, min(100, score))

    sarima_forecast = None
    if sarima and sarima.get("status") == "ok":
        sarima_forecast = {
            "forecast_6mo":  sarima["forecast_6mo"],
            "ci_80_lower":   sarima["ci_80_lower"],
            "ci_80_upper":   sarima["ci_80_upper"],
        }

    result = {
        "ktda_member_no":  member_no,
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
        "xgb_prediction":   xgb,
        "sarima_forecast":  sarima_forecast,
        "pricing_forecast": {k: v.get("forecast", [])[:6] for k, v in pricing.items()},
        "recommendations":  recs,
        "performance_score": score,
        "gnn": {
            "cluster_id":    gnn.get("cluster_id") if gnn else None,
            "anomaly_score": gnn.get("anomaly_score") if gnn else None,
            "cluster_peers": [{"ktda_member_no": p["ktda_member_no"],
                               "collection_centre": p["collection_centre"],
                               "anomaly_score": p["anomaly_score"]} for p in peers],
        } if gnn else None,
        "computed_at":      datetime.utcnow().isoformat(),
        "elapsed_seconds":  round(time.time() - t0, 2),
        "from_cache":       False,
    }

    #  Optional groq narrative 
    if narrative:
        result["narrative"] = _call_groq(
            result,
            api_key  = current_app.config["GROQ_API_KEY"],
            model = current_app.config["GROQ_MODEL"],
        )

    #  Cache result 
    db.model_outputs.replace_one(
        {"ktda_member_no": member_no},
        result,
        upsert=True,
    )

    return jsonify(result)