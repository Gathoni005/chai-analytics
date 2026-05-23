"""
api/utils.py — shared MongoDB client and ML model loader.

Both are initialised once per process and reused across requests.
"""

import os
import json
import joblib
from pathlib import Path
from functools import lru_cache

import pandas as pd
from pymongo import MongoClient
from flask import current_app


# MongoDB 

_mongo_client: MongoClient | None = None

def get_db():
    """Return the chaimterics MongoDB database, reusing the connection."""
    global _mongo_client
    if _mongo_client is None:
        uri = current_app.config["MONGODB_URI"]
        _mongo_client = MongoClient(uri, serverSelectionTimeoutMS=5000)
    return _mongo_client.get_default_database()


#  ML model loader 

_models: dict = {}

def get_models() -> dict:
    """
    Load XGBoost model, preprocessor, and feature importances once per process.
    Returns dict with keys: xgb, preprocessor, importances.
    """
    global _models
    if _models:
        return _models

    models_dir = Path(current_app.config["MODELS_DIR"])
    try:
        _models = {
            "xgb":          joblib.load(models_dir / "xgb_yield.pkl"),
            "preprocessor": joblib.load(models_dir / "preprocessor.pkl"),
            "importances":  pd.read_parquet(models_dir / "xgb_feature_importances.parquet"),
            "X":            pd.read_parquet(models_dir / "X_features.parquet"),
            "row_ids":      pd.read_parquet(models_dir / "row_ids.parquet"),
        }
    except FileNotFoundError as e:
        raise RuntimeError(f"Model file missing: {e}. Run preprocess.py and train_xgboost.py first.")
    return _models


def get_sarima_meta(member_no: str, factory_code: str) -> dict | None:
    """Load per-farm SARIMA metadata JSON. Returns None if not yet fitted."""
    models_dir = Path(current_app.config["MODELS_DIR"])
    path = models_dir / "sarima" / factory_code / f"{member_no}_meta.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def get_pricing_meta(factory_code: str) -> dict:
    """Load all three pricing SARIMA metadata dicts for a factory."""
    models_dir = Path(current_app.config["MODELS_DIR"])
    result = {}
    for label in ("monthly_rate", "minibonus_rate", "annual_bonus"):
        path = models_dir / "sarima" / "pricing" / f"{factory_code}_{label}_meta.json"
        if path.exists():
            with open(path) as f:
                result[label] = json.load(f)
    return result


#  Response helpers 

def farm_summary(farm: dict) -> dict:
    """Slim farm dict safe to return in list endpoints."""
    return {
        "ktda_member_no":    farm.get("ktda_member_no"),
        "name":              farm.get("name"),
        "owner_name":        farm.get("owner_name"),
        "factory_code":      farm.get("factory_code"),
        "factory_name":      farm.get("factory_name"),
        "collection_centre": farm.get("collection_centre"),
        "county":            farm.get("county"),
        "hectares":          farm.get("hectares"),
        "altitude_m":        farm.get("altitude_m"),
        "registered_year":   farm.get("registered_year"),
        "fairtrade_certified": farm.get("fairtrade_certified"),
        "latitude":          farm.get("latitude"),
        "longitude":         farm.get("longitude"),
    }

    def get_gnn_output(member_no: str) -> dict | None:
    """Load GNN cluster_id and anomaly_score for a farm. Returns None if not yet computed."""
        db  = get_db()
        doc = db.gnn_outputs.find_one({"ktda_member_no": member_no},
                                   {"_id": 0, "embedding": 0})
    return doc

    def get_cluster_peers(member_no: str, cluster_id: int, limit: int = 5) -> list[dict]:
    """
    Return other farms in the same cluster ordered by anomaly_score descending.
    Used to surface peer comparison in recommendations.
    """
        db   = get_db()
        docs = list(
            db.gnn_outputs.find(
                {"cluster_id": cluster_id, "ktda_member_no": {"$ne": member_no}},
                {"_id": 0, "embedding": 0},
            ).sort("anomaly_score", -1).limit(limit)
        )
    return docs