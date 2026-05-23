"""
api/utils.py — shared MongoDB client and ML model loader.

Models are loaded from MongoDB GridFS (uploaded by upload_models_to_mongo.py).
SARIMA metadata is loaded from the sarima_meta collection.
All loaders cache in memory after first call — subsequent requests are instant.
"""

import os
import io
import json
import joblib
from pathlib import Path

import gridfs
import pandas as pd
from pymongo import MongoClient
from flask import current_app


# ── MongoDB ───────────────────────────────────────────────────────────────────

_mongo_client: MongoClient | None = None

def get_db():
    """Return the chaimterics MongoDB database, reusing the connection."""
    global _mongo_client
    if _mongo_client is None:
        uri = current_app.config["MONGODB_URI"]
        _mongo_client = MongoClient(uri, serverSelectionTimeoutMS=5000)
    return _mongo_client.get_default_database()


# ── ML model loader — GridFS ──────────────────────────────────────────────────

_models: dict = {}

def _load_pkl_from_gridfs(fs: gridfs.GridFS, filename: str):
    """Load a joblib-serialised object from GridFS into memory."""
    grid_out = fs.find_one({"filename": filename})
    if grid_out is None:
        raise FileNotFoundError(
            f"'{filename}' not found in MongoDB GridFS. "
            f"Run scripts/upload_models_to_mongo.py first."
        )
    return joblib.load(io.BytesIO(grid_out.read()))


def _load_parquet_from_gridfs(fs: gridfs.GridFS, filename: str) -> pd.DataFrame:
    """Load a parquet file from GridFS into a pandas DataFrame."""
    grid_out = fs.find_one({"filename": filename})
    if grid_out is None:
        raise FileNotFoundError(
            f"'{filename}' not found in MongoDB GridFS. "
            f"Run scripts/upload_models_to_mongo.py first."
        )
    return pd.read_parquet(io.BytesIO(grid_out.read()))


def get_models() -> dict:
    """
    Load XGBoost model, preprocessor, and feature matrices once per process.
    First call loads from MongoDB GridFS (~2-3s). Subsequent calls return cache.
    """
    global _models
    if _models:
        return _models

    db = get_db()
    fs = gridfs.GridFS(db, collection="model_files")

    try:
        _models = {
            "xgb":          _load_pkl_from_gridfs(fs, "xgb_yield.pkl"),
            "preprocessor": _load_pkl_from_gridfs(fs, "preprocessor.pkl"),
            "importances":  _load_parquet_from_gridfs(fs, "xgb_feature_importances.parquet"),
            "X":            _load_parquet_from_gridfs(fs, "X_features.parquet"),
            "row_ids":      _load_parquet_from_gridfs(fs, "row_ids.parquet"),
        }
    except FileNotFoundError as e:
        raise RuntimeError(str(e))

    return _models


# ── SARIMA metadata loader ────────────────────────────────────────────────────
# SARIMA meta JSON files are stored in MongoDB sarima_meta collection.
# Upload script stores them with keys: {member_no, factory_code, ...metadata}
# Fallback: try disk if collection is empty (for local dev).

def get_sarima_meta(member_no: str, factory_code: str) -> dict | None:
    """
    Load per-farm SARIMA metadata.
    Tries MongoDB first, falls back to disk for local development.
    """
    db  = get_db()

    # Try MongoDB collection first
    doc = db.sarima_meta.find_one(
        {"ktda_member_no": member_no, "factory_code": factory_code},
        {"_id": 0}
    )
    if doc:
        return doc

    # Fallback: disk (local dev only)
    models_dir = Path(os.getenv("MODELS_DIR", "/app/models"))
    path = models_dir / "sarima" / factory_code / f"{member_no}_meta.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)

    return None


def get_pricing_meta(factory_code: str) -> dict:
    """
    Load pricing SARIMA metadata for a factory.
    Tries MongoDB first, falls back to disk.
    """
    db     = get_db()
    result = {}

    for label in ("monthly_rate", "minibonus_rate", "annual_bonus"):
        # Try MongoDB
        doc = db.sarima_pricing_meta.find_one(
            {"factory_code": factory_code, "label": label},
            {"_id": 0}
        )
        if doc:
            result[label] = doc
            continue

        # Fallback: disk
        models_dir = Path(os.getenv("MODELS_DIR", "/app/models"))
        path = models_dir / "sarima" / "pricing" / f"{factory_code}_{label}_meta.json"
        if path.exists():
            with open(path) as f:
                result[label] = json.load(f)

    return result


# ── Response helpers ──────────────────────────────────────────────────────────

def farm_summary(farm: dict) -> dict:
    """Slim farm dict safe to return in list endpoints."""
    return {
        "ktda_member_no":      farm.get("ktda_member_no"),
        "name":                farm.get("name"),
        "owner_name":          farm.get("owner_name"),
        "factory_code":        farm.get("factory_code"),
        "factory_name":        farm.get("factory_name"),
        "collection_centre":   farm.get("collection_centre"),
        "county":              farm.get("county"),
        "hectares":            farm.get("hectares"),
        "altitude_m":          farm.get("altitude_m"),
        "registered_year":     farm.get("registered_year"),
        "fairtrade_certified": farm.get("fairtrade_certified"),
        "latitude":            farm.get("latitude"),
        "longitude":           farm.get("longitude"),
    }


# ── GNN output loaders ────────────────────────────────────────────────────────

def get_gnn_output(member_no: str) -> dict | None:
    """Load GNN cluster_id and anomaly_score for a farm. Returns None if not yet computed."""
    db  = get_db()
    doc = db.gnn_outputs.find_one(
        {"ktda_member_no": member_no},
        {"_id": 0, "embedding": 0}
    )
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