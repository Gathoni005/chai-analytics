#!/usr/bin/env python3
"""
upload_models_to_mongo.py

Uploads trained model files to MongoDB so Render can load them
without needing a persistent disk.

Files uploaded:
  - models/xgb_yield.pkl
  - models/preprocessor.pkl
  - models/xgb_feature_importances.parquet
  - models/X_features.parquet
  - models/row_ids.parquet

Uses GridFS for large files (xgb, X_features) and a simple
model_files collection for smaller ones.

Run once locally after training:
    python scripts/upload_models_to_mongo.py

Re-run after every retrain to update.
"""

import os
import io
from pathlib import Path

import gridfs
from pymongo import MongoClient

MONGODB_URI = os.getenv("MONGODB_URI")
MODELS_DIR  = Path(os.getenv("MODELS_DIR",
              str(Path(__file__).resolve().parent.parent / "models")))

# Files to upload via GridFS (binary, can be large)
GRIDFS_FILES = [
    "xgb_yield.pkl",
    "preprocessor.pkl",
    "xgb_feature_importances.parquet",
    "X_features.parquet",
    "row_ids.parquet",
    "gnn_norm_params.pkl",
]


def upload_file(fs: gridfs.GridFS, name: str, path: Path) -> bool:
    if not path.exists():
        print(f"  [SKIP] {name} not found at {path}")
        return False

    size_mb = path.stat().st_size / 1024 / 1024
    print(f"  Uploading {name} ({size_mb:.1f} MB) ...", end=" ", flush=True)

    # Delete existing version
    for existing in fs.find({"filename": name}):
        fs.delete(existing._id)

    with open(path, "rb") as f:
        fs.put(f, filename=name)

    print("OK")
    return True


def main():
    print(f"Connecting to {MONGODB_URI} ...")
    client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=5000)
    db     = client.get_default_database()
    db.command("ping")
    print("Connected.\n")

    fs = gridfs.GridFS(db, collection="model_files")

    uploaded = 0
    skipped  = 0

    print("Uploading model files to MongoDB GridFS ...")
    for filename in GRIDFS_FILES:
        path = MODELS_DIR / filename
        ok   = upload_file(fs, filename, path)
        if ok:
            uploaded += 1
        else:
            skipped += 1

    print(f"\n-- Summary --")
    print(f"  Uploaded : {uploaded}")
    print(f"  Skipped  : {skipped}")

    # List what's now in GridFS
    print(f"\n-- Files now in MongoDB model_files --")
    for f in fs.find():
        size_mb = f.length / 1024 / 1024
        print(f"  {f.filename:<40} {size_mb:.1f} MB  "
              f"uploaded: {f.upload_date.strftime('%Y-%m-%d %H:%M')}")

    print("\nDone. Models are now in MongoDB.")
    print("Render will load them automatically on next deploy.")


if __name__ == "__main__":
    main()


def upload_sarima_meta(db, models_dir: Path):
    """
    Upload all SARIMA farm metadata JSON files to sarima_meta collection.
    Upload pricing SARIMA metadata to sarima_pricing_meta collection.
    """
    sarima_dir  = models_dir / "sarima"
    pricing_dir = sarima_dir / "pricing"
    uploaded    = 0

    # Per-farm SARIMA metadata
    print("\nUploading farm SARIMA metadata ...")
    for factory_dir in sarima_dir.iterdir():
        if not factory_dir.is_dir() or factory_dir.name == "pricing":
            continue
        for meta_file in factory_dir.glob("*_meta.json"):
            member_no    = meta_file.stem.replace("_meta", "")
            factory_code = factory_dir.name
            with open(meta_file) as f:
                doc = json.load(f)
            doc["ktda_member_no"] = member_no
            doc["factory_code"]   = factory_code
            db.sarima_meta.replace_one(
                {"ktda_member_no": member_no, "factory_code": factory_code},
                doc, upsert=True
            )
            uploaded += 1

    print(f"  Uploaded {uploaded} farm SARIMA metadata documents.")

    # Pricing SARIMA metadata
    print("Uploading pricing SARIMA metadata ...")
    pricing_uploaded = 0
    if pricing_dir.exists():
        for meta_file in pricing_dir.glob("*_meta.json"):
            stem         = meta_file.stem.replace("_meta", "")
            parts        = stem.split("_", 1)
            factory_code = parts[0]
            label        = parts[1] if len(parts) > 1 else stem
            with open(meta_file) as f:
                doc = json.load(f)
            doc["factory_code"] = factory_code
            doc["label"]        = label
            db.sarima_pricing_meta.replace_one(
                {"factory_code": factory_code, "label": label},
                doc, upsert=True
            )
            pricing_uploaded += 1
    print(f"  Uploaded {pricing_uploaded} pricing SARIMA metadata documents.")


if __name__ == "__main__":
    import json
    print(f"Connecting to {MONGODB_URI} ...")
    client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=5000)
    db     = client.get_default_database()
    db.command("ping")
    print("Connected.\n")

    fs = gridfs.GridFS(db, collection="model_files")
    uploaded = skipped = 0

    print("Uploading model files to MongoDB GridFS ...")
    for filename in GRIDFS_FILES:
        path = MODELS_DIR / filename
        ok   = upload_file(fs, filename, path)
        uploaded += 1 if ok else 0
        skipped  += 0 if ok else 1

    upload_sarima_meta(db, MODELS_DIR)

    print(f"\n-- GridFS files now in MongoDB --")
    for f in fs.find():
        size_mb = f.length / 1024 / 1024
        print(f"  {f.filename:<40} {size_mb:.1f} MB")

    print("\nAll done. Models and SARIMA metadata are in MongoDB.")