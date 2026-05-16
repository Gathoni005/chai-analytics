#!/usr/bin/env python3
"""
Phase 2 — Step 1: Build the farm graph.

Constructs nodes and edges from the farms collection and saves the
graph structure to MongoDB (farm_graph collection) ready for GNN training.

Edge rules (from roadmap):
  - Shared collection_centre     → weight 1.0
  - Shared factory_code          → weight 0.6
  - GPS proximity < 5km          → weight proportional to inverse distance

Run on WSL host:
    python scripts/build_graph.py
    python scripts/build_graph.py --validate
"""

import os
import math
import json
import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
from pymongo import MongoClient

MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017/chaimterics")
GPS_EDGE_THRESHOLD_KM = 5.0


def haversine_km(lat1, lon1, lat2, lon2) -> float:
    """Great-circle distance between two GPS points in kilometres."""
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi  = math.radians(lat2 - lat1)
    dlam  = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlam/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def extract_node_features(farm: dict) -> dict:
    """
    Extract normalisation-ready features for one farm node.
    Returns raw values — normalisation happens in train_gnn.py.
    """
    seasons = sorted(farm.get("historical_seasons", []),
                     key=lambda s: s["season_year"])

    # flatten all monthly kgs means and rainfall for all seasons
    season_means = []
    all_monthly_kgs = []
    all_monthly_rains = []

    for s in seasons:
        kg = [v for v in s.get("monthly_kg", []) if v is not None]
        rainfall = [v for v in s.get("season_rainfall_mm", []) if v is not None]
        if kg:
            season_means.append(float(np.mean(kg)))
            all_monthly_kgs.extend(kg)
        if rainfall:
            all_monthly_rains.extend(rainfall)

    
    # Yield history: mean monthly kg per season (up to 15 seasons)
    season_means = []
    for s in seasons:
        kg = [v for v in s.get("monthly_kg", []) if v is not None]
        if kg:
            season_means.append(float(np.mean(kg)))

    # Pad/truncate to 15 seasons
    padded = (season_means + [0.0] * 15)[:15]

    # YIELD_TREND_SLOPE: slope of season means over time (recent 5 seasons or all if <5)
    # fit y = mx + c for across 5 seasons, m is the slope
    # shows whether are improving or declining yields in recent years, which may indicate good/bad practices or external factors
    recent = season_means[-5:] if len(season_means) >= 2 else season_means
    if len(recent) >= 2:
        xs    = np.arange(len(recent), dtype=float)
        slope = float(np.polyfit(xs, recent, 1)[0])
    else:
        slope = 0.0

    # YIELD STANDARD DEVIATION (VARIABILITY) across seasons (kg)
    # shows how consistent the farm is across seasons, which may indicate resilience to external factors or variability in practices
    # σ = sqrt( (1/N) Σ(yᵢ - ȳ)² )
    # high σ means the farm swings wildly between good and bad months
    # low σ means consistent output regardless of season
    yield_std = float(np.std(all_monthly_kgs)) if len(all_monthly_kgs) > 1 else 0.0

    # SEASONAL_AMPLITUDE: standard deviation of season means (kg)
    # shows how much the farm's yield changes across seasons, which may indicate sensitivity to seasonal factors or variability in practices
    # A = (max-min) / mean 
    if all_monthly_kgs and np.mean(all_monthly_kgs) > 0:
        amplitude = float(max(all_monthly_kgs) - min(all_monthly_kgs)) / float(np.mean(all_monthly_kgs))
        
    else:
        amplitude = 0.0

    # PRUNING_RECOVERY_RATE 
    # From agronomic rules: pruning month → 45% yield, month+1 → 78%, month+2 → 108% flush
    # We detect pruning months (yield ≤ 50% of season mean)
    # then measure actual month+2 yield vs expected flush (1.08 × season mean)
    # r > 1.0 means this farm recovers faster than the model expects
    # r < 1.0 means it recovers slower — possible soil or management issue
    recovery_rates = []
    for s in seasons:
        kg = s.get("monthly_kg", [])
        s_mean = np.mean([v for v in kg if v is not None]) if kg else 0.0
        if s_mean <= 0:
            continue
        for idx, val in enumerate(kg):
            if val is None:
                continue
            if val <= 0.5 * s_mean and idx + 2 < len(kg):
                flush_actual = kg[idx + 2]
                flush_expected = 1.08 * s_mean
                if flush_actual is not None and flush_expected > 0:
                    recovery_rates.append(float(flush_actual) / flush_expected)
    pruning_recovery_rate = float(np.mean(recovery_rates)) if recovery_rates else 1.0
    # RAINFALL_RESPONSE_BETA
    # β = Σ(rᵢ - r̄)(yᵢ - ȳ) / Σ(rᵢ - r̄)²
    # this is the standard OLS slope formula from numerical methods
    # β > 0: farm yields more when it rains more (rain-dependent)
    # β ≈ 0: farm yield is independent of rainfall (good drainage / irrigation)
    # β < 0: rare, would mean yield drops with rain (waterlogging)
    # we pair each monthly yield with the corresponding monthly rainfall
    rainfall_response_beta = 0.0
    paired_kg, paired_rain = [],[]
   
    for s in seasons:
        kg = s.get("monthly_kg", [])
        rain = s.get("season_rainfall_mm", [])
        for idx in range(min(len(kg), len(rain))):
            if kg[idx] is not None and rain[idx] is not None:
                paired_kg.append(float(kg[idx]))
                paired_rain.append(float(rain[idx]))

            if len(paired_kg) >= 4:
                rain_arr = np.array(paired_rain),
                kg_arr = np.array(paired_kg),
                rain_deviation = rain_arr - np.mean(rain_arr) # (rᵢ - r̄)
                kg_deviation = kg_arr - np.mean(kg_arr) # (yᵢ - ȳ)

                denominator = np.sum(rain_deviation ** 2) # Σ(rᵢ - r̄)²
                if denominator < 1e-8:
                    rainfall_response_beta = float(np.sum(rain_deviation * kg_deviation) / denominator)
    
    #FERTILIZER_EFFICIENCY
    # η = yield increase in month after application / kg of fertiliser applied
    # compares the month+1 yield to the season mean (baseline)
    # positive η means fertiliser lifted yield above baseline
    # negative η means yield was below baseline even after application
    efficiencies = []
    for s in seasons:
        kg = s.get("monthly_kg", [])
        fert_apps = s.get("fertilizer_applications", [])
        s_mean = np.mean([v for v in kg if v is not None]) if kg else 0
        if s_mean <= 0 or not fert_apps:
            continue
            for app in fert_applications:
                quantity = float(app.get("quantity_kg", 0))
                month_idx = app.get("season_month_idx")
                if quantity <= 0 or month_idx is None:
                    continue
                next_idx = month_idx + 1
                if next_idx < len(kg) and kg[next_idx] is not None:
                    delta_yield = kg[next_idx] - s_mean
                    efficiencies.append(delta_yield / quantity)

    fertiliser_efficiency = float(np.mean(efficiencies)) if efficiencies else 0.0 

    # Average rainfall from historical seasons
    avg_rainfall = float(np.mean(all_monthly_rains)) if all_monthly_rains else 100.0

    return {
        "season_yield_means":  padded,           # 15-dim yield history
        "overall_yield_mean":  float(np.mean(season_means)) if season_means else 0.0,
        "yield_trend_slope":   slope,
        "yield_standard_deviation": yield_std,
        "seasonal_amplitude":  amplitude,
        "pruning_recovery_rate": pruning_recovery_rate,
        "rainfall_response_beta": rainfall_response_beta,
        "fertiliser_efficiency":  fertiliser_efficiency,   
        "hectares":            float(farm.get("hectares", 1.0)),
        "altitude_m":          float(farm.get("altitude_m", 1700)),
        "avg_rainfall_mm":     avg_rainfall,
        "years_active":        len(seasons),
        "fairtrade":           int(farm.get("fairtrade_certified", False)),
        "n_seasons":           len(seasons),
    }


def build_edges(farms: list[dict], nodes: list[dict], alpha: float = 0.6) -> list[dict]:
    """
    Build edge list following roadmap edge rules.
    Returns list of {src, dst, weight, edge_type} dicts.
    Edges are undirected — each pair stored once (src < dst by index).
    """
    edges = []
    n = len(farms)

    for i in range(n):
        fi = farms[i]
        for j in range(i + 1, n):
            fj = farms[j]

            max_weight = 0.0
            edge_type  = None

            # Rule 1: shared collection centre (strongest signal)
            if fi["collection_centre"] == fj["collection_centre"]:
                max_weight = 1.0
                edge_type  = "same_centre"

            # Rule 2: shared factory zone
            elif fi["factory_code"] == fj["factory_code"]:
                max_weight = 0.6
                edge_type  = "same_factory"

            # Rule 3: GPS proximity < 5km
            lat_i = fi.get("latitude")
            lon_i = fi.get("longitude")
            lat_j = fj.get("latitude")
            lon_j = fj.get("longitude")

            if all(v is not None for v in [lat_i, lon_i, lat_j, lon_j]):
                dist_km = haversine_km(lat_i, lon_i, lat_j, lon_j)
                if dist_km < GPS_EDGE_THRESHOLD_KM and dist_km > 0:
                    gps_weight = 1.0 / dist_km   # inverse distance
                    # Normalise: max weight at 0.1km → 10.0, cap at 0.8
                    gps_weight = min(0.8, gps_weight / 10.0)
                    if gps_weight > max_weight:
                        max_weight = gps_weight
                        edge_type  = "gps_proximity"

            if max_weight > 0:
                edges.append({
                    "src":        i,
                    "dst":        j,
                    "src_member": fi["ktda_member_no"],
                    "dst_member": fj["ktda_member_no"],
                    "weight":     round(max_weight, 4),
                    "edge_type":  edge_type,
                })

    return edges


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate", action="store_true",
                        help="Print graph stats and feature distributions after building")
    parser.add_argument("--alpha", type=float, default=0.6,
                        help="0.0=pure feature similarity, 1.0=pure geography. Default 0.6")
    args = parser.parse_args()

    print(f"Connecting to {MONGODB_URI} ...")
    client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=5000)
    db     = client.get_default_database()
    db.command("ping")
    print("Connected.\n")

    # Load all farms
    farms = list(db.farms.find({}, {"_id": 0}))
    print(f"Loaded {len(farms)} farms.")

    # Build node feature dicts
    print("Extracting node features ...")
    nodes = []
    for i, farm in enumerate(farms):
        features = extract_node_features(farm)
        nodes.append({
            "node_idx":          i,
            "ktda_member_no":    farm["ktda_member_no"],
            "factory_code":      farm["factory_code"],
            "collection_centre": farm["collection_centre"],
            "latitude":          farm.get("latitude"),
            "longitude":         farm.get("longitude"),
            "features":          features,
        })

    # Build edge list
    print("Building edges ...")
    edges = build_edges(farms, nodes, alpha=args.alpha)

    # Edge type breakdown
    type_counts = {}
    for e in edges:
        type_counts[e["edge_type"]] = type_counts.get(e["edge_type"], 0) + 1

    print(f"  Nodes : {len(nodes)}")
    print(f"  Edges : {len(edges)}")
    for etype, count in type_counts.items():
        print(f"    {etype:<20}: {count}")

    # Save to MongoDB
    graph_doc = {
        "_id":        "farm_graph_v1",
        "n_nodes":    len(nodes),
        "n_edges":    len(edges),
        "nodes":      nodes,
        "edges":      edges,
        "edge_types": type_counts,
        "alpha":      args.alpha,
        "built_at":   datetime.utcnow().isoformat(),
        "status":     "built",
    }

    db.farm_graph.replace_one({"_id": "farm_graph_v1"}, graph_doc, upsert=True)
    print(f"\nSaved farm_graph_v1 to MongoDB.")

    if args.validate:
        degree = {}
        for e in edges:
            degree[e["src"]] = degree.get(e["src"], 0) + 1
            degree[e["dst"]] = degree.get(e["dst"], 0) + 1
        avg_degree = sum(degree.values()) / len(nodes) if nodes else 0
        isolated   = len(nodes) - len(degree)

        print(f"\n-- Graph connectivity --")
        print(f"  Avg degree per node : {avg_degree:.1f}")
        print(f"  Max degree          : {max(degree.values()) if degree else 0}")
        print(f"  Isolated nodes      : {isolated}  {'(bad)' if isolated > 5 else '(OK)'}")

        weights = [e["weight"] for e in edges]
        w_geo   = [e["w_geo"]  for e in edges]
        w_feat  = [e["w_feat"] for e in edges]
        print(f"\n-- Edge weight distribution --")
        print(f"  Combined  mean={np.mean(weights):.3f}  std={np.std(weights):.3f}")
        print(f"  Geo       mean={np.mean(w_geo):.3f}")
        print(f"  Feature   mean={np.mean(w_feat):.3f}")

        print(f"\n-- New feature distributions ({len(nodes)} farms) --")
        for key, label in [
            ("yield_std",              "Yield std (kg)"),
            ("seasonal_amplitude",     "Seasonal amplitude"),
            ("pruning_recovery_rate",  "Pruning recovery rate"),
            ("rainfall_response_beta", "Rainfall beta (kg/mm)"),
            ("fertiliser_efficiency",  "Fertiliser efficiency"),
        ]:
            vals = [n["features"][key] for n in nodes]
            print(f"  {label:<40} mean={np.mean(vals):7.3f}  "
                  f"std={np.std(vals):.3f}  min={min(vals):.3f}  max={max(vals):.3f}")

        print(f"\n-- Top 5 edges by combined weight --")
        for e in sorted(edges, key=lambda e: e["weight"], reverse=True)[:5]:
            print(f"  {e['src_member']} <-> {e['dst_member']:<12} "
                  f"w={e['weight']:.3f}  geo={e['w_geo']:.3f}  "
                  f"feat={e['w_feat']:.3f}  type={e['edge_type']}")

    print("\nStep done. Run train_gnn.py next.")
    print("Tip: re-run with different --alpha to tune edge structure:")
    print("  python scripts/build_graph.py --alpha 0.4 --validate  # more agronomic")
    print("  python scripts/build_graph.py --alpha 0.8 --validate  # more geographic")


if __name__ == "__main__":
    main()