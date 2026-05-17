"""
Phase 2 — Step 2: Train GraphSAGE GNN on the farm graph.

Loads the farm graph from MongoDB (built by build_graph.py),
trains a 2-layer GraphSAGE model, and writes cluster_id +
anomaly_score back to MongoDB for each farm node.

Results are stored in:
  - gnn_outputs collection: per-farm cluster_id, anomaly_score, embedding
  - farm_graph_v1 document: updated with "trained" status

Run on WSL host (CPU is fine for 240 nodes):
    pip install torch torch-geometric --break-system-packages
    python scripts/train_gnn.py
    python scripts/train_gnn.py --validate
    python scripts/train_gnn.py --epochs 300 --clusters 6
"""

import os
import argparse
from datetime import datetime
from pathlib import Path
import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.data import Data


import numpy as np
from pymongo import MongoClient

MONGODB_URI = os.getenv("MONGODB_URI")
if not MONGODB_URI:
    raise RuntimeError("MONGODB_URI environment variable is not set")
DATA_DIR      = Path(os.getenv("DATA_DIR", str(Path(__file__).resolve().parent.parent / "data")))
MODELS_DIR  = Path(os.getenv("MODELS_DIR", str(Path(__file__).resolve().parent.parent / "models")))


def load_graph(db) -> tuple[np.ndarray, np.ndarray, np.ndarray, list]:
    """
    Load graph from MongoDB.
    Returns (X, edge_index, edge_weights, node_meta).
    X shape: (n_nodes, n_features)
    edge_index shape: (2, n_edges) — both directions for undirected graph
    """
    doc = db.farm_graph.find_one({"_id": "farm_graph_v1"})
    if not doc:
        raise RuntimeError("farm_graph_v1 not found. Run build_graph.py first.")

    nodes     = doc["nodes"]
    edges     = doc["edges"]
    n_nodes   = doc["n_nodes"]

    # Build feature matrix
    # Feature order: season_yield_means(15) + scalar features(12)) = 27 dims
    feature_rows = []
    for node in nodes:
        f = node["features"]
        row = (
            f["season_yield_means"] +          # 15 dims yield history
            [
                f["overall_yield_mean"],        # 1 lifetime average monthly yield
                f["yield_trend_slope"],         # 1 is farm improving or declining over time
                f["yield_standard_deviation"],  # 1 yield volatility
                f["seasonal_amplitude"],        # 1 difference between peak and trough months
                f["pruning_recovery_rate"],     # 1 how quickly farm recovers yield after pruning events (flush vs expected rate)
                f["rainfall_response_beta"],         # 1 correlation of yield with rainfall patterns (yield sensitivity to rain)
                f["fertiliser_efficiency"],     # 1 how much yield increases per unit of fertilizer used (kg per kg fertilizer)
                f["hectares"],                  # 1 farm size
                f["altitude_m"],                # 1 elevation
                f["avg_rainfall_mm"],           # 1 mean rainfall
                f["years_active"],              # 1 seasons of data available
                f["fairtrade"],                 # 1 binary flag for fairtrade certification

            ]
        )
        feature_rows.append(row)

    X = np.array(feature_rows, dtype=np.float32)

    # Normalise per feature (zero mean, unit std)
    means = X.mean(axis=0)
    stds  = X.std(axis=0)
    stds[stds < 1e-8] = 1.0   # avoid divide-by-zero for constant features
    X = (X - means) / stds

    # Build edge index — undirected: add both directions
    src_list, dst_list, weight_list = [], [], []
    for e in edges:
        src_list.extend([e["src"], e["dst"]])
        dst_list.extend([e["dst"], e["src"]])
        weight_list.extend([e["weight"], e["weight"]])

    edge_index   = np.array([src_list, dst_list], dtype=np.int64)
    edge_weights = np.array(weight_list, dtype=np.float32)

    node_meta = [
        {
            "node_idx":          n["node_idx"],
            "ktda_member_no":    n["ktda_member_no"],
            "factory_code":      n["factory_code"],
            "collection_centre": n["collection_centre"],
        }
        for n in nodes
    ]

    print(f"  Graph loaded: {n_nodes} nodes, {len(edges)} edges ({edge_index.shape[1]} directed)")
    print(f"  Feature matrix: {X.shape}")
    return X, edge_index, edge_weights, node_meta, {"means": means, "stds": stds}


def train_graphsage(X, edge_index, edge_weights, n_clusters=6,
                    epochs=200, hidden_dim=32, dropout=0.3):
    """
    2-layer GraphSAGE using PyTorch Geometric.
    Returns node embeddings (n_nodes, hidden_dim).
    """
    print(f"\nTraining GraphSAGE: {epochs} epochs, {n_clusters} clusters, hidden={hidden_dim}")

    x           = torch.FloatTensor(X)
    edge_idx    = torch.LongTensor(edge_index)
    edge_w      = torch.FloatTensor(edge_weights)
    data        = Data(x=x, edge_index=edge_idx, edge_attr=edge_w)

    class GraphSAGE(torch.nn.Module):
        def __init__(self, in_dim, hidden_dim, out_dim):
            super().__init__()
            self.conv1 = SAGEConv(in_dim,     hidden_dim)
            self.conv2 = SAGEConv(hidden_dim, out_dim)
            self.drop  = torch.nn.Dropout(dropout)

        def forward(self, x, edge_index):
            x = self.conv1(x, edge_index)
            x = F.relu(x)
            x = self.drop(x)
            x = self.conv2(x, edge_index)
            return x

    model = GraphSAGE(in_dim=X.shape[1], hidden_dim=hidden_dim, out_dim=hidden_dim)
    optim = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    # Unsupervised training: graph autoencoder loss
    # Positive pairs: connected nodes should have similar embeddings
    # Negative pairs: random unconnected nodes should differ
    src_nodes = edge_idx[0]
    dst_nodes = edge_idx[1]

    model.train()
    for epoch in range(epochs):
        optim.zero_grad()
        embeddings = model(data.x, data.edge_index)

        # Positive loss: cosine similarity of connected pairs
        emb_src = F.normalize(embeddings[src_nodes], dim=1)
        emb_dst = F.normalize(embeddings[dst_nodes], dim=1)
        pos_loss = (1 - (emb_src * emb_dst).sum(dim=1)).mean()

        # Negative sampling: random node pairs
        n        = embeddings.size(0)
        neg_src  = torch.randint(0, n, (src_nodes.size(0),))
        neg_dst  = torch.randint(0, n, (src_nodes.size(0),))
        emb_ns   = F.normalize(embeddings[neg_src], dim=1)
        emb_nd   = F.normalize(embeddings[neg_dst], dim=1)
        neg_loss = F.relu((emb_ns * emb_nd).sum(dim=1) + 0.5).mean()

        loss = pos_loss + neg_loss
        loss.backward()
        optim.step()

        if (epoch + 1) % 50 == 0:
            print(f"  Epoch {epoch+1:3d}/{epochs}  loss={loss.item():.4f}  "
                  f"pos={pos_loss.item():.4f}  neg={neg_loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        embeddings = model(data.x, data.edge_index).numpy()

    return embeddings


def cluster_embeddings(embeddings: np.ndarray, n_clusters: int) -> np.ndarray:
    """K-means clustering on GNN embeddings → cluster_id per node."""
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    return km.fit_predict(embeddings)


def compute_anomaly_scores(embeddings: np.ndarray, cluster_ids: np.ndarray,
                            node_meta: list) -> np.ndarray:
    """
    Anomaly score per node: distance from its cluster centroid,
    normalised 0–1 within each cluster. Higher = more anomalous.
    """
    scores = np.zeros(len(embeddings))
    unique_clusters = np.unique(cluster_ids)

    for cid in unique_clusters:
        mask     = cluster_ids == cid
        cluster_embs = embeddings[mask]
        centroid = cluster_embs.mean(axis=0)
        dists    = np.linalg.norm(cluster_embs - centroid, axis=1)

        # Normalise within cluster
        d_min, d_max = dists.min(), dists.max()
        if d_max - d_min > 1e-8:
            norm_dists = (dists - d_min) / (d_max - d_min)
        else:
            norm_dists = np.zeros_like(dists)

        scores[mask] = norm_dists

    return scores


def save_results(db, node_meta: list, cluster_ids: np.ndarray,
                 anomaly_scores: np.ndarray, embeddings: np.ndarray):
    """Write per-farm GNN outputs to gnn_outputs collection."""
    now = datetime.utcnow().isoformat()
    ops = []
    for i, meta in enumerate(node_meta):
        doc = {
            "ktda_member_no":    meta["ktda_member_no"],
            "factory_code":      meta["factory_code"],
            "collection_centre": meta["collection_centre"],
            "cluster_id":        int(cluster_ids[i]),
            "anomaly_score":     round(float(anomaly_scores[i]), 4),
            "embedding":         embeddings[i].tolist(),
            "computed_at":       now,
        }
        from pymongo import UpdateOne
        ops.append(UpdateOne(
            {"ktda_member_no": meta["ktda_member_no"]},
            {"$set": doc},
            upsert=True,
        ))

    if ops:
        db.gnn_outputs.bulk_write(ops)

    # Update graph doc status
    db.farm_graph.update_one(
        {"_id": "farm_graph_v1"},
        {"$set": {"status": "trained", "trained_at": now,
                  "n_clusters": int(cluster_ids.max() + 1)}},
    )

    print(f"\nSaved {len(ops)} GNN output documents to gnn_outputs collection.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",   type=int, default=200)
    parser.add_argument("--clusters", type=int, default=6,
                        help="Number of farm clusters (default 6 = 3 per factory)")
    parser.add_argument("--hidden",   type=int, default=32)
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--skip-gnn", action="store_true",
                        help="Skip GNN, just run K-means on raw features (useful if PyG not installed)")
    args = parser.parse_args()

    print(f"Connecting to {MONGODB_URI} ...")
    client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=5000)
    db     = client.get_default_database()
    db.command("ping")
    print("Connected.\n")

    X, edge_index, edge_weights, node_meta, norm_params = load_graph(db)

    if args.skip_gnn:
        print("\n[--skip-gnn] Running K-means on raw normalised features only.")
        embeddings = X
    else:
        embeddings = train_graphsage(
            X, edge_index, edge_weights,
            n_clusters = args.clusters,
            epochs     = args.epochs,
            hidden_dim = args.hidden,
        )

    print("\nClustering embeddings ...")
    cluster_ids = cluster_embeddings(embeddings, n_clusters=args.clusters)

    print("Computing anomaly scores ...")
    anomaly_scores = compute_anomaly_scores(embeddings, cluster_ids, node_meta)

    # Cluster breakdown
    unique, counts = np.unique(cluster_ids, return_counts=True)
    print(f"\n-- Cluster breakdown --")
    for cid, cnt in zip(unique, counts):
        members_in_cluster = [m["ktda_member_no"] for m, c in zip(node_meta, cluster_ids) if c == cid]
        centres = list(set(m["collection_centre"] for m in node_meta
                           if cluster_ids[node_meta.index(m)] == cid))
        print(f"  Cluster {cid}: {cnt} farms  centres={centres[:3]}")

    # Top anomalies
    top_anomaly_idx = np.argsort(anomaly_scores)[-5:][::-1]
    print(f"\n-- Top 5 anomalous farms --")
    for idx in top_anomaly_idx:
        m = node_meta[idx]
        print(f"  {m['ktda_member_no']:<12} cluster={cluster_ids[idx]}  "
              f"anomaly_score={anomaly_scores[idx]:.3f}  centre={m['collection_centre']}")

    save_results(db, node_meta, cluster_ids, anomaly_scores, embeddings)

    # Save norm params for inference
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    import joblib
    joblib.dump({
        "means":      norm_params["means"].tolist(),
        "stds":       norm_params["stds"].tolist(),
        "n_clusters": args.clusters,
        "hidden_dim": args.hidden,
    }, MODELS_DIR / "gnn_norm_params.pkl")
    print(f"Saved normalisation params to models/gnn_norm_params.pkl")

    if args.validate:
        print(f"\n-- Anomaly score distribution --")
        print(f"  Mean  : {anomaly_scores.mean():.3f}")
        print(f"  Std   : {anomaly_scores.std():.3f}")
        print(f"  Max   : {anomaly_scores.max():.3f}")
        high_anomaly = (anomaly_scores > 0.7).sum()
        print(f"  > 0.7 : {high_anomaly} farms flagged as high anomaly")

    print("\nPhase 2 GNN training done.")
    print("Next: the API /insights endpoint will now include cluster_id and anomaly_score.")


if __name__ == "__main__":
    main()