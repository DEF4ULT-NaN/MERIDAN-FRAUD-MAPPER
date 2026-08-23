"""
Phase C — GraphSAGE node-classification model for the Fraud Network Mapper.

Reads `data/graph.json` (Phase B output), trains a 2-layer GraphSAGE to
predict per-account fraud labels, validates on a stratified hold-out split,
and saves:
  - `models/gnn.pt`           model state_dict + scaler + feature schema
  - `models/training_history.json`
  - `data/graph_gnn_scores.json`  same graph shape, `risk_score` swapped for
                                  GNN probabilities — ready for Phase D

Run as a script:
    python scripts/train_gnn.py
        [--graph data/graph.json]
        [--out models/gnn.pt]
        [--epochs 200]
        [--hidden 64]
        [--lr 0.01]
        [--seed 42]

Or import the public functions:
    from train_gnn import build_pyg_data, build_model, train_model, evaluate,
                          predict_scores, load_trained_model
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv


# ---------------------------------------------------------------------------
# Public config
# ---------------------------------------------------------------------------
@dataclass
class TrainConfig:
    hidden_dim: int = 64
    epochs: int = 200
    lr: float = 0.01
    dropout: float = 0.5
    weight_decay: float = 5e-4
    seed: int = 42
    test_size: float = 0.2
    patience: int = 30                # early-stopping patience (epochs)


# Feature column order — frozen here so training and inference agree.
FEATURE_COLUMNS: list[str] = [
    "n_logins",
    "n_transactions",
    "login_unique_ips",
    "login_unique_devices",
    "tx_total_amount",
    "tx_avg_amount",
    "tx_max_amount",
    "degree",
    "weighted_degree",
    "shared_ip_count",
    "shared_device_count",
    "shared_phone_count",
    "shared_bank_count",
    "account_age_days",
]


# ---------------------------------------------------------------------------
# Data conversion
# ---------------------------------------------------------------------------
def build_pyg_data(graph_payload: dict) -> tuple[Data, list[str]]:
    """Convert Phase B's graph.json payload into a torch_geometric.data.Data.

    Returns the Data object and the ordered list of node IDs so callers
    can map predictions back to account_ids.
    """
    nodes = graph_payload["nodes"]
    edges = graph_payload["edges"]

    # Stable node-id ordering
    node_ids: list[str] = [n["id"] for n in nodes]
    id_to_idx = {nid: i for i, nid in enumerate(node_ids)}

    # Feature matrix — fill missing values with 0.0 (defensive)
    x = np.zeros((len(nodes), len(FEATURE_COLUMNS)), dtype=np.float32)
    y = np.zeros(len(nodes), dtype=np.int64)
    for i, n in enumerate(nodes):
        for j, col in enumerate(FEATURE_COLUMNS):
            v = n.get("features", {}).get(col, 0.0)
            x[i, j] = float(v) if v is not None else 0.0
        y[i] = int(n.get("is_fraud", 0))

    # Edge index — undirected, so add both directions
    src_list: list[int] = []
    dst_list: list[int] = []
    edge_weights: list[float] = []
    for e in edges:
        s = id_to_idx[e["source"]]
        d = id_to_idx[e["target"]]
        w = float(e.get("weight", 1))
        src_list += [s, d]
        dst_list += [d, s]
        edge_weights += [w, w]

    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
    edge_weight = torch.tensor(edge_weights, dtype=torch.float32)

    data = Data(
        x=torch.from_numpy(x),
        edge_index=edge_index,
        edge_weight=edge_weight,
        y=torch.from_numpy(y),
        num_nodes=len(nodes),
    )
    return data, node_ids


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class FraudGraphSAGE(nn.Module):
    """2-layer GraphSAGE for binary node classification.

    Note: vanilla `SAGEConv` in PyG 2.8+ uses a mean aggregator that
    treats each neighbour equally — it does not accept per-edge weights
    in `forward`.  Edge weights are therefore not used here; the
    graph topology (who's connected to whom) is the dominant signal
    in planted-ring fraud, and the planted rings are fully-connected
    cliques anyway.  Edge weights remain in `Data.edge_weight` for
    inspection / future variants.
    """

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int = 2,
                 dropout: float = 0.5):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden_dim, aggr="mean")
        self.conv2 = SAGEConv(hidden_dim, out_dim, aggr="mean")
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_weight: torch.Tensor | None = None) -> torch.Tensor:
        h = self.conv1(x, edge_index)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.conv2(h, edge_index)
        return h


def build_model(in_dim: int, hidden_dim: int, out_dim: int = 2,
                dropout: float = 0.5) -> FraudGraphSAGE:
    return FraudGraphSAGE(in_dim=in_dim, hidden_dim=hidden_dim,
                          out_dim=out_dim, dropout=dropout)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_model(
    data: Data,
    cfg: TrainConfig | None = None,
    *,
    feature_scaler: StandardScaler | None = None,
) -> tuple[nn.Module, StandardScaler, list[dict]]:
    """Train FraudGraphSAGE.  Returns (model, fitted_scaler, history).

    The StandardScaler is fitted on the *training* feature matrix only,
    to prevent val/test leakage.  If `feature_scaler` is passed in, it
    is used instead (so inference can apply the same transform).
    """
    cfg = cfg or TrainConfig()
    _set_seed(cfg.seed)

    x = data.x.numpy()
    y = data.y.numpy()

    # Stratified split
    idx = np.arange(len(y))
    train_idx, val_idx = train_test_split(
        idx, test_size=cfg.test_size, random_state=cfg.seed,
        stratify=y,
    )

    # Scale features — fit on train only
    if feature_scaler is None:
        scaler = StandardScaler()
        scaler.fit(x[train_idx])
    else:
        scaler = feature_scaler
    x_scaled = scaler.transform(x)

    data.x = torch.from_numpy(x_scaled).float()

    # pos_weight for class imbalance (num_normal / num_fraud)
    n_fraud = int(y[train_idx].sum())
    n_normal = int(len(train_idx) - n_fraud)
    pos_weight = torch.tensor([n_normal / max(n_fraud, 1)], dtype=torch.float32)

    model = build_model(in_dim=x_scaled.shape[1], hidden_dim=cfg.hidden_dim,
                        dropout=cfg.dropout)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr,
                           weight_decay=cfg.weight_decay)
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor([1.0, pos_weight.item()]))

    train_mask = torch.zeros(len(y), dtype=torch.bool)
    val_mask = torch.zeros(len(y), dtype=torch.bool)
    train_mask[train_idx] = True
    val_mask[val_idx] = True

    best_val = float("inf")
    best_state = None
    patience_left = cfg.patience
    history: list[dict] = []

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        opt.zero_grad()
        logits = model(data.x, data.edge_index, data.edge_weight)
        loss = loss_fn(logits[train_mask], data.y[train_mask])
        loss.backward()
        opt.step()

        # Validation
        model.eval()
        with torch.no_grad():
            v_logits = model(data.x, data.edge_index, data.edge_weight)
            v_loss = loss_fn(v_logits[val_mask], data.y[val_mask]).item()
        history.append({"epoch": epoch, "train_loss": loss.item(),
                        "val_loss": v_loss})

        if v_loss < best_val - 1e-4:
            best_val = v_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            patience_left = cfg.patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"  early-stop at epoch {epoch} (best val_loss={best_val:.4f})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, scaler, history


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate(model: nn.Module, data: Data) -> dict[str, float]:
    """Compute accuracy / precision / recall / F1 / ROC-AUC on the full graph.

    The metric is over ALL nodes (not just val split) because we also
    use it to compute the 'ring-detection' number for the planted rings —
    see main() for that.
    """
    model.eval()
    with torch.no_grad():
        logits = model(data.x, data.edge_index, data.edge_weight)
        probs = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
        preds = logits.argmax(dim=1).cpu().numpy()
    y = data.y.cpu().numpy()

    out: dict[str, float] = {
        "accuracy": float(accuracy_score(y, preds)),
        "roc_auc": float(roc_auc_score(y, probs)),
        "fraud_recall": 0.0,
        "fraud_precision": 0.0,
        "fraud_f1": 0.0,
        "macro_f1": float(f1_score(y, preds, average="macro")),
    }
    p, r, f, _ = precision_recall_fscore_support(
        y, preds, labels=[0, 1], zero_division=0
    )
    out["fraud_precision"] = float(p[1])
    out["fraud_recall"] = float(r[1])
    out["fraud_f1"] = float(f[1])
    return out


def ring_detection_score(
    model: nn.Module,
    data: Data,
    node_ids: list[str],
    rings: list[dict],
) -> dict[str, Any]:
    """For each planted ring, what fraction of its members land in the
    top-K predictions (K = total fraud accounts)?  Returns per-ring and
    aggregate numbers.
    """
    model.eval()
    with torch.no_grad():
        logits = model(data.x, data.edge_index, data.edge_weight)
        probs = F.softmax(logits, dim=1)[:, 1].cpu().numpy()

    n_fraud = int(data.y.sum().item())
    # Indices of the top-K predicted fraud nodes
    top_idx = np.argsort(-probs)[:n_fraud]
    top_ids = {node_ids[i] for i in top_idx}

    per_ring: list[dict] = []
    rings_full: list[str] = []
    for r in rings:
        members = set(r["account_ids"])
        hit = members & top_ids
        frac = len(hit) / max(len(members), 1)
        per_ring.append({
            "ring_id": r["ring_id"],
            "size": len(members),
            "hit": len(hit),
            "fraction": round(frac, 3),
            "recovered": frac >= 0.8,
        })
        if frac >= 0.8:
            rings_full.append(r["ring_id"])
    return {
        "per_ring": per_ring,
        "rings_recovered": len(rings_full),
        "rings_total": len(rings),
        "ring_recovery_rate": round(len(rings_full) / max(len(rings), 1), 3),
    }


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
def predict_scores(
    model: nn.Module, data: Data, node_ids: list[str]
) -> dict[str, float]:
    """Return {account_id: probability_fraud} for every node."""
    model.eval()
    with torch.no_grad():
        logits = model(data.x, data.edge_index, data.edge_weight)
        probs = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
    return {nid: float(p) for nid, p in zip(node_ids, probs)}


def load_trained_model(path: Path) -> tuple[nn.Module, StandardScaler, dict]:
    """Load a saved model from disk.  Returns (model, scaler, metadata)."""
    blob = torch.load(path, weights_only=False, map_location="cpu")
    meta = blob["meta"]
    scaler = StandardScaler()
    scaler.mean_ = np.array(blob["scaler_mean"], dtype=np.float32)
    scaler.scale_ = np.array(blob["scaler_scale"], dtype=np.float32)
    model = build_model(
        in_dim=meta["in_dim"],
        hidden_dim=meta["hidden_dim"],
        out_dim=meta.get("out_dim", 2),
        dropout=meta.get("dropout", 0.5),
    )
    model.load_state_dict(blob["state_dict"])
    model.eval()
    return model, scaler, meta


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train GraphSAGE on the Phase B graph.json output."
    )
    parser.add_argument("--graph", type=Path, default=Path("data/graph.json"))
    parser.add_argument("--out", type=Path, default=Path("models/gnn.pt"))
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = TrainConfig(
        hidden_dim=args.hidden, epochs=args.epochs, lr=args.lr, seed=args.seed,
    )

    print(f"Loading {args.graph}…")
    payload = json.loads(args.graph.read_text())
    data, node_ids = build_pyg_data(payload)
    rings = payload.get("rings", [])
    print(f"  nodes : {data.num_nodes}  edges : {data.edge_index.shape[1] // 2}")
    print(f"  fraud : {int(data.y.sum())}  normal : {data.num_nodes - int(data.y.sum())}")

    print("Training GraphSAGE…")
    model, scaler, history = train_model(data, cfg)

    print("Evaluating on full graph…")
    metrics = evaluate(model, data)
    ring = ring_detection_score(model, data, node_ids, rings)
    for k, v in metrics.items():
        print(f"  {k:18s}: {v:.3f}")
    print(f"  ring_recovery     : {ring['rings_recovered']}/{ring['rings_total']} "
          f"rings >=80% recovered")

    # Save model
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "scaler_mean": scaler.mean_.tolist(),
            "scaler_scale": scaler.scale_.tolist(),
            "meta": {
                "in_dim": data.x.shape[1],
                "hidden_dim": cfg.hidden_dim,
                "out_dim": 2,
                "dropout": cfg.dropout,
                "feature_columns": FEATURE_COLUMNS,
                "node_ids": node_ids,
                "metrics": metrics,
                "ring_detection": ring,
            },
        },
        args.out,
    )
    history_path = args.out.parent / "training_history.json"
    history_path.write_text(json.dumps(history, indent=2))
    print(f"\nWrote {args.out}")
    print(f"Wrote {history_path}")

    # Save a graph.json clone with GNN scores swapped in
    scores = predict_scores(model, data, node_ids)
    gnn_payload = json.loads(json.dumps(payload))   # deep copy
    for n in gnn_payload["nodes"]:
        n["risk_score"] = round(scores.get(n["id"], 0.0), 4)
    gnn_payload["scoring"]["method"] = "gnn_graphsage"
    out_graph = args.graph.parent / "graph_gnn_scores.json"
    out_graph.write_text(json.dumps(gnn_payload, indent=2))
    print(f"Wrote {out_graph}")


if __name__ == "__main__":
    main()
