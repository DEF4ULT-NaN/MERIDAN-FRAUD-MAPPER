"""
Phase B — Graph construction, node features, and rule-based risk scorer
for the Fraud Network Mapper.

Reads the Phase A outputs (in `data/` by default) and produces:
  - an in-memory `networkx.Graph` with accounts as nodes and collapsed
    shared-attribute edges carrying a `weight` and a `reasons` list
  - a JSON serialisation at `data/graph.json` that Phase D (FastAPI) and
    Phase E (React) consume directly
  - rule-based per-node risk scores (the fallback path the GNN will
    eventually replace — see CLAUDE.md §2 fallback rule)

Run as a script:
    python scripts/build_graph.py [--data-dir data] [--out data/graph.json]

Or import the public functions:
    from build_graph import load_dataframes, build_graph, rule_based_score
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import networkx as nx
import pandas as pd


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------
@dataclass
class DataBundle:
    """All inputs needed to build the graph, loaded once."""

    accounts: pd.DataFrame
    logins: pd.DataFrame
    transactions: pd.DataFrame
    shared_edges: pd.DataFrame
    rings: list[dict] = field(default_factory=list)


@dataclass
class RiskWeights:
    """Weights for the rule-based risk score. Tunable via CLI in Phase D.

    Only graph-derived features participate — they are the discriminative
    signal in the planted-ring data. Per-account stats (logins / tx / age)
    are computed and exposed as `features` for the GNN (Phase C) and the
    explain panel, but they would add pure noise to this scorer: fraud
    and normal accounts were generated with identical random ranges for
    those variables, so any weight on them shifts both classes equally.
    """

    degree: float = 0.20
    weighted_degree: float = 0.40
    shared_ip: float = 0.15
    shared_device: float = 0.10
    shared_phone: float = 0.08
    shared_bank: float = 0.07


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------
def load_dataframes(data_dir: Path) -> DataBundle:
    """Load the Phase A outputs into a `DataBundle`."""
    data_dir = Path(data_dir)
    # `keep_default_na=False` keeps empty CSV cells as "" instead of NaN,
    # so empty `ring_id` values don't become a non-JSON-compliant float
    # when the payload is serialised for the API.
    accounts = pd.read_csv(
        data_dir / "accounts.csv", dtype={"ring_id": str}, keep_default_na=False
    )
    logins = pd.read_csv(data_dir / "logins.csv")
    transactions = pd.read_csv(data_dir / "transactions.csv")
    shared_edges = pd.read_csv(data_dir / "shared_attributes.csv")
    rings_path = data_dir / "fraud_rings.json"
    rings: list[dict] = []
    if rings_path.exists():
        rings = json.loads(rings_path.read_text())
    return DataBundle(
        accounts=accounts,
        logins=logins,
        transactions=transactions,
        shared_edges=shared_edges,
        rings=rings,
    )


# ---------------------------------------------------------------------------
# B1 — Graph construction
# ---------------------------------------------------------------------------
# Human-readable reason labels for each attribute type. Kept short because
# they end up in the frontend's click-to-explain panel.
_REASON_LABEL = {
    "ip": "shared ip {value}",
    "device": "shared device {value}",
    "phone": "shared phone {value}",
    "bank_account": "shared bank account {value}",
}


def build_graph(bundle: DataBundle) -> nx.Graph:
    """Convert tabular data into a `networkx.Graph`.

    Nodes: accounts.  Edges: one per unique (src, dst) pair, carrying
    `weight` (count of shared attributes), `reasons` (list of strings),
    and `attribute_types` (set of {ip, device, phone, bank_account}).
    """
    g = nx.Graph()

    # --- nodes ---
    for _, row in bundle.accounts.iterrows():
        g.add_node(
            row.account_id,
            # NOTE: must use `row["name"]` (column) — `row.name` is pandas'
            # built-in attribute that returns the row's index label
            # (0, 1, 2, ...), which silently clobbers the real name column
            # when accounts.csv has a "name" header.
            name=row["name"],
            email=row["email"],
            age_days=int(row["age_days"]),
            is_fraud=int(row["is_fraud"]),
            ring_id=row["ring_id"] or "",
        )

    # --- edges (collapse shared_attributes.csv) ---
    # Group rows by unordered pair so an account pair with several shared
    # attributes becomes a single weighted edge.
    edge_groups: dict[tuple[str, str], list[dict]] = {}
    for _, row in bundle.shared_edges.iterrows():
        a, b = row.src, row.dst
        key = tuple(sorted((a, b)))
        edge_groups.setdefault(key, []).append(
            {"attribute": row.attribute, "value": row.value}
        )

    for (a, b), items in edge_groups.items():
        types = {it["attribute"] for it in items}
        reasons = [
            _REASON_LABEL[it["attribute"]].format(value=it["value"])
            for it in items
        ]
        g.add_edge(
            a,
            b,
            weight=len(items),
            reasons=reasons,
            attribute_types=sorted(types),
        )

    # --- B2: node features ---
    _attach_node_features(g, bundle)

    return g


def _attach_node_features(g: nx.Graph, bundle: DataBundle) -> None:
    """Compute per-account features from raw events + graph topology."""
    # Per-account login aggregations
    login_grp = bundle.logins.groupby("account_id")
    n_logins = login_grp.size().to_dict()
    unique_ips = login_grp["ip_address"].nunique().to_dict()
    unique_devices = login_grp["device_id"].nunique().to_dict()

    # Per-account transaction aggregations
    tx_grp = bundle.transactions.groupby("account_id")
    n_tx = tx_grp.size().to_dict()
    tx_total = tx_grp["amount"].sum().to_dict()
    tx_avg = tx_grp["amount"].mean().to_dict()
    tx_max = tx_grp["amount"].max().to_dict()

    for node, attrs in g.nodes(data=True):
        aid = node
        # Per-incident-edge counts of each attribute type
        shared_counts = {"ip": 0, "device": 0, "phone": 0, "bank_account": 0}
        for nbr, eattrs in g.adj[aid].items():
            for t in eattrs.get("attribute_types", []):
                shared_counts[t] = shared_counts.get(t, 0) + 1

        # weighted_degree: sum of edge weights (= total shared attrs)
        # over neighbours (count each shared attr, not each neighbour)
        weighted_degree = sum(
            eattrs.get("weight", 1) for eattrs in g.adj[aid].values()
        )

        attrs["features"] = {
            "n_logins": int(n_logins.get(aid, 0)),
            "n_transactions": int(n_tx.get(aid, 0)),
            "login_unique_ips": int(unique_ips.get(aid, 0)),
            "login_unique_devices": int(unique_devices.get(aid, 0)),
            "tx_total_amount": float(tx_total.get(aid, 0.0)),
            "tx_avg_amount": float(tx_avg.get(aid, 0.0)),
            "tx_max_amount": float(tx_max.get(aid, 0.0)),
            "degree": int(g.degree(aid)),
            "weighted_degree": int(weighted_degree),
            "shared_ip_count": int(shared_counts["ip"]),
            "shared_device_count": int(shared_counts["device"]),
            "shared_phone_count": int(shared_counts["phone"]),
            "shared_bank_count": int(shared_counts["bank_account"]),
            "account_age_days": int(attrs.get("age_days", 0)),
        }


# ---------------------------------------------------------------------------
# B3 — Rule-based risk scorer
# ---------------------------------------------------------------------------
def _min_max(values: dict[str, float]) -> dict[str, float]:
    """Return {node_id: normalized_value ∈ [0,1]} from raw values.

    Accepts a dict so we never accidentally iterate over keys in `min()`/`max()`
    and try to subtract strings.  Returns all zeros when every value is equal
    (avoids division-by-zero and keeps the score flat on degenerate inputs).
    """
    if not values:
        return {}
    lo = min(values.values())
    hi = max(values.values())
    if hi == lo:
        return {k: 0.0 for k in values}
    return {k: (v - lo) / (hi - lo) for k, v in values.items()}


def _sigmoid(x: float, k: float = 6.0, midpoint: float = 0.5) -> float:
    """Squash a value into (0, 1) with a sharp-ish curve for demo contrast.

    Standard logistic sigmoid centred on `midpoint` with steepness `k`:
        sigma(x) = 1 / (1 + exp(-k * (x - midpoint)))
    so values above the midpoint map toward 1.0 and values below toward 0.0.
    """
    z = k * (x - midpoint)
    if z >= 0:
        ez = math.exp(-z)
        return 1.0 / (1.0 + ez)
    ez = math.exp(z)
    return ez / (1.0 + ez)


def rule_based_score(
    g: nx.Graph,
    weights: RiskWeights | None = None,
) -> dict[str, float]:
    """Compute a 0.0–1.0 risk score per node.

    Each component is min-max normalised across the graph so the weights
    stay interpretable, then passed through a sigmoid for crisp demo
    contrast.  Planted fraud-ring members should land near 0.9–1.0;
    isolated normal accounts near 0.0.

    Only graph-derived signals are used (degree, weighted_degree, and the
    per-attribute shared counts). Per-account stats (logins, tx, age) are
    available as `features` for the GNN and the explain panel but are
    deliberately excluded here — they were generated with identical
    random ranges for fraud and normal accounts, so weighting them would
    just add noise.
    """
    w = weights or RiskWeights()
    nodes = list(g.nodes)

    # Pull feature vectors out once
    feats = {n: g.nodes[n].get("features", {}) for n in nodes}

    raw: dict[str, dict[str, float]] = {}
    for n in nodes:
        f = feats[n]
        raw[n] = {
            "degree": float(f.get("degree", 0)),
            "weighted_degree": float(f.get("weighted_degree", 0)),
            "shared_ip": float(f.get("shared_ip_count", 0)),
            "shared_device": float(f.get("shared_device_count", 0)),
            "shared_phone": float(f.get("shared_phone_count", 0)),
            "shared_bank": float(f.get("shared_bank_count", 0)),
        }

    # Min-max normalise each component across all nodes
    norm: dict[str, dict[str, float]] = {}
    for comp in raw[nodes[0]].keys():
        m = {n: raw[n][comp] for n in nodes}
        scaled = _min_max(m)
        for n, v in scaled.items():
            norm.setdefault(n, {})[comp] = v

    scores: dict[str, float] = {}
    for n in nodes:
        c = norm[n]
        x = (
            w.degree * c["degree"]
            + w.weighted_degree * c["weighted_degree"]
            + w.shared_ip * c["shared_ip"]
            + w.shared_device * c["shared_device"]
            + w.shared_phone * c["shared_phone"]
            + w.shared_bank * c["shared_bank"]
        )
        scores[n] = round(_sigmoid(x), 4)

    return scores


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------
def serialize_graph(
    g: nx.Graph,
    scores: dict[str, float],
    rings: list[dict],
    weights: RiskWeights,
) -> dict[str, Any]:
    """Produce a JSON-friendly dict matching Phase D / Phase E expectations."""
    nodes = []
    for n, attrs in g.nodes(data=True):
        nodes.append(
            {
                "id": n,
                "name": attrs.get("name", ""),
                "email": attrs.get("email", ""),
                "is_fraud": int(attrs.get("is_fraud", 0)),
                "ring_id": attrs.get("ring_id", "") or None,
                "age_days": int(attrs.get("age_days", 0)),
                "features": attrs.get("features", {}),
                "risk_score": float(scores.get(n, 0.0)),
            }
        )

    edges = []
    for u, v, attrs in g.edges(data=True):
        edges.append(
            {
                "source": u,
                "target": v,
                "weight": int(attrs.get("weight", 1)),
                "reasons": list(attrs.get("reasons", [])),
                "attribute_types": list(attrs.get("attribute_types", [])),
            }
        )

    return {
        "nodes": nodes,
        "edges": edges,
        "rings": rings,
        "scoring": {
            "method": "rule_based",
            "weights": {
                "degree": weights.degree,
                "weighted_degree": weights.weighted_degree,
                "shared_ip": weights.shared_ip,
                "shared_device": weights.shared_device,
                "shared_phone": weights.shared_phone,
                "shared_bank": weights.shared_bank,
            },
        },
        "stats": {
            "n_nodes": len(nodes),
            "n_edges": len(edges),
            "n_rings": len(rings),
            "n_fraud_accounts": sum(1 for n in nodes if n["is_fraud"]),
        },
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a fraud-network graph from Phase A outputs."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Directory containing accounts.csv, logins.csv, etc.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output JSON path. Defaults to <data-dir>/graph.json.",
    )
    args = parser.parse_args()

    print(f"Loading data from {args.data_dir}…")
    bundle = load_dataframes(args.data_dir)

    print("Building graph…")
    g = build_graph(bundle)

    print("Computing rule-based risk scores…")
    weights = RiskWeights()
    scores = rule_based_score(g, weights)

    out_path = args.out or (args.data_dir / "graph.json")
    payload = serialize_graph(g, scores, bundle.rings, weights)
    out_path.write_text(json.dumps(payload, indent=2))

    stats = payload["stats"]
    print(
        f"  nodes : {stats['n_nodes']}  "
        f"edges : {stats['n_edges']}  "
        f"rings : {stats['n_rings']}  "
        f"fraud : {stats['n_fraud_accounts']}"
    )
    # Quick contrast check
    fraud_scores = [n["risk_score"] for n in payload["nodes"] if n["is_fraud"]]
    normal_scores = [n["risk_score"] for n in payload["nodes"] if not n["is_fraud"]]
    if fraud_scores and normal_scores:
        mean_fraud = sum(fraud_scores) / len(fraud_scores)
        mean_normal = sum(normal_scores) / len(normal_scores)
        print(
            f"  mean risk | fraud : {mean_fraud:.3f}  "
            f"normal : {mean_normal:.3f}  ratio : "
            f"{(mean_fraud / max(mean_normal, 1e-9)):.2f}x"
        )

    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
