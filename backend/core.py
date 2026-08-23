"""
Phase D — core orchestration for the FastAPI backend.

Holds the in-memory session store and the loaded GNN model.  Imports
the Phase B (graph) and Phase C (GNN) modules so the API layer is a
thin shell over already-tested logic.
"""

from __future__ import annotations

import json
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

# Make sibling scripts/ importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from build_graph import (  # noqa: E402
    DataBundle,
    RiskWeights,
    build_graph,
    load_dataframes,
    rule_based_score,
    serialize_graph,
)
from train_gnn import (  # noqa: E402
    FEATURE_COLUMNS,
    build_pyg_data,
    load_trained_model,
    predict_scores,
)


MODEL_PATH = PROJECT_ROOT / "models" / "gnn.pt"
SESSION_TTL_SECONDS = 30 * 60


# ---------------------------------------------------------------------------
# Session store
# ---------------------------------------------------------------------------
@dataclass
class Session:
    """Everything needed to answer follow-up requests for one upload."""

    session_id: str
    created_at: float
    bundle: DataBundle
    graph_payload: dict
    scores: dict[str, float] = field(default_factory=dict)
    scoring_method: str = "rule_based"


class SessionStore:
    """Thread-safe-enough in-memory session cache.

    For a single-process demo server, the GIL makes dict ops safe.  In
    production this would be Redis or a database.
    """

    def __init__(self, ttl_seconds: int = SESSION_TTL_SECONDS):
        self.ttl = ttl_seconds
        self._store: dict[str, Session] = {}

    def create(self, bundle: DataBundle, graph_payload: dict) -> Session:
        sid = uuid.uuid4().hex
        sess = Session(
            session_id=sid,
            created_at=time.time(),
            bundle=bundle,
            graph_payload=graph_payload,
        )
        self._store[sid] = sess
        return sess

    def get(self, sid: str) -> Session:
        self._gc()
        sess = self._store.get(sid)
        if sess is None:
            raise KeyError(f"unknown or expired session: {sid}")
        return sess

    def _gc(self) -> None:
        now = time.time()
        expired = [k for k, v in self._store.items() if now - v.created_at > self.ttl]
        for k in expired:
            del self._store[k]


# ---------------------------------------------------------------------------
# Model holder
# ---------------------------------------------------------------------------
@dataclass
class ModelHolder:
    """Wraps the trained GNN so /score can answer without reloading."""

    loaded: bool = False
    model: Any = None
    scaler: StandardScaler | None = None
    meta: dict = field(default_factory=dict)

    def load(self, path: Path = MODEL_PATH) -> bool:
        if not path.exists():
            self.loaded = False
            return False
        model, scaler, meta = load_trained_model(path)
        self.model = model
        self.scaler = scaler
        self.meta = meta
        self.loaded = True
        return True


# ---------------------------------------------------------------------------
# Upload auto-detection
# ---------------------------------------------------------------------------
# Canonical column names -> list of acceptable aliases found in user CSVs.
# Order matters: when the SAME alias appears in two canonical entries
# (e.g. "value" in both `amount` and `value`), the FIRST canonical entry
# in this dict wins.  More specific / narrower meanings therefore go
# first so they don't get hijacked by generic ones.
COLUMN_ALIASES: dict[str, list[str]] = {
    "account_id":    ["account_id", "accountid", "user_id", "userid", "id"],
    "name":          ["name", "full_name", "user_name"],
    "email":         ["email", "email_address"],
    "age_days":      ["age_days", "account_age_days", "age"],
    "is_fraud":      ["is_fraud", "fraud", "label", "is_fraudster"],
    "ring_id":       ["ring_id", "ring", "cluster_id"],
    "login_id":      ["login_id", "session_id", "event_id"],
    "timestamp":     ["timestamp", "ts", "datetime", "event_time"],
    "ip_address":    ["ip_address", "src_ip", "ip"],
    "device_id":     ["device_id", "device_fingerprint", "device"],
    "user_agent":    ["user_agent", "ua"],
    "success":       ["success", "login_success", "ok"],
    "tx_id":         ["tx_id", "transaction_id"],
    "bank_account":  ["bank_account", "account_number", "iban", "bank"],
    "amount":        ["amount", "tx_amount"],  # `value` is reserved for shared_attributes
    "counterparty":  ["counterparty", "recipient", "payee"],
    "currency":      ["currency", "ccy"],
    "src":           ["src", "source", "account_a", "account_1", "from"],
    "dst":           ["dst", "destination", "account_b", "account_2", "to"],
    "attribute":     ["shared_attribute", "attribute", "attr", "type"],
    "value":         ["shared_value", "shared", "value"],
}


def _resolve_columns(df: pd.DataFrame, file_kind: str) -> tuple[dict[str, str], list[str]]:
    """Map df columns to canonical names using COLUMN_ALIASES.

    Returns ({canonical: original}, list of unmapped original columns).

    Resolution rule: collect every (alias, canonical) pair across all
    canonical entries, then sort by alias length descending and assign
    greedily.  This way the most specific alias (`shared_value`) wins
    over a generic one (`value`) regardless of dict order.
    """
    lower_to_orig = {c.lower().strip(): c for c in df.columns}
    candidates: list[tuple[str, str]] = []
    for canonical, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            if alias in lower_to_orig:
                candidates.append((alias, canonical))
    candidates.sort(key=lambda ac: -len(ac[0]))

    mapping: dict[str, str] = {}
    used: set[str] = set()
    for alias, canonical in candidates:
        orig = lower_to_orig[alias]
        if orig in used or canonical in mapping:
            continue
        mapping[canonical] = orig
        used.add(orig)
    unmapped = [c for c in df.columns if c not in used]
    return mapping, unmapped


# Required canonical columns per file kind
_REQUIRED = {
    "accounts":         ["account_id"],
    "logins":           ["account_id"],
    "transactions":     ["account_id"],
    "shared_attributes": ["src", "dst", "attribute", "value"],
}


def load_uploaded_files(files: dict[str, pd.DataFrame]) -> DataBundle:
    """Build a DataBundle from the user's uploaded files.

    `files` is a dict like `{"accounts": df, "logins": df, ...}`.  Each
    DataFrame is normalised (columns renamed via auto-detect, light type
    coercion) and then wrapped in a `DataBundle`.  Missing required
    columns raise a ValueError so the API layer can return HTTP 422.
    """
    normalised: dict[str, pd.DataFrame] = {}
    detected: dict[str, dict[str, str]] = {}

    for kind, df in files.items():
        if kind not in _REQUIRED:
            continue
        mapping, unmapped = _resolve_columns(df, kind)
        detected[kind] = mapping
        missing = [c for c in _REQUIRED[kind] if c not in mapping]
        if missing:
            raise ValueError(
                f"file '{kind}' missing required columns: {missing}; "
                f"got {list(df.columns)}"
            )
        # Build a renamed copy with only canonical columns + extras
        rename = {orig: canon for canon, orig in mapping.items()}
        norm = df.rename(columns=rename)
        # Coerce types where it matters
        if "account_id" in norm.columns:
            norm["account_id"] = norm["account_id"].astype(str)
        if "src" in norm.columns:
            norm["src"] = norm["src"].astype(str)
        if "dst" in norm.columns:
            norm["dst"] = norm["dst"].astype(str)
        if "age_days" in norm.columns:
            norm["age_days"] = pd.to_numeric(norm["age_days"], errors="coerce").fillna(30).astype(int)
        if "is_fraud" in norm.columns:
            norm["is_fraud"] = pd.to_numeric(norm["is_fraud"], errors="coerce").fillna(0).astype(int)
        normalised[kind] = norm

    # Default is_fraud / ring_id / age_days if missing
    if "accounts" in normalised:
        for col, default in (("is_fraud", 0), ("ring_id", ""), ("age_days", 365)):
            if col not in normalised["accounts"].columns:
                normalised["accounts"][col] = default
        for col, default in (("name", ""), ("email", "")):
            if col not in normalised["accounts"].columns:
                normalised["accounts"][col] = default

    return DataBundle(
        accounts=normalised.get("accounts", pd.DataFrame(columns=["account_id"])),
        logins=normalised.get("logins", pd.DataFrame(columns=["account_id"])),
        transactions=normalised.get("transactions", pd.DataFrame(columns=["account_id"])),
        shared_edges=normalised.get(
            "shared_attributes", pd.DataFrame(columns=["src", "dst", "attribute", "value"])
        ),
        rings=[],
    ), detected


# ---------------------------------------------------------------------------
# Build / score / explain
# ---------------------------------------------------------------------------
def build_graph_for_session(bundle: DataBundle, weights: RiskWeights | None = None
                            ) -> tuple[dict, dict[str, float]]:
    """Build graph + rule-based scores.  Returns (payload, scores)."""
    g = build_graph(bundle)
    weights = weights or RiskWeights()
    scores = rule_based_score(g, weights)
    payload = serialize_graph(g, scores, bundle.rings, weights)
    return payload, scores


def score_with_gnn(graph_payload: dict, holder: ModelHolder) -> dict[str, float]:
    """Run the loaded GNN over a graph payload and return {id: prob}."""
    if not holder.loaded:
        raise RuntimeError("GNN model is not loaded")
    data, node_ids = build_pyg_data(graph_payload)
    x_scaled = holder.scaler.transform(data.x.numpy())
    data.x = torch.from_numpy(x_scaled).float()
    return predict_scores(holder.model, data, node_ids)


def explain_account(graph_payload: dict, account_id: str,
                    holder: ModelHolder | None = None) -> dict[str, Any]:
    """Return a human-readable explanation for one account.

    The narrative is built from the node's risk score, its top incident
    edges (by weight), and a one-line cluster summary if the account is
    part of a known ring.  Designed to be both API-consumable (JSON) and
    drop-in text for the frontend's click-to-explain panel.
    """
    nodes_by_id = {n["id"]: n for n in graph_payload["nodes"]}
    if account_id not in nodes_by_id:
        raise KeyError(f"unknown account_id: {account_id}")

    node = nodes_by_id[account_id]
    score = float(node.get("risk_score", 0.0))
    score_pct = round(score * 100, 1)

    # Collect incident edges
    incident: list[tuple[dict, float]] = []
    for e in graph_payload["edges"]:
        if e["source"] == account_id or e["target"] == account_id:
            incident.append((e, float(e.get("weight", 1))))
    incident.sort(key=lambda t: -t[1])

    top_reasons: list[str] = []
    for e, _ in incident[:6]:
        # Prefer the strongest single reason per edge
        top_reasons.append(e["reasons"][0])

    # Cluster membership
    neighbours = {
        (e["target"] if e["source"] == account_id else e["source"])
        for e in graph_payload["edges"]
        if e["source"] == account_id or e["target"] == account_id
    }
    neighbour_ids = [nid for nid in neighbours if nid in nodes_by_id]
    cluster_size = 1 + len(neighbour_ids)
    cluster_risk = (
        sum(nodes_by_id[n].get("risk_score", 0.0) for n in neighbour_ids)
        / max(len(neighbour_ids), 1)
    )

    method = graph_payload.get("scoring", {}).get("method", "rule_based")

    headline = (
        f"Account {account_id} has a {score_pct}% fraud risk "
        f"({method})."
    )
    if incident:
        shared_count = len(incident)
        attrs = set()
        for e, _ in incident:
            for t in e.get("attribute_types", []):
                attrs.add(t)
        cluster_part = (
            f" It is connected to {shared_count} other account(s) "
            f"via shared {', '.join(sorted(attrs))}."
        )
    else:
        cluster_part = " It has no shared-attribute connections."

    if node.get("ring_id"):
        cluster_part += f" It belongs to cluster {node['ring_id']}."

    reasons_block = "\n".join(f"  - {r}" for r in top_reasons) or "  (no shared attributes)"

    narrative = (
        f"{headline}{cluster_part}\n\n"
        f"Top reasons:\n{reasons_block}\n\n"
        f"Cluster size: {cluster_size}.  Average neighbour risk: {cluster_risk*100:.1f}%."
    )

    return {
        "account_id": account_id,
        "name": node.get("name", ""),
        "risk_score": score,
        "risk_method": method,
        "ring_id": node.get("ring_id"),
        "cluster_size": cluster_size,
        "avg_neighbour_risk": round(cluster_risk, 4),
        "top_reasons": top_reasons,
        "incident_edges": [
            {"source": e["source"], "target": e["target"],
             "weight": int(e["weight"]), "reasons": e["reasons"][:5]}
            for e, _ in incident[:8]
        ],
        "narrative": narrative,
    }


def explain_ring(graph_payload: dict, ring_id: str) -> dict[str, Any]:
    """Return a plain-English summary for a planted ring."""
    for ring in graph_payload.get("rings", []):
        if ring.get("ring_id") == ring_id:
            members = ring.get("account_ids", [])
            nodes_by_id = {n["id"]: n for n in graph_payload["nodes"]}
            member_nodes = [nodes_by_id[m] for m in members if m in nodes_by_id]
            if not member_nodes:
                raise KeyError(f"ring {ring_id} has no resolvable members")
            risks = [float(n.get("risk_score", 0.0)) for n in member_nodes]
            avg_risk = sum(risks) / len(risks)
            top_member = max(member_nodes, key=lambda n: n.get("risk_score", 0.0))
            tightness = ring.get("tightness", "unknown")
            narrative = (
                f"Cluster {ring_id} contains {len(members)} accounts "
                f"({tightness} configuration).  Average risk score: "
                f"{avg_risk*100:.1f}%.  Highest-risk member: "
                f"{top_member['id']} at {top_member.get('risk_score', 0)*100:.1f}%."
            )
            return {
                "ring_id": ring_id,
                "size": len(members),
                "tightness": tightness,
                "avg_risk_score": round(avg_risk, 4),
                "top_member_id": top_member["id"],
                "members": [
                    {"id": n["id"], "risk_score": n.get("risk_score", 0.0)}
                    for n in member_nodes
                ],
                "shared_ips": ring.get("shared_ips", []),
                "shared_devices": ring.get("shared_devices", []),
                "shared_phones": ring.get("shared_phones", []),
                "shared_banks": ring.get("shared_banks", []),
                "narrative": narrative,
            }
    raise KeyError(f"unknown ring_id: {ring_id}")
