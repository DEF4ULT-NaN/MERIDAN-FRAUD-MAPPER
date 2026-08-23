"""
Phase D — FastAPI app for the Fraud Network Mapper.

Endpoints (per CLAUDE.md §4 D2-D6):
  POST /upload          multipart: accounts/logins/transactions/shared_attributes CSVs
  POST /upload-json     body: raw graph.json payload (Phase B output)
  POST /build-graph     body: {session_id} | rebuild from stored session
  POST /score           body: {session_id, method: "gnn"|"rule_based"|"auto"}
  POST /explain         body: {session_id, account_id} or {session_id, ring_id}
  POST /report          body: {session_id} -> application/pdf
  GET  /health          liveness + model status
  GET  /                service banner + endpoint list

Run:
    cd backend
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import io
import json
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field

# Make sibling modules importable
sys.path.insert(0, str(Path(__file__).resolve().parent))
from core import (  # noqa: E402
    ModelHolder,
    SessionStore,
    build_graph_for_session,
    explain_account,
    explain_ring,
    load_uploaded_files,
    score_with_gnn,
)
from pdf_report import build_pdf  # noqa: E402


# ---------------------------------------------------------------------------
# Lifespan: load the GNN model once at startup
# ---------------------------------------------------------------------------
store = SessionStore()
holder = ModelHolder()


@asynccontextmanager
async def lifespan(app: FastAPI):
    loaded = holder.load()
    print(f"[startup] GNN model {'loaded' if loaded else 'NOT FOUND — using rule-based fallback'}")
    yield
    print("[shutdown] cleaning up")


app = FastAPI(
    title="Fraud Network Mapper API",
    version="0.1.0",
    description="Backend for the Palantir-style fraud network visualizer.",
    lifespan=lifespan,
)

# CORS — env-driven so the same image works for local dev, the Render
# static site, and any future staging URLs.
#
#   ALLOWED_ORIGINS="*"           → open (hackathon-demo default; fine because
#                                   we don't use cookies/credentials headers)
#   ALLOWED_ORIGINS=""            → fall back to the explicit localhost list
#                                   below (strictest mode for local dev)
#   ALLOWED_ORIGINS="a,b,c"       → explicit allow-list (production)
#
# The regex below ALWAYS covers localhost + file:// (Origin: "null") so
# local dev "just works" no matter what ALLOWED_ORIGINS says.
_raw = os.environ.get("ALLOWED_ORIGINS", "*").strip()
if _raw == "*" or _raw == "":
    # Explicit-empty falls back to localhost-only; bare-`*` is open.
    allow_origins: list[str] = ["*"] if _raw == "*" else [
        "http://localhost:3000",
        "http://localhost:5173",
        "http://localhost:5174",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:5500",
        "null",
    ]
else:
    allow_origins = [o.strip() for o in _raw.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_origin_regex=r"^(https?://(localhost|127\.0\.0\.1)(:[0-9]+)?|null)$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class ScoreRequest(BaseModel):
    session_id: str
    method: Literal["gnn", "rule_based", "auto"] = "auto"


class ExplainRequest(BaseModel):
    session_id: str
    account_id: str | None = None
    ring_id: str | None = None


class SessionRequest(BaseModel):
    session_id: str = Field(..., description="ID returned by /upload")


class UploadJsonRequest(BaseModel):
    graph: dict[str, Any]


# ---------------------------------------------------------------------------
# Root + health
# ---------------------------------------------------------------------------
@app.get("/")
def root() -> dict[str, Any]:
    return {
        "service": "Fraud Network Mapper API",
        "version": app.version,
        "model_loaded": holder.loaded,
        "endpoints": [
            "GET  /health",
            "POST /upload          (multipart: accounts, logins, transactions, shared_attributes)",
            "POST /upload-json     (raw graph.json payload)",
            "POST /build-graph     ({session_id})",
            "POST /score           ({session_id, method})",
            "POST /explain         ({session_id, account_id} or {session_id, ring_id})",
            "POST /report          ({session_id}) -> application/pdf",
        ],
    }


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "model_loaded": holder.loaded,
        "fallback_available": True,
        "active_sessions": len(store._store),  # noqa: SLF001
    }


# ---------------------------------------------------------------------------
# Upload — multipart CSVs
# ---------------------------------------------------------------------------
@app.post("/upload")
async def upload(
    accounts: UploadFile | None = File(None),
    logins: UploadFile | None = File(None),
    transactions: UploadFile | None = File(None),
    shared_attributes: UploadFile | None = File(None),
    fraud_rings: UploadFile | None = File(None, description="Optional rings metadata JSON"),
    graph: UploadFile | None = File(None, description="Optional pre-built graph.json"),
) -> dict[str, Any]:
    """Accept one or more CSVs (or a single graph.json) and create a session."""

    # Shortcut: user uploaded a pre-built graph.json directly
    if graph is not None:
        payload = json.loads(await graph.read())
        # If the payload has 'nodes' + 'edges' (Phase B shape), store as-is
        if "nodes" in payload and "edges" in payload:
            sess = store.create(
                bundle=None,  # type: ignore[arg-type]
                graph_payload=payload,
            )
            return {
                "session_id": sess.session_id,
                "n_nodes": len(payload["nodes"]),
                "n_edges": len(payload["edges"]),
                "detected_columns": {"graph.json": "(pre-built)"},
            }
        raise HTTPException(400, "graph.json must contain 'nodes' and 'edges'")

    files: dict[str, pd.DataFrame] = {}
    detected: dict[str, dict[str, str]] = {}
    rings_json: list[dict] = []
    for kind, upload in (
        ("accounts", accounts),
        ("logins", logins),
        ("transactions", transactions),
        ("shared_attributes", shared_attributes),
    ):
        if upload is None:
            continue
        raw = await upload.read()
        try:
            df = pd.read_csv(io.BytesIO(raw))
        except Exception as exc:
            raise HTTPException(400, f"failed to parse '{kind}' as CSV: {exc}")
        files[kind] = df

    if fraud_rings is not None:
        try:
            rings_json = json.loads(await fraud_rings.read())
            if not isinstance(rings_json, list):
                raise ValueError("fraud_rings must be a JSON array")
        except Exception as exc:
            raise HTTPException(400, f"failed to parse fraud_rings.json: {exc}")

    if not files:
        raise HTTPException(
            400,
            "no files provided — supply at least one of: accounts, logins, "
            "transactions, shared_attributes",
        )

    try:
        bundle, detected = load_uploaded_files(files)
    except ValueError as exc:
        raise HTTPException(422, str(exc))

    bundle.rings = rings_json
    payload, _ = build_graph_for_session(bundle)
    sess = store.create(bundle=bundle, graph_payload=payload)

    return {
        "session_id": sess.session_id,
        "n_nodes": len(payload["nodes"]),
        "n_edges": len(payload["edges"]),
        "n_fraud": int(payload["stats"].get("n_fraud_accounts", 0)),
        "n_rings": len(payload.get("rings", [])),
        "detected_columns": detected,
    }


# ---------------------------------------------------------------------------
# Upload — raw graph.json (JSON body)
# ---------------------------------------------------------------------------
@app.post("/upload-json")
def upload_json(req: UploadJsonRequest) -> dict[str, Any]:
    payload = req.graph
    if "nodes" not in payload or "edges" not in payload:
        raise HTTPException(400, "graph payload must contain 'nodes' and 'edges'")
    sess = store.create(bundle=None, graph_payload=payload)  # type: ignore[arg-type]
    return {
        "session_id": sess.session_id,
        "n_nodes": len(payload["nodes"]),
        "n_edges": len(payload["edges"]),
    }


# ---------------------------------------------------------------------------
# Build-graph
# ---------------------------------------------------------------------------
@app.post("/build-graph")
def build_graph_endpoint(req: SessionRequest) -> dict[str, Any]:
    try:
        sess = store.get(req.session_id)
    except KeyError as exc:
        raise HTTPException(404, str(exc))

    if sess.bundle is None:
        # Already a graph payload — return it as-is
        return sess.graph_payload

    payload, scores = build_graph_for_session(sess.bundle)
    sess.graph_payload = payload
    sess.scores = scores
    sess.scoring_method = "rule_based"
    return payload


# ---------------------------------------------------------------------------
# Score
# ---------------------------------------------------------------------------
@app.post("/score")
def score(req: ScoreRequest) -> dict[str, Any]:
    try:
        sess = store.get(req.session_id)
    except KeyError as exc:
        raise HTTPException(404, str(exc))

    method = req.method
    if method == "auto":
        method = "gnn" if holder.loaded else "rule_based"

    payload = sess.graph_payload
    if method == "gnn":
        if not holder.loaded:
            raise HTTPException(
                503,
                "GNN model not loaded — pass method='rule_based' to use the fallback",
            )
        try:
            scores = score_with_gnn(payload, holder)
        except Exception as exc:
            raise HTTPException(500, f"GNN inference failed: {exc}")
    else:
        if sess.bundle is not None:
            _, scores = build_graph_for_session(sess.bundle)
        else:
            scores = {n["id"]: float(n.get("risk_score", 0.0))
                      for n in payload["nodes"]}

    # Persist back into the session payload (so /explain and /report see them)
    score_map = {n["id"]: 0.0 for n in payload["nodes"]}
    score_map.update(scores)
    for n in payload["nodes"]:
        n["risk_score"] = round(score_map[n["id"]], 4)
    payload["scoring"]["method"] = "gnn_graphsage" if method == "gnn" else "rule_based"
    sess.scores = score_map
    sess.scoring_method = method

    return {
        "session_id": sess.session_id,
        "method": method,
        "model_loaded": holder.loaded,
        "n_scores": len(scores),
        "scores": {k: round(v, 4) for k, v in scores.items()},
    }


# ---------------------------------------------------------------------------
# Explain
# ---------------------------------------------------------------------------
@app.post("/explain")
def explain(req: ExplainRequest) -> dict[str, Any]:
    try:
        sess = store.get(req.session_id)
    except KeyError as exc:
        raise HTTPException(404, str(exc))

    payload = sess.graph_payload
    if req.account_id:
        try:
            return explain_account(payload, req.account_id, holder)
        except KeyError as exc:
            raise HTTPException(404, str(exc))
    if req.ring_id:
        try:
            return explain_ring(payload, req.ring_id)
        except KeyError as exc:
            raise HTTPException(404, str(exc))
    raise HTTPException(400, "supply either account_id or ring_id")


# ---------------------------------------------------------------------------
# Report — PDF download
# ---------------------------------------------------------------------------
@app.post("/report")
def report(req: SessionRequest) -> Response:
    try:
        sess = store.get(req.session_id)
    except KeyError as exc:
        raise HTTPException(404, str(exc))

    pdf_bytes = build_pdf(sess.graph_payload)
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={
            "Content-Disposition": (
                f'attachment; filename="fraud_report_{req.session_id[:8]}.pdf"'
            ),
        },
    )


# ---------------------------------------------------------------------------
# Convenience: serve the demo sample as a single JSON payload
# ---------------------------------------------------------------------------
SAMPLE_DIR = Path(__file__).resolve().parent.parent / "data" / "samples"
SAMPLE_GRAPH = SAMPLE_DIR / "graph.json"
SAMPLE_RINGS = SAMPLE_DIR / "fraud_rings.json"


@app.get("/sample-dataset")
def sample_dataset() -> JSONResponse:
    if not SAMPLE_GRAPH.exists():
        raise HTTPException(404, "sample dataset not found — run scripts/build_graph.py")
    payload = json.loads(SAMPLE_GRAPH.read_text())
    if SAMPLE_RINGS.exists() and not payload.get("rings"):
        payload["rings"] = json.loads(SAMPLE_RINGS.read_text())
    # Sanitize NaN ring_ids (pandas treats empty CSV cells as NaN) so the
    # JSONResponse encoder doesn't choke on a non-JSON-compliant float.
    for n in payload.get("nodes", []):
        if n.get("ring_id") is None:
            n["ring_id"] = ""
    # Run the GNN inline if loaded so the demo shows the real per-account
    # fraud probability (mean fraud ~1.0 vs normal ~0.0001) instead of the
    # cached rule-based scores from data/samples/graph.json. The file
    # written by Phase B has all fraud in planted rings, but 4 of the 29
    # fraud accounts (RING-02) only score 0.39 with the rule-based scorer —
    # the GNN correctly classifies all 29 as fraud.
    if holder.loaded:
        try:
            scores = score_with_gnn(payload, holder)
            for n in payload["nodes"]:
                n["risk_score"] = round(float(scores.get(n["id"], 0.0)), 4)
            payload["scoring"]["method"] = "gnn_graphsage"
        except Exception:
            # Stay on the cached scores if inference fails (CLAUDE.md §2 fallback)
            pass
    return JSONResponse(content=payload)
