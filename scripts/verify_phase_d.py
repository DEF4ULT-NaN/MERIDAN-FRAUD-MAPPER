"""Phase D — End-to-End Verification (script form for clean quoting)."""
import json
import sys
from pathlib import Path

import httpx

BASE = "http://127.0.0.1:8000"
PROJECT = Path(r"C:\Users\Kushal Baroi\OneDrive\Documents\HACKATHON")
DATA = PROJECT / "data"

passed, failed = 0, 0


def check(name, ok, detail=""):
    global passed, failed
    tag = "PASS" if ok else "FAIL"
    print(f"  [{tag}] {name} {detail}")
    if ok:
        passed += 1
    else:
        failed += 1


print("=" * 70)
print("Phase D - End-to-End Verification")
print("=" * 70)

# --- 1. /health ---
r = httpx.get(f"{BASE}/health", timeout=10)
check("1. /health -> 200", r.status_code == 200, f"(status={r.status_code})")
body = r.json()
check(
    "1a. model_loaded",
    body.get("model_loaded") is True,
    f"(model_loaded={body.get('model_loaded')})",
)

# --- 2. /upload ---
print("\n[2] POST /upload (multipart)")
files = {
    "accounts": (
        "accounts.csv",
        (DATA / "accounts.csv").read_bytes(),
        "text/csv",
    ),
    "logins": (
        "logins.csv",
        (DATA / "logins.csv").read_bytes(),
        "text/csv",
    ),
    "transactions": (
        "transactions.csv",
        (DATA / "transactions.csv").read_bytes(),
        "text/csv",
    ),
    "shared_attributes": (
        "shared_attributes.csv",
        (DATA / "shared_attributes.csv").read_bytes(),
        "text/csv",
    ),
    "fraud_rings": (
        "fraud_rings.json",
        (DATA / "fraud_rings.json").read_bytes(),
        "application/json",
    ),
}
r = httpx.post(f"{BASE}/upload", files=files, timeout=30)
check(
    "2. /upload -> 200",
    r.status_code == 200,
    f"(status={r.status_code}, body={r.text[:200] if r.status_code != 200 else ''})",
)
if r.status_code == 200:
    up = r.json()
    sid = up["session_id"]
    print(
        f"     session_id = {sid}\n"
        f"     n_nodes={up['n_nodes']} n_edges={up['n_edges']} "
        f"n_fraud={up['n_fraud']} n_rings={up['n_rings']}"
    )
    check(
        "2a. n_nodes > 0",
        up.get("n_nodes", 0) > 0,
        f"(n_nodes={up.get('n_nodes')})",
    )
    check(
        "2b. n_rings == 5",
        up.get("n_rings") == 5,
        f"(n_rings={up.get('n_rings')})",
    )

# --- 3. /build-graph ---
print("\n[3] POST /build-graph")
r = httpx.post(f"{BASE}/build-graph", json={"session_id": sid}, timeout=30)
check("3. /build-graph -> 200", r.status_code == 200, f"(status={r.status_code})")
bg = None
if r.status_code == 200:
    bg = r.json()
    check(
        "3a. n_nodes echoed",
        bg.get("stats", {}).get("n_nodes", 0) > 0,
        f"(n_nodes={bg.get('stats', {}).get('n_nodes')})",
    )

# --- 4. /score method=auto ---
print("\n[4] POST /score method=auto")
r = httpx.post(
    f"{BASE}/score", json={"session_id": sid, "method": "auto"}, timeout=30
)
check("4. /score(auto) -> 200", r.status_code == 200, f"(status={r.status_code})")
if r.status_code == 200 and bg is not None:
    sc = r.json()
    print(
        f"     method={sc['method']}  model_loaded={sc['model_loaded']}  "
        f"n_scores={sc['n_scores']}"
    )
    scores = sc["scores"]
    by_id = {n["id"]: n for n in bg["nodes"]}
    fraud_scores = [
        scores[i] for i, n in by_id.items() if n["is_fraud"]
    ]
    normal_scores = [
        scores[i] for i, n in by_id.items() if not n["is_fraud"]
    ]
    mean_fraud = sum(fraud_scores) / len(fraud_scores)
    mean_normal = sum(normal_scores) / len(normal_scores)
    print(
        f"     mean fraud risk = {mean_fraud:.4f}, mean normal risk = "
        f"{mean_normal:.4f}, ratio = {mean_fraud / max(mean_normal, 1e-9):.1f}x"
    )
    check(
        "4a. mean fraud > 0.9",
        mean_fraud > 0.9,
        f"(mean_fraud={mean_fraud:.4f})",
    )
    check(
        "4b. mean normal < 0.1",
        mean_normal < 0.1,
        f"(mean_normal={mean_normal:.4f})",
    )

# --- 5. /score method=rule_based ---
print("\n[5] POST /score method=rule_based")
r = httpx.post(
    f"{BASE}/score",
    json={"session_id": sid, "method": "rule_based"},
    timeout=30,
)
check(
    "5. /score(rule_based) -> 200",
    r.status_code == 200,
    f"(status={r.status_code})",
)
if r.status_code == 200:
    sc5 = r.json()
    check(
        "5a. method=rule_based",
        sc5.get("method") == "rule_based",
        f"(method={sc5.get('method')})",
    )

# --- 6. /explain account_id ---
print("\n[6] POST /explain account_id")
fraud_id = next(n["id"] for n in bg["nodes"] if n["is_fraud"])
r = httpx.post(
    f"{BASE}/explain",
    json={"session_id": sid, "account_id": fraud_id},
    timeout=10,
)
check(
    "6. /explain(account) -> 200",
    r.status_code == 200,
    f"(status={r.status_code}, acc={fraud_id})",
)
if r.status_code == 200:
    ex = r.json()
    check(
        "6a. narrative present",
        bool(ex.get("narrative")),
        f"(narrative len={len(ex.get('narrative', ''))})",
    )
    print(
        f"     risk_score={ex['risk_score']}  cluster_size={ex['cluster_size']}  "
        f"top_reasons={len(ex['top_reasons'])}"
    )

# --- 7. /explain ring_id ---
print("\n[7] POST /explain ring_id")
ring_id = bg["rings"][0]["ring_id"]
r = httpx.post(
    f"{BASE}/explain",
    json={"session_id": sid, "ring_id": ring_id},
    timeout=10,
)
check(
    "7. /explain(ring) -> 200",
    r.status_code == 200,
    f"(status={r.status_code}, ring={ring_id})",
)
if r.status_code == 200:
    er = r.json()
    check(
        "7a. ring size >= 4",
        er.get("size", 0) >= 4,
        f"(size={er.get('size')})",
    )
    print(
        f"     ring={er['ring_id']}  size={er['size']}  "
        f"avg_risk={er['avg_risk_score']}"
    )

# --- 8. /report (PDF) ---
print("\n[8] POST /report")
r = httpx.post(f"{BASE}/report", json={"session_id": sid}, timeout=10)
check("8. /report -> 200", r.status_code == 200, f"(status={r.status_code})")
if r.status_code == 200:
    check(
        "8a. content-type=application/pdf",
        r.headers.get("content-type", "").startswith("application/pdf"),
        f"(ct={r.headers.get('content-type')})",
    )
    check(
        "8b. magic bytes %PDF-",
        r.content[:4] == b"%PDF",
        f"(magic={r.content[:4]!r})",
    )
    print(f"     pdf size = {len(r.content)} bytes")

# --- 9. /sample-dataset ---
print("\n[9] GET /sample-dataset")
r = httpx.get(f"{BASE}/sample-dataset", timeout=10)
check(
    "9. /sample-dataset -> 200",
    r.status_code == 200,
    f"(status={r.status_code})",
)
if r.status_code == 200:
    sd = r.json()
    check(
        "9a. n_nodes=109 (sample)",
        sd["stats"]["n_nodes"] == 109,
        f"(n_nodes={sd['stats']['n_nodes']})",
    )
    check(
        "9b. rings=5",
        len(sd.get("rings", [])) == 5,
        f"(rings={len(sd.get('rings', []))})",
    )
    nan_rings = [
        n["id"]
        for n in sd["nodes"]
        if isinstance(n.get("ring_id"), float)
    ]
    check(
        "9c. no NaN ring_ids",
        not nan_rings,
        f"(found {len(nan_rings)} NaN)",
    )

# --- 10. Error paths ---
print("\n[10] Error paths")
r = httpx.post(
    f"{BASE}/explain",
    json={"session_id": "does-not-exist", "account_id": "X"},
    timeout=5,
)
check(
    "10a. unknown session -> 404",
    r.status_code == 404,
    f"(status={r.status_code})",
)

r = httpx.post(
    f"{BASE}/explain",
    json={"session_id": sid, "account_id": "ACC-NOPE"},
    timeout=5,
)
check(
    "10b. unknown account_id -> 404",
    r.status_code == 404,
    f"(status={r.status_code})",
)

# --- 11. Fallback path ---
print("\n[11] Fallback path")
print(
    "     Note: with GNN loaded, score(method=rule_based) must still work"
)
r = httpx.post(
    f"{BASE}/score",
    json={"session_id": sid, "method": "rule_based"},
    timeout=10,
)
check(
    "11. rule_based fallback works while GNN loaded",
    r.status_code == 200,
    f"(status={r.status_code})",
)

print()
print("=" * 70)
print(f"TOTAL: {passed} passed, {failed} failed")
print("=" * 70)
sys.exit(0 if failed == 0 else 1)
