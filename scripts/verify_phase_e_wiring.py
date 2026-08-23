"""Simulate the exact sequence the browser runs after 'Use sample dataset'."""
import json, sys, httpx

BASE = "http://127.0.0.1:8000"
client = httpx.Client(timeout=30)

passed = failed = 0
def check(name, ok, detail=""):
    global passed, failed
    tag = "PASS" if ok else "FAIL"
    print(f"  [{tag}] {name} {detail}")
    passed += 1 if ok else 0
    failed += 1 if not ok else 0

print("=" * 70)
print("Phase E wiring - simulated browser flow")
print("=" * 70)

# 1. /sample-dataset (with Origin: null to mimic file://)
print("\n[1] GET /sample-dataset (Origin: null)")
r = client.get(f"{BASE}/sample-dataset", headers={"Origin": "null"})
check("1. /sample-dataset -> 200", r.status_code == 200, f"(status={r.status_code})")
check("1a. ACAO header allows null", r.headers.get("access-control-allow-origin") == "null",
      f"(ACAO={r.headers.get('access-control-allow-origin')!r})")
sample = r.json()
check("1b. n_nodes=109", sample["stats"]["n_nodes"] == 109, f"(n_nodes={sample['stats']['n_nodes']})")
check("1c. n_rings=5", sample["stats"]["n_rings"] == 5, f"(rings={sample['stats']['n_rings']})")

# 2. POST /upload-json (the sample payload wrapped in {graph: ...})
print("\n[2] POST /upload-json")
r = client.post(f"{BASE}/upload-json",
                json={"graph": sample},
                headers={"Origin": "null"})
check("2. /upload-json -> 200", r.status_code == 200, f"(status={r.status_code})")
session = r.json()
sid = session["session_id"]
print(f"     session_id = {sid}")
print(f"     n_nodes = {session['n_nodes']}, n_edges = {session['n_edges']}")

# 3. Simulate the adapter (Python port of adaptBackendPayloadToGraph)
print("\n[3] Shape adapter (Python port)")
id_to_node = {}
nodes = []
for n in sample["nodes"]:
    risk = round((n.get("risk_score") or 0) * 100)
    feats = n.get("features") or {}
    node = {
        "id": n["id"], "name": n.get("name") or n["id"],
        "risk": risk, "riskBand": "high" if risk>=70 else "mid" if risk>=40 else "low",
        "degree": feats.get("degree", 0), "distinctAttrTypes": 0,
        "accountAgeDays": feats.get("account_age_days", 0),
        "ring": n.get("ring_id") or -1, "seedFraud": bool(n.get("is_fraud")),
    }
    id_to_node[n["id"]] = node
    nodes.append(node)
edges = [{"source": e["source"], "target": e["target"],
          "shared": e.get("attribute_types") or [],
          "weight": e.get("weight", 1),
          "reasons": e.get("reasons") or []} for e in sample["edges"]]
counts = {}
for e in edges:
    t = len(e["shared"])
    counts[e["source"]] = max(counts.get(e["source"], 0), t)
    counts[e["target"]] = max(counts.get(e["target"], 0), t)
for n in nodes:
    n["distinctAttrTypes"] = counts.get(n["id"], 0)
check("3a. adapter produces 109 nodes", len(nodes) == 109, f"(got {len(nodes)})")
check("3b. adapter produces edges", len(edges) > 0, f"(got {len(edges)} edges)")
high_risk = [n for n in nodes if n["riskBand"] == "high"]
check("3c. fraud accounts land in 'high' band", len(high_risk) == 29,
      f"(got {len(high_risk)} high-risk)")

# 4. POST /explain with a fraud account
print("\n[4] POST /explain (fraud account)")
fraud_id = next(n["id"] for n in sample["nodes"] if n.get("is_fraud"))
r = client.post(f"{BASE}/explain",
                json={"session_id": sid, "account_id": fraud_id},
                headers={"Origin": "null"})
check("4. /explain -> 200", r.status_code == 200, f"(status={r.status_code})")
if r.status_code == 200:
    expl = r.json()
    check("4a. narrative present", bool(expl.get("narrative")),
          f"(len={len(expl.get('narrative',''))})")
    check("4b. risk_method present", "risk_method" in expl,
          f"(method={expl.get('risk_method')})")
    print(f"     risk_score={expl['risk_score']} method={expl['risk_method']}")
    print(f"     narrative preview: {expl['narrative'][:120]!r}…")

# 5. POST /report (PDF)
print("\n[5] POST /report (PDF download)")
r = client.post(f"{BASE}/report",
                json={"session_id": sid},
                headers={"Origin": "null", "Accept": "application/pdf"})
check("5. /report -> 200", r.status_code == 200, f"(status={r.status_code})")
if r.status_code == 200:
    check("5a. content-type=application/pdf",
          r.headers.get("content-type","").startswith("application/pdf"),
          f"(ct={r.headers.get('content-type')})")
    check("5b. magic bytes %PDF-", r.content[:4] == b"%PDF",
          f"(magic={r.content[:4]!r})")
    print(f"     pdf size={len(r.content)} bytes")

# 6. Error path: stale session
print("\n[6] Stale session -> 404")
r = client.post(f"{BASE}/explain",
                json={"session_id": "00000000000000000000000000000000",
                      "account_id": fraud_id},
                headers={"Origin": "null"})
check("6. unknown session -> 404", r.status_code == 404, f"(status={r.status_code})")

print()
print("=" * 70)
print(f"TOTAL: {passed} passed, {failed} failed")
print("=" * 70)
sys.exit(0 if failed == 0 else 1)