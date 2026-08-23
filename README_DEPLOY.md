# Fraud Network Mapper — Deployment Guide

Production deploys the two halves of the app to two different platforms:

| Half | Platform | Type | URL pattern | Cost |
|---|---|---|---|---|
| **Backend** (FastAPI + GNN) | **Railway** | Web Service (Docker) | `https://<name>.up.railway.app` | ~$3-5/mo of free credit |
| **Frontend** (single HTML + d3) | **Render** | Static Site | `https://<name>.onrender.com` | Free |

Why split: Railway gives us a Docker build for Python+PyTorch (needed for the
trained GNN). Render's Static Site is free, has no cold-start, and the HTML
is fully self-contained — no build step, no Node, no bundler.

---

## 1. Deploy the backend to Railway

### Option A — via GitHub (recommended)

1. Push this repo to GitHub (private is fine).
2. In Railway: **New Project → Deploy from GitHub repo → pick this repo**.
3. Railway auto-detects `backend/railway.toml` and starts the Docker build.
4. First build takes 3–5 minutes (CPU-only PyTorch wheel is ~700 MB).
5. Once deployed, go to **Settings → Networking → Generate Domain**.
   You'll get something like `https://fraud-mapper-api.up.railway.app`.
6. Smoke-test:
   ```bash
   curl https://fraud-mapper-api.up.railway.app/health
   # → {"status":"ok","model_loaded":true,"fallback_available":true,...}
   ```
7. (Optional) Lock down CORS by setting an env var:
   - Go to **Variables → New Variable**
   - Name: `ALLOWED_ORIGINS`
   - Value: `https://fraud-network-mapper.onrender.com`
   - This replaces the default wildcard (`*`) with an explicit allow-list.

### Option B — via Railway CLI

```bash
npm install -g @railway/cli
railway login
railway init    # creates a new project
railway up      # builds + deploys from the current dir
railway domain  # prints the public URL
```

The CLI reads `backend/railway.toml` automatically.

### Files the backend brings along

`backend/Dockerfile` copies (relative to repo root):

| What | Size | Why |
|---|---|---|
| `backend/` | ~30 KB | FastAPI source |
| `models/gnn.pt` | 18 KB | trained GraphSAGE weights |
| `data/samples/` | ~80 KB | the 109-node demo dataset |
| `scripts/` | ~50 KB | graph construction (runtime) + GNN utils |

Excluded by `.dockerignore`:
- Full training CSVs in `data/*.csv` (~600 KB) — not needed at runtime
- Test/verification scripts
- Local caches / `__pycache__`

---

## 2. Deploy the frontend to Render

1. In Render: **New → Static Site → Connect your GitHub repo**.
2. **Root Directory**: `frontend`
3. **Build Command**: *(leave blank — there's no build step)*
4. **Publish Directory**: `.` (the `frontend/` folder)
5. Render auto-detects `frontend/render.yaml`. Click **Create Static Site**.
6. After ~30 seconds you'll get a URL like `https://fraud-network-mapper.onrender.com`.

### What gets served

Just the two files in `frontend/`:
- `fraud-network-mapper.html` (335 KB, all of d3 inlined)
- `render.yaml` (header config)

That's it. No npm install, no bundler, no compile step.

---

## 3. Wire the two halves together

After both deploys are live, the frontend needs to know the backend URL.

**Edit `frontend/fraud-network-mapper.html`** — find this block near the top of the inline script:

```js
const PRODUCTION_DEFAULT = 'https://fraud-mapper-api.up.railway.app';
```

Replace the URL with your **actual** Railway domain, then commit + push:

```bash
git add frontend/fraud-network-mapper.html
git commit -m "Point frontend at Railway backend"
git push
```

Render will auto-redeploy in ~30 seconds. Verify by opening your Render URL
and clicking "Use sample dataset" — the graph should appear with **109
accounts**, **5 rings**, **29 red fraud nodes** with real names
("Monica Herrera", etc.).

---

## 4. Demo-time overrides (no rebuild needed)

You can repoint the frontend at a different backend **without redeploying** by
using the `?api=` URL query parameter:

```
https://fraud-network-mapper.onrender.com/?api=https://my-staging.up.railway.app
```

The bootstrap script reads this query param first, before falling back to
the hardcoded Railway URL. Useful for:

- Pointing at a staging backend during a dry-run
- Demoing from a localhost backend on your laptop (e.g.
  `?api=http://192.168.1.42:8000` if your laptop is on the same Wi-Fi as
  the judge's phone)

Resolution order at runtime:
1. `?api=https://...` URL parameter (highest priority)
2. `window.__API_BASE__` injected by an optional `config.js` (extension point)
3. `PRODUCTION_DEFAULT` baked into the HTML at deploy time
4. `http://127.0.0.1:8000` if the page is opened from `file://` or `localhost`

---

## 5. Troubleshooting

### Network error / 502 from the frontend

Railway's free tier **sleeps the container after 15 min of no requests**.
The first `/health` call after a sleep takes 30–60 seconds to wake up.
Hit `https://<railway>/health` in your browser to wake it before the demo.

### CORS error in the browser console

You see `Access to fetch ... has been blocked by CORS policy`.

**Fix**: set `ALLOWED_ORIGINS` on Railway to your exact Render URL:
```
ALLOWED_ORIGINS=https://fraud-network-mapper.onrender.com
```
Redeploys in ~30 sec.

### "model_loaded: false"

The trained model didn't load. Check `backend/Dockerfile` actually copies
`models/gnn.pt` into the image:
```bash
railway run bash -c "ls -la /app/models/"
```

### Build fails with "no space left on device"

`backend/Dockerfile` installs CPU-only PyTorch specifically to avoid this.
If you still hit it:
- Make sure `.dockerignore` is in place (excludes training CSVs)
- Verify `FROM python:3.11-slim` (not the default `python:3.11` which is huge)

### PDF report says "session not found"

The `SessionStore` is **in-memory**. Restarting the backend clears all
sessions. The `/sample-dataset` → `/upload-json` → `/explain` flow always
works in one browser session. If you uploaded a CSV earlier and try to
re-explain later, restart loses it. This is by design — see `core.py:SessionStore`.

---

## 6. Local dev vs production parity

Both deploy scripts give you the same behavior as the deployed apps:

```bash
# Backend locally
cd backend && python -m uvicorn main:app --host 0.0.0.0 --port 8000
# OR with the production CORS default:
cd backend && ALLOWED_ORIGINS="*" python -m uvicorn main:app --host 0.0.0.0 --port 8000

# Frontend locally
cd frontend && python -m http.server 5500
# OR via npm:
cd frontend && npm install && npm run serve
```

Then open `http://127.0.0.1:5500/fraud-network-mapper.html`. The
`API_BASE` bootstrap auto-detects localhost and points at
`http://127.0.0.1:8000` — no env vars to set.

---

## 7. What this gives the demo

| Click | URL hit | Returns |
|---|---|---|
| Page load | GET `/health` | `{status:"ok",model_loaded:true,...}` |
| "Use sample dataset" | GET `/sample-dataset` → POST `/upload-json` | 109-node graph, session_id |
| Click red node | POST `/explain` | plain-English fraud narrative |
| "Generate report" | POST `/report` | application/pdf (3 KB) |

The judge sees the dashboard go from `0` to real numbers in <2 seconds, then
can click any red node and watch a Palantir-style side panel populate with
a GNN-generated explanation.
