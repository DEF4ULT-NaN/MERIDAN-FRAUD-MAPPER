# Deploy to Production — Step-by-Step Checklist

This is the **short, ordered checklist**. For longer explanations see `README_DEPLOY.md`.

**Total time**: ~15 min for first deploy (mostly waiting on Docker build + Render static-site provisioning). Subsequent deploys are <1 min because both platforms auto-rebuild on `git push`.

---

## Prerequisites (do these once)

- [ ] **GitHub account** — create a new repo (e.g. `fraud-network-mapper`). Private is fine.
- [ ] **Railway account** — sign up at https://railway.app (free $5/mo credit).
- [ ] **Render account** — sign up at https://render.com (free tier, no card required).
- [ ] **Git** installed locally: `git --version`.

---

## Step 1 — Push code to GitHub

```bash
cd "C:\Users\Kushal Baroi\OneDrive\Documents\HACKATHON"

# Initialize git (skip if you already have a repo)
git init
git branch -M main

# Tell git who you are (only the first time)
git config user.name "Your Name"
git config user.email "you@example.com"

# Stage everything — .gitignore already filters out caches, venvs, etc.
git add .
git status        # review the list — should NOT contain .venv/, __pycache__/, data/graph.json

# Commit
git commit -m "Initial deploy — Fraud Network Mapper (Phase E + deployment)"

# Add your GitHub repo as origin and push
git remote add origin https://github.com/<your-username>/<repo-name>.git
git push -u origin main
```

**Expected output**: push succeeds, GitHub shows your files. The `models/gnn.pt` file (18 KB) **must** appear in the repo — if it's missing, your `.gitignore` is too aggressive.

---

## Step 2 — Deploy backend to Railway

1. **Open** https://railway.app → **New Project**.
2. Choose **Deploy from GitHub repo** → select your `<repo-name>`.
3. Railway auto-detects `backend/railway.toml`. Wait for the first build (~3-5 min the first time because PyTorch installs).
4. Watch the **Deployments** tab — you should see:
   - `Building…` → `Deploying…` → `Success`
   - The build log will say `[startup] GNN model loaded` near the bottom — that's the moment you know it worked.
5. **Settings → Networking → Generate Domain** → click **Generate**.
   Railway gives you a URL like `https://fraud-mapper-api.up.railway.app`. **Copy this — you'll need it in Step 4.**
6. **Sanity test** in a new terminal:
   ```bash
   curl https://fraud-mapper-api.up.railway.app/health
   ```
   Expected:
   ```json
   {"status":"ok","model_loaded":true,"fallback_available":true,"active_sessions":0}
   ```
   If `model_loaded:false`, scroll up in Railway logs and find the Python traceback — most common cause is the COPY step didn't include `models/gnn.pt`.
7. (Optional) Lock down CORS: **Variables → New Variable**:
   - Name: `ALLOWED_ORIGINS`
   - Value: `https://fraud-network-mapper.onrender.com`
   - (You'll fill in the real Render URL in Step 4. For now, leaving it as the wildcard `*` default is fine.)

---

## Step 3 — Deploy frontend to Render

1. **Open** https://render.com → **New + → Static Site**.
2. Connect the same GitHub repo.
3. Configure:
   - **Name**: `fraud-network-mapper` (or anything you like)
   - **Branch**: `main`
   - **Root Directory**: `frontend`
   - **Build Command**: *(leave blank)*
   - **Publish Directory**: `.` *(just a dot)*
   - **Auto-Deploy**: `Yes`
4. Click **Create Static Site**. Render provisions it in ~30 seconds.
5. Once live, Render gives you a URL like `https://fraud-network-mapper.onrender.com`. **Copy this.**

---

## Step 4 — Wire them together

The frontend needs to know where the backend lives. Open `frontend/fraud-network-mapper.html` and find this line (near the top of the inline script, in the API layer block):

```js
const PRODUCTION_DEFAULT = 'https://fraud-mapper-api.up.railway.app';
```

Replace with your **actual** Railway URL from Step 2:

```js
const PRODUCTION_DEFAULT = 'https://YOUR-ACTUAL-NAME.up.railway.app';
```

**Save** the file, then:

```bash
cd "C:\Users\Kushal Baroi\OneDrive\Documents\HACKATHON"
git add frontend/fraud-network-mapper.html
git commit -m "Wire frontend at Railway backend URL"
git push
```

Render auto-redeploys in ~30 seconds. The Railway backend does NOT need to redeploy — only the HTML changed.

---

## Step 5 — Verify the full demo flow

Open your Render URL in a browser (Chrome or Edge):

```
https://fraud-network-mapper.onrender.com
```

Click **"Use sample dataset"**.

| What you should see | What it means |
|---|---|
| Processing animation runs 5 steps in ~2 sec | Frontend is talking to backend |
| Graph appears with **109 nodes** | Correct sample dataset |
| **29 nodes are red** (the fraud rings) | GNN classified them correctly |
| Dashboard: **109 / 73 / 5 / 29 / ~30%** | Stats are real |
| Scoring engine card shows **GNN** | Real GNN inference ran (not fallback) |
| Node names like **"Monica Herrera"** | Sample data is loaded correctly |
| Click any red node → side panel opens with narrative | `/explain` endpoint is working |
| Click **Generate Report** → PDF downloads | `/report` endpoint is working |

Open **DevTools (F12) → Network tab** while clicking. You should see:
- `GET /sample-dataset` → 200
- `POST /upload-json` → 200
- All `200`, no `(blocked)` or CORS errors

---

## Step 6 — (Optional) Lock down CORS for production

Once you have the real Render URL, lock CORS down from the wildcard default:

1. Go to Railway → your project → **Variables**.
2. Set `ALLOWED_ORIGINS=https://fraud-network-mapper.onrender.com` (your actual Render URL — no trailing slash).
3. Save. Railway auto-redeploys the backend in ~30 seconds.

This replaces the demo-friendly `*` with an explicit allow-list. No CORS changes on Render needed.

---

## Subsequent deploys (after the first one)

Every time you change code:

```bash
git add .
git commit -m "What I changed"
git push
```

Both platforms auto-rebuild:
- **Railway** rebuilds only if files matching `watchPatterns` in `backend/railway.toml` change (backend code, model, sample data, scripts).
- **Render** rebuilds on any push to `frontend/`.

Typical round-trip: 30 sec (Render) to 3 min (Railway if PyTorch wheels get re-pulled).

---

## Common problems + fixes

### "Service unavailable" / "Application failed to start"
Click into the Railway deployment → **Logs** tab → scroll up. Most common causes:
- `ModuleNotFoundError: No module named 'core'` → Dockerfile COPY path is wrong
- `FileNotFoundError: models/gnn.pt` → model file not tracked in git (check `git ls-files models/`)

### "Network error" on first page load
Railway free tier sleeps after 15 min idle. **Wake it up**: open `https://<railway>/health` in a browser tab. The first request after sleep takes 30-60 sec.

### CORS error: "blocked by CORS policy"
Either:
- You forgot Step 6 (no `ALLOWED_ORIGINS` env var set, but the wildcard `*` default should still work…), OR
- You're using a query-string override `?api=...` pointing at a backend with `ALLOWED_ORIGINS` not including your origin

Fix: in Railway Variables, set `ALLOWED_ORIGINS=*` (or your exact origin).

### Graph doesn't render, dashboard shows zeros
The frontend is talking to the wrong backend OR the backend has no model. Check the browser console — you should see:
- `[wire] API_BASE = https://your-app.up.railway.app`
- `[wire] adapt: ... 109 nodes`

If `API_BASE` is `http://127.0.0.1:8000`, your hardcoded `PRODUCTION_DEFAULT` URL is wrong.

### "Cannot read properties of undefined" or "split is not a function"
Your browser cached an older version of the HTML. **Hard refresh** (Ctrl+Shift+R), or open in an Incognito window.

---

## Rollback (if a deploy breaks)

**Railway**: Deployments tab → click the last green deployment → **Redeploy**. Old version is live in ~30 sec.

**Render**: Manual deploys tab → pick an earlier commit → **Deploy**.

The previous version of `models/gnn.pt` and the previous HTML are still in git history, so everything is recoverable.

---

## Total cost

| Service | Plan | Cost |
|---|---|---|
| Railway | Free tier (Hobby) | $5/mo of credit, this app uses ~$3/mo |
| Render | Static Site | Free (no card required) |
| GitHub | Private repo | Free |
| **Total** | | **$0/mo** while within Railway's $5 credit |

Once you exceed Railway's free credit, the cheapest production plan is **$5/mo**. For the hackathon demo the free tier is fine.

---

## What you've got at the end

A judge opens `https://fraud-network-mapper.onrender.com` in their browser, clicks **"Use sample dataset"**, and within 2 seconds sees:

1. A graph of 109 accounts lighting up
2. 29 red fraud accounts clustered into 5 rings
3. A dashboard with real GNN risk scores
4. Click any red node → plain-English explanation of why it was flagged
5. Click "Generate Report" → downloads a PDF they can hand to their fraud team

All running on `https://`, all server-rendered endpoints responding in <100ms, no local setup needed.

That's the whole demo. 🎉
