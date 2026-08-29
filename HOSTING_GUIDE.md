# Hosting the App So You Can Use It On Your Phone

Two options, in order of effort: test on your phone today over your home
WiFi (no deployment needed), then properly host it in the cloud so it works
from anywhere, not just your home network.

---

## Option A — Test on your phone right now (same WiFi, no hosting)

1. Find your laptop's local IP address:
   - **Mac/Linux:** `ifconfig | grep "inet "` (look for something like `192.168.1.42`)
   - **Windows:** `ipconfig` (look for "IPv4 Address")

2. Run the server bound to all network interfaces, not just localhost:
   ```bash
   uvicorn app.api:app --host 0.0.0.0 --port 8000
   ```

3. On your phone (connected to the **same WiFi**), open a browser and go to:
   ```
   http://<your-laptop-ip>:8000
   ```
   e.g. `http://192.168.1.42:8000`

**Limitation:** PWA install (the "Add to Home Screen" step that makes it feel
like a real app) generally requires HTTPS. Plain `http://` on your local
network usually still lets you browse and use the app in the phone browser,
but the installability may be inconsistent across browsers. `localhost`
itself is exempted from the HTTPS requirement, but your phone connecting to
your laptop's IP is not "localhost" from the phone's point of view. For
reliable PWA install, use Option B.

---

## Option B — Real hosting (works from anywhere, proper PWA install)

Using **Render** (has a free tier for both a web service and a small
Postgres database, and gives you free HTTPS automatically — required for
reliable PWA installability). Render docs: https://render.com/docs

### Step 1 — Put your project on GitHub
If it isn't already:
```bash
git init
git add .
git commit -m "Initial commit"
```
Create a new repo on GitHub, then:
```bash
git remote add origin https://github.com/<you>/<repo-name>.git
git push -u origin main
```

**Important:** your `.gitignore` already excludes `.env` and `*.db` — keep
it that way, never commit real credentials.

### Step 2 — Create the Postgres database on Render
1. Render dashboard → **New → PostgreSQL**
2. Give it a name, pick the free tier
3. Once created, copy the **Internal Database URL** — you'll need it in Step 4

### Step 3 — Create the web service
1. Render dashboard → **New → Web Service**
2. Connect your GitHub repo
3. **Build command:**
   ```
   pip install -r requirements.txt
   ```
4. **Start command:**
   ```
   uvicorn app.api:app --host 0.0.0.0 --port $PORT
   ```
   (Render sets `$PORT` automatically — don't hardcode a port number.)

### Step 4 — Set environment variables
In the web service's **Environment** tab, add:
```
DATABASE_URL=<the Internal Database URL from Step 2>
MODEL_PATH=model.joblib
```

### Step 5 — Get your trained model onto the server
Render's filesystem is ephemeral (wiped on redeploy), so `model.joblib`
needs to either:
- Be **committed to your git repo** (simplest — it's just a file; a few MB
  is fine for git), or
- Be regenerated on startup by adding a step to your start command, e.g.:
  ```
  python -m app.init_db && python -m app.train_model --model-out model.joblib && uvicorn app.api:app --host 0.0.0.0 --port $PORT
  ```
  (only sensible once your database already has real historical data loaded)

Simplest path: commit `model.joblib` to git after training it locally.

### Step 6 — Initialize the database on Render
Once deployed, you need tables created and data ingested. Render gives you
a **Shell** tab on the web service — open it and run:
```bash
python -m app.init_db
python -m app.ingest_csv sample_data/sample_results.csv --competition "Premier League"
python -m app.elo
```
(Substitute your real season CSVs instead of the sample file.)

### Step 7 — Visit your app
Render gives you a URL like `https://your-app-name.onrender.com` — open
that on your phone. Because it's real HTTPS, the browser should offer
"Add to Home Screen" / "Install app" properly.

**Free tier note:** Render's free web services spin down after inactivity
and take ~30-60 seconds to wake up on the next request — fine for personal
testing, just expect the first load after a while to be slow.

---

## Installing as an app on your phone

**Android (Chrome):** open the site → tap the **⋮** menu → **"Install app"**
(or **"Add to Home Screen"**). It'll appear as a normal app icon.

**iPhone (Safari):** open the site → tap the **Share** icon → **"Add to
Home Screen"**. iOS support for PWAs is more limited than Android's (no
install prompt banner, some offline features behave differently), but this
still gives you a home-screen icon that opens without browser chrome.

---

## Quick comparison

| | Option A (local WiFi) | Option B (Render) |
|---|---|---|
| Setup time | 2 minutes | 20-30 minutes |
| Works away from home | No | Yes |
| Reliable PWA install | Not guaranteed | Yes (real HTTPS) |
| Cost | Free | Free tier available |
| Needs your laptop running | Yes, always | No |
