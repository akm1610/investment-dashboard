# Investment Dashboard – Deployment Guide

> **Disclaimer**: This tool is for informational purposes only and does not
> constitute financial advice.

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Prerequisites](#prerequisites)
3. [Quick Start (Automated)](#quick-start-automated)
4. [Manual Setup](#manual-setup)
5. [Environment Variables](#environment-variables)
6. [Verification Commands](#verification-commands)
7. [API Reference](#api-reference)
8. [Troubleshooting](#troubleshooting)
9. [Docker Deployment](#docker-deployment)
10. [Production Deployment](#production-deployment)

---

## System Overview

```
┌─────────────────────────────────────────────────────┐
│                  Investment Dashboard                │
│                                                     │
│  ┌─────────────────┐      ┌─────────────────────┐  │
│  │  React Frontend  │ ──▶ │  Flask REST API      │  │
│  │  localhost:3000  │     │  localhost:9000      │  │
│  └─────────────────┘      └──────────┬──────────┘  │
│                                       │              │
│                           ┌──────────▼──────────┐  │
│                           │  ML / Scoring Engine │  │
│                           │  (yfinance, sklearn) │  │
│                           └─────────────────────┘  │
└─────────────────────────────────────────────────────┘
```

| Service | URL | Description |
|---------|-----|-------------|
| React Dashboard | `http://localhost:3000` | Web UI for stock analysis |
| Flask API | `http://localhost:9000` | REST API for predictions |

---

## Prerequisites

| Tool | Minimum version | Install |
|------|----------------|---------|
| Python | 3.10+ | https://python.org |
| pip | 22+ | `python -m ensurepip --upgrade` |
| Node.js | 18+ | https://nodejs.org |
| npm | 9+ | bundled with Node.js |

Verify you have everything:

```bash
python3 --version   # >= 3.10
pip3 --version      # >= 22
node --version      # >= 18
npm --version       # >= 9
```

---

## Quick Start (Automated)

The `start.sh` script handles everything in one step:

```bash
# Clone the repository (first time only)
git clone https://github.com/akm1610/investment-dashboard.git
cd investment-dashboard

# Make the scripts executable
chmod +x start.sh stop.sh

# Start everything (installs deps + kills port conflicts + starts services)
./start.sh
```

The script will:
1. Kill any existing processes on ports 8000–9000 and 3000
2. Install Python dependencies from `requirements.txt`
3. Install Node dependencies in `dashboard/`
4. Start the Flask API on port **9000** (background, logged to `logs/flask_api.log`)
5. Start the React dashboard on port **3000** (background, logged to `logs/react_dashboard.log`)
6. Wait for both services to be ready and print a summary

> **Restart without reinstalling deps (faster):**
> ```bash
> ./stop.sh && ./start.sh --no-install
> ```

---

## Manual Setup

If you prefer to set things up step by step:

### Step 1 – Create a Python virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate.bat     # Windows
```

### Step 2 – Install Python dependencies

```bash
pip install -r requirements.txt
```

### Step 3 – Kill conflicting processes

```bash
# macOS / Linux – kill anything on ports 8000-9000 and 3000
for port in $(seq 8000 9000) 3000; do
  lsof -ti tcp:$port | xargs kill -9 2>/dev/null || true
done
```

### Step 4 – Start the Flask API

```bash
# Port 9000 (default, can be overridden via API_PORT env var)
API_PORT=9000 python3 flask_api.py
```

The API logs to stdout. You should see:
```
INFO flask_api – Starting Investment API on port 9000
INFO werkzeug – Running on http://0.0.0.0:9000
```

Leave this terminal open (or run with `nohup` / a process manager).

### Step 5 – Start the React Dashboard

Open a **new terminal**:

```bash
cd dashboard
npm install   # first time only
npm run dev   # starts on port 3000
```

You should see:
```
  VITE ready in XXX ms
  ➜  Local:   http://localhost:3000/
```

Open `http://localhost:3000` in your browser.

---

## Environment Variables

Copy `.env.example` to `.env` and adjust values as needed:

```bash
cp .env.example .env
```

| Variable | Default | Description |
|----------|---------|-------------|
| `API_PORT` | `9000` | Port the Flask REST API listens on |
| `DASHBOARD_PORT` | `3000` | Port the React dev server listens on |
| `NEWS_API_KEY` | _(empty)_ | Optional [NewsAPI](https://newsapi.org) key for richer sentiment data |

---

## Verification Commands

### Health check

```bash
curl http://localhost:9000/health
```

Expected response:
```json
{
  "status": "healthy",
  "timestamp": "2026-03-20T16:25:55.084Z"
}
```

### Single stock prediction

```bash
curl http://localhost:9000/predict/AAPL | python3 -m json.tool
```

Expected response:
```json
{
  "ticker": "AAPL",
  "price": 248.20,
  "scores": {
    "fundamentals": 9.5,
    "technicals": 6.5,
    "risk": 6.0,
    "ml": 9.2,
    "sentiment": 7.5,
    "etf": 9.5,
    "total": 8.09
  },
  "signal": "BUY",
  "confidence": 61.8,
  "timestamp": "2026-03-20T16:25:55.084Z"
}
```

### Portfolio analysis

```bash
curl -s -X POST http://localhost:9000/portfolio \
  -H "Content-Type: application/json" \
  -d '{"tickers": ["AAPL", "MSFT", "GOOGL", "NVDA"]}' \
  | python3 -m json.tool
```

### Portfolio optimisation

```bash
curl -s -X POST http://localhost:9000/portfolio/optimize \
  -H "Content-Type: application/json" \
  -d '{"tickers": ["AAPL", "MSFT", "GOOGL", "NVDA", "TSLA"], "exclude_sell": true}' \
  | python3 -m json.tool
```

### Run the full test suite

```bash
python3 -m pytest tests/ -v
```

---

## API Reference

### `GET /health`

Returns API liveness status.

**Response 200**
```json
{ "status": "healthy", "timestamp": "<ISO-8601>" }
```

---

### `GET /predict/<ticker>`

Returns a comprehensive prediction for a single stock.

**Parameters**
| Name | In | Type | Description |
|------|----|------|-------------|
| ticker | path | string | Stock symbol (e.g. `AAPL`) |

**Response 200**
```json
{
  "ticker": "AAPL",
  "price": 248.20,
  "scores": {
    "fundamentals": 9.5,
    "technicals": 6.5,
    "risk": 6.0,
    "ml": 9.2,
    "sentiment": 7.5,
    "etf": 9.5,
    "total": 8.09
  },
  "signal": "BUY",
  "confidence": 61.8,
  "timestamp": "<ISO-8601>"
}
```

**Signals**
| Signal | Score range | Meaning |
|--------|------------|---------|
| `BUY` | ≥ 7.0 | Positive outlook |
| `HOLD` | 5.0 – 6.9 | Neutral; monitor |
| `SELL` | < 5.0 | Negative outlook |

**Errors**
| Code | Cause |
|------|-------|
| 400 | Ticker contains non-alpha characters |
| 500 | Data fetch or model error |

---

### `POST /portfolio`

Analyse a list of stocks and return ranked results.

**Request body**
```json
{ "tickers": ["AAPL", "MSFT", "GOOGL"] }
```

**Response 200**
```json
{
  "results": [ /* sorted by total score descending */ ],
  "summary": {
    "strong_buy": ["AAPL"],
    "buy": ["MSFT"],
    "hold": [],
    "sell": []
  },
  "errors": []
}
```

---

### `POST /portfolio/optimize`

Return score-weighted portfolio allocation.

**Request body**
```json
{
  "tickers": ["AAPL", "MSFT", "GOOGL", "NVDA"],
  "exclude_sell": true
}
```

**Response 200**
```json
{
  "weights": { "AAPL": 0.28, "MSFT": 0.25, "GOOGL": 0.27, "NVDA": 0.20 },
  "scores":  { "AAPL": 8.09, "MSFT": 7.32, "GOOGL": 8.38, "NVDA": 7.69 },
  "signals": { "AAPL": "BUY", "MSFT": "BUY", "GOOGL": "BUY", "NVDA": "BUY" },
  "errors": []
}
```

---

## Troubleshooting

### Flask API won't start

**Symptom**: `Address already in use` or API unreachable.

**Fix**:
```bash
# Find and kill the process using port 9000
lsof -ti tcp:9000 | xargs kill -9

# Or use a different port
API_PORT=9001 python3 flask_api.py
```

---

### React dashboard shows "API offline"

**Symptom**: Red dot and "API offline" in the top right corner.

**Cause**: The Flask API is not running or not reachable at `localhost:9000`.

**Fix**:
1. Check that the Flask API is running: `curl http://localhost:9000/health`
2. Restart the API: `API_PORT=9000 python3 flask_api.py`
3. Check `logs/flask_api.log` for errors.

---

### `ModuleNotFoundError` when starting the API

**Fix**:
```bash
pip install -r requirements.txt
```

If using a virtual environment, activate it first:
```bash
source .venv/bin/activate
pip install -r requirements.txt
```

---

### React dashboard won't start on port 3000

**Symptom**: `EADDRINUSE` error.

**Fix**:
```bash
lsof -ti tcp:3000 | xargs kill -9
cd dashboard && npm run dev
```

---

### `npm install` fails

**Symptom**: `ERESOLVE` or peer dependency errors.

**Fix**:
```bash
cd dashboard
rm -rf node_modules package-lock.json
npm install
```

---

### Slow or missing stock data

**Symptom**: Predictions take a long time or return errors.

**Cause**: `yfinance` makes live network calls. The first call for a ticker is
always slower.

**Fix**:
- Ensure you have internet access.
- Wait for the data fetch to complete (up to 30 s for the first call).
- Check for yfinance rate-limiting (add delays between batch calls).

---

### Tests failing

```bash
# Run a single test file
python3 -m pytest tests/test_flask_api.py -v

# Run with more output
python3 -m pytest tests/ -v --tb=short

# Run only fast tests (skip network calls)
python3 -m pytest tests/ -v -k "not integration"
```

---

## Docker Deployment

Docker Compose starts both the Flask API and the React dashboard in isolated containers.

### Prerequisites

- [Docker](https://docs.docker.com/get-docker/) 24+
- [Docker Compose](https://docs.docker.com/compose/install/) v2+

### Quick start

```bash
# (Optional) create a .env file from the template
cp .env.example .env

# Build images and start both services in the background
docker compose up --build -d
```

Services will be available at:

| Service | URL |
|---------|-----|
| Flask API | `http://localhost:9000` |
| React Dashboard | `http://localhost:3000` |

### Useful commands

```bash
# View live logs
docker compose logs -f

# Stop and remove containers
docker compose down

# Rebuild after code changes
docker compose up --build -d
```

---

## Production Deployment


### Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `API_PORT` | `9000` | Flask API port |
| `FLASK_ENV` | `production` | Flask environment |

### Using a process manager (Supervisor)

```ini
# /etc/supervisor/conf.d/investment-api.conf
[program:investment-api]
command=python3 /path/to/flask_api.py
environment=API_PORT="9000",FLASK_ENV="production"
autostart=true
autorestart=true
stdout_logfile=/var/log/investment-api.log
stderr_logfile=/var/log/investment-api-error.log
```

### Building the React dashboard for production

```bash
cd dashboard
npm run build   # output in dashboard/dist/
```

Serve the `dist/` folder with any static file server (nginx, Apache, Caddy).

### Nginx reverse proxy example

```nginx
server {
    listen 80;
    server_name your-domain.com;

    # React static files
    location / {
        root /path/to/investment-dashboard/dashboard/dist;
        try_files $uri $uri/ /index.html;
    }

    # Flask API proxy
    location /api/ {
        proxy_pass http://127.0.0.1:9000/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

---

## Stopping All Services

```bash
./stop.sh
```

Or manually:

```bash
# Kill Flask API
kill $(cat logs/flask_api.pid)

# Kill React dev server
kill $(cat logs/react_dashboard.pid)
```
