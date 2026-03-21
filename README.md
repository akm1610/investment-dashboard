# Investment Dashboard

A non-predictive **Long-Term Investment Analysis System** for disciplined investors.  
The system helps analyse company fundamentals, track portfolio health, and enforce disciplined decision-making through data-driven insights.

---

## Features

| Module | Description |
|--------|-------------|
| `src/data_fetcher.py` | Download financials and price history from Yahoo Finance |
| `analysis_engine.py` _(planned)_ | Compute health scores and ratios |
| `portfolio_manager.py` _(planned)_ | Track holdings, allocation, and rebalancing |
| `app.py` _(planned)_ | Streamlit dashboard integrating everything |

---

## Streamlit Dashboard

The primary interface is a **multi-page Streamlit app** (`src/app.py`) that integrates
all analysis—fundamental, technical, ML, and **sentiment**—for any ticker symbol.

### Pages

| Page | Icon | Description |
|------|------|-------------|
| Company Analysis | 🔍 | Fundamental scores, ratios, price chart, pre-trade checklist, and **Sentiment Analysis panel** |
| Portfolio Overview | 💼 | Holdings, allocation, and rebalancing |
| Pre-Trade Checklist | ✅ | 7-item decision gate |
| Investment Journal | 📓 | Timestamped thesis records |
| Risk & Recommendations | 🛡️ | Risk profile assessment and curated watchlists |
| Strategy Backtesting | 📈 | Historical strategy testing and validation |
| Sentiment Analysis | 📰 | Real-time news sentiment, analyst consensus, and insider activity for any ticker |

### Running the Streamlit App

```bash
# From the repository root
streamlit run src/app.py
```

Open <http://localhost:8501> in your browser.

---

## Unified Quick Start

The fastest way to run the **entire system** (Flask API + Streamlit dashboard)
from a **single terminal** after cloning:

```bash
git clone https://github.com/akm1610/investment-dashboard.git
cd investment-dashboard
chmod +x run_all.sh
./run_all.sh
```

That single command:
1. Creates and activates a Python virtual environment (`.venv/`) automatically.
2. Installs all Python dependencies from `requirements.txt`.
3. Detects and resolves port conflicts interactively.
4. Starts the **Flask prediction API** on port **9000** in the background.
5. Starts the **Streamlit dashboard** on port **8501** in the background.
6. Opens the dashboard in your browser automatically.
7. Prints a status table with all service URLs.
8. Shuts down both services cleanly when you press **Ctrl+C**.

### Available commands

| Command | Description |
|---------|-------------|
| `./run_all.sh` | Install deps and start all services (default) |
| `./run_all.sh --no-install` | Skip pip install for faster restart |
| `./run_all.sh --force` | Auto-kill port conflicts without prompting (for CI/scripts) |
| `./run_all.sh --api-only` | Start only the Flask API |
| `./run_all.sh --streamlit-only` | Start only the Streamlit dashboard |
| `./run_all.sh stop` | Stop all managed services |
| `./run_all.sh restart` | Stop then start all services |
| `./run_all.sh status` | Show which services are running |
| `./run_all.sh logs` | Tail live logs from all services |

### Makefile shortcuts (requires `make`)

```bash
make all        # start everything (default)
make stop       # stop all services
make restart    # stop + restart
make status     # check running services
make logs       # tail live logs
make test       # run test suite
make install    # install Python deps only
make check      # verify ports are free
make help       # print all targets
```

Override ports on the command line:

```bash
make all API_PORT=9001 STREAMLIT_PORT=8502
```

### Service URLs

| Service | Default URL | Description |
|---------|-------------|-------------|
| Streamlit dashboard | <http://localhost:8501> | Full interactive UI |
| Flask API | <http://localhost:9000> | REST prediction API |
| API health check | <http://localhost:9000/health> | Liveness probe |
| Stock prediction | <http://localhost:9000/predict/AAPL> | Example endpoint |

### Port conflicts

If a port is already in use, `run_all.sh` will show the occupying PID and ask
whether to kill it.  You can also free ports manually:

```bash
# Find what is using port 9000
lsof -i :9000

# Kill by PID (replace 1234 with the actual PID)
kill -9 1234

# Or use a different port
API_PORT=9001 ./run_all.sh
```

### Logs

All service logs are written to `logs/`:

```
logs/flask_api.log   – Flask API output
logs/streamlit.log   – Streamlit output
```

### Windows notes

`run_all.sh` and the Makefile are designed for **macOS and Linux**.
On Windows, use [Git Bash](https://git-scm.com/downloads) or
[WSL](https://learn.microsoft.com/en-us/windows/wsl/) to run `./run_all.sh`.

Alternatively, open two terminals and start each service manually:

```powershell
# Terminal 1 – Flask API
python flask_api.py

# Terminal 2 – Streamlit dashboard
streamlit run src/app.py
```

---

## Quick Start (manual setup)

### 1 · Clone & create a virtual environment

```bash
git clone https://github.com/akm1610/investment-dashboard.git
cd investment-dashboard
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

### 2 · Install dependencies

```bash
pip install -r requirements.txt
```

### 3 · Use the data fetcher

```python
from src.data_fetcher import (
    fetch_stock_data,
    fetch_company_info,
    calculate_basic_ratios,
    batch_fetch,
)

# Historical price data (5-year default)
prices = fetch_stock_data("AAPL")
print(prices.tail())

# Company overview
info = fetch_company_info("MSFT")
print(info["longName"], info["sector"])

# Key financial ratios
ratios = calculate_basic_ratios("GOOGL")
for name, value in ratios.items():
    print(f"{name}: {value}")

# Fetch data for multiple tickers at once
results = batch_fetch(["AAPL", "MSFT", "AMZN"], fetch_type="ratios")
```

---

## Module: `src/data_fetcher.py`

### Functions

| Function | Signature | Description |
|----------|-----------|-------------|
| `fetch_stock_data` | `(ticker, period='5y') → DataFrame` | Historical OHLCV data |
| `fetch_financial_statements` | `(ticker) → dict[str, DataFrame]` | Annual + quarterly financials |
| `fetch_company_info` | `(ticker) → dict` | Company overview and sector |
| `fetch_cash_flow_data` | `(ticker, quarterly=False) → DataFrame` | Cash flow statement |
| `fetch_balance_sheet` | `(ticker, quarterly=False) → DataFrame` | Balance sheet |
| `fetch_income_statement` | `(ticker, quarterly=False) → DataFrame` | Income statement |
| `calculate_basic_ratios` | `(ticker) → dict[str, float\|None]` | P/E, P/B, D/E, ROE, ROA, … |
| `batch_fetch` | `(tickers_list, fetch_type, …) → dict` | Concurrent multi-ticker fetch |

### Ratios computed by `calculate_basic_ratios`

| Ratio | Key |
|-------|-----|
| Price-to-Earnings (trailing) | `pe_ratio` |
| Price-to-Book | `pb_ratio` |
| Debt-to-Equity | `debt_to_equity` |
| Return on Equity | `roe` |
| Return on Assets | `roa` |
| Current Ratio | `current_ratio` |
| Gross Margin | `gross_margin` |
| Operating Margin | `operating_margin` |
| EPS (TTM) | `eps_ttm` |

---

## Validating the Recommendation System

Run the end-to-end validation script to confirm the system is working correctly with real market data:

```bash
python scripts/validate_system.py
```

The script tests six real stocks (AAPL, NVDA, INTC, RELIANCE.NS, INFY.NS, SBIN.NS) and prints:

- **Current prices** sourced live from Yahoo Finance
- **Fundamental scores** — P/E, ROE, Debt/Equity and more
- **Technical scores** — RSI, MACD, price vs 200-day MA and other indicators
- **Risk metrics** — annualised volatility, Sharpe ratio, max drawdown
- **ML predictions** — signal (BUY / HOLD / SELL) and confidence from the ensemble
- **3–5 key reasons** explaining each score with specific numbers
- **Benchmark comparison** vs S&P 500 (and NIFTY 50 for Indian stocks)

Sample output excerpt:

```
======================================================================
AAPL → Final Score: 7.2/10
Current Price: $192.35
======================================================================

Fundamentals: 7.5/10
  ├─ pe_ratio: 28.5000
  ├─ roe: 0.9200
  └─ debt_to_equity: 1.8000

Technicals: 6.8/10
  ├─ RSI(14): 62.0000
  ├─ MACD Histogram: 0.4500
  └─ Price vs SMA200: +8.20%

...

KEY REASONS:
  1. P/E ratio of 28.5: Premium valuation
  2. RSI(14) at 62: Neutral momentum
  3. Price is 8.2% above 200-day MA (uptrend)
  4. Annualized volatility 22.50%: Moderate risk
  5. ML models predict BUY with 68% confidence

BENCHMARK COMPARISON:
  OUTPERFORMS S&P 500 (+0.7 points)
======================================================================
```

---

## Running Tests

```bash
pytest tests/ -v
```

---

## NewsAPI Integration

The sentiment scoring engine can use live news headlines from [NewsAPI](https://newsapi.org)
to produce a more accurate sentiment signal.

### Setup

1. Obtain a free API key from <https://newsapi.org>.
2. Copy `.env.example` to `.env` and set your key:

   ```ini
   NEWS_API_KEY=your_newsapi_key_here
   ```

3. The Streamlit app, Flask API, and scoring engine will automatically load the `.env` file
   via `python-dotenv` (installed with `pip install -r requirements.txt`).

### How it works

| Scenario | Behaviour |
|----------|-----------|
| `NEWS_API_KEY` is set | Headlines are fetched from `newsapi.org/v2/everything` for the ticker and scored with keyword-based polarity |
| `NEWS_API_KEY` is absent or API call fails | Falls back to yfinance news headlines for the same ticker |
| No matching keyword in any headline | Sentiment returns `None`; the overall score stays at neutral 5.0 |

Responses are cached for **15 minutes** to avoid exhausting the free-tier rate limit (100 requests/day).

### Where sentiment appears

- **Company Analysis page** – an embedded **Sentiment Analysis** panel shows a gauge,
  verdict badge (Bullish / Neutral / Bearish), analyst consensus, and a headlines table
  with per-headline polarity whenever you analyse a ticker.
- **Sentiment Analysis page** – a dedicated page for an in-depth view: polarity distribution
  chart, headline count breakdown, and a natural-language summary.
- **Scoring engine** – `score_sentiment(ticker)` contributes 5 % to the composite score.
- **Flask API** – `GET /predict/<ticker>` response (`scores.sentiment`) and
  `GET /sentiment/<ticker>` response (`sentiment_score`, `news_api_active`, `headlines`).

### Public API

```python
from src.scoring_engine import get_news_sentiment, get_news_headlines

# Returns a float in [-1, +1] or None if no headlines match
score = get_news_sentiment("AAPL")

# Returns a list of dicts: title, url, published_at, source, polarity
headlines = get_news_headlines("AAPL", max_articles=10)
```

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `API_PORT` | `9000` | Port the Flask API listens on |
| `API_KEY` | _(empty)_ | Optional API key for request authentication |
| `NEWS_API_KEY` | _(empty)_ | [NewsAPI](https://newsapi.org) key for live headline sentiment |
| `RATELIMIT_DEFAULT` | `200 per day;60 per minute` | Flask-Limiter rate limit string |

Copy `.env.example` to `.env` and fill in values before starting the server.

---

## Running the Flask Backend API

The backend API (`flask_api.py`) fetches **live, real-time data** from Yahoo Finance
for any requested ticker.  No API key is needed for stock price and fundamental
data — yfinance handles that automatically.  An optional [NewsAPI](https://newsapi.org)
key enables richer news headline sentiment (free tier: 100 requests/day).

### Step 1 — Set up environment variables (optional)

```bash
cp .env.example .env
# Edit .env to set NEWS_API_KEY if you want live news headlines
```

> **Getting a free NewsAPI key**
> 1. Visit <https://newsapi.org/register> and create a free account.
> 2. Copy the generated key into `.env`:
>    ```bash
>    NEWS_API_KEY=your_key_here
>    ```
> 3. If you skip this step, sentiment falls back to yfinance news headlines automatically.

### Step 2 — Start the API server

```bash
python flask_api.py
```

The server starts on port **9000** by default.  You should see:

```
INFO flask_api – Starting Investment API on port 9000
```

### Step 3 — Test a live prediction

```bash
curl http://localhost:9000/predict/AAPL
```

A successful response returns **real market data** including the current price,
component scores derived from live fundamentals and technicals, a BUY/HOLD/SELL
signal, and the most recent news headlines — all stamped with the current UTC
timestamp.

#### Example response fields

| Field | Description |
|-------|-------------|
| `price` | Latest closing price from Yahoo Finance |
| `scores.fundamentals` | Scored from live P/E, ROE, margins, debt ratios |
| `scores.technicals` | Scored from live RSI, MACD, SMA200 comparison |
| `scores.sentiment` | Keyword polarity of recent news headlines |
| `scores.ml` | ML ensemble signal (defaults to neutral when models not trained) |
| `news_headlines` | Up to 5 recent headlines (NewsAPI or yfinance fallback) |
| `data_source` | Always `"Yahoo Finance (live)"` |

### Error handling

| Scenario | HTTP status | Response |
|----------|-------------|----------|
| Invalid ticker (numbers, too long) | `400` | `{"error": "…"}` |
| Ticker not found / delisted | `500` | `{"error": "…", "ticker": "XYZ"}` |
| Rate limit exceeded | `429` | Flask-Limiter default response |

---

## Running the Frontend Dashboard

The static dashboard lives in the `frontend/` folder at the root of this repository.

### Step 1 — Confirm you are in the project root

```bash
# You should see a list that includes "frontend"
ls
```

If you do **not** see `frontend` in the output, navigate to the project root first:

```bash
cd /path/to/investment-dashboard
```

### Step 2 — Start the HTTP server

```bash
cd frontend
python3 -m http.server 3000
```

Then open <http://localhost:3000> in your browser.

### Troubleshooting

#### `cd: no such file or directory: frontend`

You are not in the project root directory.  Run `ls` to check your current location.  
Navigate to the folder that **contains** the `frontend` directory before running `cd frontend`.

#### `OSError: [Errno 48] Address already in use`

Port 3000 is occupied by another process.  You have two options:

**Option A — kill the process on port 3000**

```bash
# Find the process ID (PID) using the port
lsof -i :3000

# Kill it (replace <PID> with the number shown above)
kill -9 <PID>

# Then retry
python3 -m http.server 3000
```

**Option B — use a different port**

```bash
python3 -m http.server 3001
```

Then open <http://localhost:3001> in your browser.

---

## Project Structure

```
investment-dashboard/
├── frontend/
│   └── index.html          # Static dashboard UI
├── src/
│   ├── __init__.py
│   └── data_fetcher.py
├── tests/
│   └── test_data_fetcher.py
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Disclaimer

This tool is for **informational purposes only** and does **not** constitute financial advice.  
All investment decisions should be made independently after consulting qualified professionals.