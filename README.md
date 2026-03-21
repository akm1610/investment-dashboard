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

## Quick Start

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

3. The Flask API and scoring engine will automatically load the `.env` file
   via `python-dotenv` (installed with `pip install -r requirements.txt`).

### How it works

| Scenario | Behaviour |
|----------|-----------|
| `NEWS_API_KEY` is set | Headlines are fetched from `newsapi.org/v2/everything` for the ticker and scored with keyword-based polarity |
| `NEWS_API_KEY` is absent or API call fails | Falls back to yfinance news headlines for the same ticker |
| No matching keyword in any headline | Sentiment returns `None`; the overall score stays at neutral 5.0 |

Responses are cached for **15 minutes** to avoid exhausting the free-tier rate limit (100 requests/day).

The live news sentiment feeds directly into:
- `score_sentiment(ticker)` → contributes 5 % to the composite score
- `GET /predict/<ticker>` response (`scores.sentiment`)
- `GET /sentiment/<ticker>` response (`sentiment_score`, `news_api_active`, `headlines`)

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