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

## Running Tests

```bash
pytest tests/ -v
```

---

## Project Structure

```
investment-dashboard/
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