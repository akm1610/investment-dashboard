"""
src/components/investment_journal.py
--------------------------------------
Page 4 – maintain timestamped investment thesis records for accountability
and learning.

Components rendered:
* Create New Entry expander (ticker, thesis, conviction, tags)
* Filter & search bar (by ticker, date range, conviction, full-text)
* Thesis entries list (expandable rows with full thesis & notes)
* Entry statistics (chart of entries by ticker and conviction)
* Export options (JSON & CSV download)
"""

from __future__ import annotations

import sys
import os

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import io
import json
from datetime import date, datetime, timezone
from typing import List

import pandas as pd
import plotly.express as px
import streamlit as st

import portfolio_manager as pm


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _entries_to_df(entries: List[dict]) -> pd.DataFrame:
    """Convert journal entries list to a display DataFrame."""
    if not entries:
        return pd.DataFrame()
    rows = []
    for e in entries:
        ts_raw = e.get("timestamp", "")
        try:
            ts = datetime.fromisoformat(ts_raw.replace("Z", "+00:00"))
            date_str = ts.strftime("%Y-%m-%d %H:%M")
        except Exception:
            date_str = ts_raw[:10] if ts_raw else "Unknown"

        thesis_text = e.get("thesis", "")
        rows.append(
            {
                "Symbol": e.get("symbol", ""),
                "Date": date_str,
                "Tags": ", ".join(e.get("tags", [])),
                "Thesis Preview": (thesis_text[:80] + "…") if len(thesis_text) > 80 else thesis_text,
                "_thesis": thesis_text,
                "_timestamp": ts_raw,
            }
        )
    df = pd.DataFrame(rows)
    return df


def _filter_entries(
    entries: List[dict],
    ticker_filter: List[str],
    start_date: date | None,
    end_date: date | None,
    conviction_filter: List[str],
    search_text: str,
) -> List[dict]:
    """Apply user-specified filters to the list of journal entries."""
    result = entries

    if ticker_filter:
        result = [e for e in result if e.get("symbol", "") in ticker_filter]

    if start_date and end_date:
        start_dt = datetime(start_date.year, start_date.month, start_date.day, tzinfo=timezone.utc)
        end_dt = datetime(end_date.year, end_date.month, end_date.day, 23, 59, 59, tzinfo=timezone.utc)
        filtered = []
        for e in result:
            try:
                ts = datetime.fromisoformat(e["timestamp"].replace("Z", "+00:00"))
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                if start_dt <= ts <= end_dt:
                    filtered.append(e)
            except Exception:
                pass
        result = filtered

    if conviction_filter:
        tags_lower = [t.lower() for t in conviction_filter]
        result = [
            e for e in result
            if any(tag in tags_lower for tag in e.get("tags", []))
        ]

    if search_text.strip():
        q = search_text.strip().lower()
        result = [
            e for e in result
            if q in e.get("thesis", "").lower()
            or q in e.get("symbol", "").lower()
        ]

    return result


# ---------------------------------------------------------------------------
# Public page function
# ---------------------------------------------------------------------------


def page_investment_journal() -> None:
    """Render the Investment Journal page."""
    st.title("📓 Investment Journal")
    st.caption("Maintain timestamped investment thesis records for accountability and learning.")

    portfolio = st.session_state.portfolio

    # --- Create new entry ---
    with st.expander("➕ Create New Entry", expanded=False):
        with st.form("journal_new_entry_form"):
            j_col1, j_col2 = st.columns(2)
            j_ticker = j_col1.text_input("Ticker Symbol", placeholder="e.g. AAPL")
            j_conviction = j_col2.radio(
                "Conviction Level",
                options=["LOW", "MEDIUM", "HIGH"],
                horizontal=True,
                index=1,
            )
            j_thesis = st.text_area(
                "Investment Thesis",
                height=150,
                placeholder=(
                    "Why am I buying this stock?\n"
                    "What are the key catalysts?\n"
                    "What could go wrong?\n"
                    "Time horizon?"
                ),
            )
            j_tags_raw = st.text_input(
                "Additional Tags (comma-separated)",
                placeholder="e.g. growth, moat, dividend",
            )
            j_notes = st.text_area("Notes (optional)", height=60)

            if st.form_submit_button("💾 Save Entry", type="primary"):
                ticker_clean = j_ticker.strip().upper()
                if not ticker_clean:
                    st.error("Please enter a ticker symbol.")
                elif not j_thesis.strip():
                    st.error("Please enter an investment thesis.")
                else:
                    tags = [j_conviction.lower()]
                    extra_tags = [t.strip() for t in j_tags_raw.split(",") if t.strip()]
                    tags.extend(extra_tags)
                    full_thesis = j_thesis.strip()
                    if j_notes.strip():
                        full_thesis += f"\n\n**Notes:** {j_notes.strip()}"
                    pm.add_journal_entry(portfolio, ticker_clean, full_thesis, tags=tags)
                    pm.save_portfolio(portfolio)
                    st.success(f"✅ Entry for **{ticker_clean}** saved.")
                    st.rerun()

    # --- Get all entries ---
    all_entries = pm.get_journal(portfolio)

    if not all_entries:
        st.info("No journal entries yet.  Use the form above to create your first entry.")
        return

    # --- Filters ---
    st.subheader("Filter & Search")
    all_tickers = sorted({e.get("symbol", "") for e in all_entries if e.get("symbol")})
    all_convictions = ["LOW", "MEDIUM", "HIGH"]

    f_col1, f_col2, f_col3, f_col4 = st.columns([2, 2, 1, 2])

    ticker_filter: List[str] = f_col1.multiselect(
        "Filter by Ticker",
        options=all_tickers,
        key="journal_ticker_filter",
    )

    today = date.today()
    date_range = f_col2.date_input(
        "Date Range",
        value=[date(2000, 1, 1), today],
        key="journal_date_filter",
    )
    start_date: date | None = date_range[0] if len(date_range) >= 1 else None
    end_date: date | None = date_range[1] if len(date_range) >= 2 else today

    conviction_filter: List[str] = f_col3.multiselect(
        "Conviction",
        options=all_convictions,
        key="journal_conviction_filter",
    )

    search_text: str = f_col4.text_input(
        "Full-text search",
        placeholder="Search thesis …",
        key="journal_search",
    )

    # Apply filters
    filtered_entries = _filter_entries(
        all_entries,
        ticker_filter=ticker_filter,
        start_date=start_date,
        end_date=end_date,
        conviction_filter=conviction_filter,
        search_text=search_text,
    )

    st.write(f"Showing **{len(filtered_entries)}** of **{len(all_entries)}** entries")

    # --- Display entries ---
    st.subheader("Thesis Entries")
    if filtered_entries:
        for entry in reversed(filtered_entries):
            sym = entry.get("symbol", "?")
            ts = entry.get("timestamp", "")[:10]
            tags = entry.get("tags", [])
            conviction_disp = next(
                (t.upper() for t in tags if t.upper() in all_convictions), "N/A"
            )
            label = f"**{sym}** — {ts}  |  Conviction: {conviction_disp}"
            with st.expander(label, expanded=False):
                st.markdown(entry.get("thesis", ""))
                if tags:
                    st.caption("Tags: " + ", ".join(tags))

                # Delete button
                del_key = f"del_{sym}_{ts}"
                if st.button("🗑️ Delete Entry", key=del_key, type="secondary"):
                    # Find and remove the specific entry
                    journal = portfolio.get("journal", [])
                    try:
                        journal.remove(entry)
                        pm.save_portfolio(portfolio)
                        st.success(f"Entry deleted.")
                        st.rerun()
                    except ValueError:
                        st.error("Could not find entry to delete.")
    else:
        st.info("No entries match the current filters.")

    st.markdown("---")

    # --- Statistics ---
    st.subheader("Statistics")
    s_col1, s_col2 = st.columns(2)

    # Entries per ticker
    ticker_counts = {}
    for e in all_entries:
        sym = e.get("symbol", "")
        ticker_counts[sym] = ticker_counts.get(sym, 0) + 1
    ticker_df = (
        pd.DataFrame(list(ticker_counts.items()), columns=["Ticker", "Entries"])
        .sort_values("Entries", ascending=False)
        .head(10)
    )
    fig_tickers = px.bar(
        ticker_df,
        x="Ticker",
        y="Entries",
        title="Most-Analysed Tickers",
        text="Entries",
    )
    fig_tickers.update_layout(height=300, margin={"l": 20, "r": 20, "t": 40, "b": 20})
    s_col1.plotly_chart(fig_tickers, use_container_width=True)

    # Entries per conviction level
    conviction_counts = {"LOW": 0, "MEDIUM": 0, "HIGH": 0}
    for e in all_entries:
        for tag in e.get("tags", []):
            if tag.upper() in conviction_counts:
                conviction_counts[tag.upper()] += 1
                break
    conv_df = pd.DataFrame(
        list(conviction_counts.items()), columns=["Conviction", "Entries"]
    )
    fig_conv = px.pie(
        conv_df,
        names="Conviction",
        values="Entries",
        title="Entries by Conviction Level",
        color="Conviction",
        color_discrete_map={"LOW": "#EF5350", "MEDIUM": "#FFA726", "HIGH": "#66BB6A"},
        hole=0.45,
    )
    fig_conv.update_layout(height=300, margin={"l": 20, "r": 20, "t": 40, "b": 20})
    s_col2.plotly_chart(fig_conv, use_container_width=True)

    total_entries = len(all_entries)
    unique_tickers = len(set(e.get("symbol", "") for e in all_entries))
    avg_per_ticker = total_entries / unique_tickers if unique_tickers else 0.0
    st.caption(
        f"Total entries: **{total_entries}** | Unique tickers: **{unique_tickers}** "
        f"| Avg entries per ticker: **{avg_per_ticker:.1f}**"
    )

    st.markdown("---")

    # --- Export options ---
    st.subheader("Export")
    exp_col1, exp_col2 = st.columns(2)

    # JSON export
    json_str = json.dumps(all_entries, indent=2, default=str)
    exp_col1.download_button(
        "📥 Download as JSON",
        data=json_str,
        file_name="investment_journal.json",
        mime="application/json",
    )

    # CSV export
    entries_df = _entries_to_df(all_entries)
    if not entries_df.empty:
        csv_export = entries_df[["Symbol", "Date", "Tags", "Thesis Preview"]].to_csv(index=False)
        exp_col2.download_button(
            "📥 Download as CSV",
            data=csv_export,
            file_name="investment_journal.csv",
            mime="text/csv",
        )
