from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import yfinance as yf


TICKER_PROXY_FALLBACKS = {
    "^GSPC": ["SPY"],
    "^DJI": ["DIA"],
    "^IXIC": ["QQQ"],
    "^RUT": ["IWM"],
}


class RegimeDataLoader:
    """Loads monthly price data for regime analysis."""

    def __init__(self, ticker: str = "^GSPC", start_date: str = "1990-01-01", end_date: str | None = None):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date or dt.datetime.now().strftime("%Y-%m-%d")
        self.resolved_ticker = ticker

    def _download_price_frame(self, ticker: str) -> pd.DataFrame:
        return yf.download(
            ticker,
            start=self.start_date,
            end=self.end_date,
            progress=False,
            auto_adjust=True,
        )

    def _candidate_tickers(self) -> list[str]:
        candidates = [self.ticker]
        for fallback in TICKER_PROXY_FALLBACKS.get(self.ticker, []):
            if fallback not in candidates:
                candidates.append(fallback)
        return candidates

    def fetch_monthly_prices(self) -> pd.DataFrame:
        attempted: list[str] = []

        for candidate in self._candidate_tickers():
            attempted.append(candidate)
            df = self._download_price_frame(candidate)
            if df is None or df.empty:
                continue

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            if "Close" not in df.columns:
                continue

            monthly = df[["Close"]].resample("MS").last().dropna()
            if monthly.empty:
                continue

            monthly.rename(columns={"Close": "price"}, inplace=True)
            monthly["log_price"] = np.log(monthly["price"].astype(float))
            monthly["return_1m"] = monthly["price"].pct_change()
            monthly["requested_ticker"] = self.ticker
            monthly["resolved_ticker"] = candidate
            self.resolved_ticker = candidate
            return monthly.dropna()

        attempted_text = ", ".join(attempted) if attempted else self.ticker
        raise ValueError(
            f"No price data returned for {self.ticker}. "
            f"Tried: {attempted_text}. Yahoo Finance may be rate-limiting the request."
        )
