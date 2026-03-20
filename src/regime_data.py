from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import yfinance as yf


class RegimeDataLoader:
    """Loads monthly price data for regime analysis."""

    def __init__(self, ticker: str = "^GSPC", start_date: str = "1990-01-01", end_date: str | None = None):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date or dt.datetime.now().strftime("%Y-%m-%d")

    def fetch_monthly_prices(self) -> pd.DataFrame:
        df = yf.download(self.ticker, start=self.start_date, end=self.end_date, progress=False, auto_adjust=True)
        if df.empty:
            raise ValueError(f"No price data returned for {self.ticker}.")

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        if "Close" not in df.columns:
            raise ValueError("Expected 'Close' column in downloaded price data.")

        monthly = df[["Close"]].resample("MS").last().dropna()
        monthly.rename(columns={"Close": "price"}, inplace=True)
        monthly["log_price"] = np.log(monthly["price"].astype(float))
        monthly["return_1m"] = monthly["price"].pct_change()
        return monthly.dropna()
