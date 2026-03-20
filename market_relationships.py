from __future__ import annotations

import datetime as dt

import pandas as pd
import yfinance as yf


DEFAULT_MARKET_INDICATORS = {
    "^IXIC": "NASDAQ",
    "^DJI": "Dow Jones",
    "^RUT": "Russell 2000",
    "^VIX": "VIX",
    "JPY=X": "USD/JPY",
    "^TNX": "US 10Y Yield",
    "^N225": "Nikkei 225",
    "^FTSE": "FTSE 100",
    "GC=F": "Gold",
}


def _download_close_prices(tickers: list[str], start_date: str, end_date: str | None = None) -> pd.DataFrame:
    end = end_date or dt.datetime.now().strftime("%Y-%m-%d")
    df = yf.download(tickers, start=start_date, end=end, progress=False, auto_adjust=True)
    if df.empty:
        raise ValueError("No market data returned from Yahoo Finance.")

    if isinstance(df.columns, pd.MultiIndex):
        if "Close" in df.columns.get_level_values(0):
            df = df["Close"]
        elif "Adj Close" in df.columns.get_level_values(0):
            df = df["Adj Close"]

    if isinstance(df, pd.Series):
        df = df.to_frame()

    return df.dropna(how="all")


def load_market_relationship_frame(
    base_ticker: str,
    start_date: str,
    indicator_map: dict[str, str] | None = None,
) -> pd.DataFrame:
    indicator_map = indicator_map or DEFAULT_MARKET_INDICATORS
    tickers = [base_ticker] + [ticker for ticker in indicator_map if ticker != base_ticker]
    prices = _download_close_prices(tickers, start_date=start_date)
    monthly = prices.resample("MS").last().dropna(how="all")
    monthly_returns = monthly.pct_change().dropna(how="all")

    rename_map = {base_ticker: "base_asset"}
    rename_map.update(indicator_map)
    monthly_returns.rename(columns=rename_map, inplace=True)
    return monthly_returns.dropna(how="all")


def compute_lead_lag_relationships(
    monthly_returns: pd.DataFrame,
    base_col: str = "base_asset",
    max_lag: int = 6,
) -> pd.DataFrame:
    if base_col not in monthly_returns.columns:
        raise ValueError(f"{base_col} not found in relationship frame.")

    base_series = monthly_returns[base_col]
    rows = []

    for col in monthly_returns.columns:
        if col == base_col:
            continue

        indicator = monthly_returns[col]
        best_corr = None
        best_lag = 0

        lag_profile = {}
        for lag in range(-max_lag, max_lag + 1):
            shifted = indicator.shift(lag)
            valid = base_series.notna() & shifted.notna()
            if valid.sum() < 24:
                continue

            corr = base_series[valid].corr(shifted[valid])
            lag_profile[lag] = corr
            if best_corr is None or abs(corr) > abs(best_corr):
                best_corr = corr
                best_lag = lag

        if best_corr is None:
            continue

        rows.append(
            {
                "indicator": col,
                "same_month_corr": lag_profile.get(0),
                "best_corr": best_corr,
                "best_lag_months": best_lag,
                "relationship_type": "先行" if best_lag > 0 else "遅行" if best_lag < 0 else "同時",
                "impact_direction": "押し上げ寄り" if best_corr > 0 else "逆方向",
            }
        )

    return pd.DataFrame(rows).sort_values(by="best_corr", key=lambda s: s.abs(), ascending=False)


def compute_lag_profile(monthly_returns: pd.DataFrame, indicator_col: str, base_col: str = "base_asset", max_lag: int = 6) -> pd.DataFrame:
    base_series = monthly_returns[base_col]
    indicator = monthly_returns[indicator_col]
    rows = []
    for lag in range(-max_lag, max_lag + 1):
        shifted = indicator.shift(lag)
        valid = base_series.notna() & shifted.notna()
        corr = base_series[valid].corr(shifted[valid]) if valid.sum() >= 24 else None
        rows.append(
            {
                "lag_months": lag,
                "correlation": corr,
                "meaning": "先行" if lag > 0 else "遅行" if lag < 0 else "同時",
            }
        )
    return pd.DataFrame(rows)
