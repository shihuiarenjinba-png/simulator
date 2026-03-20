from __future__ import annotations

import datetime as dt

import pandas as pd
import yfinance as yf


DEFAULT_MARKET_INDICATORS = {
    "SPY": "SPY ETF",
    "QQQ": "QQQ ETF",
    "DIA": "DIA ETF",
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

TICKER_ALIASES = {
    "SP500": "^GSPC",
    "S&P500": "^GSPC",
    "S&P 500": "^GSPC",
    "SPY": "SPY",
    "GSPC": "^GSPC",
    "^GSPC": "^GSPC",
    "QQQ": "QQQ",
    "DIA": "DIA",
    "NASDAQ": "^IXIC",
    "IXIC": "^IXIC",
    "^IXIC": "^IXIC",
    "DOW": "^DJI",
    "DJI": "^DJI",
    "^DJI": "^DJI",
    "RUSSELL": "^RUT",
    "RUSSELL2000": "^RUT",
    "RUT": "^RUT",
    "^RUT": "^RUT",
    "VIX": "^VIX",
    "^VIX": "^VIX",
    "USDJPY": "JPY=X",
    "USD/JPY": "JPY=X",
    "JPY=X": "JPY=X",
    "TNX": "^TNX",
    "^TNX": "^TNX",
    "N225": "^N225",
    "NIKKEI": "^N225",
    "NIKKEI225": "^N225",
    "NIKKEI 225": "^N225",
    "^N225": "^N225",
    "FTSE": "^FTSE",
    "^FTSE": "^FTSE",
    "GOLD": "GC=F",
    "GC=F": "GC=F",
}

DISPLAY_NAMES = {
    "^GSPC": "S&P 500",
    "SPY": "SPY ETF",
    "QQQ": "QQQ ETF",
    "DIA": "DIA ETF",
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


def normalize_ticker(user_input: str) -> str:
    cleaned = (user_input or "").strip()
    if not cleaned:
        raise ValueError("Ticker is empty.")
    return TICKER_ALIASES.get(cleaned.upper(), cleaned)


def describe_ticker(user_input: str) -> dict[str, str]:
    normalized = normalize_ticker(user_input)
    return {
        "input": user_input,
        "normalized": normalized,
        "label": DISPLAY_NAMES.get(normalized, normalized),
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
    base_ticker = normalize_ticker(base_ticker)
    indicator_map = indicator_map or DEFAULT_MARKET_INDICATORS
    tickers = [base_ticker] + [ticker for ticker in indicator_map if ticker != base_ticker]
    prices = _download_close_prices(tickers, start_date=start_date)
    monthly = prices.resample("MS").last().dropna(how="all")
    monthly_returns = monthly.pct_change().dropna(how="all")

    rename_map = {base_ticker: "base_asset"}
    rename_map.update(indicator_map)
    monthly_returns.rename(columns=rename_map, inplace=True)

    if "base_asset" not in monthly_returns.columns and base_ticker in monthly_returns.columns:
        monthly_returns.rename(columns={base_ticker: "base_asset"}, inplace=True)

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
