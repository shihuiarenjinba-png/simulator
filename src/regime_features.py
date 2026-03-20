from __future__ import annotations

import numpy as np
import pandas as pd
import pywt


def annualized_growth_from_window(log_prices: pd.Series, window_months: int) -> pd.Series:
    rolling_change = log_prices - log_prices.shift(window_months)
    return np.exp(rolling_change * (12.0 / window_months)) - 1.0


def rolling_wavelet_energy(series: pd.Series, window: int, wavelet: str = "db2", level: int = 2) -> pd.DataFrame:
    values = series.to_numpy(dtype=float)
    detail_energy_1 = np.full(len(values), np.nan)
    detail_energy_2 = np.full(len(values), np.nan)
    approx_energy = np.full(len(values), np.nan)

    for idx in range(window - 1, len(values)):
        segment = values[idx - window + 1 : idx + 1]
        coeffs = pywt.wavedec(segment, wavelet=wavelet, level=level)
        approx_energy[idx] = np.mean(np.square(coeffs[0]))
        detail_energy_1[idx] = np.mean(np.square(coeffs[-1]))
        detail_energy_2[idx] = np.mean(np.square(coeffs[-2])) if len(coeffs) > 2 else np.nan

    return pd.DataFrame(
        {
            "wavelet_approx_energy": approx_energy,
            "wavelet_detail_energy_1": detail_energy_1,
            "wavelet_detail_energy_2": detail_energy_2,
        },
        index=series.index,
    )


def build_regime_features(
    price_df: pd.DataFrame,
    growth_target: float = 0.06,
    trend_window_months: int = 120,
    wavelet_window_months: int = 48,
    wavelet_name: str = "db2",
) -> pd.DataFrame:
    df = price_df.copy()
    df["realized_growth_10y"] = annualized_growth_from_window(df["log_price"], trend_window_months)
    df["trend_gap"] = df["realized_growth_10y"] - growth_target
    df["price_to_trend"] = df["log_price"] - df["log_price"].rolling(trend_window_months).mean()
    df["vol_12m"] = df["return_1m"].rolling(12).std() * np.sqrt(12.0)
    df["momentum_12m"] = df["price"].pct_change(12)
    df["drawdown_12m"] = df["price"] / df["price"].rolling(12).max() - 1.0

    wavelet_features = rolling_wavelet_energy(
        df["return_1m"].fillna(0.0),
        window=wavelet_window_months,
        wavelet=wavelet_name,
    )
    df = df.join(wavelet_features)

    feature_columns = [
        "return_1m",
        "realized_growth_10y",
        "trend_gap",
        "price_to_trend",
        "vol_12m",
        "momentum_12m",
        "drawdown_12m",
        "wavelet_approx_energy",
        "wavelet_detail_energy_1",
        "wavelet_detail_energy_2",
    ]
    return df[feature_columns].dropna()
