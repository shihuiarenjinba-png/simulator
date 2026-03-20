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


def detect_dynamic_growth_targets(
    log_prices: pd.Series,
    candidate_targets: list[float],
    min_window_months: int = 24,
    max_window_months: int = 180,
) -> pd.DataFrame:
    target_values = np.full(len(log_prices), np.nan)
    target_windows = np.full(len(log_prices), np.nan)
    realized_growth = np.full(len(log_prices), np.nan)
    target_distance = np.full(len(log_prices), np.nan)

    values = log_prices.to_numpy(dtype=float)
    for idx in range(max_window_months, len(values)):
        current = values[idx]
        best_choice = None

        for window in range(min_window_months, max_window_months + 1):
            past = values[idx - window]
            annualized_growth = np.exp((current - past) * (12.0 / window)) - 1.0
            for target in candidate_targets:
                distance = abs(annualized_growth - target)
                score = (distance, window)
                if best_choice is None or score < best_choice[0]:
                    best_choice = (score, target, window, annualized_growth)

        if best_choice is None:
            continue

        _, target, window, annualized_growth = best_choice
        target_values[idx] = target
        target_windows[idx] = window
        realized_growth[idx] = annualized_growth
        target_distance[idx] = annualized_growth - target

    return pd.DataFrame(
        {
            "active_growth_target": target_values,
            "active_growth_window_months": target_windows,
            "matched_growth_rate": realized_growth,
            "target_distance": target_distance,
        },
        index=log_prices.index,
    )


def build_regime_features(
    price_df: pd.DataFrame,
    growth_target: float = 0.06,
    trend_window_months: int = 120,
    wavelet_window_months: int = 48,
    wavelet_name: str = "db2",
    growth_target_mode: str = "fixed",
    candidate_targets: list[float] | None = None,
    dynamic_target_min_window: int = 24,
    dynamic_target_max_window: int = 180,
    extra_features: pd.DataFrame | None = None,
) -> pd.DataFrame:
    df = price_df.copy()
    df["realized_growth_reference"] = annualized_growth_from_window(df["log_price"], trend_window_months)
    if growth_target_mode == "auto":
        dynamic_targets = detect_dynamic_growth_targets(
            df["log_price"],
            candidate_targets=candidate_targets or [0.03, 0.06],
            min_window_months=dynamic_target_min_window,
            max_window_months=dynamic_target_max_window,
        )
        df = df.join(dynamic_targets)
        df["trend_gap"] = df["target_distance"]
    else:
        df["active_growth_target"] = growth_target
        df["active_growth_window_months"] = trend_window_months
        df["matched_growth_rate"] = df["realized_growth_reference"]
        df["target_distance"] = df["realized_growth_reference"] - growth_target
        df["trend_gap"] = df["target_distance"]

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

    extra_feature_columns: list[str] = []
    if extra_features is not None and not extra_features.empty:
        aligned_extra = extra_features.copy()
        aligned_extra = aligned_extra.loc[~aligned_extra.index.duplicated(keep="first")].sort_index()
        aligned_extra = aligned_extra.add_prefix("factor_")
        df = df.join(aligned_extra, how="left")
        extra_feature_columns = aligned_extra.columns.tolist()

    feature_columns = [
        "return_1m",
        "realized_growth_reference",
        "matched_growth_rate",
        "active_growth_target",
        "active_growth_window_months",
        "trend_gap",
        "target_distance",
        "price_to_trend",
        "vol_12m",
        "momentum_12m",
        "drawdown_12m",
        "wavelet_approx_energy",
        "wavelet_detail_energy_1",
        "wavelet_detail_energy_2",
    ]
    feature_columns.extend(extra_feature_columns)
    return df[feature_columns].dropna()
