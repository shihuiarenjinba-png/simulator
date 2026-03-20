from __future__ import annotations

import argparse
import json
import os
import sys

import pandas as pd

current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, "src")
if src_dir not in sys.path:
    sys.path.append(src_dir)

from src.regime_data import RegimeDataLoader
from src.regime_features import build_regime_features
from src.regime_model import WaveletHMMRegimeModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Wavelet/HMM regime research runner")
    parser.add_argument("--ticker", default="^GSPC", help="Yahoo Finance ticker. Default: ^GSPC")
    parser.add_argument("--start-date", default="1990-01-01", help="Analysis start date")
    parser.add_argument("--growth-target", type=float, default=0.06, help="Annual long-run growth target")
    parser.add_argument("--trend-window", type=int, default=120, help="Trend window in months")
    parser.add_argument("--wavelet-window", type=int, default=48, help="Wavelet rolling window in months")
    parser.add_argument("--wavelet", default="db2", help="Wavelet family name")
    parser.add_argument("--states", type=int, default=3, help="Number of HMM states")
    parser.add_argument("--output-csv", default="outputs/regime_states.csv", help="Where to save the regime table")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    loader = RegimeDataLoader(ticker=args.ticker, start_date=args.start_date)
    prices = loader.fetch_monthly_prices()

    features = build_regime_features(
        prices,
        growth_target=args.growth_target,
        trend_window_months=args.trend_window,
        wavelet_window_months=args.wavelet_window,
        wavelet_name=args.wavelet,
    )

    model = WaveletHMMRegimeModel(n_states=args.states)
    result = model.fit_predict(features)

    output_path = os.path.join(current_dir, args.output_csv)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    result.to_csv(output_path)

    recent = model.summarize_recent_regime(result)
    print("Saved regime table to:", output_path)
    print(json.dumps(recent, indent=2))
    print("\nRecent regime counts:")
    print(result["regime_label"].tail(12).value_counts())
    print("\nPreview:")
    print(result.tail(5)[["return_1m", "trend_gap", "vol_12m", "state", "regime_label"]])


if __name__ == "__main__":
    main()
