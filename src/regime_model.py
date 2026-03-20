from __future__ import annotations

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler


class WaveletHMMRegimeModel:
    """Fits a Gaussian HMM to wavelet and trend features."""

    def __init__(self, n_states: int = 3, random_state: int = 42):
        self.n_states = n_states
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.model = GaussianHMM(
            n_components=n_states,
            covariance_type="full",
            n_iter=500,
            random_state=random_state,
        )
        self.feature_columns: list[str] = []

    def fit_predict(self, features: pd.DataFrame) -> pd.DataFrame:
        if features.empty:
            raise ValueError("Features are empty. Cannot fit regime model.")

        self.feature_columns = list(features.columns)
        scaled = self.scaler.fit_transform(features)
        self.model.fit(scaled)
        hidden_states = self.model.predict(scaled)
        state_probs = self.model.predict_proba(scaled)

        result = features.copy()
        result["state"] = hidden_states
        for idx in range(self.n_states):
            result[f"state_{idx}_prob"] = state_probs[:, idx]

        state_summary = result.groupby("state").agg(
            avg_return=("return_1m", "mean"),
            avg_trend_gap=("trend_gap", "mean"),
            avg_vol=("vol_12m", "mean"),
        )
        state_summary["risk_score"] = (
            state_summary["avg_vol"].rank(pct=True)
            + (-state_summary["avg_return"]).rank(pct=True)
            + state_summary["avg_trend_gap"].rank(pct=True)
        ) / 3.0
        order = state_summary["risk_score"].sort_values().index.tolist()
        labels = {}
        if len(order) >= 3:
            labels[order[0]] = "supportive_uptrend"
            labels[order[-1]] = "fragile_downturn"
            for state in order[1:-1]:
                labels[state] = "transition"
        else:
            for state in order:
                labels[state] = f"state_{state}"

        result["regime_label"] = result["state"].map(labels)
        return result

    @staticmethod
    def summarize_recent_regime(result: pd.DataFrame) -> dict[str, object]:
        latest = result.iloc[-1]
        current_state_prob = float(latest.get(f"state_{int(latest['state'])}_prob", 0.0))
        return {
            "date": result.index[-1].strftime("%Y-%m-%d"),
            "state": int(latest["state"]),
            "regime_label": latest["regime_label"],
            "return_1m": float(latest["return_1m"]),
            "trend_gap": float(latest["trend_gap"]),
            "active_growth_target": float(latest.get("active_growth_target", 0.0)),
            "active_growth_window_months": float(latest.get("active_growth_window_months", 0.0)),
            "current_state_prob": current_state_prob,
        }
