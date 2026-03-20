from __future__ import annotations

import datetime as dt

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from src.data_loader import DataLoader
from src.market_relationships import normalize_ticker
from src.regime_data import RegimeDataLoader


MACRO_DISPLAY_NAMES = {
    "DI_Coincident_level": "景気一致指数",
    "DI_Coincident_change": "景気一致指数の前月差",
    "CI_Coincident_level": "CI一致指数",
    "CI_Coincident_change": "CI一致指数の前月差",
    "VIX_level": "VIX水準",
    "VIX_change_pct": "VIX変化率",
    "USD_JPY_level": "USD/JPY水準",
    "USD_JPY_change_pct": "USD/JPY変化率",
}


def load_macro_feature_frame(start_date: str, end_date: str | None = None) -> pd.DataFrame:
    end = end_date or dt.datetime.now().strftime("%Y-%m-%d")
    loader = DataLoader()
    loader.start_date = start_date
    loader.end_date = end

    df_cabinet = loader.fetch_cabinet_office_data()
    df_market = loader.fetch_market_data()
    df_macro = pd.concat([df_cabinet, df_market], axis=1).sort_index()

    features = pd.DataFrame(index=df_macro.index)

    if "DI_Coincident" in df_macro.columns:
        features["DI_Coincident_level"] = df_macro["DI_Coincident"]
        features["DI_Coincident_change"] = df_macro["DI_Coincident"].diff()

    if "CI_Coincident" in df_macro.columns:
        features["CI_Coincident_level"] = df_macro["CI_Coincident"]
        features["CI_Coincident_change"] = df_macro["CI_Coincident"].diff()

    if "VIX" in df_macro.columns:
        features["VIX_level"] = df_macro["VIX"]
        features["VIX_change_pct"] = df_macro["VIX"].pct_change()

    if "USD_JPY" in df_macro.columns:
        features["USD_JPY_level"] = df_macro["USD_JPY"]
        features["USD_JPY_change_pct"] = df_macro["USD_JPY"].pct_change()

    return features.dropna(how="all")


def load_macro_relationship_frame(base_ticker: str, start_date: str, end_date: str | None = None) -> pd.DataFrame:
    ticker = normalize_ticker(base_ticker)
    price_loader = RegimeDataLoader(ticker=ticker, start_date=start_date, end_date=end_date)
    price_df = price_loader.fetch_monthly_prices()[["return_1m"]].rename(columns={"return_1m": "base_asset"})
    macro_features = load_macro_feature_frame(start_date=start_date, end_date=end_date)
    frame = price_df.join(macro_features, how="left")
    return frame[frame["base_asset"].notna()]


def compute_macro_lead_lag_relationships(
    frame: pd.DataFrame,
    base_col: str = "base_asset",
    max_lag: int = 12,
) -> pd.DataFrame:
    if base_col not in frame.columns:
        raise ValueError(f"{base_col} not found in macro frame.")

    base_series = frame[base_col]
    rows = []

    for col in frame.columns:
        if col == base_col:
            continue

        indicator = frame[col]
        best_corr = None
        best_lag = 0
        same_month_corr = None

        for lag in range(0, max_lag + 1):
            shifted = indicator.shift(lag)
            valid = base_series.notna() & shifted.notna()
            if valid.sum() < 24:
                continue

            corr = base_series[valid].corr(shifted[valid])
            if lag == 0:
                same_month_corr = corr

            if best_corr is None or abs(corr) > abs(best_corr):
                best_corr = corr
                best_lag = lag

        if best_corr is None:
            continue

        rows.append(
            {
                "indicator": col,
                "indicator_label": MACRO_DISPLAY_NAMES.get(col, col),
                "same_month_corr": same_month_corr,
                "best_corr": best_corr,
                "best_lag_months": best_lag,
                "impact_direction": "押し上げ寄り" if best_corr > 0 else "逆方向",
            }
        )

    return pd.DataFrame(rows).sort_values(by="best_corr", key=lambda s: s.abs(), ascending=False)


def build_macro_regression_dataset(
    frame: pd.DataFrame,
    relationship_table: pd.DataFrame,
    base_col: str = "base_asset",
    top_n: int = 4,
) -> pd.DataFrame:
    if base_col not in frame.columns:
        raise ValueError(f"{base_col} not found in macro frame.")

    selected = relationship_table.head(top_n)
    regression_df = pd.DataFrame(index=frame.index)
    regression_df[base_col] = frame[base_col]

    for row in selected.itertuples(index=False):
        lag = int(row.best_lag_months)
        feature_name = f"{row.indicator}_Lead{lag}"
        regression_df[feature_name] = frame[row.indicator].shift(lag)

    return regression_df.dropna()


def fit_macro_regression(dataset: pd.DataFrame, base_col: str = "base_asset") -> dict[str, object] | None:
    if dataset is None or dataset.empty or base_col not in dataset.columns or len(dataset) < 24:
        return None

    X = dataset.drop(columns=[base_col])
    y = dataset[base_col]
    if X.empty:
        return None

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    X_scaled = x_scaler.fit_transform(X)
    y_scaled = y_scaler.fit_transform(y.to_numpy().reshape(-1, 1)).ravel()

    model = LinearRegression()
    model.fit(X_scaled, y_scaled)

    contributions = []
    for name, coef in zip(X.columns, model.coef_):
        contributions.append(
            {
                "factor": name,
                "factor_label": MACRO_DISPLAY_NAMES.get(name.split("_Lead")[0], name),
                "weight": coef,
                "impact": abs(coef),
            }
        )
    contributions.sort(key=lambda item: item["impact"], reverse=True)

    return {
        "r_squared": model.score(X_scaled, y_scaled),
        "samples": len(dataset),
        "contributions": contributions,
    }


def build_macro_regime_features(
    base_ticker: str,
    start_date: str,
    max_lag: int = 12,
    top_n: int = 4,
) -> pd.DataFrame:
    frame = load_macro_relationship_frame(base_ticker=base_ticker, start_date=start_date)
    relationship_table = compute_macro_lead_lag_relationships(frame, max_lag=max_lag)
    if relationship_table.empty:
        return pd.DataFrame()

    regression_df = build_macro_regression_dataset(frame, relationship_table, top_n=top_n)
    if regression_df.empty:
        return pd.DataFrame()

    return regression_df.drop(columns=["base_asset"], errors="ignore")
