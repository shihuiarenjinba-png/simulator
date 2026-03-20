from __future__ import annotations

import os
import sys

import pandas as pd
import plotly.express as px
import streamlit as st

current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, "src")
if src_dir not in sys.path:
    sys.path.append(src_dir)

from src.regime_data import RegimeDataLoader
from src.regime_features import build_regime_features
from src.regime_model import WaveletHMMRegimeModel


st.set_page_config(page_title="Market Factor Lab", layout="wide")


@st.cache_data(show_spinner=False)
def load_factor_data() -> pd.DataFrame:
    from src.data_loader import DataLoader

    loader = DataLoader()
    return loader.get_merged_data()


@st.cache_data(show_spinner=False)
def load_regime_data(ticker: str, start_date: str) -> pd.DataFrame:
    loader = RegimeDataLoader(ticker=ticker, start_date=start_date)
    return loader.fetch_monthly_prices()


def render_factor_tab() -> None:
    st.subheader("Economic Factor Attribution")
    st.write("景気指数や市場指標に対して、ファクターの先行・遅行関係と寄与度を確認します。")

    if st.button("ファクターデータを取得して分析する", type="primary"):
        with st.spinner("データ取得とラグ分析を進めています..."):
            df_master = load_factor_data()

            if df_master.empty:
                st.error("データ取得に失敗しました。ネットワーク接続やデータソースを確認してください。")
                return

            st.success(f"{len(df_master)} ヶ月分のデータを取得しました。")
            st.dataframe(df_master.tail(12), use_container_width=True)

            available_targets = [col for col in ["DI_Coincident", "CI_Coincident", "VIX", "USD_JPY"] if col in df_master.columns]
            if not available_targets:
                st.error("分析対象となるターゲット指標が見つかりませんでした。")
                return

            from src.analyzer import EconomicAnalyzer
            from src.modeler import EconomicModeler

            analyzer = EconomicAnalyzer(df_master)
            lag_results = analyzer.analyze_multi_targets(available_targets)

            target = st.selectbox("分析対象", available_targets, index=0, key="factor_target")
            if target not in lag_results:
                st.warning("このターゲットのラグ分析結果がありません。")
                return

            lag_table = pd.DataFrame(lag_results[target]).T.reset_index().rename(columns={"index": "factor"})
            st.markdown("**最適ラグの候補**")
            st.dataframe(lag_table, use_container_width=True)

            df_aligned = analyzer.get_lagged_dataset(target)
            if df_aligned is None or df_aligned.empty:
                st.warning("モデリング用データが作成できませんでした。")
                return

            modeler = EconomicModeler(df_aligned)
            results = modeler.train_model()
            if not results:
                st.warning("モデル学習に必要なデータが不足しています。")
                return

            contribution_df = pd.DataFrame(results["contributions"])
            fig = px.bar(
                contribution_df,
                x="factor",
                y="weight",
                color="weight",
                title=f"{target} に対するファクター寄与",
            )
            st.plotly_chart(fig, use_container_width=True)
            st.metric("R^2", f"{results['r_squared']:.3f}")
    else:
        st.info("左のボタンで既存の経済ファクター分析を実行できます。")


def render_regime_tab() -> None:
    st.subheader("Wavelet / HMM Regime Research")
    st.write("長期トレンド、Wavelet 特徴量、HMM によるレジーム推定を試す研究用ビューです。")

    col1, col2, col3, col4 = st.columns(4)
    ticker = col1.text_input("Ticker", value="^GSPC")
    start_date = col2.text_input("Start Date", value="1990-01-01")
    growth_target = col3.number_input("Growth Target", min_value=-0.2, max_value=0.3, value=0.06, step=0.01)
    n_states = col4.number_input("HMM States", min_value=2, max_value=6, value=3, step=1)

    col5, col6, col7 = st.columns(3)
    trend_window = col5.slider("Trend Window (months)", min_value=36, max_value=180, value=120, step=12)
    wavelet_window = col6.slider("Wavelet Window (months)", min_value=24, max_value=96, value=48, step=12)
    wavelet_name = col7.selectbox("Wavelet", ["db2", "db4", "haar", "sym4"], index=0)

    if st.button("レジーム分析を実行する", type="primary"):
        with st.spinner("価格データ取得とレジーム推定を進めています..."):
            prices = load_regime_data(ticker, start_date)
            features = build_regime_features(
                prices,
                growth_target=growth_target,
                trend_window_months=trend_window,
                wavelet_window_months=wavelet_window,
                wavelet_name=wavelet_name,
            )
            model = WaveletHMMRegimeModel(n_states=int(n_states))
            result = model.fit_predict(features)

            output_dir = os.path.join(current_dir, "outputs")
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, "regime_states.csv")
            result.to_csv(output_path)

            latest = model.summarize_recent_regime(result)
            metric1, metric2, metric3 = st.columns(3)
            metric1.metric("Latest Regime", latest["regime_label"])
            metric2.metric("Trend Gap", f"{latest['trend_gap']:.3f}")
            metric3.metric("State Confidence", f"{latest['current_state_prob']:.3f}")

            chart_df = result.reset_index().rename(columns={"index": "Date"})
            price_chart = px.line(chart_df, x="Date", y="return_1m", color="regime_label", title="Monthly Return and Regime")
            st.plotly_chart(price_chart, use_container_width=True)

            st.markdown("**Recent regime states**")
            st.dataframe(
                result.tail(24)[["return_1m", "trend_gap", "vol_12m", "state", "regime_label"]],
                use_container_width=True,
            )
            st.caption(f"CSV saved to {output_path}")
    else:
        st.info("パラメータを調整してから実行すると、結果を CSV に保存します。")


def render_notes_tab() -> None:
    st.subheader("Model Notes")
    st.write("この研究版では、以下の論点を今後拡張できます。")
    st.markdown(
        """
        - 市場データに加えて、景気先行指数、一致指数、VIX、為替、金利などを同時に使う
        - 各系列の lead / lag を探索して、HMM 入力特徴量へ組み込む
        - 固定の 6% ではなく、expanding window で長期トレンドを更新する
        - バブル期は即弱気にせず、下落確率を徐々に上げる設計にする
        """
    )


def main() -> None:
    st.title("Market Factor Lab")
    st.caption("既存のファクター分析と、Wavelet / HMM レジーム研究を1つの画面から試せます。")

    factor_tab, regime_tab, notes_tab = st.tabs(
        ["Factor Attribution", "Regime Research", "Notes"]
    )

    with factor_tab:
        render_factor_tab()
    with regime_tab:
        render_regime_tab()
    with notes_tab:
        render_notes_tab()


if __name__ == "__main__":
    main()
