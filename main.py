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
from src.market_relationships import (
    DEFAULT_MARKET_INDICATORS,
    compute_lag_profile,
    compute_lead_lag_relationships,
    load_market_relationship_frame,
)
from src.regime_model import WaveletHMMRegimeModel


st.set_page_config(page_title="Market Factor Lab", layout="wide")

WAVELET_OPTIONS = {
    "db2": "db2: なめらかな短期変動を見る基本形",
    "db4": "db4: 少し滑らかで中期変動も拾いやすい",
    "haar": "haar: 段差や急変の検出向け",
    "sym4": "sym4: 対称性が高くノイズに比較的強い",
}

REGIME_LABELS_JA = {
    "supportive_uptrend": "上昇を支える局面",
    "transition": "転換・様子見局面",
    "fragile_downturn": "下落が壊れやすく強い局面",
}


@st.cache_data(show_spinner=False)
def load_factor_data() -> pd.DataFrame:
    from src.data_loader import DataLoader

    loader = DataLoader()
    return loader.get_merged_data()


@st.cache_data(show_spinner=False)
def load_regime_data(ticker: str, start_date: str) -> pd.DataFrame:
    loader = RegimeDataLoader(ticker=ticker, start_date=start_date)
    return loader.fetch_monthly_prices()


@st.cache_data(show_spinner=False)
def load_relationship_data(base_ticker: str, start_date: str) -> pd.DataFrame:
    return load_market_relationship_frame(base_ticker=base_ticker, start_date=start_date)


def render_factor_tab() -> None:
    st.subheader("株価中心の影響マップ")
    st.write("基準となる株価指数を中心にして、周辺の市場・経済指標がどう連動し、先行または遅行しているかを見ます。")

    top_col1, top_col2 = st.columns([1, 1])
    base_ticker = top_col1.text_input("中心に置く株価指数", value="^GSPC")
    relationship_start = top_col2.text_input("分析開始日", value="2000-01-01", key="relationship_start")

    if st.button("影響マップを作成する", type="primary"):
        with st.spinner("周辺指標との相関・先行遅行を計算しています..."):
            relationship_df = load_relationship_data(base_ticker, relationship_start)
            relationship_table = compute_lead_lag_relationships(relationship_df, base_col="base_asset", max_lag=6)

            if relationship_table.empty:
                st.warning("相関を計算できるだけのデータが集まりませんでした。")
                return

            fig = px.bar(
                relationship_table,
                x="best_corr",
                y="indicator",
                color="relationship_type",
                orientation="h",
                hover_data=["same_month_corr", "best_lag_months", "impact_direction"],
                title=f"{base_ticker} に対する周辺指標の影響",
            )
            fig.update_layout(yaxis_title="周辺指標", xaxis_title="最大相関")
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("**相関・先行遅行テーブル**")
            st.dataframe(relationship_table, use_container_width=True)

            selected_indicator = st.selectbox("ラグの形を見る指標", relationship_table["indicator"].tolist(), key="lag_profile_indicator")
            lag_profile = compute_lag_profile(relationship_df, selected_indicator, max_lag=6)
            lag_fig = px.line(
                lag_profile,
                x="lag_months",
                y="correlation",
                markers=True,
                title=f"{selected_indicator} と {base_ticker} の先行・遅行プロファイル",
            )
            lag_fig.update_layout(xaxis_title="ラグ(月)  正なら先行 / 負なら遅行", yaxis_title="相関")
            st.plotly_chart(lag_fig, use_container_width=True)

            with st.expander("現在使っている周辺指標"):
                st.write(DEFAULT_MARKET_INDICATORS)
    else:
        st.info("まずは株価指数を1つ決めて、周辺の株価指数や経済指標との関係を見られます。")

    st.divider()
    st.subheader("既存のファクター寄与分析")
    st.write("Fama-French 系ファクターと景気指標の関係も、従来どおり確認できます。")

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
    st.subheader("Wavelet / HMM レジーム研究")
    st.write("長期成長率、Wavelet の窓、隠れ状態モデルを組み合わせて、いまの市場局面を見ます。")

    with st.expander("用語の意味"):
        st.markdown(
            """
            - `Wavelet`: 値動きを周波数ごとに分けて、短期ノイズと中期のうねりを分けて見る方法です。
            - `db2 / db4 / haar / sym4`: Wavelet の種類で、値動きのどの形を強調するかが少し変わります。
            - `HMM`: 隠れマルコフモデルです。表面の価格変動の裏にある「上昇局面・転換局面・下落局面」のような見えない状態を推定します。
            - `グロースターゲット`: 長期で達成したい年率成長の基準です。
            """
        )

    col1, col2, col3, col4 = st.columns(4)
    ticker = col1.text_input("Ticker", value="^GSPC")
    start_date = col2.text_input("Start Date", value="1990-01-01")
    growth_mode = col3.radio("成長率の決め方", options=["fixed", "auto"], format_func=lambda x: "固定値" if x == "fixed" else "自動特定", horizontal=True)
    n_states = col4.number_input("隠れ状態の数", min_value=2, max_value=6, value=3, step=1)

    if growth_mode == "fixed":
        growth_target = st.number_input("固定の長期成長率", min_value=-0.2, max_value=0.3, value=0.06, step=0.01)
        candidate_targets = [growth_target]
    else:
        growth_target = 0.06
        candidate_targets = st.multiselect(
            "自動特定の候補利率",
            options=[0.03, 0.06, 0.09, 0.12],
            default=[0.03, 0.06],
            format_func=lambda x: f"{int(x * 100)}%",
        )
        if not candidate_targets:
            candidate_targets = [0.03, 0.06]

    col5, col6, col7 = st.columns(3)
    trend_window = col5.slider("基準トレンド窓(月)", min_value=36, max_value=180, value=120, step=12)
    wavelet_window = col6.slider("Wavelet 窓(月)", min_value=24, max_value=96, value=48, step=12)
    wavelet_name = col7.selectbox("Wavelet の種類", list(WAVELET_OPTIONS.keys()), index=0, format_func=lambda x: WAVELET_OPTIONS[x])

    if st.button("レジーム分析を実行する", type="primary"):
        with st.spinner("価格データ取得とレジーム推定を進めています..."):
            prices = load_regime_data(ticker, start_date)
            features = build_regime_features(
                prices,
                growth_target=growth_target,
                trend_window_months=trend_window,
                wavelet_window_months=wavelet_window,
                wavelet_name=wavelet_name,
                growth_target_mode=growth_mode,
                candidate_targets=candidate_targets,
            )
            model = WaveletHMMRegimeModel(n_states=int(n_states))
            result = model.fit_predict(features)
            result["regime_label_ja"] = result["regime_label"].map(REGIME_LABELS_JA).fillna(result["regime_label"])

            output_dir = os.path.join(current_dir, "outputs")
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, "regime_states.csv")
            result.to_csv(output_path)

            latest = model.summarize_recent_regime(result)
            metric1, metric2, metric3, metric4 = st.columns(4)
            metric1.metric("現在の局面", REGIME_LABELS_JA.get(latest["regime_label"], latest["regime_label"]))
            metric2.metric("成長率とのズレ", f"{latest['trend_gap']:.3f}")
            metric3.metric("状態の確信度", f"{latest['current_state_prob']:.3f}")
            metric4.metric("採用された成長目標", f"{latest['active_growth_target']:.2%}")

            st.caption(
                f"今回の判定では、直近で年率 {latest['active_growth_target']:.2%} に最も近い窓として "
                f"{int(latest['active_growth_window_months'])} ヶ月が採用されました。"
            )

            chart_df = result.reset_index().rename(columns={"index": "Date"})
            price_chart = px.line(chart_df, x="Date", y="matched_growth_rate", color="regime_label_ja", title="成長率の到達状況とレジーム")
            price_chart.update_layout(yaxis_title="年率換算成長率", xaxis_title="日付")
            st.plotly_chart(price_chart, use_container_width=True)

            target_chart = px.line(
                chart_df,
                x="Date",
                y=["active_growth_target", "matched_growth_rate"],
                title="目標成長率と実現成長率",
            )
            target_chart.update_layout(yaxis_title="年率", xaxis_title="日付")
            st.plotly_chart(target_chart, use_container_width=True)

            st.markdown("**Recent regime states**")
            st.dataframe(
                result.tail(24)[
                    [
                        "return_1m",
                        "matched_growth_rate",
                        "active_growth_target",
                        "active_growth_window_months",
                        "trend_gap",
                        "vol_12m",
                        "state",
                        "regime_label_ja",
                    ]
                ],
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
        - 株価指数を中心にして、周辺の市場・経済指標との相関マップを広げる
        - 各系列の先行・遅行を自動探索し、HMM 入力特徴量へ組み込む
        - 固定の 6% だけでなく、3% や 6% を自動達成する窓を切り出して使う
        - バブル期は即弱気にせず、下落確率を徐々に上げる設計にする
        """
    )


def main() -> None:
    st.title("Market Factor Lab")
    st.caption("既存のファクター分析と、Wavelet / HMM レジーム研究を1つの画面から試せます。")

    factor_tab, regime_tab, notes_tab = st.tabs(
        ["影響マップ", "レジーム研究", "設計メモ"]
    )

    with factor_tab:
        render_factor_tab()
    with regime_tab:
        render_regime_tab()
    with notes_tab:
        render_notes_tab()


if __name__ == "__main__":
    main()
