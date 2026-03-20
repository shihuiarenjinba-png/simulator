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
from src.factor_library import FACTOR_DATASETS, FIVE_FACTOR_COLUMNS, PORTFOLIO_DATASETS
from src.macro_analysis import (
    build_macro_regime_features,
    build_macro_regression_dataset,
    compute_macro_lead_lag_relationships,
    fit_macro_regression,
    load_macro_relationship_frame,
)

try:
    from src.market_relationships import (
        DEFAULT_MARKET_INDICATORS,
        compute_lag_profile,
        compute_lead_lag_relationships,
        describe_ticker,
        load_market_relationship_frame,
    )
    MARKET_RELATIONSHIPS_IMPORT_ERROR = None
except Exception as exc:
    DEFAULT_MARKET_INDICATORS = {}
    MARKET_RELATIONSHIPS_IMPORT_ERROR = exc

    def load_market_relationship_frame(*args, **kwargs):
        raise RuntimeError("market_relationships module is unavailable") from MARKET_RELATIONSHIPS_IMPORT_ERROR

    def compute_lead_lag_relationships(*args, **kwargs):
        raise RuntimeError("market_relationships module is unavailable") from MARKET_RELATIONSHIPS_IMPORT_ERROR

    def compute_lag_profile(*args, **kwargs):
        raise RuntimeError("market_relationships module is unavailable") from MARKET_RELATIONSHIPS_IMPORT_ERROR

    def describe_ticker(*args, **kwargs):
        raise RuntimeError("market_relationships module is unavailable") from MARKET_RELATIONSHIPS_IMPORT_ERROR


st.set_page_config(page_title="Market Factor Lab", layout="wide")

WAVELET_OPTIONS = {
    "db2": "db2: なめらかな短期変動を見る基本形",
    "db4": "db4: 少し滑らかで中期変動も拾いやすい",
    "haar": "haar: 段差や急変の検出向け",
    "sym4": "sym4: 対称性が高くノイズに比較的強い",
}

BASE_TICKER_PRESETS = {
    "S&P 500 (^GSPC)": "^GSPC",
    "SPY (ETF)": "SPY",
    "Dow Jones (^DJI)": "^DJI",
    "NASDAQ (^IXIC)": "^IXIC",
    "Nikkei 225 (^N225)": "^N225",
    "Custom": "__custom__",
}

FACTOR_SOURCE_PRESETS = list(FACTOR_DATASETS.keys())
PORTFOLIO_SOURCE_PRESETS = list(PORTFOLIO_DATASETS.keys())

REGIME_LABELS_JA = {
    "supportive_uptrend": "上昇を支える局面",
    "transition": "転換・様子見局面",
    "fragile_downturn": "下落が壊れやすく強い局面",
}


@st.cache_data(show_spinner=False)
def load_factor_data(factor_source: str, portfolio_source: str) -> pd.DataFrame:
    from src.data_loader import DataLoader

    loader = DataLoader(factor_source=factor_source, portfolio_source=portfolio_source)
    return loader.get_merged_data()


@st.cache_data(show_spinner=False)
def load_factor_feature_frame(factor_source: str) -> pd.DataFrame:
    df = load_factor_data(factor_source, "なし")
    available_factor_cols = [col for col in FIVE_FACTOR_COLUMNS if col in df.columns]
    return df[available_factor_cols].dropna() if available_factor_cols else pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_regime_data(ticker: str, start_date: str) -> pd.DataFrame:
    loader = RegimeDataLoader(ticker=ticker, start_date=start_date)
    return loader.fetch_monthly_prices()


@st.cache_data(show_spinner=False)
def load_relationship_data(base_ticker: str, start_date: str) -> pd.DataFrame:
    return load_market_relationship_frame(base_ticker=base_ticker, start_date=start_date)


@st.cache_data(show_spinner=False)
def load_macro_relationship_data(base_ticker: str, start_date: str) -> pd.DataFrame:
    return load_macro_relationship_frame(base_ticker=base_ticker, start_date=start_date)


def ensure_app_state() -> None:
    st.session_state.setdefault("factor_df_master", None)
    st.session_state.setdefault("factor_lag_results", None)
    st.session_state.setdefault("factor_available_targets", [])
    st.session_state.setdefault("factor_source_name", FACTOR_SOURCE_PRESETS[0])
    st.session_state.setdefault("portfolio_source_name", "なし")
    st.session_state.setdefault("factor_comparison_rows", [])
    st.session_state.setdefault("factor_weight_rows", [])
    st.session_state.setdefault("relationship_df", None)
    st.session_state.setdefault("relationship_table", None)
    st.session_state.setdefault("relationship_ticker_info", None)
    st.session_state.setdefault("macro_relationship_df", None)
    st.session_state.setdefault("macro_relationship_table", None)
    st.session_state.setdefault("macro_regression_result", None)
    st.session_state.setdefault("macro_ticker_info", None)


def get_selected_relationship_ticker() -> str:
    preset_label = st.session_state.get("relationship_base_preset", list(BASE_TICKER_PRESETS.keys())[0])
    preset_value = BASE_TICKER_PRESETS.get(preset_label, "^GSPC")
    if preset_value == "__custom__":
        return st.session_state.get("relationship_custom_ticker", "^GSPC")
    return preset_value


def render_factor_tab() -> None:
    st.subheader("株価中心の影響マップ")
    st.write("基準となる株価指数を中心にして、周辺の市場・経済指標がどう連動し、先行または遅行しているかを見ます。")

    if MARKET_RELATIONSHIPS_IMPORT_ERROR is not None:
        st.warning(
            "影響マップ機能の追加ファイルがまだ反映されていないため、この部分はいったん停止しています。"
            " ただし下の既存分析とレジーム研究は使えます。"
        )
        st.caption(f"読み込みエラー: {MARKET_RELATIONSHIPS_IMPORT_ERROR}")
    else:
        top_col1, top_col2, top_col3 = st.columns([1.2, 1.2, 1])
        preset_label = top_col1.selectbox("中心に置く株価指数", list(BASE_TICKER_PRESETS.keys()), index=0, key="relationship_base_preset")
        preset_value = BASE_TICKER_PRESETS[preset_label]
        if preset_value == "__custom__":
            base_ticker = top_col2.text_input("カスタムTicker", value="^GSPC", key="relationship_custom_ticker")
        else:
            base_ticker = preset_value
            top_col2.text_input("選択中のTicker", value=base_ticker, disabled=True, key="relationship_selected_ticker")

        relationship_start = top_col3.text_input("分析開始日", value="2000-01-01", key="relationship_start")
        st.caption("例: `^GSPC`, `SPY`, `^DJI`, `NASDAQ`, `N225`, `VIX`, `USD/JPY` のような入力でも自動解釈します。")

        if st.button("影響マップを作成する", type="primary"):
            with st.spinner("周辺指標との相関・先行遅行を計算しています..."):
                ticker_info = describe_ticker(base_ticker)
                relationship_df = load_relationship_data(base_ticker, relationship_start)
                relationship_table = compute_lead_lag_relationships(relationship_df, base_col="base_asset", max_lag=6)
                st.session_state["relationship_df"] = relationship_df
                st.session_state["relationship_table"] = relationship_table
                st.session_state["relationship_ticker_info"] = ticker_info

        relationship_df = st.session_state.get("relationship_df")
        relationship_table = st.session_state.get("relationship_table")
        relationship_ticker_info = st.session_state.get("relationship_ticker_info")

        if relationship_df is not None and relationship_table is not None:
            if relationship_ticker_info:
                st.success(
                    f"入力 `{relationship_ticker_info['input']}` を "
                    f"`{relationship_ticker_info['label']}` (`{relationship_ticker_info['normalized']}`) として分析しています。"
                )

            if relationship_table.empty:
                st.warning("相関を計算できるだけのデータが集まりませんでした。")
            else:
                fig = px.bar(
                    relationship_table,
                    x="best_corr",
                    y="indicator",
                    color="relationship_type",
                    orientation="h",
                    hover_data=["same_month_corr", "best_lag_months", "impact_direction"],
                    title="中心指数に対する周辺指標の影響",
                )
                fig.update_layout(yaxis_title="周辺指標", xaxis_title="最大相関")
                st.plotly_chart(fig, width="stretch")

                selected_indicator = st.selectbox(
                    "ラグの形を見る指標",
                    relationship_table["indicator"].tolist(),
                    key="lag_profile_indicator",
                )
                lag_profile = compute_lag_profile(relationship_df, selected_indicator, max_lag=6)
                lag_fig = px.line(
                    lag_profile,
                    x="lag_months",
                    y="correlation",
                    markers=True,
                    title=f"{selected_indicator} と {base_ticker} の先行・遅行プロファイル",
                )
                lag_fig.update_layout(xaxis_title="ラグ(月)  正なら先行 / 負なら遅行", yaxis_title="相関")
                st.plotly_chart(lag_fig, width="stretch")

                st.markdown("**相関・先行遅行テーブル**")
                st.dataframe(relationship_table, width="stretch")

                with st.expander("現在使っている周辺指標"):
                    st.write(DEFAULT_MARKET_INDICATORS)
        else:
            st.info("まずは株価指数を1つ決めて、周辺の株価指数や経済指標との関係を見られます。")

    st.divider()
    st.subheader("マルチファクター寄与分析")
    st.write("Fama-French 系ファクターを使って、複数の景気・市場指標を並べて比較できます。")
    factor_col1, factor_col2 = st.columns(2)
    factor_source = factor_col1.selectbox("ファクター地域 / ソース", FACTOR_SOURCE_PRESETS, index=0, key="factor_source_select")
    portfolio_source = factor_col2.selectbox("追加ポートフォリオデータ", PORTFOLIO_SOURCE_PRESETS, index=0, key="portfolio_source_select")
    with st.expander("この欄は何を見るのか"):
        st.markdown(
            """
            - ここは個別銘柄を入れる欄ではありません。
            - 既に組み込まれている景気指標や市場指標、あるいはポートフォリオ収益率に対して、Fama-French ファクターがどれくらい効いているかを見ます。
            - `Japan 5 Factors` や `North America 5 Factors` を切り替えて、地域ごとの差も比べられます。
            - 複数の対象を同時に選んで、どのファクターが効き方の違いを生んでいるかを比較できます。
            - 個別銘柄や自分のポートフォリオに広げるなら、次の段階で別入力欄を追加するのが自然です。
            """
        )

    if st.button("ファクターデータを取得 / 更新する", type="primary"):
        with st.spinner("データ取得とラグ分析を進めています..."):
            df_master = load_factor_data(factor_source, portfolio_source)

            if df_master.empty:
                st.error("データ取得に失敗しました。ネットワーク接続やデータソースを確認してください。")
                st.session_state["factor_df_master"] = None
                st.session_state["factor_lag_results"] = None
                st.session_state["factor_available_targets"] = []
                st.session_state["factor_comparison_rows"] = []
                st.session_state["factor_weight_rows"] = []
            else:
                available_targets = [col for col in df_master.columns if col not in FIVE_FACTOR_COLUMNS + ["RF"]]
                if not available_targets:
                    st.error("分析対象となるターゲット指標が見つかりませんでした。")
                    st.session_state["factor_df_master"] = None
                    st.session_state["factor_lag_results"] = None
                    st.session_state["factor_available_targets"] = []
                    st.session_state["factor_comparison_rows"] = []
                    st.session_state["factor_weight_rows"] = []
                else:
                    from src.analyzer import EconomicAnalyzer

                    analyzer = EconomicAnalyzer(df_master, factors=FIVE_FACTOR_COLUMNS)
                    lag_results = analyzer.analyze_multi_targets(available_targets)
                    st.session_state["factor_df_master"] = df_master
                    st.session_state["factor_lag_results"] = lag_results
                    st.session_state["factor_available_targets"] = available_targets
                    st.session_state["factor_source_name"] = factor_source
                    st.session_state["portfolio_source_name"] = portfolio_source

    factor_df_master = st.session_state.get("factor_df_master")
    factor_lag_results = st.session_state.get("factor_lag_results")
    factor_available_targets = st.session_state.get("factor_available_targets", [])

    if factor_df_master is not None and factor_lag_results:
        st.success(
            f"{len(factor_df_master)} ヶ月分のデータを取得済みです。"
            f" ファクター: {st.session_state.get('factor_source_name')} / 追加データ: {st.session_state.get('portfolio_source_name')}"
        )
        selected_targets = st.multiselect(
            "比較したい分析対象データ",
            factor_available_targets,
            default=factor_available_targets[: min(2, len(factor_available_targets))],
            key="factor_target_multi",
        )
        st.dataframe(factor_df_master.tail(12), width="stretch")

        if not selected_targets:
            st.info("少なくとも1つ分析対象を選ぶと、下に寄与分析が出ます。")
        else:
            from src.analyzer import EconomicAnalyzer
            from src.modeler import EconomicModeler

            analyzer = EconomicAnalyzer(factor_df_master, factors=FIVE_FACTOR_COLUMNS)
            analyzer.results = factor_lag_results
            factor_tabs = st.tabs(selected_targets)
            comparison_rows = []
            weight_rows = []

            for tab, target in zip(factor_tabs, selected_targets):
                with tab:
                    if target not in factor_lag_results:
                        st.warning(f"{target} のラグ分析結果がありません。")
                        continue

                    lag_table = pd.DataFrame(factor_lag_results[target]).T.reset_index().rename(columns={"index": "factor"})
                    st.markdown("**最適ラグの候補**")
                    st.dataframe(lag_table, width="stretch")

                    df_aligned = analyzer.get_lagged_dataset(target)
                    if df_aligned is None or df_aligned.empty:
                        st.warning("モデリング用データが作成できませんでした。")
                        continue

                    modeler = EconomicModeler(df_aligned)
                    results = modeler.train_model()
                    if not results:
                        st.warning("モデル学習に必要なデータが不足しています。")
                        continue

                    comparison_rows.append(
                        {
                            "target": target,
                            "r_squared": results["r_squared"],
                            "samples": len(df_aligned),
                        }
                    )
                    for item in results["contributions"]:
                        weight_rows.append(
                            {
                                "target": target,
                                "factor": item["factor"],
                                "weight": item["weight"],
                                "impact": item["impact"],
                            }
                        )

                    contribution_df = pd.DataFrame(results["contributions"])
                    fig = px.bar(
                        contribution_df,
                        x="factor",
                        y="weight",
                        color="weight",
                        title=f"{target} に対するファクター寄与",
                    )
                    st.plotly_chart(fig, width="stretch")
                    st.metric("R^2", f"{results['r_squared']:.3f}")

            st.session_state["factor_comparison_rows"] = comparison_rows
            st.session_state["factor_weight_rows"] = weight_rows

            if comparison_rows:
                st.markdown("**比較サマリー**")
                comparison_df = pd.DataFrame(comparison_rows).sort_values("r_squared", ascending=False)
                st.dataframe(comparison_df, width="stretch")

            if weight_rows:
                st.markdown("**ファクター感応度の比較**")
                weight_df = pd.DataFrame(weight_rows)
                pivot_df = weight_df.pivot(index="target", columns="factor", values="weight").reset_index()
                heatmap_df = weight_df.pivot(index="target", columns="factor", values="weight")
                st.dataframe(pivot_df, width="stretch")
                heatmap = px.imshow(
                    heatmap_df,
                    aspect="auto",
                    color_continuous_scale="RdBu",
                    origin="lower",
                    title="ターゲット別ファクター荷重ヒートマップ",
                )
                st.plotly_chart(heatmap, width="stretch")
    else:
        st.info("上のボタンでデータを取得すると、ここで分析対象を切り替えられます。")

    st.divider()
    st.subheader("マクロ指数影響分析")
    st.write("市場に対して、景気指数や為替・ボラティリティといったマクロ系列がどの程度影響しているかを見ます。")
    macro_target_ticker = get_selected_relationship_ticker()
    macro_start_date = st.session_state.get("relationship_start", "2000-01-01")
    st.caption(
        f"現在は `{macro_target_ticker}` を対象に、景気一致指数・CI一致指数・VIX・USD/JPY の先行相関を調べます。"
    )

    macro_col1, macro_col2 = st.columns(2)
    macro_max_lag = macro_col1.slider("マクロ先行ラグ上限(月)", min_value=1, max_value=18, value=12, step=1, key="macro_max_lag")
    macro_top_n = macro_col2.slider("回帰に使う上位マクロ系列数", min_value=2, max_value=8, value=4, step=1, key="macro_top_n")

    if st.button("マクロ指数分析を実行する", type="primary"):
        with st.spinner("マクロ指数との先行相関と回帰寄与を計算しています..."):
            macro_frame = load_macro_relationship_data(macro_target_ticker, macro_start_date)
            macro_table = compute_macro_lead_lag_relationships(macro_frame, max_lag=macro_max_lag)
            macro_regression_df = build_macro_regression_dataset(macro_frame, macro_table, top_n=macro_top_n) if not macro_table.empty else pd.DataFrame()
            macro_regression_result = fit_macro_regression(macro_regression_df)
            st.session_state["macro_relationship_df"] = macro_frame
            st.session_state["macro_relationship_table"] = macro_table
            st.session_state["macro_regression_result"] = macro_regression_result
            st.session_state["macro_ticker_info"] = describe_ticker(macro_target_ticker)

    macro_relationship_table = st.session_state.get("macro_relationship_table")
    macro_regression_result = st.session_state.get("macro_regression_result")
    macro_ticker_info = st.session_state.get("macro_ticker_info")

    if macro_relationship_table is not None:
        if macro_ticker_info:
            st.success(
                f"マクロ分析対象は `{macro_ticker_info['label']}` (`{macro_ticker_info['normalized']}`) です。"
            )

        if macro_relationship_table.empty:
            st.warning("マクロ系列との相関を計算できるだけのデータが集まりませんでした。")
        else:
            macro_fig = px.bar(
                macro_relationship_table,
                x="best_corr",
                y="indicator_label",
                color="impact_direction",
                orientation="h",
                hover_data=["same_month_corr", "best_lag_months"],
                title="マクロ系列の先行影響",
            )
            macro_fig.update_layout(yaxis_title="マクロ系列", xaxis_title="最大相関")
            st.plotly_chart(macro_fig, width="stretch")

            st.markdown("**マクロ先行相関テーブル**")
            st.dataframe(
                macro_relationship_table[
                    ["indicator_label", "same_month_corr", "best_corr", "best_lag_months", "impact_direction"]
                ],
                width="stretch",
            )

            if macro_regression_result:
                st.markdown("**マクロ回帰の寄与度**")
                macro_contrib_df = pd.DataFrame(macro_regression_result["contributions"])
                contrib_fig = px.bar(
                    macro_contrib_df,
                    x="factor_label",
                    y="weight",
                    color="weight",
                    title="マクロ系列の回帰寄与",
                )
                st.plotly_chart(contrib_fig, width="stretch")
                metric_col1, metric_col2 = st.columns(2)
                metric_col1.metric("R^2", f"{macro_regression_result['r_squared']:.3f}")
                metric_col2.metric("サンプル数", f"{macro_regression_result['samples']}")
    else:
        st.info("上のボタンでマクロ指数分析を実行すると、ここに先行相関と回帰寄与が表示されます。")


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
    with st.expander("この画面の使い方"):
        st.markdown(
            """
            - `Ticker`: 分析したい中心資産です。`^GSPC` や `N225` のように入れます。
            - `成長率の決め方`: 固定値は常に同じ基準で見ます。自動特定は 3% や 6% に近い窓を毎回探します。
            - `隠れ状態の数`: 市場局面を何種類に分けるかです。まずは 3 が無難です。
            - `基準トレンド窓`: 長期基準を見る長さです。
            - `Wavelet 窓`: 値動きの波形を見る長さです。
            - `レジーム分析を実行する`: 下のグラフと表を更新して、CSV も保存します。
            """
        )
    st.caption("ここで出る線や局面は探索的な研究結果です。これだけで平均回帰が正式に証明されたことにはなりません。正式な検証には、外部期間テストや感度分析が必要です。")

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

    use_factor_features = st.checkbox("レジーム推定にマルチファクター特徴量も加える", value=False)
    regime_factor_source = None
    if use_factor_features:
        regime_factor_source = st.selectbox("レジーム側で使うファクター地域", FACTOR_SOURCE_PRESETS, index=0, key="regime_factor_source")

    use_macro_features = st.checkbox("レジーム推定にマクロ指数特徴量も加える", value=True)
    macro_feature_top_n = None
    macro_feature_max_lag = None
    if use_macro_features:
        macro_col1, macro_col2 = st.columns(2)
        macro_feature_top_n = macro_col1.slider("レジームに入れる上位マクロ系列数", min_value=2, max_value=8, value=4, step=1, key="regime_macro_top_n")
        macro_feature_max_lag = macro_col2.slider("マクロ先行ラグ上限(月)", min_value=1, max_value=18, value=12, step=1, key="regime_macro_max_lag")

    if st.button("レジーム分析を実行する", type="primary"):
        with st.spinner("価格データ取得とレジーム推定を進めています..."):
            prices = load_regime_data(ticker, start_date)
            extra_feature_frames = []
            if use_factor_features and regime_factor_source:
                factor_features = load_factor_feature_frame(regime_factor_source)
                if factor_features is not None and not factor_features.empty:
                    extra_feature_frames.append(factor_features)
            if use_macro_features and macro_feature_top_n is not None and macro_feature_max_lag is not None:
                macro_features = build_macro_regime_features(
                    base_ticker=ticker,
                    start_date=start_date,
                    max_lag=macro_feature_max_lag,
                    top_n=macro_feature_top_n,
                )
                if macro_features is not None and not macro_features.empty:
                    extra_feature_frames.append(macro_features)

            extra_features = None
            if extra_feature_frames:
                extra_features = pd.concat(extra_feature_frames, axis=1)
                extra_features = extra_features.loc[~extra_features.index.duplicated(keep="first")].sort_index()

            features = build_regime_features(
                prices,
                growth_target=growth_target,
                trend_window_months=trend_window,
                wavelet_window_months=wavelet_window,
                wavelet_name=wavelet_name,
                growth_target_mode=growth_mode,
                candidate_targets=candidate_targets,
                extra_features=extra_features,
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
            st.plotly_chart(price_chart, width="stretch")

            target_chart = px.line(
                chart_df,
                x="Date",
                y=["active_growth_target", "matched_growth_rate"],
                title="目標成長率と実現成長率",
            )
            target_chart.update_layout(yaxis_title="年率", xaxis_title="日付")
            st.plotly_chart(target_chart, width="stretch")

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
                width="stretch",
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
    ensure_app_state()
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
