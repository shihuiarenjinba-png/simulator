"""Export market regime analysis report as static HTML for GitHub Pages.

Reads the bundled `regime_states.csv` (~315 months of monthly market state
data from 2000 to 2026 with HMM 3-state probabilities + wavelet features),
produces four interactive plotly charts, and writes the whole thing to
`docs/index.html`.

The CSV is the pre-computed output of `research_regime.py` — using it here
keeps the report fast (no HMM refit per build), reproducible (no yfinance
calls), and reachable from Pages without any API key.

Local run:
    python tools/export_report.py
"""
from __future__ import annotations

import pathlib

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = ROOT / "regime_states.csv"
OUT = ROOT / "docs" / "index.html"

REGIME_COLORS = {
    "bull": "#10b981",
    "bear": "#ef4444",
    "transition": "#f59e0b",
    "neutral": "#94a3b8",
    "recovery": "#0ea5e9",
}


def load_states() -> pd.DataFrame:
    df = pd.read_csv(DATA, parse_dates=["Date"]).set_index("Date").sort_index()
    if "regime_label" in df.columns:
        df["regime_label"] = df["regime_label"].astype(str).fillna("neutral")
    return df


def fig_cumulative_by_regime(df: pd.DataFrame) -> go.Figure:
    cum = (1 + df["return_1m"].fillna(0)).cumprod()
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=cum.index, y=cum, mode="lines",
        name="Cumulative return", line=dict(color="#0f172a", width=2),
        hovertemplate="%{x|%Y-%m}<br>wealth=%{y:.3f}<extra></extra>",
    ))
    if "regime_label" in df.columns:
        for label in df["regime_label"].dropna().unique():
            sub = df[df["regime_label"] == label]
            fig.add_trace(go.Scatter(
                x=sub.index, y=cum.loc[sub.index], mode="markers",
                name=str(label),
                marker=dict(size=6, color=REGIME_COLORS.get(label, "#64748b")),
                hovertemplate="%{x|%Y-%m}<br>" + str(label) + "<extra></extra>",
            ))
    fig.update_layout(
        title="Cumulative Return Colored by Regime Label",
        xaxis_title="Date", yaxis_title="Wealth (1.0 = start)",
        template="plotly_white", height=440,
        legend=dict(orientation="h", y=-0.18),
        hovermode="closest",
    )
    return fig


def fig_state_probs(df: pd.DataFrame) -> go.Figure:
    prob_cols = [c for c in df.columns if c.startswith("state_") and c.endswith("_prob")]
    fig = go.Figure()
    palette = ["#0ea5e9", "#f59e0b", "#ef4444", "#10b981", "#8b5cf6"]
    for i, col in enumerate(prob_cols):
        fig.add_trace(go.Scatter(
            x=df.index, y=df[col], mode="lines", stackgroup="one",
            name=col.replace("_prob", "").replace("state_", "State "),
            line=dict(color=palette[i % len(palette)], width=0),
        ))
    fig.update_layout(
        title="HMM State Posterior Probabilities (stacked, total = 1.0)",
        xaxis_title="Date", yaxis_title="Probability",
        template="plotly_white", height=380,
        legend=dict(orientation="h", y=-0.18),
        hovermode="x unified",
    )
    fig.update_yaxes(range=[0, 1])
    return fig


def fig_return_distribution_by_regime(df: pd.DataFrame) -> go.Figure:
    if "regime_label" not in df.columns:
        return go.Figure()
    fig = px.box(
        df.reset_index(),
        x="regime_label", y="return_1m", color="regime_label",
        title="Monthly Return Distribution per Regime Label",
        template="plotly_white",
        color_discrete_map=REGIME_COLORS,
    )
    fig.update_layout(height=380, showlegend=False)
    fig.update_xaxes(title="Regime")
    fig.update_yaxes(title="1-month return")
    return fig


def fig_vol_and_drawdown(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if "vol_12m" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["vol_12m"], mode="lines",
            name="12-month volatility",
            line=dict(color="#0ea5e9", width=1.5),
        ))
    if "drawdown_12m" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["drawdown_12m"], mode="lines",
            name="12-month drawdown",
            line=dict(color="#ef4444", width=1.5), yaxis="y2",
        ))
    fig.update_layout(
        title="12-Month Realized Volatility and Drawdown",
        xaxis_title="Date",
        yaxis=dict(title="Volatility", side="left"),
        yaxis2=dict(title="Drawdown", overlaying="y", side="right",
                    showgrid=False, range=[-0.6, 0.05]),
        template="plotly_white", height=380,
        legend=dict(orientation="h", y=-0.22),
        hovermode="x unified",
    )
    return fig


HTML_HEAD = """<!doctype html>
<html lang="ja"><head><meta charset="utf-8">
<title>Market Regime — Live Report</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="description" content="HMM-based market regime analysis with state probabilities, regime-colored returns, and volatility.">
<meta property="og:title" content="Market Regime — Live Report">
<meta property="og:description" content="3-state HMM posteriors, regime-colored cumulative returns, regime return distribution, volatility/drawdown.">
<style>
*{box-sizing:border-box}
html,body{margin:0;background:#f8fafc;color:#0f172a;
  font-family:system-ui,-apple-system,"Hiragino Sans","Yu Gothic","Noto Sans CJK JP",sans-serif;
  -webkit-text-size-adjust:100%}
.wrap{max-width:960px;margin:0 auto;padding:24px 18px 64px}
header{display:flex;align-items:baseline;gap:10px;flex-wrap:wrap;margin-bottom:24px}
h1{font-size:24px;margin:0;font-weight:700;letter-spacing:-.01em}
.sub{color:#64748b;margin:0;font-size:14px}
.badges{margin:6px 0 18px;display:flex;gap:6px;flex-wrap:wrap}
.badge{display:inline-block;background:#e0f2fe;color:#0369a1;
  padding:3px 10px;border-radius:999px;font-size:12px;font-weight:600}
.badge.alt{background:#f1f5f9;color:#334155}
section{background:#fff;border:1px solid #e2e8f0;border-radius:12px;
  padding:18px 22px;margin-bottom:14px;box-shadow:0 1px 3px rgb(0 0 0 /.04)}
section h2{font-size:15px;margin:0 0 12px;font-weight:700;color:#0f172a}
.stats{font-size:12.5px;border-collapse:collapse;width:100%}
.stats th,.stats td{padding:6px 10px;text-align:right;border-bottom:1px solid #e2e8f0;
  font-variant-numeric:tabular-nums}
.stats th{background:#f1f5f9;font-weight:600;text-align:left}
footer{text-align:center;color:#94a3b8;font-size:12px;margin-top:32px;line-height:1.7}
footer a{color:#0ea5e9}
@media(prefers-color-scheme:dark){
  body{background:#0a0f1a;color:#f1f5f9}
  .sub{color:#94a3b8}
  .badge{background:#0c4a6e;color:#7dd3fc}
  .badge.alt{background:#1e293b;color:#cbd5e1}
  section{background:#0f172a;border-color:#1f2a3a;box-shadow:0 1px 3px rgb(0 0 0 /.3)}
  section h2{color:#f1f5f9}
  .stats th{background:#1e293b;color:#cbd5e1}
  .stats td{border-color:#1f2a3a}
  footer a{color:#38bdf8}
}
@media(max-width:480px){
  .wrap{padding:16px 12px 48px}
  h1{font-size:20px}
}
</style></head><body><div class="wrap">
"""

HTML_FOOTER = """
<footer>
Auto-generated from `regime_states.csv` ・ <a href="https://github.com/shihuiarenjinba-png/simulator" target="_blank" rel="noopener">Source on GitHub</a><br>
Data: HMM 3-state regime model (`research_regime.py`) over monthly market features
</footer>
</div></body></html>
"""


def render_html(figs: dict[str, go.Figure], df: pd.DataFrame) -> str:
    start = df.index.min().strftime("%Y-%m")
    end = df.index.max().strftime("%Y-%m")
    build_ts = pd.Timestamp.now("UTC").strftime("%Y-%m-%d %H:%M UTC")
    summary_cols = [c for c in ["return_1m", "vol_12m", "momentum_12m", "drawdown_12m",
                                 "trend_gap", "realized_growth_10y"] if c in df.columns]
    summary = df[summary_cols].describe().round(4).to_html(classes="stats", border=0)
    regime_counts = ""
    if "regime_label" in df.columns:
        counts = df["regime_label"].value_counts().to_frame("months")
        counts["share %"] = (counts["months"] / len(df) * 100).round(1)
        regime_counts = counts.to_html(classes="stats", border=0)

    parts = [HTML_HEAD]
    parts.append(f'''<header>
<h1>🌊 Market Regime — Live Report</h1>
</header>
<p class="sub">Bundled HMM-classified market state data — period: <b>{start}</b> 〜 <b>{end}</b> ({len(df)} months)</p>
<div class="badges">
  <span class="badge">3-state HMM</span>
  <span class="badge alt">Wavelet features</span>
  <span class="badge alt">No API key</span>
  <span class="badge alt">Build: {build_ts}</span>
</div>''')
    for title, fig in figs.items():
        snippet = fig.to_html(include_plotlyjs="cdn", full_html=False,
                              config={"displaylogo": False, "responsive": True})
        parts.append(f'<section><h2>{title}</h2>{snippet}</section>')
    if regime_counts:
        parts.append(f'<section><h2>Regime occupancy</h2>{regime_counts}</section>')
    parts.append(f'<section><h2>Descriptive Statistics</h2>{summary}</section>')
    parts.append(HTML_FOOTER)
    return "".join(parts)


def main() -> None:
    df = load_states()
    figs = {
        "Cumulative Return (regime-colored)": fig_cumulative_by_regime(df),
        "HMM State Probabilities": fig_state_probs(df),
        "Return Distribution by Regime": fig_return_distribution_by_regime(df),
        "Volatility & Drawdown (12-month)": fig_vol_and_drawdown(df),
    }
    html = render_html(figs, df)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(html, encoding="utf-8")
    (OUT.parent / ".nojekyll").write_text("", encoding="utf-8")
    print(f"WROTE {OUT} ({OUT.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
