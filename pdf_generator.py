import io
import os
import streamlit as st
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib import colors

def create_pdf_report(payload, figs_dict):
    buffer = io.BytesIO()
    
    # 1. ドキュメント設定
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=40, leftMargin=40,
        topMargin=40, bottomMargin=40,
        title="Portfolio Report"
    )
    
    # 2. 日本語フォント登録 (超堅牢版)
    font_filename = "ipaexg.ttf"
    font_name = 'IPAexGothic'
    
    # 3つの場所を順番に探す
    possible_paths = [
        os.path.join(os.getcwd(), font_filename),  # 1. カレントディレクトリ (最優先)
        os.path.join(os.path.dirname(__file__), font_filename), # 2. このファイルの隣
        font_filename # 3. ファイル名のみ (運任せ)
    ]
    
    font_registered = False
    for path in possible_paths:
        if os.path.exists(path):
            try:
                pdfmetrics.registerFont(TTFont(font_name, path))
                font_registered = True
                # デバッグ用: どこで見つけたかログに残す（本番では消してもOK）
                print(f"Font loaded from: {path}") 
                break
            except Exception as e:
                print(f"Font load failed at {path}: {e}")
                continue
    
    if not font_registered:
        # 万が一見つからない場合はエラーではなく、英語フォント(Helvetica)に切り替えてPDF生成を続行させる
        # これにより「データが空」エラーは回避できる
        st.warning(f"⚠️ 日本語フォント ({font_filename}) が読み込めませんでした。文字化けする可能性がありますが、標準フォントで生成します。")
        font_name = 'Helvetica' # 英語フォントへフォールバック

    # 3. スタイル定義
    styles = getSampleStyleSheet()
    
    # フォールバックしたフォント名を使う
    title_style = ParagraphStyle('JpTitle', parent=styles['Title'], fontName=font_name, fontSize=24, leading=30, spaceAfter=20)
    heading_style = ParagraphStyle('JpHeading', parent=styles['Heading2'], fontName=font_name, fontSize=14, leading=18, spaceBefore=15, spaceAfter=10, textColor=colors.darkblue)
    normal_style = ParagraphStyle('JpNormal', parent=styles['Normal'], fontName=font_name, fontSize=10.5, leading=16, spaceAfter=10)
    alert_style = ParagraphStyle('JpAlert', parent=styles['Normal'], fontName=font_name, fontSize=10, leading=14, textColor=colors.firebrick, spaceAfter=10)
    small_style = ParagraphStyle('JpSmall', parent=styles['Normal'], fontName=font_name, fontSize=9, leading=12, textColor=colors.gray, spaceAfter=5)

    # 4. コンテンツ構築
    story = []

    # --- ヘッダー ---
    story.append(Paragraph("Portfolio Analysis Report", title_style)) # 英語タイトルにしておく（文字化け回避）
    story.append(Paragraph(f"Date: {payload.get('date', '-')}", normal_style))
    story.append(Spacer(1, 20))

    # --- 第1章: サマリー ---
    story.append(Paragraph("1. Analysis Summary", heading_style))
    
    summary_text = f"""
    CAGR: <b>{payload['metrics']['CAGR']}</b><br/>
    Risk (Vol): <b>{payload['metrics']['Vol']}</b><br/>
    Sharpe Ratio: <b>{payload['metrics']['Sharpe']}</b><br/>
    Max Drawdown: <b>{payload['metrics']['MaxDD']}</b>
    """
    story.append(Paragraph(summary_text, normal_style))
    
    if 'mc_stats' in payload:
        story.append(Paragraph(f"<b>Future Simulation (20Y):</b> {payload['mc_stats']}", small_style))

    # AI詳細レビュー (日本語が含まれる場合はフォントがないと文字化けします)
    if 'detailed_review' in payload:
        story.append(Spacer(1, 5))
        for line in payload['detailed_review'].split('\n'):
            story.append(Paragraph(line, normal_style))

    story.append(Spacer(1, 10))

    # --- 第2章: AI診断 ---
    story.append(Paragraph("2. AI Diagnosis", heading_style))
    diag = payload.get('diagnosis', {})
    if diag:
        story.append(Paragraph(f"<b>Type: {diag.get('type', '-')}</b>", normal_style))
        story.append(Paragraph(f"Diversification: {diag.get('diversification_comment', '-')}", normal_style))
        story.append(Paragraph(f"Risk: {diag.get('risk_comment', '-')}", alert_style))
        story.append(Paragraph(f"Action: {diag.get('action_plan', '-')}", normal_style))

    if 'factor_comment' in payload:
        story.append(Spacer(1, 10))
        story.append(Paragraph("<b>Factor Analysis</b>", normal_style))
        story.append(Paragraph(payload['factor_comment'], normal_style))

    story.append(PageBreak())

    # --- 第3章: チャート ---
    story.append(Paragraph("3. Chart Analysis", heading_style))
    story.append(Spacer(1, 10))

    plot_order = ['allocation', 'correlation', 'monte_carlo', 'cumulative', 'drawdown', 'factors', 'attribution']
    
    title_map = {
        'allocation': 'Asset Allocation',
        'correlation': 'Correlation Matrix',
        'monte_carlo': 'Monte Carlo Simulation',
        'cumulative': 'Cumulative Return',
        'drawdown': 'Drawdown',
        'factors': 'Factor Exposure',
        'attribution': 'Risk/Return Attribution'
    }

    for key in plot_order:
        if key in figs_dict:
            story.append(Paragraph(title_map.get(key, key), heading_style))
            
            try:
                fig = figs_dict[key]
                img_bytes = fig.to_image(format="png", width=900, height=500, scale=2)
                img_io = io.BytesIO(img_bytes)
                
                im = RLImage(img_io, width=460, height=255) 
                story.append(im)
                story.append(Spacer(1, 15))
                
                if key in ['monte_carlo', 'drawdown', 'correlation']: 
                    story.append(PageBreak())
                    
            except Exception as e:
                story.append(Paragraph(f"Graph Error: {e}", alert_style))

    try:
        doc.build(story)
        buffer.seek(0)
        return buffer
    except Exception as e:
        st.error(f"PDF Build Error: {e}")
        return None
