from io import BytesIO
from typing import Any, Dict, Optional

import pandas as pd
import streamlit as st

from chat import answer_inventory_question, answer_inventory_with_llm, initialize_chat_state
from constants import (
    FORECAST_COEFFICIENT_COLUMNS,
    FORECAST_PRODUCT_COLUMNS,
    NO_ORDER_COLUMNS,
    ORDER_CANDIDATE_COLUMNS,
    OVERSTOCK_COLUMNS,
    PLAN_COLUMNS,
    RISK_COLUMNS,
    TABLE_COLUMNS,
)
from inventory import adjust_order_quantity, build_download_bytes, prepare_display_df


def render_summary(
    metrics_df: pd.DataFrame,
    order_needed_df: pd.DataFrame,
    optimized_df: pd.DataFrame,
    risk_df: pd.DataFrame,
    overstock_df: pd.DataFrame,
    budget_limit: Optional[float],
) -> None:
    """画面上部のKPIを表示する。"""
    review_candidate_count = len(
        metrics_df[
            metrics_df["need_order"] | (metrics_df["risk_level"].isin(["高", "中"])) | (metrics_df["overstock_note"] != "")
        ]
    )
    total_order_items = len(order_needed_df)
    optimized_cost = int(optimized_df["estimated_order_cost"].sum()) if not optimized_df.empty else 0
    total_inventory_value = int(metrics_df["inventory_value"].sum()) if "inventory_value" in metrics_df.columns else 0
    total_monthly_holding_cost = int(metrics_df["monthly_holding_cost"].sum()) if "monthly_holding_cost" in metrics_df.columns else 0
    overstock_cost = int(overstock_df["excess_stock_cost"].sum()) if not overstock_df.empty else 0
    stockout_risk_cost = int(risk_df["stockout_risk_cost"].sum()) if "stockout_risk_cost" in risk_df.columns else 0
    high_risk_count = int((risk_df["risk_level"] == "高").sum()) if not risk_df.empty else 0
    budget_text = "上限なし" if budget_limit is None or budget_limit <= 0 else f"{int(budget_limit):,}円"
    forecast_items = int(metrics_df["forecast_daily_sales"].notna().sum()) if "forecast_daily_sales" in metrics_df.columns else 0
    policy_label = str(metrics_df["order_policy_label"].iloc[0]) if "order_policy_label" in metrics_df.columns else "都度発注"

    st.subheader("サマリー")
    col1, col2, col3 = st.columns(3)
    col1.metric("見直し候補", f"{review_candidate_count}商品")
    col2.metric("欠品高リスク", f"{high_risk_count}商品")
    col3.metric("今回の発注予定額", f"{optimized_cost:,}円")
    col4, col5, col6 = st.columns(3)
    col4.metric("総在庫金額", f"{total_inventory_value:,}円")
    col5.metric("過剰在庫候補額", f"{overstock_cost:,}円")
    col6.metric("毎月の保管コスト目安", f"{total_monthly_holding_cost:,}円")
    st.caption(
        f"現在の発注方式: {policy_label} / 発注対象: {total_order_items}商品 / 予測適用: {forecast_items}商品 / "
        f"予算設定: {budget_text} / 欠品高リスク額の目安: {stockout_risk_cost:,}円"
    )


def build_forecast_plain_summary(forecast_df: pd.DataFrame) -> str:
    """予測結果を現場向けの短い文章にまとめる。"""
    available_df = forecast_df[forecast_df["forecast_daily_sales"].notna()].copy()
    if available_df.empty:
        return "今回は需要予測を使える商品がまだありません。"

    increased_df = available_df[available_df["forecast_diff"] > 0].sort_values("forecast_diff", ascending=False)
    decreased_df = available_df[available_df["forecast_diff"] < 0].sort_values("forecast_diff")

    period_ready_count = int(available_df["forecast_period_demand"].notna().sum()) if "forecast_period_demand" in available_df.columns else 0
    summary = [f"需要予測を使えるのは {len(available_df)} 商品です。"]
    if period_ready_count > 0:
        summary.append(f"このうち {period_ready_count} 商品は、保護期間の累積需要も計算できています。")
    if not increased_df.empty:
        top_up = increased_df.iloc[0]
        summary.append(
            f"実績平均より需要が増えそうなのは {len(increased_df)} 商品で、特に {top_up['product_name']} が上振れしそうです。"
        )
    if not decreased_df.empty:
        top_down = decreased_df.iloc[0]
        summary.append(
            f"一方で、{top_down['product_name']} は実績平均より落ち着く見込みです。"
        )
    return " ".join(summary)


def inject_pop_ui_styles() -> None:
    """親しみやすく、チャット導線が見えやすい配色と装飾を適用する。"""
    st.markdown(
        """
        <style>
        :root {
            --bg-main: linear-gradient(180deg, #f7fafc 0%, #eef4f8 52%, #e8f0f6 100%);
            --panel: rgba(255, 255, 255, 0.96);
            --panel-strong: #f8fbfd;
            --line: rgba(86, 112, 134, 0.18);
            --text-main: #20313f;
            --text-soft: #506474;
            --accent: #2f80c4;
            --accent-strong: #24679d;
            --accent-pale: #dbeefe;
            --mint: #dff5ef;
            --yellow: #fff2bf;
            --shadow-soft: 0 10px 28px rgba(58, 89, 112, 0.08);
            --shadow-strong: 0 18px 40px rgba(58, 89, 112, 0.10);
        }

        .stApp {
            background: var(--bg-main);
        }

        [data-testid="stAppViewContainer"] {
            background:
                radial-gradient(circle at top left, rgba(255, 255, 255, 0.92) 0%, rgba(255, 255, 255, 0) 34%),
                linear-gradient(180deg, #f7fafc 0%, #eef4f8 52%, #e8f0f6 100%);
        }

        [data-testid="stHeader"] {
            background: rgba(247, 250, 252, 0.86);
            border-bottom: 1px solid rgba(86, 112, 134, 0.10);
            backdrop-filter: blur(8px);
        }

        .block-container {
            padding-top: 2rem;
            padding-bottom: 3rem;
        }

        .stApp,
        .stApp div,
        .stApp label,
        .stApp p,
        .stApp li,
        .stApp span,
        .stApp small {
            color: var(--text-main);
        }

        .stApp h1,
        .stApp h2,
        .stApp h3,
        .stApp h4,
        .stApp h5,
        .stApp h6 {
            color: var(--text-main) !important;
        }

        .stCaption,
        .stMarkdown,
        .stMetricLabel,
        .stMetricValue,
        [data-testid="stMarkdownContainer"] p,
        [data-testid="stMarkdownContainer"] li {
            color: var(--text-main) !important;
        }

        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #f9fcfe 0%, #eef5fa 100%);
            border-left: 1px solid var(--line);
        }

        [data-testid="stSidebar"] *,
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] span {
            color: var(--text-main) !important;
        }

        [data-testid="stFileUploader"],
        [data-testid="stMetric"],
        [data-testid="stDataFrame"],
        [data-testid="stExpander"],
        div[data-testid="stChatMessage"] {
            background: var(--panel);
            border: 1px solid var(--line);
            border-radius: 20px;
            box-shadow: var(--shadow-soft);
        }

        div[data-testid="stChatMessage"] {
            padding: 0.4rem 0.8rem;
        }

        [data-testid="stFileUploader"] section,
        [data-testid="stFileUploaderDropzone"] {
            background: linear-gradient(180deg, #fdfefe 0%, #f2f7fb 100%) !important;
            border: 1px dashed rgba(86, 112, 134, 0.28) !important;
            color: var(--text-main) !important;
        }

        [data-testid="stFileUploader"] small,
        [data-testid="stFileUploader"] span,
        [data-testid="stFileUploader"] p,
        [data-testid="stFileUploader"] label {
            color: var(--text-main) !important;
        }

        .stCodeBlock,
        .stCode,
        pre,
        code {
            background: #f7fafc !important;
            color: #244158 !important;
        }

        pre {
            border: 1px solid rgba(86, 112, 134, 0.16) !important;
            border-radius: 18px !important;
        }

        .stCodeBlock code,
        .stCode code,
        pre code {
            color: #244158 !important;
        }

        div[role="radiogroup"] {
            gap: 0.7rem;
            padding: 0.35rem;
            background: rgba(255, 255, 255, 0.86);
            border: 1px solid var(--line);
            border-radius: 999px;
        }

        div[role="radiogroup"] label {
            border-radius: 999px !important;
            padding: 0.25rem 0.35rem !important;
        }

        div[role="radiogroup"] label > div {
            border-radius: 999px;
            padding: 0.55rem 1rem;
            background: rgba(255, 255, 255, 0.88);
            border: 1px solid rgba(86, 112, 134, 0.10);
            color: var(--text-main) !important;
            font-weight: 700;
        }

        div[role="radiogroup"] label[data-checked="true"] > div {
            background: linear-gradient(135deg, #dcefff 0%, #bddfff 100%);
            color: #1f4461 !important;
            border-color: rgba(47, 128, 196, 0.26);
            box-shadow: 0 8px 18px rgba(47, 128, 196, 0.14);
        }

        .hero-card {
            background:
                radial-gradient(circle at top left, rgba(255,255,255,0.98) 0%, rgba(255,255,255,0.88) 34%, rgba(228,241,251,0.94) 100%);
            border: 1px solid var(--line);
            border-radius: 28px;
            padding: 1.4rem 1.5rem 1.2rem;
            box-shadow: var(--shadow-strong);
            margin-bottom: 1rem;
        }

        .hero-badge {
            display: inline-block;
            background: #e8f4ff;
            color: #2a628f !important;
            border-radius: 999px;
            padding: 0.35rem 0.8rem;
            font-size: 0.84rem;
            font-weight: 700;
            margin-bottom: 0.65rem;
        }

        .hero-title {
            font-size: 2rem;
            font-weight: 800;
            line-height: 1.25;
            margin-bottom: 0.45rem;
            color: var(--text-main) !important;
        }

        .hero-text {
            color: var(--text-soft) !important;
            font-size: 1rem;
            line-height: 1.7;
            margin-bottom: 0.9rem;
        }

        .chat-shell {
            background: linear-gradient(180deg, rgba(248, 251, 253, 0.96) 0%, rgba(255, 255, 255, 0.98) 100%);
            border: 1px solid rgba(86, 112, 134, 0.14);
            border-radius: 24px;
            padding: 1rem 1rem 0.35rem;
            margin-bottom: 0.85rem;
            box-shadow: var(--shadow-soft);
        }

        .stButton > button, .stDownloadButton > button {
            border-radius: 14px;
            border: 1px solid rgba(82, 137, 123, 0.36);
            background:
                linear-gradient(180deg, #eef9f4 0%, #d9f2e7 52%, #c3e8d7 100%);
            color: #1f4d42;
            font-weight: 800;
            letter-spacing: 0.01em;
            padding: 0.72rem 1.15rem;
            min-height: 2.9rem;
            box-shadow:
                0 4px 0 #8abda9,
                0 10px 22px rgba(122, 177, 156, 0.28);
            transition:
                transform 120ms ease,
                box-shadow 120ms ease,
                background 120ms ease,
                border-color 120ms ease;
            cursor: pointer;
        }

        .stButton > button:hover, .stDownloadButton > button:hover {
            background:
                linear-gradient(180deg, #f6fcf9 0%, #e3f7ee 48%, #cfefdf 100%);
            border-color: rgba(82, 137, 123, 0.46);
            color: #143d34;
            transform: translateY(-1px);
            box-shadow:
                0 5px 0 #8abda9,
                0 14px 26px rgba(122, 177, 156, 0.30);
        }

        .stButton > button:active, .stDownloadButton > button:active {
            transform: translateY(3px);
            box-shadow:
                0 1px 0 #8abda9,
                0 4px 10px rgba(122, 177, 156, 0.22);
            background:
                linear-gradient(180deg, #cbe8da 0%, #b8dccb 100%);
        }

        .stButton > button:focus-visible, .stDownloadButton > button:focus-visible {
            outline: 3px solid rgba(138, 202, 180, 0.44);
            outline-offset: 2px;
        }

        [data-testid="stChatInput"] textarea,
        [data-testid="stTextInput"] input,
        [data-testid="stNumberInput"] input {
            color: var(--text-main) !important;
            background: rgba(255, 255, 255, 0.98) !important;
            border: 1px solid rgba(86, 112, 134, 0.18) !important;
            caret-color: var(--text-main) !important;
            pointer-events: auto !important;
        }

        [data-testid="stChatInput"] textarea::placeholder,
        [data-testid="stTextInput"] input::placeholder,
        [data-testid="stNumberInput"] input::placeholder {
            color: #8091a0 !important;
        }

        [data-testid="stSelectbox"] label,
        [data-testid="stSelectbox"] div,
        [data-testid="stSelectbox"] span {
            color: #20313f !important;
        }

        [data-testid="stSelectbox"] [data-baseweb="select"] > div,
        [data-testid="stSelectbox"] [data-baseweb="select"] input,
        [data-testid="stSelectbox"] [data-baseweb="select"] span {
            background: #fdfefe !important;
            color: #20313f !important;
        }

        [data-testid="stSelectbox"] [data-baseweb="select"] > div {
            border: 1px solid rgba(86, 112, 134, 0.24) !important;
            border-radius: 14px !important;
            box-shadow: 0 6px 16px rgba(58, 89, 112, 0.08);
        }

        div[role="listbox"] {
            background: #ffffff !important;
            border: 1px solid rgba(86, 112, 134, 0.18) !important;
        }

        div[role="option"] {
            background: #ffffff !important;
            color: #20313f !important;
        }

        div[role="option"]:hover {
            background: #eef5fa !important;
            color: #18354b !important;
        }

        [data-testid="stFileUploader"] div,
        [data-testid="stFileUploaderDropzone"] div,
        [data-testid="stFileUploaderDropzoneInstructions"],
        [data-testid="stFileUploaderFileName"] {
            background: transparent !important;
            color: var(--text-main) !important;
        }

        [data-testid="stExpander"] details,
        [data-testid="stExpander"] summary {
            background: #f8fbfd !important;
            color: var(--text-main) !important;
        }

        [data-testid="stCodeBlock"] {
            background: #f7fafc !important;
            border: 1px solid rgba(86, 112, 134, 0.16) !important;
            border-radius: 18px !important;
        }

        [data-testid="stCodeBlock"] * {
            background: transparent !important;
            color: #244158 !important;
        }

        [data-baseweb="tab-list"] {
            gap: 0.4rem;
            background: linear-gradient(180deg, #dfeaf3 0%, #d4e3ef 100%);
            border: 1px solid rgba(86, 112, 134, 0.20);
            border-radius: 999px;
            padding: 0.4rem;
            box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.65);
        }

        [data-baseweb="tab"] {
            background: rgba(255, 255, 255, 0.58) !important;
            color: #3f586d !important;
            border-radius: 999px !important;
            border: 1px solid rgba(86, 112, 134, 0.12) !important;
            font-weight: 700 !important;
            padding: 0.45rem 1rem !important;
        }

        [aria-selected="true"][data-baseweb="tab"] {
            background: linear-gradient(180deg, #ffffff 0%, #f4f9fd 100%) !important;
            color: #18354b !important;
            border: 1px solid rgba(47, 128, 196, 0.28) !important;
            box-shadow: 0 6px 16px rgba(58, 89, 112, 0.12);
        }

        [data-baseweb="tab"]:hover {
            background: rgba(255, 255, 255, 0.82) !important;
            color: var(--text-main) !important;
        }

        [data-testid="stSidebar"] [data-testid="stRadio"] > div {
            gap: 0.55rem;
        }

        [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.55rem;
        }

        [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] label {
            margin: 0;
            min-height: 72px;
            padding: 0.8rem 0.75rem;
            border: 1px solid rgba(114, 171, 152, 0.34);
            border-radius: 16px;
            background: linear-gradient(180deg, rgba(245, 252, 248, 0.96) 0%, rgba(231, 246, 238, 0.98) 100%);
            box-shadow:
                0 4px 0 rgba(170, 210, 195, 0.92),
                0 10px 20px rgba(122, 177, 156, 0.12);
            transition: transform 0.12s ease, box-shadow 0.12s ease, border-color 0.12s ease, background 0.12s ease;
        }

        [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] label:hover {
            transform: translateY(-1px);
            border-color: rgba(94, 160, 138, 0.48);
            background: linear-gradient(180deg, rgba(250, 254, 252, 0.98) 0%, rgba(236, 249, 242, 1) 100%);
            box-shadow:
                0 5px 0 rgba(170, 210, 195, 0.92),
                0 14px 24px rgba(122, 177, 156, 0.16);
        }

        [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] label[data-checked="true"] {
            transform: translateY(3px);
            border-color: rgba(72, 140, 118, 0.90);
            background: linear-gradient(180deg, rgba(208, 238, 224, 0.98) 0%, rgba(191, 229, 212, 1) 100%);
            box-shadow:
                0 1px 0 rgba(126, 183, 163, 0.96),
                0 6px 12px rgba(122, 177, 156, 0.16);
        }

        [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] label:active {
            transform: translateY(3px);
            box-shadow:
                0 1px 0 rgba(126, 183, 163, 0.96),
                0 5px 10px rgba(122, 177, 156, 0.16);
        }

        [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] label p {
            margin: 0;
            font-size: 0.96rem;
            font-weight: 700;
            line-height: 1.35;
            text-align: center;
            color: var(--text-main);
        }

        [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] label[data-checked="true"] p {
            color: #1f5a49;
        }

        .demand-mode-indicator {
            margin-top: 0.45rem;
            padding: 0.8rem 0.9rem;
            border-radius: 14px;
            border: 1px solid rgba(47, 128, 196, 0.22);
            background: linear-gradient(180deg, rgba(255, 255, 255, 0.95) 0%, rgba(244, 249, 253, 0.98) 100%);
            box-shadow: 0 10px 24px rgba(58, 89, 112, 0.07);
        }

        .demand-mode-indicator-label {
            font-size: 0.74rem;
            font-weight: 700;
            letter-spacing: 0.04em;
            color: var(--text-soft);
        }

        .demand-mode-indicator-value {
            margin-top: 0.18rem;
            font-size: 1rem;
            font-weight: 800;
            color: var(--accent-strong);
        }

        .demand-mode-indicator-note {
            margin-top: 0.24rem;
            font-size: 0.82rem;
            line-height: 1.45;
            color: var(--text-soft);
        }

        @media (max-width: 1100px) {
            [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] {
                grid-template-columns: 1fr;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_pop_hero() -> None:
    """難しい印象をやわらげる導入エリアを表示する。"""
    st.markdown(
        """
        <div class="hero-card">
            <div class="hero-badge">在庫発注アシスタント</div>
            <div class="hero-title">在庫の確認と発注判断を、<br>ひとつの画面でシンプルに。</div>
            <div class="hero-text">
                会話、最適化結果、一覧テーブルを切り替えながら、
                今の在庫状況と次回発注計画を落ち着いて確認できます。
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _sanitize_sheet_name(value: str, used_names: set[str]) -> str:
    """Excel シート名の制約に合わせて重複しない名前を返す。"""
    invalid_chars = ['\\', '/', '*', '?', ':', '[', ']']
    sanitized = str(value).strip()
    for char in invalid_chars:
        sanitized = sanitized.replace(char, "_")
    sanitized = sanitized[:31] or "sheet"

    candidate = sanitized
    suffix = 1
    while candidate in used_names:
        suffix_text = f"_{suffix}"
        candidate = f"{sanitized[: 31 - len(suffix_text)]}{suffix_text}"
        suffix += 1
    used_names.add(candidate)
    return candidate


def _format_schedule_date(value: pd.Timestamp) -> str:
    """日付を画像イメージに近い形式で整形する。"""
    weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    ts = pd.Timestamp(value)
    return f"{ts.strftime('%Y-%m-%d')} ({weekdays[ts.weekday()]})"


def build_product_order_sheet(product_row: pd.Series, forecast_date: pd.Timestamp, periods: int = 4) -> pd.DataFrame:
    """商品ごとの簡易発注シミュレーション表を作る。"""
    review_cycle_days = int(max(float(product_row.get("review_cycle_days", 0) or 0), 1))
    lead_time_days = int(max(float(product_row.get("lead_time_days", 0) or 0), 0))
    safety_days = float(max(product_row.get("safety_days", 0) or 0, 0))
    demand_per_day = float(max(product_row.get("demand_basis_value", 0) or 0, 0))
    order_unit = float(max(product_row.get("order_unit", 1) or 1, 1))
    min_order_qty = float(max(product_row.get("min_order_qty", 0) or 0, 0))
    raw_max_stock = product_row.get("max_stock", float("inf"))
    max_stock = float("nan") if pd.isna(raw_max_stock) or raw_max_stock == float("inf") else float(raw_max_stock)
    current_stock = float(max(product_row.get("current_stock", 0) or 0, 0))

    cycle_stock = demand_per_day * (lead_time_days + review_cycle_days)
    safety_stock = demand_per_day * safety_days
    standard_stock = cycle_stock + safety_stock

    review_dates = [pd.Timestamp(forecast_date).normalize() + pd.Timedelta(days=review_cycle_days * idx) for idx in range(periods)]
    scheduled_arrivals: Dict[pd.Timestamp, float] = {}
    ending_stock = current_stock
    rows = []

    for review_date in review_dates:
        arrival_qty = float(scheduled_arrivals.pop(review_date, 0))
        available_stock = ending_stock + arrival_qty
        cycle_demand = demand_per_day * review_cycle_days
        base_order_qty = max(0.0, standard_stock - available_stock)
        order_qty, _ = adjust_order_quantity(
            base_order_qty,
            available_stock,
            order_unit,
            min_order_qty,
            max_stock,
        )

        arrival_date = review_date + pd.Timedelta(days=lead_time_days)
        if order_qty > 0:
            scheduled_arrivals[arrival_date] = scheduled_arrivals.get(arrival_date, 0.0) + float(order_qty)

        ending_stock = max(available_stock - cycle_demand, 0.0)
        rows.append(
            {
                "日付": review_date.date().isoformat(),
                "在庫数": round(available_stock, 1),
                "需要予測": round(demand_per_day, 1),
                "サイクル在庫": round(cycle_stock, 1),
                "安全在庫": round(safety_stock, 1),
                "標準在庫": round(standard_stock, 1),
                "発注日": "☑" if order_qty > 0 else "",
                "推奨発注量": int(order_qty),
                "入荷数": int(arrival_qty),
                "欠品": "☑" if available_stock < cycle_demand and cycle_demand > 0 else "",
            }
        )

    return pd.DataFrame(rows)


def build_order_sheet_workbook_bytes(metrics_df: pd.DataFrame, forecast_date: pd.Timestamp) -> bytes:
    """商品ごとの発注計画を Excel の複数シートで出力する。"""
    output = BytesIO()
    used_sheet_names: set[str] = set()

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        summary_rows = []
        sorted_metrics_df = metrics_df.sort_values(["need_order", "priority_score", "product_name"], ascending=[False, False, True]).reset_index(drop=True)

        for _, row in sorted_metrics_df.iterrows():
            schedule_df = build_product_order_sheet(row, forecast_date)
            next_order_row = schedule_df[schedule_df["推奨発注量"] > 0].head(1)
            next_order_date = next_order_row.iloc[0]["日付"] if not next_order_row.empty else ""
            next_order_qty = int(next_order_row.iloc[0]["推奨発注量"]) if not next_order_row.empty else 0

            summary_rows.append(
                {
                    "商品ID": row["product_id"],
                    "商品名": row["product_name"],
                    "仕入先": row["supplier"],
                    "カテゴリ": row["category"],
                    "現在庫": round(float(row["current_stock"]), 1),
                    "需要基準": row["demand_basis_label"],
                    "需要基準値": round(float(row["demand_basis_value"]), 2),
                    "次回発注日": next_order_date,
                    "次回発注量": next_order_qty,
                    "発注点": round(float(row["reorder_point"]), 1),
                    "目標在庫量": round(float(row["target_stock"]), 1),
                    "発注要否": "要" if bool(row["need_order"]) else "不要",
                }
            )

            sheet_name = _sanitize_sheet_name(str(row["product_name"]), used_sheet_names)
            detail_start_row = 0
            detail_df = pd.DataFrame(
                [
                    ["商品ID", row["product_id"]],
                    ["商品名", row["product_name"]],
                    ["仕入先", row["supplier"]],
                    ["カテゴリ", row["category"]],
                    ["現在庫", round(float(row["current_stock"]), 1)],
                    ["需要基準", row["demand_basis_label"]],
                    ["需要基準値", round(float(row["demand_basis_value"]), 2)],
                    ["発注点", round(float(row["reorder_point"]), 1)],
                    ["目標在庫量", round(float(row["target_stock"]), 1)],
                    ["次回発注日", next_order_date],
                    ["次回発注量", next_order_qty],
                ],
                columns=["項目", "値"],
            )
            detail_df.to_excel(writer, sheet_name=sheet_name, index=False, startrow=detail_start_row)
            schedule_df.to_excel(writer, sheet_name=sheet_name, index=False, startrow=len(detail_df) + 3)

            worksheet = writer.sheets[sheet_name]
            worksheet.column_dimensions["A"].width = 18
            worksheet.column_dimensions["B"].width = 20
            for column_letter in ["C", "D", "E", "F", "G", "H", "I", "J"]:
                worksheet.column_dimensions[column_letter].width = 14

        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_excel(writer, sheet_name="一覧", index=False)
        summary_sheet = writer.sheets["一覧"]
        for column_letter in ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L"]:
            summary_sheet.column_dimensions[column_letter].width = 18

    output.seek(0)
    return output.getvalue()


def render_order_sheet_tab(metrics_df: pd.DataFrame, forecast_date: pd.Timestamp, order_policy: str) -> None:
    """商品ごとの発注計画シートを表示する。"""
    st.subheader("発注計画シート")
    st.caption(f"現在の発注方式: {order_policy} / 1商品ずつシート形式で確認できます。")

    if metrics_df.empty:
        st.info("表示できる商品がありません。")
        return

    try:
        workbook_bytes = build_order_sheet_workbook_bytes(metrics_df, forecast_date)
    except ImportError:
        workbook_bytes = b""
        st.warning("Excel出力には `openpyxl` のインストールが必要です。`pip install -r requirements.txt` 後に使えます。")
    else:
        st.download_button(
            "全商品をExcelでダウンロード",
            data=workbook_bytes,
            file_name="order_sheets.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

    sorted_metrics_df = metrics_df.sort_values(["need_order", "priority_score", "product_name"], ascending=[False, False, True]).reset_index(drop=True)
    product_options = sorted_metrics_df.apply(
        lambda row: f"{row['product_name']} | {row['supplier']} | 在庫 {round(float(row['current_stock']), 1)}",
        axis=1,
    ).tolist()
    selector_key = "order_sheet_product_selectbox"
    current_selection = st.session_state.get(selector_key, product_options[0])
    if current_selection not in product_options:
        current_selection = product_options[0]

    current_index = product_options.index(current_selection)

    selector_col1, selector_col2, selector_col3 = st.columns([1, 4, 1])
    with selector_col1:
        if st.button("前の商品", use_container_width=True, disabled=current_index <= 0):
            st.session_state[selector_key] = product_options[current_index - 1]
            st.rerun()
    with selector_col2:
        selected_label = st.selectbox(
            "表示する商品",
            product_options,
            index=current_index,
            key=selector_key,
        )
    with selector_col3:
        if st.button("次の商品", use_container_width=True, disabled=current_index >= len(product_options) - 1):
            st.session_state[selector_key] = product_options[current_index + 1]
            st.rerun()

    selected_index = product_options.index(selected_label)
    st.caption(f"{selected_index + 1} / {len(product_options)} 商品を表示中")
    selected_row = sorted_metrics_df.iloc[selected_index]

    schedule_df = build_product_order_sheet(selected_row, forecast_date)
    next_order_row = schedule_df[schedule_df["推奨発注量"] > 0].head(1)
    if next_order_row.empty:
        next_order_date_text = "発注不要"
        next_order_qty_text = "0"
    else:
        next_order_date_text = _format_schedule_date(pd.Timestamp(next_order_row.iloc[0]["日付"]))
        next_order_qty_text = f"{float(next_order_row.iloc[0]['推奨発注量']):,.1f}"

    header_left, header_right = st.columns([3, 1])
    with header_left:
        st.markdown(f"### {selected_row['product_name']}")
        st.caption(
            f"カテゴリ: {selected_row['category']} / 仕入先: {selected_row['supplier']} / 需要基準: {selected_row['demand_basis_label']}"
        )
        st.markdown("**次回発注日**")
        st.markdown(f"<div style='font-size:2.2rem;font-weight:800;margin:0.2rem 0 1.4rem;'>{next_order_date_text}</div>", unsafe_allow_html=True)
    with header_right:
        st.markdown("**次回発注量**")
        st.markdown(f"<div style='font-size:2.2rem;font-weight:800;margin-top:0.2rem;'>{next_order_qty_text}</div>", unsafe_allow_html=True)

    st.dataframe(schedule_df, use_container_width=True, hide_index=True)

    detail_col1, detail_col2, detail_col3, detail_col4 = st.columns(4)
    detail_col1.metric("現在庫", f"{float(selected_row['current_stock']):,.1f}")
    detail_col2.metric("発注点", f"{float(selected_row['reorder_point']):,.1f}")
    detail_col3.metric("目標在庫量", f"{float(selected_row['target_stock']):,.1f}")
    if pd.isna(selected_row["days_left"]) or selected_row["days_left"] == float("inf"):
        days_left_text = "∞"
    else:
        days_left_text = f"{float(selected_row['days_left']):,.1f}"
    detail_col4.metric("在庫保有日数", days_left_text)

    st.download_button(
        "この商品の発注計画CSVをダウンロード",
        data=schedule_df.to_csv(index=False).encode("utf-8-sig"),
        file_name=f"order_sheet_{selected_row['product_id']}.csv",
        mime="text/csv",
    )

    with st.expander("この商品の計算根拠を見る", expanded=False):
        reason_lines = [
            f"需要予測の基準値: {float(selected_row['demand_basis_value']):,.2f} / 日",
            f"リードタイム: {int(selected_row['lead_time_days'])}日",
            f"発注見直し周期: {int(max(float(selected_row.get('review_cycle_days', 0) or 0), 1))}日",
            f"安全在庫日数: {float(selected_row['safety_days']):,.1f}日",
            f"推奨発注量の調整: 発注単位 {int(selected_row['order_unit'])}, 最低発注数 {int(selected_row['min_order_qty'])}",
        ]
        for line in reason_lines:
            st.write(f"- {line}")


def render_planning_tab(
    metrics_df: pd.DataFrame,
    optimized_df: pd.DataFrame,
    skipped_df: pd.DataFrame,
    risk_df: pd.DataFrame,
    overstock_df: pd.DataFrame,
    forecast_date: pd.Timestamp,
    order_policy: str,
) -> None:
    """最適化結果を表示する。"""
    render_order_sheet_tab(metrics_df, forecast_date, order_policy)

    st.subheader("優先度の高いもの")
    if optimized_df.empty:
        st.info("現在の条件では優先度の高いものはありません。")
    else:
        display_df = prepare_display_df(optimized_df, PLAN_COLUMNS)
        st.dataframe(display_df, use_container_width=True)
        st.download_button(
            "優先度の高いもの一覧をCSVダウンロード",
            data=build_download_bytes(optimized_df, PLAN_COLUMNS),
            file_name="next_order_plan.csv",
            mime="text/csv",
        )

    st.subheader("今回は優先しなかったもの")
    if skipped_df.empty:
        st.info("今回は優先しなかったものはありません。")
    else:
        st.dataframe(prepare_display_df(skipped_df, PLAN_COLUMNS), use_container_width=True)

    st.subheader("欠品リスク一覧")
    if risk_df.empty:
        st.info("欠品リスクが高い商品はありません。")
    else:
        st.dataframe(prepare_display_df(risk_df, RISK_COLUMNS), use_container_width=True)

    st.subheader("過剰在庫一覧")
    if overstock_df.empty:
        st.info("過剰在庫の可能性が高い商品はありません。")
    else:
        st.dataframe(prepare_display_df(overstock_df, OVERSTOCK_COLUMNS), use_container_width=True)


def render_tables(metrics_df: pd.DataFrame, order_needed_df: pd.DataFrame, no_order_df: pd.DataFrame) -> None:
    """一覧テーブルを表示する。"""
    st.subheader("全商品一覧")
    st.dataframe(prepare_display_df(metrics_df, TABLE_COLUMNS), use_container_width=True)

    st.subheader("優先度付き発注一覧")
    if order_needed_df.empty:
        st.info("発注が必要な商品はありません。")
    else:
        order_display = prepare_display_df(order_needed_df, ORDER_CANDIDATE_COLUMNS)
        st.dataframe(order_display, use_container_width=True)
        st.download_button(
            "優先度付き発注一覧をCSVダウンロード",
            data=build_download_bytes(order_needed_df, ORDER_CANDIDATE_COLUMNS),
            file_name="next_order_plan.csv",
            mime="text/csv",
        )

    st.subheader("発注不要商品一覧")
    if no_order_df.empty:
        st.info("発注不要の商品はありません。")
    else:
        st.dataframe(prepare_display_df(no_order_df, NO_ORDER_COLUMNS), use_container_width=True)


def render_forecast_tab(
    forecast_result: Dict[str, Any],
    forecast_date: pd.Timestamp,
) -> None:
    """需要予測の概要を表示する。"""
    st.subheader("需要予測の見方")
    st.caption(f"予測対象日: {forecast_date.date()} / 発注計算は需要予測優先です")
    st.caption("定期発注では、翌日予測だけでなくリードタイム + 発注周期 + 安全日数の累積需要を日換算して発注計算に使います。")
    st.caption("需要予測を作れない商品は、自動で実績平均日販にフォールバックします。")
    st.write(forecast_result["message"])

    for note in forecast_result.get("notes", [])[:5]:
        st.caption(note)

    forecast_df = forecast_result.get("forecast_df")
    if isinstance(forecast_df, pd.DataFrame) and not forecast_df.empty:
        st.info(build_forecast_plain_summary(forecast_df))
        st.caption("詳しい理由は、チャットで「お茶の予測はなぜ？」「需要予測を使うと何が変わる？」のように聞けます。")
        st.dataframe(prepare_display_df(forecast_df, FORECAST_PRODUCT_COLUMNS), use_container_width=True)
    else:
        st.info("予測一覧はまだありません。")

    coefficients_df = forecast_result.get("coefficients_df")
    with st.expander("係数の詳細を見る", expanded=False):
        st.caption("担当者向けの通常利用では見なくても大丈夫です。モデルの確認や説明資料づくりに使えます。")
        if isinstance(coefficients_df, pd.DataFrame) and not coefficients_df.empty:
            st.dataframe(coefficients_df[FORECAST_COEFFICIENT_COLUMNS], use_container_width=True)
        else:
            st.info("係数を表示できるモデルはまだ作成されていません。")


def render_chat_section(
    raw_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    order_needed_df: pd.DataFrame,
    optimized_df: pd.DataFrame,
    risk_df: pd.DataFrame,
    overstock_df: pd.DataFrame,
    use_llm_chat: bool,
    openai_api_key: str,
) -> None:
    """チャットUIを表示する。"""
    initialize_chat_state()

    st.subheader("チャットで相談")
    st.caption("気になる商品名や予算、予測の意味をそのまま入力すると、次に見るべき内容を返します。")

    for message in st.session_state.chat_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["dataframe"] is not None:
                st.dataframe(message["dataframe"], use_container_width=True)

    prompt = st.chat_input("チャットボットに質問")
    if prompt:
        st.session_state.chat_messages.append({"role": "user", "content": prompt, "dataframe": None})
        if use_llm_chat:
            try:
                answer = answer_inventory_with_llm(
                    prompt,
                    raw_df,
                    metrics_df,
                    order_needed_df,
                    optimized_df,
                    risk_df,
                    overstock_df,
                    openai_api_key,
                )
            except Exception as exc:
                answer = {
                    "content": f"GPT連携でエラーが発生したため、ルールベース回答に切り替えます。詳細: {exc}",
                    "dataframe": None,
                }
        else:
            answer = answer_inventory_question(
                prompt,
                raw_df,
                metrics_df,
                order_needed_df,
                optimized_df,
                risk_df,
                overstock_df,
            )
        st.session_state.chat_messages.append(
            {"role": "assistant", "content": answer["content"], "dataframe": answer["dataframe"]}
        )
        st.rerun()
