import os
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import streamlit as st

from chat import reset_chat_state
from constants import SAMPLE_CSV
from csv_loader import (
    apply_optional_defaults,
    convert_numeric_columns,
    load_csv_with_fallbacks,
    normalize_cell_values,
    normalize_column_names,
    prepare_forecast_external_df,
    prepare_forecast_history_df,
    sanitize_numeric_values,
    validate_columns,
    validate_required_values,
)
from forecasting import generate_demand_forecast
from inventory import calculate_inventory_metrics, optimize_order_plan
from ui import (
    inject_pop_ui_styles,
    render_chat_section,
    render_forecast_tab,
    render_planning_tab,
    render_pop_hero,
    render_summary,
    render_tables,
)


def get_openai_api_key_default() -> str:
    """secrets.toml が無い環境でも API キー初期値を安全に取得する。"""
    session_value = str(st.session_state.get("openai_api_key", "")).strip()
    if session_value:
        return session_value

    try:
        secret_value = str(st.secrets.get("OPENAI_API_KEY", "")).strip()
        if secret_value:
            return secret_value
    except Exception:
        pass

    return str(os.getenv("OPENAI_API_KEY", "")).strip()


def main() -> None:
    st.set_page_config(page_title="在庫発注最適化アシスタント", layout="wide")
    inject_pop_ui_styles()

    st.title("在庫発注計画アシスタント")
    st.write(
        "CSVを読み込むと、次回の発注判断を一覧とチャットの両方で確認できます。需要予測は必要なときだけ追加して使えます。"
    )
    render_pop_hero()

    uploaded_file = st.file_uploader("在庫CSVを選択してください", type="csv")
    st.caption("まずは在庫CSVだけで使い始められます。需要予測を使いたい場合のみ、下の追加CSVを読み込みます。")
    with st.expander("需要予測を使う場合の追加CSV", expanded=False):
        sales_history_file = st.file_uploader("販売履歴CSV", type="csv")
        external_factors_file = st.file_uploader("外部要因CSV", type="csv")

    if uploaded_file is None:
        reset_chat_state()
        st.info("まずはCSVをアップロードしてください。")
        st.subheader("サンプルCSV形式")
        st.code(SAMPLE_CSV, language="csv")
        st.caption("任意列として order_lot, min_order_qty, max_stock, unit_cost, priority_weight, supplier_id, category, location も使えます。")
        return

    current_file_id = "|".join(
        [
            f"{uploaded_file.name}-{uploaded_file.size}",
            f"{sales_history_file.name}-{sales_history_file.size}" if sales_history_file else "no-history",
            f"{external_factors_file.name}-{external_factors_file.size}" if external_factors_file else "no-external",
        ]
    )
    if st.session_state.get("uploaded_file_id") != current_file_id:
        st.session_state.uploaded_file_id = current_file_id
        reset_chat_state()

    try:
        df, load_notes = load_csv_with_fallbacks(uploaded_file)
        df, column_notes = normalize_column_names(df)
        df = normalize_cell_values(df)
    except Exception as exc:
        st.error(f"CSVの読み込み中にエラーが発生しました: {exc}")
        return

    order_policy = st.sidebar.radio(
        "発注方式",
        options=["都度発注", "定期発注"],
        horizontal=True,
        help="都度発注は発注点ベース、定期発注は次回見直しまで持たせる目標在庫量ベースで計算します。",
    )

    missing_columns = validate_columns(df, order_policy)
    if missing_columns:
        st.error("必須列が不足しています。以下の列を含めてください:")
        st.write(missing_columns)
        return

    df = apply_optional_defaults(df)
    df, conversion_errors = convert_numeric_columns(df)
    if conversion_errors:
        st.error("数値変換エラーが発生しました。CSVの該当列を確認してください。")
        for message in conversion_errors:
            st.write(f"- {message}")
        return

    required_value_errors = validate_required_values(df, order_policy)
    if required_value_errors:
        st.error("必須項目の空欄が見つかりました。CSVの該当行を確認してください。")
        for message in required_value_errors:
            st.write(f"- {message}")
        return

    df["product_id"] = df["product_id"].astype(str)
    df = sanitize_numeric_values(df)
    normalization_notes = load_notes + column_notes
    if normalization_notes:
        with st.expander("CSV読み込みメモ", expanded=False):
            for note in normalization_notes:
                st.write(f"- {note}")

    history_df: Optional[pd.DataFrame] = None
    external_df: Optional[pd.DataFrame] = None
    forecast_extra_notes: List[str] = []

    if sales_history_file is not None and external_factors_file is not None:
        history_df, history_notes, history_errors = prepare_forecast_history_df(sales_history_file)
        external_df, external_notes, external_errors = prepare_forecast_external_df(external_factors_file)
        forecast_extra_notes.extend(history_notes + external_notes)

        if history_errors:
            st.error("販売履歴CSVに問題があります。")
            for message in history_errors:
                st.write(f"- {message}")
            return
        if external_errors:
            st.error("外部要因CSVに問題があります。")
            for message in external_errors:
                st.write(f"- {message}")
            return

    default_forecast_date = pd.Timestamp.today().normalize() + pd.Timedelta(days=1)
    forecast_date_input_args: Dict[str, Any] = {"value": default_forecast_date.date()}
    if external_df is not None and not external_df.empty:
        available_dates = external_df["date"].dropna().sort_values()
        if not available_dates.empty:
            earliest_external_date = pd.Timestamp(available_dates.iloc[0]).normalize()
            latest_external_date = pd.Timestamp(available_dates.iloc[-1]).normalize()
            default_forecast_date = latest_external_date
            forecast_date_input_args = {
                "value": default_forecast_date.date(),
                "min_value": earliest_external_date.date(),
                "max_value": latest_external_date.date(),
            }

    supplier_options = sorted(df["supplier"].astype(str).unique().tolist())
    selected_suppliers = st.sidebar.multiselect(
        "対象の仕入先",
        options=supplier_options,
        default=supplier_options,
    )
    forecast_mode = st.sidebar.radio(
        "発注計算に使う需要",
        options=["実績平均", "需要予測", "大きい方"],
        horizontal=True,
        help="需要予測が使えない商品は自動で実績平均にフォールバックします。",
    )
    demand_mode_notes = {
        "実績平均": "通常の平均日販をそのまま使います。",
        "需要予測": "予測値を優先し、使えない商品は実績平均に戻します。",
        "大きい方": "実績平均と予測値を比べて、大きい方を採用します。",
    }
    st.sidebar.markdown(
        f"""
        <div class="demand-mode-indicator">
            <div class="demand-mode-indicator-label">現在の選択</div>
            <div class="demand-mode-indicator-value">{forecast_mode}</div>
            <div class="demand-mode-indicator-note">{demand_mode_notes[forecast_mode]}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    forecast_date = pd.Timestamp(
        st.sidebar.date_input(
            "予測対象日",
            **forecast_date_input_args,
        )
    )
    if external_df is not None and not external_df.empty:
        st.sidebar.caption(
            f"外部要因CSVで選べる予測日は {earliest_external_date.date()} 〜 {latest_external_date.date()} です。"
        )
    budget_limit = st.sidebar.number_input(
        "予算上限（円）",
        min_value=0,
        value=50000,
        step=1000,
        help="0 を指定すると予算上限なしで全候補を採用します。",
    )
    api_key_default = get_openai_api_key_default()
    with st.sidebar.expander("チャットの詳細設定", expanded=False):
        openai_api_key = st.text_input(
            "OpenAI APIキー",
            value=api_key_default,
            type="password",
            help="設定するとチャットで GPT の関数呼び出しを使えます。未設定時は従来のルールベース回答です。",
        ).strip()
        st.session_state.openai_api_key = openai_api_key
        use_llm_chat = st.toggle(
            "GPTチャットを使う",
            value=bool(openai_api_key),
            disabled=not bool(openai_api_key),
            help="APIキー設定後に有効化できます。",
        )
    st.sidebar.caption("CSVに任意列がない場合は、発注単位1、最小発注数0、原価1000円、月次保管コスト率2%、重要度1.0で計算します。")
    if order_policy == "定期発注":
        st.sidebar.caption("定期発注では review_cycle_days を使って、次回見直しまで持つ目標在庫量を計算します。")

    filtered_df = df[df["supplier"].astype(str).isin(selected_suppliers)].copy()
    if filtered_df.empty:
        st.warning("選択中の仕入先に該当する商品がありません。")
        return

    forecast_result: Dict[str, Any] = {
        "enabled": False,
        "message": "販売履歴CSVと外部要因CSVをアップロードすると需要予測を利用できます。",
        "forecast_df": pd.DataFrame(),
        "coefficients_df": pd.DataFrame(),
        "notes": [],
    }

    if sales_history_file is not None and external_factors_file is not None:
        normalization_notes.extend(forecast_extra_notes)
        if history_df is not None and external_df is not None:
            forecast_result = generate_demand_forecast(filtered_df, history_df, external_df, forecast_date)
            forecast_df = forecast_result.get("forecast_df")
            if isinstance(forecast_df, pd.DataFrame) and not forecast_df.empty:
                filtered_df = filtered_df.merge(
                    forecast_df[
                        [
                            "product_id",
                            "forecast_daily_sales",
                            "forecast_period_demand",
                            "forecast_horizon_days",
                            "forecast_effective_daily_sales",
                            "forecast_model_group",
                            "forecast_reason_summary",
                        ]
                    ],
                    on="product_id",
                    how="left",
                )
    elif sales_history_file is not None or external_factors_file is not None:
        forecast_result["message"] = "需要予測を使うには、販売履歴CSVと外部要因CSVの両方をアップロードしてください。"

    if len(normalization_notes) > len(load_notes + column_notes):
        with st.expander("追加CSV読み込みメモ", expanded=False):
            for note in normalization_notes[len(load_notes + column_notes):]:
                st.write(f"- {note}")

    if "forecast_daily_sales" not in filtered_df.columns:
        filtered_df["forecast_daily_sales"] = np.nan
    if "forecast_model_group" not in filtered_df.columns:
        filtered_df["forecast_model_group"] = ""
    if "forecast_reason_summary" not in filtered_df.columns:
        filtered_df["forecast_reason_summary"] = ""
    if "forecast_period_demand" not in filtered_df.columns:
        filtered_df["forecast_period_demand"] = np.nan
    if "forecast_horizon_days" not in filtered_df.columns:
        filtered_df["forecast_horizon_days"] = (
            filtered_df["lead_time_days"] + filtered_df["review_cycle_days"] + filtered_df["safety_days"]
        ).clip(lower=1)
    if "forecast_effective_daily_sales" not in filtered_df.columns:
        filtered_df["forecast_effective_daily_sales"] = np.nan

    forecast_basis_column = "forecast_effective_daily_sales" if order_policy == "定期発注" else "forecast_daily_sales"
    if forecast_mode == "需要予測":
        filtered_df["selected_daily_sales"] = filtered_df[forecast_basis_column].fillna(filtered_df["avg_daily_sales"])
    elif forecast_mode == "大きい方":
        filtered_df["selected_daily_sales"] = np.maximum(
            filtered_df["avg_daily_sales"].astype(float),
            filtered_df[forecast_basis_column].fillna(0).astype(float),
        )
    else:
        filtered_df["selected_daily_sales"] = filtered_df["avg_daily_sales"].astype(float)

    try:
        metrics_df = calculate_inventory_metrics(filtered_df, "selected_daily_sales", forecast_mode, order_policy)
    except Exception as exc:
        st.error(f"在庫指標の計算中に予期せぬエラーが発生しました: {exc}")
        return

    metrics_df = metrics_df.sort_values(["priority_score", "days_left"], ascending=[False, True]).reset_index(drop=True)
    order_needed_df = metrics_df[metrics_df["need_order"]].copy().reset_index(drop=True)
    no_order_df = metrics_df[~metrics_df["need_order"]].copy().reset_index(drop=True)
    optimized_df, skipped_df = optimize_order_plan(order_needed_df, budget_limit)
    risk_df = metrics_df[metrics_df["risk_level"].isin(["高", "中"])].copy()
    risk_df = risk_df.sort_values(["priority_score", "days_left"], ascending=[False, True]).reset_index(drop=True)
    overstock_df = metrics_df[metrics_df["overstock_note"] != ""].copy()
    overstock_df = overstock_df.sort_values("excess_stock", ascending=False).reset_index(drop=True)

    render_summary(metrics_df, order_needed_df, optimized_df, risk_df, overstock_df, budget_limit)

    render_chat_section(
        filtered_df,
        metrics_df,
        order_needed_df,
        optimized_df,
        risk_df,
        overstock_df,
        use_llm_chat,
        openai_api_key,
    )

    planning_tab, table_tab, forecast_tab = st.tabs(["おすすめ発注", "一覧", "需要予測"])

    with planning_tab:
        render_planning_tab(optimized_df, skipped_df, risk_df, overstock_df, order_policy)

    with table_tab:
        render_tables(metrics_df, order_needed_df, no_order_df)

    with forecast_tab:
        render_forecast_tab(forecast_result, forecast_mode, forecast_date)


if __name__ == "__main__":
    main()
