import math
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

REQUIRED_COLUMNS = [
    "product_id",
    "product_name",
    "current_stock",
    "avg_daily_sales",
    "lead_time_days",
    "safety_days",
]

OPTIONAL_DEFAULTS: Dict[str, Any] = {
    "order_unit": 1,
    "min_order_qty": 0,
    "unit_cost": 1000,
    "priority_weight": 1.0,
    "supplier": "未設定",
}

NUMERIC_COLUMNS = [
    "current_stock",
    "avg_daily_sales",
    "lead_time_days",
    "safety_days",
    "order_unit",
    "min_order_qty",
    "unit_cost",
    "priority_weight",
]

TABLE_COLUMNS = [
    "product_name",
    "supplier",
    "current_stock",
    "avg_daily_sales",
    "lead_time_days",
    "safety_days",
    "reorder_point",
    "base_recommended_order",
    "adjusted_order",
    "estimated_order_cost",
    "days_left",
    "priority_score",
    "need_order",
]

PLAN_COLUMNS = [
    "product_name",
    "supplier",
    "adjusted_order",
    "estimated_order_cost",
    "days_left",
    "priority_score",
    "selection_reason",
]

ORDER_CANDIDATE_COLUMNS = [
    "product_name",
    "supplier",
    "adjusted_order",
    "estimated_order_cost",
    "days_left",
    "priority_score",
]

RISK_COLUMNS = [
    "product_name",
    "supplier",
    "current_stock",
    "days_left",
    "lead_time_days",
    "adjusted_order",
    "priority_score",
    "risk_level",
]

OVERSTOCK_COLUMNS = [
    "product_name",
    "supplier",
    "current_stock",
    "avg_daily_sales",
    "days_left",
    "reorder_point",
    "excess_stock",
    "overstock_note",
]

SAMPLE_CSV = """product_id,product_name,current_stock,avg_daily_sales,lead_time_days,safety_days,order_unit,min_order_qty,unit_cost,priority_weight,supplier
1,ミネラルウォーター,20,3.5,5,3,12,24,110,1.2,飲料仕入先A
2,お茶,8,2.0,7,2,24,24,95,1.1,飲料仕入先A
3,コーヒー,50,1.2,10,5,10,20,380,1.4,飲料仕入先B
4,カップ麺,5,4.0,3,2,12,24,180,1.6,食品仕入先C
5,スポーツドリンク,80,1.0,4,2,24,24,140,0.8,飲料仕入先A
"""

WELCOME_MESSAGE = """在庫アシスタントです。こんな聞き方ができます。

- 「今日のおすすめ発注を見せて」
- 「予算30000円で発注案を出して」
- 「欠品リスクが高い商品を見せて」
- 「過剰在庫を教えて」
- 「お茶の発注理由は？」
- 「安全在庫を5日にしたらどうなる？」
"""


def validate_columns(df: pd.DataFrame) -> List[str]:
    """必要な列がすべて含まれているかをチェックする。"""
    return [col for col in REQUIRED_COLUMNS if col not in df.columns]


def apply_optional_defaults(df: pd.DataFrame) -> pd.DataFrame:
    """任意列がない場合でも動くようにデフォルト値を補完する。"""
    df = df.copy()
    for column, default_value in OPTIONAL_DEFAULTS.items():
        if column not in df.columns:
            df[column] = default_value
        else:
            df[column] = df[column].fillna(default_value)
    return df


def convert_numeric_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """数値列を変換し、変換に失敗した行をエラーとして返す。"""
    df = df.copy()
    errors: List[str] = []

    for col in NUMERIC_COLUMNS:
        if col not in df.columns:
            continue
        converted = pd.to_numeric(df[col], errors="coerce")
        invalid = df[col].notna() & converted.isna()
        for index in df[invalid].index:
            value = df.at[index, col]
            errors.append(f"行 {index + 1}: 列 '{col}' に数値として扱えない値 '{value}' が含まれています。")
        df[col] = converted

    return df, errors


def sanitize_numeric_values(df: pd.DataFrame) -> pd.DataFrame:
    """ロジック上破綻しないように数値列を下限補正する。"""
    df = df.copy()
    df["current_stock"] = df["current_stock"].clip(lower=0)
    df["avg_daily_sales"] = df["avg_daily_sales"].clip(lower=0)
    df["lead_time_days"] = df["lead_time_days"].clip(lower=0)
    df["safety_days"] = df["safety_days"].clip(lower=0)
    df["order_unit"] = df["order_unit"].clip(lower=1)
    df["min_order_qty"] = df["min_order_qty"].clip(lower=0)
    df["unit_cost"] = df["unit_cost"].clip(lower=0)
    df["priority_weight"] = df["priority_weight"].clip(lower=0.1)
    return df


def round_order_quantity(base_qty: float, order_unit: float, min_order_qty: float) -> int:
    """発注単位と最小発注数を考慮して発注量を丸める。"""
    if base_qty <= 0:
        return 0

    adjusted_qty = max(base_qty, min_order_qty)
    rounded_qty = math.ceil(adjusted_qty / order_unit) * order_unit
    return int(rounded_qty)


def calculate_priority_score(df: pd.DataFrame) -> pd.Series:
    """欠品リスクと重要度を組み合わせた優先度スコアを計算する。"""
    days_divisor = np.maximum(df["days_left"].replace(np.inf, 9999), 0.5)
    urgency_component = np.maximum(df["reorder_point"] - df["current_stock"], 0) + df["avg_daily_sales"]
    lead_time_component = 1 + (df["lead_time_days"] / np.maximum(df["lead_time_days"].max(), 1))
    return (df["priority_weight"] * urgency_component * lead_time_component) / days_divisor


def calculate_inventory_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """在庫関連指標と発注条件をまとめて計算する。"""
    df = df.copy()
    df["safety_stock"] = df["avg_daily_sales"] * df["safety_days"]
    df["reorder_point"] = df["avg_daily_sales"] * df["lead_time_days"] + df["safety_stock"]
    df["base_recommended_order"] = np.ceil(
        np.maximum(0, df["reorder_point"] - df["current_stock"])
    ).astype(int)
    df["adjusted_order"] = [
        round_order_quantity(base_qty, order_unit, min_order_qty)
        for base_qty, order_unit, min_order_qty in zip(
            df["base_recommended_order"], df["order_unit"], df["min_order_qty"]
        )
    ]
    df["days_left"] = np.where(
        df["avg_daily_sales"] == 0,
        np.inf,
        df["current_stock"] / df["avg_daily_sales"],
    )
    df["need_order"] = df["adjusted_order"] > 0
    df["estimated_order_cost"] = df["adjusted_order"] * df["unit_cost"]
    df["priority_score"] = calculate_priority_score(df).round(2)
    df["risk_level"] = np.select(
        [
            df["days_left"] <= df["lead_time_days"],
            df["days_left"] <= (df["lead_time_days"] + df["safety_days"]),
        ],
        ["高", "中"],
        default="低",
    )
    df["excess_stock"] = np.maximum(0, df["current_stock"] - df["reorder_point"] * 1.5).round(1)
    df["overstock_note"] = np.where(
        (df["days_left"] > (df["lead_time_days"] + df["safety_days"] + 14))
        & (df["current_stock"] > df["reorder_point"]),
        "在庫日数が長く、過剰在庫の可能性があります",
        "",
    )
    return df


def prepare_display_df(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """表示用に数値や無限大を整える。"""
    display_df = df[columns].copy()
    if "days_left" in display_df.columns:
        display_df["days_left"] = display_df["days_left"].apply(format_days_left)
    if "estimated_order_cost" in display_df.columns:
        display_df["estimated_order_cost"] = display_df["estimated_order_cost"].map(lambda x: f"{int(x):,}円")
    return display_df


def initialize_chat_state() -> None:
    """チャット履歴を初期化する。"""
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = [
            {"role": "assistant", "content": WELCOME_MESSAGE, "dataframe": None}
        ]


def reset_chat_state() -> None:
    """CSV更新時にチャット履歴をリセットする。"""
    st.session_state.chat_messages = [
        {"role": "assistant", "content": WELCOME_MESSAGE, "dataframe": None}
    ]


def format_days_left(value: float) -> str:
    """在庫が持つ日数を見やすく整形する。"""
    if value == np.inf:
        return "∞"
    return f"{value:.1f}日"


def extract_product_name(message: str, metrics_df: pd.DataFrame) -> Optional[str]:
    """メッセージ内に含まれる商品名を推定する。"""
    lowered = message.lower()
    for product_name in metrics_df["product_name"].astype(str):
        if product_name.lower() in lowered:
            return product_name
    return None


def extract_safety_days(message: str) -> Optional[int]:
    """シミュレーション用の安全日数を抽出する。"""
    match = re.search(r"(\d+)\s*日", message)
    if match:
        return int(match.group(1))
    return None


def extract_budget(message: str) -> Optional[int]:
    """メッセージ内から予算を抽出する。"""
    normalized = message.replace(",", "")
    match = re.search(r"予算\s*(\d+)", normalized)
    if match:
        return int(match.group(1))
    match = re.search(r"(\d+)\s*円", normalized)
    if match:
        return int(match.group(1))
    return None


def build_summary_message(metrics_df: pd.DataFrame, order_needed_df: pd.DataFrame, optimized_df: pd.DataFrame) -> str:
    """サマリー用の応答文を作る。"""
    total_items = len(metrics_df)
    total_order_items = len(order_needed_df)
    total_order_cost = int(order_needed_df["estimated_order_cost"].sum())
    optimized_cost = int(optimized_df["estimated_order_cost"].sum())
    most_urgent = metrics_df.iloc[0]

    return (
        f"全 {total_items} 商品のうち、発注が必要なのは {total_order_items} 商品です。"
        f" 発注候補の総額は {total_order_cost:,}円で、現在の予算条件で採用されているのは"
        f" {len(optimized_df)} 件、合計 {optimized_cost:,}円です。"
        f" もっとも在庫切れが近いのは {most_urgent['product_name']} で、残りは"
        f" {format_days_left(float(most_urgent['days_left']))} です。"
    )


def build_product_message(row: pd.Series) -> str:
    """単一商品の状況説明を作る。"""
    need_order_text = "発注候補に入っています" if bool(row["need_order"]) else "現時点では発注不要です"
    return (
        f"{row['product_name']} は仕入先 {row['supplier']} の商品です。現在在庫は {row['current_stock']}、"
        f" 平均販売数は 1日 {row['avg_daily_sales']}、発注点は {row['reorder_point']:.1f} です。"
        f" 基本推奨発注数は {int(row['base_recommended_order'])} 個ですが、発注単位 {int(row['order_unit'])}"
        f" と最小発注数 {int(row['min_order_qty'])} を考慮すると {int(row['adjusted_order'])} 個になります。"
        f" 推定発注金額は {int(row['estimated_order_cost']):,}円、在庫が持つ見込みは"
        f" {format_days_left(float(row['days_left']))} で、{need_order_text}。"
    )


def build_reason_message(row: pd.Series) -> str:
    """発注理由の説明を作る。"""
    return (
        f"{row['product_name']} の安全在庫は {row['avg_daily_sales']} × {row['safety_days']}日 = "
        f"{row['safety_stock']:.1f} です。発注点は 平均販売数 {row['avg_daily_sales']} × "
        f"リードタイム {row['lead_time_days']}日 + 安全在庫 {row['safety_stock']:.1f} = "
        f"{row['reorder_point']:.1f} です。そこから基本推奨発注数は {int(row['base_recommended_order'])} 個になり、"
        f" 発注単位 {int(row['order_unit'])} と最小発注数 {int(row['min_order_qty'])} を反映した最終候補は"
        f" {int(row['adjusted_order'])} 個です。優先度スコアは {row['priority_score']:.2f} です。"
    )


def build_simulation_message(simulated_df: pd.DataFrame, safety_days: int) -> str:
    """安全在庫日数のシミュレーション結果を作る。"""
    order_needed_df = simulated_df[simulated_df["need_order"]].copy()
    order_needed_df = order_needed_df.sort_values(
        ["priority_score", "days_left"], ascending=[False, True]
    ).reset_index(drop=True)
    total_recommended_order = int(order_needed_df["adjusted_order"].sum())

    if order_needed_df.empty:
        return f"安全在庫日数を {safety_days}日にすると、発注が必要な商品はありません。"

    top_product = order_needed_df.iloc[0]["product_name"]
    return (
        f"安全在庫日数を {safety_days}日にすると、発注が必要な商品は {len(order_needed_df)} 件、"
        f" 推奨発注数の合計は {total_recommended_order} 個になります。"
        f" もっとも優先度が高いのは {top_product} です。"
    )


def build_budget_plan_message(optimized_df: pd.DataFrame, budget: int, skipped_df: pd.DataFrame) -> str:
    """予算付き発注案の説明を作る。"""
    used_budget = int(optimized_df["estimated_order_cost"].sum())
    skipped_count = len(skipped_df)
    if optimized_df.empty:
        return f"予算 {budget:,}円 では採用できる発注候補がありませんでした。"
    return (
        f"予算 {budget:,}円 の範囲で {len(optimized_df)} 件を採用し、使用額は {used_budget:,}円 です。"
        f" 予算や優先度の都合で見送られた候補は {skipped_count} 件あります。"
    )


def build_help_message() -> str:
    """サポートしている質問例を返す。"""
    return (
        "まだその質問には対応しきれていません。"
        " まずは「今日のおすすめ発注」「予算30000円で発注案」「欠品リスク」「過剰在庫」"
        " 「商品名の状況」「発注理由」「安全在庫を5日にしたらどうなる？」のように聞いてみてください。"
    )


def optimize_order_plan(order_needed_df: pd.DataFrame, budget: Optional[float]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """予算内で優先度の高い商品を採用する。"""
    candidates = order_needed_df.copy()
    if candidates.empty:
        candidates["selected"] = pd.Series(dtype=bool)
        candidates["selection_reason"] = pd.Series(dtype=str)
        return candidates, candidates.copy()

    candidates = candidates.sort_values(
        ["priority_score", "days_left", "estimated_order_cost"],
        ascending=[False, True, True],
    ).reset_index(drop=True)

    if budget is None or budget <= 0:
        selected = candidates.copy()
        selected["selected"] = True
        selected["selection_reason"] = "予算上限なしのため採用"
        skipped = candidates.iloc[0:0].copy()
        skipped["selected"] = pd.Series(dtype=bool)
        skipped["selection_reason"] = pd.Series(dtype=str)
        return selected, skipped

    remaining_budget = float(budget)
    selected_rows: List[pd.Series] = []
    skipped_rows: List[pd.Series] = []

    for _, row in candidates.iterrows():
        cost = float(row["estimated_order_cost"])
        row_copy = row.copy()
        if cost <= remaining_budget:
            remaining_budget -= cost
            row_copy["selected"] = True
            row_copy["selection_reason"] = "優先度が高く、予算内に収まるため採用"
            selected_rows.append(row_copy)
        else:
            row_copy["selected"] = False
            row_copy["selection_reason"] = "予算上限を超えるため今回は見送り"
            skipped_rows.append(row_copy)

    selected_df = pd.DataFrame(selected_rows) if selected_rows else candidates.iloc[0:0].copy()
    skipped_df = pd.DataFrame(skipped_rows) if skipped_rows else candidates.iloc[0:0].copy()
    return selected_df.reset_index(drop=True), skipped_df.reset_index(drop=True)


def build_download_bytes(df: pd.DataFrame, columns: List[str]) -> bytes:
    """CSVダウンロード用のバイト列を作る。"""
    return df[columns].to_csv(index=False).encode("utf-8-sig")


def answer_inventory_question(
    message: str,
    raw_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    order_needed_df: pd.DataFrame,
    optimized_df: pd.DataFrame,
    risk_df: pd.DataFrame,
    overstock_df: pd.DataFrame,
) -> Dict[str, Any]:
    """チャットメッセージに対する回答を生成する。"""
    normalized = message.strip().lower()
    product_name = extract_product_name(message, metrics_df)

    if any(keyword in normalized for keyword in ["サマリー", "summary", "全体", "概要"]):
        return {"content": build_summary_message(metrics_df, order_needed_df, optimized_df), "dataframe": None}

    if any(keyword in normalized for keyword in ["予算", "おすすめ発注案", "発注案"]):
        budget = extract_budget(message)
        optimized_budget_df, skipped_budget_df = optimize_order_plan(order_needed_df, budget)
        return {
            "content": build_budget_plan_message(
                optimized_budget_df,
                int(budget) if budget is not None else 0,
                skipped_budget_df,
            ) if budget is not None else "現在の予算設定に基づくおすすめ発注案を表示します。",
            "dataframe": prepare_display_df(optimized_budget_df, PLAN_COLUMNS) if not optimized_budget_df.empty else None,
        }

    if any(keyword in normalized for keyword in ["今日", "おすすめ発注", "推奨発注"]):
        if optimized_df.empty:
            return {"content": "現在の条件では採用された発注候補はありません。", "dataframe": None}
        return {
            "content": f"現在の条件で採用されているおすすめ発注は {len(optimized_df)} 件です。",
            "dataframe": prepare_display_df(optimized_df, PLAN_COLUMNS),
        }

    if any(keyword in normalized for keyword in ["欠品リスク", "危険", "危ない", "在庫切れ"]):
        if risk_df.empty:
            return {"content": "現在、欠品リスクが高い商品はありません。", "dataframe": None}
        return {
            "content": f"欠品リスクが高い商品を {len(risk_df)} 件表示します。",
            "dataframe": prepare_display_df(risk_df, RISK_COLUMNS),
        }

    if any(keyword in normalized for keyword in ["過剰在庫", "余って", "在庫多い"]):
        if overstock_df.empty:
            return {"content": "現在、過剰在庫の可能性が高い商品はありません。", "dataframe": None}
        return {
            "content": f"過剰在庫の可能性がある商品を {len(overstock_df)} 件表示します。",
            "dataframe": prepare_display_df(overstock_df, OVERSTOCK_COLUMNS),
        }

    if any(keyword in normalized for keyword in ["安全在庫", "シミュ", "したらどう", "変わる"]):
        safety_days = extract_safety_days(message)
        if safety_days is None:
            return {
                "content": "シミュレーションする日数が読み取れませんでした。例えば「安全在庫を5日にしたらどうなる？」と聞いてください。",
                "dataframe": None,
            }
        simulated_input = raw_df.copy()
        simulated_input["safety_days"] = safety_days
        simulated_df = calculate_inventory_metrics(simulated_input)
        simulated_df = simulated_df.sort_values(["priority_score", "days_left"], ascending=[False, True]).reset_index(drop=True)
        simulated_order_df = simulated_df[simulated_df["need_order"]].copy()
        return {
            "content": build_simulation_message(simulated_df, safety_days),
            "dataframe": prepare_display_df(simulated_order_df, PLAN_COLUMNS) if not simulated_order_df.empty else None,
        }

    if product_name is not None and any(keyword in normalized for keyword in ["理由", "なぜ", "根拠"]):
        row = metrics_df.loc[metrics_df["product_name"] == product_name].iloc[0]
        return {"content": build_reason_message(row), "dataframe": None}

    if product_name is not None:
        row = metrics_df.loc[metrics_df["product_name"] == product_name].iloc[0]
        display_df = prepare_display_df(
            metrics_df.loc[metrics_df["product_name"] == product_name],
            TABLE_COLUMNS,
        )
        return {"content": build_product_message(row), "dataframe": display_df}

    return {"content": build_help_message(), "dataframe": None}


def render_summary(
    metrics_df: pd.DataFrame,
    order_needed_df: pd.DataFrame,
    optimized_df: pd.DataFrame,
    budget_limit: Optional[float],
) -> None:
    """画面上部のKPIを表示する。"""
    total_items = len(metrics_df)
    total_order_items = len(order_needed_df)
    optimized_cost = int(optimized_df["estimated_order_cost"].sum()) if not optimized_df.empty else 0
    budget_text = "上限なし" if budget_limit is None or budget_limit <= 0 else f"{int(budget_limit):,}円"

    st.subheader("サマリー")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("商品数", total_items)
    col2.metric("発注が必要な商品数", total_order_items)
    col3.metric("採用された発注候補", len(optimized_df))
    col4.metric("採用済み発注額", f"{optimized_cost:,}円")
    st.caption(f"現在の予算設定: {budget_text}")


def render_planning_tab(
    optimized_df: pd.DataFrame,
    skipped_df: pd.DataFrame,
    risk_df: pd.DataFrame,
    overstock_df: pd.DataFrame,
) -> None:
    """最適化結果を表示する。"""
    st.subheader("今日のおすすめ発注")
    if optimized_df.empty:
        st.info("現在の条件では採用された発注候補はありません。")
    else:
        display_df = prepare_display_df(optimized_df, PLAN_COLUMNS)
        st.dataframe(display_df, use_container_width=True)
        st.download_button(
            "おすすめ発注案をCSVダウンロード",
            data=build_download_bytes(optimized_df, PLAN_COLUMNS),
            file_name="optimized_order_plan.csv",
            mime="text/csv",
        )

    st.subheader("見送り候補")
    if skipped_df.empty:
        st.info("見送り候補はありません。")
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


def render_tables(metrics_df: pd.DataFrame, order_needed_df: pd.DataFrame) -> None:
    """一覧テーブルを表示する。"""
    st.subheader("全商品一覧")
    st.dataframe(prepare_display_df(metrics_df, TABLE_COLUMNS), use_container_width=True)

    st.subheader("発注候補一覧")
    if order_needed_df.empty:
        st.info("発注が必要な商品はありません。")
    else:
        order_display = prepare_display_df(order_needed_df, ORDER_CANDIDATE_COLUMNS)
        st.dataframe(order_display, use_container_width=True)
        st.download_button(
            "発注候補をCSVダウンロード",
            data=build_download_bytes(order_needed_df, ORDER_CANDIDATE_COLUMNS),
            file_name="recommended_orders.csv",
            mime="text/csv",
        )


def render_chat_section(
    raw_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    order_needed_df: pd.DataFrame,
    optimized_df: pd.DataFrame,
    risk_df: pd.DataFrame,
    overstock_df: pd.DataFrame,
) -> None:
    """チャットUIを表示する。"""
    st.subheader("在庫アシスタント")
    st.caption(
        "会話で在庫状況やおすすめ発注を確認できます。例: 予算30000円で発注案を出して / 欠品リスクを見せて"
    )

    initialize_chat_state()

    for message in st.session_state.chat_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["dataframe"] is not None:
                st.dataframe(message["dataframe"], use_container_width=True)

    prompt = st.chat_input("在庫について質問してください")
    if prompt:
        st.session_state.chat_messages.append({"role": "user", "content": prompt, "dataframe": None})
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


def main() -> None:
    st.set_page_config(page_title="在庫発注最適化アシスタント", layout="wide")

    st.title("在庫発注最適化アシスタント")
    st.write(
        "CSVから在庫状況を読み取り、発注単位・最小発注数・原価・重要度・予算を考慮したおすすめ発注案を確認できます。"
    )

    uploaded_file = st.file_uploader("CSVファイルを選択してください", type="csv")

    if uploaded_file is None:
        reset_chat_state()
        st.info("まずはCSVをアップロードしてください。")
        st.subheader("サンプルCSV形式")
        st.code(SAMPLE_CSV, language="csv")
        st.caption("任意列として order_unit, min_order_qty, unit_cost, priority_weight, supplier も使えます。")
        return

    current_file_id = f"{uploaded_file.name}-{uploaded_file.size}"
    if st.session_state.get("uploaded_file_id") != current_file_id:
        st.session_state.uploaded_file_id = current_file_id
        reset_chat_state()

    try:
        df = pd.read_csv(uploaded_file)
    except Exception as exc:
        st.error(f"CSVの読み込み中にエラーが発生しました: {exc}")
        return

    missing_columns = validate_columns(df)
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

    df = sanitize_numeric_values(df)

    supplier_options = sorted(df["supplier"].astype(str).unique().tolist())
    selected_suppliers = st.sidebar.multiselect(
        "対象の仕入先",
        options=supplier_options,
        default=supplier_options,
    )
    budget_limit = st.sidebar.number_input(
        "予算上限（円）",
        min_value=0,
        value=50000,
        step=1000,
        help="0 を指定すると予算上限なしで全候補を採用します。",
    )
    st.sidebar.caption("CSVに任意列がない場合は、発注単位1、最小発注数0、原価1000円、重要度1.0で計算します。")

    filtered_df = df[df["supplier"].astype(str).isin(selected_suppliers)].copy()
    if filtered_df.empty:
        st.warning("選択中の仕入先に該当する商品がありません。")
        return

    try:
        metrics_df = calculate_inventory_metrics(filtered_df)
    except Exception as exc:
        st.error(f"在庫指標の計算中に予期せぬエラーが発生しました: {exc}")
        return

    metrics_df = metrics_df.sort_values(["priority_score", "days_left"], ascending=[False, True]).reset_index(drop=True)
    order_needed_df = metrics_df[metrics_df["need_order"]].copy().reset_index(drop=True)
    optimized_df, skipped_df = optimize_order_plan(order_needed_df, budget_limit)
    risk_df = metrics_df[metrics_df["risk_level"].isin(["高", "中"])].copy()
    risk_df = risk_df.sort_values(["priority_score", "days_left"], ascending=[False, True]).reset_index(drop=True)
    overstock_df = metrics_df[metrics_df["overstock_note"] != ""].copy()
    overstock_df = overstock_df.sort_values("excess_stock", ascending=False).reset_index(drop=True)

    render_summary(metrics_df, order_needed_df, optimized_df, budget_limit)

    planning_tab, chat_tab, table_tab = st.tabs(["最適化", "チャット", "テーブル"])
    with planning_tab:
        render_planning_tab(optimized_df, skipped_df, risk_df, overstock_df)
    with chat_tab:
        render_chat_section(filtered_df, metrics_df, order_needed_df, optimized_df, risk_df, overstock_df)
    with table_tab:
        render_tables(metrics_df, order_needed_df)


if __name__ == "__main__":
    main()
