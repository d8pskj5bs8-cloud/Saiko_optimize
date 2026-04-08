import json
import re
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import streamlit as st

from constants import (
    FORECAST_PRODUCT_COLUMNS,
    LLM_SYSTEM_PROMPT,
    OPENAI_MODEL,
    OVERSTOCK_COLUMNS,
    PLAN_COLUMNS,
    RISK_COLUMNS,
    TABLE_COLUMNS,
    WELCOME_MESSAGE,
)
from inventory import calculate_inventory_metrics, format_days_left, optimize_order_plan, prepare_display_df

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


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
    overstock_cost = int(metrics_df.get("excess_stock_cost", pd.Series(dtype=float)).sum()) if "excess_stock_cost" in metrics_df.columns else 0
    high_risk_count = int((metrics_df["risk_level"] == "高").sum()) if "risk_level" in metrics_df.columns else 0
    policy_label = str(metrics_df["order_policy_label"].iloc[0]) if "order_policy_label" in metrics_df.columns else "都度発注"

    return (
        f"全 {total_items} 商品のうち、発注が必要なのは {total_order_items} 商品です。"
        f" 現在の発注方式は {policy_label} です。"
        f" 発注が必要なものの総額は {total_order_cost:,}円で、現在の予算条件で優先されているのは"
        f" {len(optimized_df)} 件、合計 {optimized_cost:,}円です。"
        f" 見直し余地のある過剰在庫候補額は {overstock_cost:,}円、欠品高リスクは {high_risk_count} 商品です。"
        f" もっとも在庫切れが近いのは {most_urgent['product_name']} で、残りは"
        f" {format_days_left(float(most_urgent['days_left']))} です。"
    )


def build_product_message(row: pd.Series) -> str:
    """単一商品の状況説明を作る。"""
    need_order_text = "優先度付き発注一覧に入っています" if bool(row["need_order"]) else "現時点では発注不要です"
    demand_basis_text = f"{row['demand_basis_label']}日販 {row['demand_basis_value']:.2f}"
    forecast_text = ""
    if not pd.isna(row.get("forecast_daily_sales", np.nan)):
        forecast_text = f" 需要予測は {float(row['forecast_daily_sales']):.2f} です。"
    if row.get("order_policy_label") == "定期発注" and not pd.isna(row.get("forecast_period_demand", np.nan)):
        forecast_text += (
            f" 定期発注では {int(row.get('forecast_horizon_days', 0))} 日分の累積需要 "
            f"{float(row['forecast_period_demand']):.2f} を日換算した"
            f" {float(row.get('forecast_effective_daily_sales', row['forecast_daily_sales'])):.2f} を使っています。"
        )
    planning_text = (
        f"{row['planning_target_label']}は {row['planning_target_value']:.1f} です。"
        if row.get("order_policy_label") == "定期発注"
        else f"発注点は {row['planning_target_value']:.1f} です。"
    )
    return (
        f"{row['product_name']} は仕入先 {row['supplier']} の商品です。現在在庫は {row['current_stock']}、"
        f" 実績平均日販は {row['avg_daily_sales']}、計算に使っている需要は {demand_basis_text}、"
        f" 発注方式は {row.get('order_policy_label', '都度発注')} で、{planning_text}{forecast_text}"
        f" 基本推奨発注数は {int(row['base_recommended_order'])} 個ですが、発注単位 {int(row['order_unit'])}"
        f" と最小発注数 {int(row['min_order_qty'])} を考慮すると {int(row['adjusted_order'])} 個になります。"
        f" 推定発注金額は {int(row['estimated_order_cost']):,}円、在庫が持つ見込みは"
        f" {format_days_left(float(row['days_left']))} で、{need_order_text}。"
        f" {row.get('no_order_reason', '')}"
    )


def build_reason_message(row: pd.Series) -> str:
    """発注理由の説明を作る。"""
    if row.get("order_policy_label") == "定期発注":
        target_expression = (
            f"目標在庫量は 需要 {row['demand_basis_value']:.2f} × "
            f"(リードタイム {row['lead_time_days']}日 + 発注周期 {row['review_cycle_days']}日 + 安全在庫日数 {row['safety_days']}日)"
            f" = {row['target_stock']:.1f} です。"
        )
        if not pd.isna(row.get("forecast_period_demand", np.nan)):
            target_expression += (
                f" 需要予測ベースでは、この {int(row.get('forecast_horizon_days', 0))} 日間の累積需要を"
                f" 日換算した値を使っています。"
            )
    else:
        target_expression = (
            f"発注点は 需要 {row['demand_basis_value']:.2f} × リードタイム {row['lead_time_days']}日 + "
            f"安全在庫 {row['safety_stock']:.1f} = {row['reorder_point']:.1f} です。"
        )
    return (
        f"{row['product_name']} の安全在庫は {row['demand_basis_label']}日販 {row['demand_basis_value']:.2f} × {row['safety_days']}日 = "
        f"{row['safety_stock']:.1f} です。発注方式は {row.get('order_policy_label', '都度発注')} で、{target_expression}"
        f" そこから基本推奨発注数は {int(row['base_recommended_order'])} 個になり、"
        f" 発注単位 {int(row['order_unit'])} と最小発注数 {int(row['min_order_qty'])} を反映した最終候補は"
        f" {int(row['adjusted_order'])} 個です。優先度スコアは {row['priority_score']:.2f} です。"
        f" {row.get('forecast_reason_summary', '')}"
    )


def build_simulation_message(simulated_df: pd.DataFrame, safety_days: int) -> str:
    """安全在庫日数のシミュレーション結果を作る。"""
    order_needed_df = simulated_df[simulated_df["need_order"]].copy()
    order_needed_df = order_needed_df.sort_values(["priority_score", "days_left"], ascending=[False, True]).reset_index(drop=True)
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
        return f"予算 {budget:,}円 では優先できる発注対象がありませんでした。"
    return (
        f"予算 {budget:,}円 の範囲で {len(optimized_df)} 件を優先し、使用額は {used_budget:,}円 です。"
        f" 予算や優先度の都合で今回は優先しなかったものは {skipped_count} 件あります。"
    )


def build_help_message() -> str:
    """サポートしている質問例を返す。"""
    return (
        "まだその質問には対応しきれていません。"
        " まずは「次回のおすすめ発注」「予算30000円で発注案」「欠品リスク」「過剰在庫」"
        " 「商品名の状況」「発注理由」「この予測はなぜ？」「需要予測を使うと何が変わる？」"
        " 「安全在庫を5日にしたらどうなる？」のように聞いてみてください。"
    )


def build_llm_inventory_summary(
    metrics_df: pd.DataFrame,
    order_needed_df: pd.DataFrame,
    optimized_df: pd.DataFrame,
    risk_df: pd.DataFrame,
    overstock_df: pd.DataFrame,
) -> Dict[str, Any]:
    """LLM向けに全体サマリーをJSONで返す。"""
    most_urgent = metrics_df.iloc[0] if not metrics_df.empty else None
    return {
        "total_items": len(metrics_df),
        "order_needed_items": len(order_needed_df),
        "selected_order_items": len(optimized_df),
        "selected_order_cost": int(optimized_df["estimated_order_cost"].sum()) if not optimized_df.empty else 0,
        "risk_items": len(risk_df),
        "overstock_items": len(overstock_df),
        "most_urgent_product": None
        if most_urgent is None
        else {
            "product_name": str(most_urgent["product_name"]),
            "days_left": None if float(most_urgent["days_left"]) == np.inf else round(float(most_urgent["days_left"]), 1),
            "risk_level": str(most_urgent["risk_level"]),
            "adjusted_order": int(most_urgent["adjusted_order"]),
        },
    }


def serialize_product_row(row: pd.Series) -> Dict[str, Any]:
    """商品行をLLMが扱いやすい辞書に整形する。"""
    days_left = float(row["days_left"])
    return {
        "product_name": str(row["product_name"]),
        "supplier": str(row["supplier"]),
        "order_policy_label": str(row.get("order_policy_label", "都度発注")),
        "current_stock": float(row["current_stock"]),
        "avg_daily_sales": float(row["avg_daily_sales"]),
        "forecast_daily_sales": None if pd.isna(row.get("forecast_daily_sales", np.nan)) else round(float(row["forecast_daily_sales"]), 2),
        "forecast_effective_daily_sales": None
        if pd.isna(row.get("forecast_effective_daily_sales", np.nan))
        else round(float(row["forecast_effective_daily_sales"]), 2),
        "forecast_period_demand": None
        if pd.isna(row.get("forecast_period_demand", np.nan))
        else round(float(row["forecast_period_demand"]), 2),
        "forecast_horizon_days": None
        if pd.isna(row.get("forecast_horizon_days", np.nan))
        else int(float(row["forecast_horizon_days"])),
        "lead_time_days": float(row["lead_time_days"]),
        "safety_days": float(row["safety_days"]),
        "review_cycle_days": float(row.get("review_cycle_days", 0)),
        "safety_stock": round(float(row["safety_stock"]), 1),
        "reorder_point": round(float(row["reorder_point"]), 1),
        "target_stock": round(float(row.get("target_stock", row["reorder_point"])), 1),
        "planning_target_label": str(row.get("planning_target_label", "発注点")),
        "planning_target_value": round(float(row.get("planning_target_value", row["reorder_point"])), 1),
        "base_recommended_order": int(row["base_recommended_order"]),
        "adjusted_order": int(row["adjusted_order"]),
        "estimated_order_cost": int(row["estimated_order_cost"]),
        "days_left": None if days_left == np.inf else round(days_left, 1),
        "demand_basis_label": str(row.get("demand_basis_label", "実績平均")),
        "demand_basis_value": round(float(row.get("demand_basis_value", row["avg_daily_sales"])), 2),
        "priority_score": round(float(row["priority_score"]), 2),
        "risk_level": str(row["risk_level"]),
        "need_order": bool(row["need_order"]),
        "selection_reason": row.get("selection_reason"),
        "forecast_model_group": str(row.get("forecast_model_group", "")),
        "overstock_note": str(row["overstock_note"]),
        "forecast_reason_summary": str(row.get("forecast_reason_summary", "")),
        "no_order_reason": str(row.get("no_order_reason", "")),
    }


def find_product_rows(metrics_df: pd.DataFrame, product_query: str) -> pd.DataFrame:
    """商品名の部分一致で候補を探す。"""
    normalized = product_query.strip().lower()
    if not normalized:
        return metrics_df.iloc[0:0].copy()

    matched = metrics_df[
        metrics_df["product_name"].astype(str).str.lower().str.contains(normalized, regex=False)
    ].copy()
    return matched.reset_index(drop=True)


def execute_inventory_tool(
    name: str,
    arguments: Dict[str, Any],
    raw_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    order_needed_df: pd.DataFrame,
    optimized_df: pd.DataFrame,
    risk_df: pd.DataFrame,
    overstock_df: pd.DataFrame,
) -> str:
    """LLMからのツール呼び出しを実行し、JSON文字列で返す。"""
    if name == "get_inventory_summary":
        return json.dumps(
            build_llm_inventory_summary(metrics_df, order_needed_df, optimized_df, risk_df, overstock_df),
            ensure_ascii=False,
        )

    if name == "get_product_info":
        product_query = str(arguments.get("product_query", "")).strip()
        matched = find_product_rows(metrics_df, product_query)
        if matched.empty:
            return json.dumps(
                {"found": False, "message": f"'{product_query}' に一致する商品が見つかりませんでした。"},
                ensure_ascii=False,
            )
        return json.dumps(
            {
                "found": True,
                "match_count": len(matched),
                "products": [serialize_product_row(row) for _, row in matched.head(5).iterrows()],
            },
            ensure_ascii=False,
        )

    if name == "get_order_plan":
        budget = arguments.get("budget")
        budget_value = None if budget in (None, "") else float(budget)
        selected_df, skipped_df = optimize_order_plan(order_needed_df, budget_value)
        return json.dumps(
            {
                "budget": budget_value,
                "selected_count": len(selected_df),
                "selected_cost": int(selected_df["estimated_order_cost"].sum()) if not selected_df.empty else 0,
                "skipped_count": len(skipped_df),
                "selected_items": [serialize_product_row(row) for _, row in selected_df.head(10).iterrows()],
                "skipped_items": [serialize_product_row(row) for _, row in skipped_df.head(10).iterrows()],
            },
            ensure_ascii=False,
        )

    if name == "get_risk_items":
        risk_level = str(arguments.get("risk_level", "all"))
        if risk_level == "all":
            filtered = risk_df.copy()
        else:
            filtered = risk_df[risk_df["risk_level"] == risk_level].copy()
        return json.dumps(
            {
                "risk_level": risk_level,
                "count": len(filtered),
                "items": [serialize_product_row(row) for _, row in filtered.head(10).iterrows()],
            },
            ensure_ascii=False,
        )

    if name == "simulate_safety_days":
        safety_days = int(arguments["safety_days"])
        simulated_input = raw_df.copy()
        simulated_input["safety_days"] = safety_days
        demand_column = "selected_daily_sales" if "selected_daily_sales" in simulated_input.columns else "avg_daily_sales"
        demand_label = str(simulated_input["demand_basis_label"].iloc[0]) if "demand_basis_label" in simulated_input.columns else "実績平均"
        order_policy = str(metrics_df["order_policy_label"].iloc[0]) if "order_policy_label" in metrics_df.columns else "都度発注"
        simulated_df = calculate_inventory_metrics(simulated_input, demand_column, demand_label, order_policy)
        simulated_df = simulated_df.sort_values(["priority_score", "days_left"], ascending=[False, True]).reset_index(drop=True)
        simulated_order_df = simulated_df[simulated_df["need_order"]].copy().reset_index(drop=True)
        return json.dumps(
            {
                "safety_days": safety_days,
                "order_needed_count": len(simulated_order_df),
                "total_order_units": int(simulated_order_df["adjusted_order"].sum()) if not simulated_order_df.empty else 0,
                "items": [serialize_product_row(row) for _, row in simulated_order_df.head(10).iterrows()],
            },
            ensure_ascii=False,
        )

    return json.dumps({"error": f"unknown tool: {name}"}, ensure_ascii=False)


def get_openai_tools() -> List[Dict[str, Any]]:
    """Responses API に渡すツール定義を返す。"""
    return [
        {
            "type": "function",
            "name": "get_inventory_summary",
            "description": "在庫全体の要約、件数、重要な商品、欠品リスク件数を取得する。",
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False,
            },
        },
        {
            "type": "function",
            "name": "get_product_info",
            "description": "指定した商品名に一致する商品の在庫状況と発注情報を取得する。部分一致で検索する。",
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {
                    "product_query": {
                        "type": "string",
                        "description": "商品名またはその一部。",
                    }
                },
                "required": ["product_query"],
                "additionalProperties": False,
            },
        },
        {
            "type": "function",
            "name": "get_order_plan",
            "description": "必要に応じて予算を指定し、おすすめ発注案を取得する。budget が null なら予算上限なし。",
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {
                    "budget": {
                        "type": ["number", "null"],
                        "description": "予算上限（円）。不要なら null。",
                    }
                },
                "required": ["budget"],
                "additionalProperties": False,
            },
        },
        {
            "type": "function",
            "name": "get_risk_items",
            "description": "欠品リスクの高い商品一覧を取得する。risk_level は all, 高, 中 のいずれか。",
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {
                    "risk_level": {
                        "type": "string",
                        "enum": ["all", "高", "中"],
                        "description": "取得したいリスクレベル。",
                    }
                },
                "required": ["risk_level"],
                "additionalProperties": False,
            },
        },
        {
            "type": "function",
            "name": "simulate_safety_days",
            "description": "安全在庫日数を変更した場合の発注状況を試算する。",
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {
                    "safety_days": {
                        "type": "integer",
                        "description": "試算したい安全在庫日数。",
                    }
                },
                "required": ["safety_days"],
                "additionalProperties": False,
            },
        },
    ]


def answer_inventory_with_llm(
    message: str,
    raw_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    order_needed_df: pd.DataFrame,
    optimized_df: pd.DataFrame,
    risk_df: pd.DataFrame,
    overstock_df: pd.DataFrame,
    api_key: str,
) -> Dict[str, Any]:
    """OpenAI Responses API と function calling を使って回答する。"""
    if OpenAI is None:
        return {
            "content": "OpenAI SDK が見つからないため GPT 連携を使えません。`pip install -r requirements.txt` を実行してください。",
            "dataframe": None,
        }

    client = OpenAI(api_key=api_key)
    tools = get_openai_tools()
    conversation_input: List[Dict[str, Any]] = [
        {"role": "system", "content": [{"type": "input_text", "text": LLM_SYSTEM_PROMPT}]},
        {"role": "user", "content": [{"type": "input_text", "text": message}]},
    ]

    for _ in range(5):
        response = client.responses.create(
            model=OPENAI_MODEL,
            input=conversation_input,
            tools=tools,
        )

        function_calls = [item for item in response.output if item.type == "function_call"]
        if not function_calls:
            return {"content": response.output_text or build_help_message(), "dataframe": None}

        for tool_call in function_calls:
            arguments = json.loads(tool_call.arguments)
            result = execute_inventory_tool(
                tool_call.name,
                arguments,
                raw_df,
                metrics_df,
                order_needed_df,
                optimized_df,
                risk_df,
                overstock_df,
            )
            conversation_input.append(tool_call.model_dump())
            conversation_input.append(
                {
                    "type": "function_call_output",
                    "call_id": tool_call.call_id,
                    "output": result,
                }
            )

    return {
        "content": "ツール呼び出しが多くなりすぎたため、回答を完了できませんでした。質問を少し具体的にしてもう一度試してください。",
        "dataframe": None,
    }


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
    forecast_available = metrics_df["forecast_daily_sales"].notna().any()

    if any(keyword in normalized for keyword in ["何が変わる", "どう変わる", "予測を使うと", "予測使うと"]):
        if not forecast_available:
            return {"content": "まだ需要予測は適用されていません。販売履歴CSVと外部要因CSVの両方をアップロードすると、実績平均との違いを確認できます。", "dataframe": None}
        changed_df = metrics_df[
            metrics_df["forecast_daily_sales"].notna()
            & (metrics_df["forecast_daily_sales"].round(2) != metrics_df["avg_daily_sales"].round(2))
        ].copy()
        if changed_df.empty:
            return {"content": "今回は予測値と実績平均日販に大きな差がなく、発注判断への影響は小さめです。", "dataframe": None}
        changed_df["forecast_impact"] = (changed_df["forecast_daily_sales"] - changed_df["avg_daily_sales"]).abs()
        changed_df = changed_df.sort_values("forecast_impact", ascending=False)
        top_row = changed_df.iloc[0]
        return {
            "content": (
                f"需要予測を使うと、{len(changed_df)} 商品で実績平均と違う需要を見込めます。"
                f" 特に {top_row['product_name']} は実績平均 {top_row['avg_daily_sales']:.2f} に対して"
                f" 予測 {top_row['forecast_daily_sales']:.2f} なので、発注判断が変わりやすい商品です。"
            ),
            "dataframe": prepare_display_df(changed_df.head(10), FORECAST_PRODUCT_COLUMNS),
        }

    if product_name is not None and any(keyword in normalized for keyword in ["予測", "需要", "天気", "なぜ", "理由"]):
        row = metrics_df.loc[metrics_df["product_name"] == product_name].iloc[0]
        if not forecast_available:
            return {"content": "まだ需要予測は適用されていません。販売履歴CSVと外部要因CSVをアップロードすると確認できます。", "dataframe": None}
        if pd.isna(row.get("forecast_daily_sales", np.nan)):
            return {"content": f"{product_name} は履歴や外部要因が不足しているため、今回は予測を作れていません。実績平均日販を使って判断しています。", "dataframe": None}
        reason_text = row.get("forecast_reason_summary", "") or "主な要因はまだ整理できていません。"
        period_text = ""
        if not pd.isna(row.get("forecast_period_demand", np.nan)):
            period_text = (
                f" 定期発注向けには {int(row.get('forecast_horizon_days', 0))} 日累積の予測需要"
                f" {float(row['forecast_period_demand']):.2f} も使っています。"
            )
        return {
            "content": (
                f"{product_name} の予測日販は {float(row['forecast_daily_sales']):.2f} です。"
                f" 実績平均日販 {float(row['avg_daily_sales']):.2f} と比べると差は {float(row['forecast_diff']):+.2f} で、"
                f"{reason_text}"
                f" 発注計算では {row['demand_basis_label']} を使っています。"
                f"{period_text}"
            ),
            "dataframe": prepare_display_df(
                metrics_df.loc[metrics_df["product_name"] == product_name],
                TABLE_COLUMNS,
            ),
        }

    if any(keyword in normalized for keyword in ["予測", "需要", "天気"]):
        if not forecast_available:
            return {"content": "まだ需要予測は適用されていません。販売履歴CSVと外部要因CSVをアップロードすると確認できます。", "dataframe": None}
        forecast_df = metrics_df[metrics_df["forecast_daily_sales"].notna()].copy()
        increased_df = forecast_df[forecast_df["forecast_diff"] > 0].sort_values("forecast_diff", ascending=False)
        content = (
            f"需要予測が適用されているのは {len(forecast_df)} 商品です。"
            f" 実績平均より需要増と見込まれるのは {len(increased_df)} 商品で、"
            f" もっとも上振れが大きいのは {increased_df.iloc[0]['product_name']} です。"
            if not increased_df.empty
            else f"需要予測が適用されているのは {len(forecast_df)} 商品です。大きな需要増は見込まれていません。"
        )
        return {"content": content, "dataframe": prepare_display_df(forecast_df, FORECAST_PRODUCT_COLUMNS)}

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

    if any(keyword in normalized for keyword in ["今日", "次回", "おすすめ発注", "推奨発注"]):
        if optimized_df.empty:
            return {"content": "現在の条件では優先度の高いものはありません。", "dataframe": None}
        return {
            "content": f"現在の条件で優先度の高いものは {len(optimized_df)} 件です。",
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
        demand_column = "selected_daily_sales" if "selected_daily_sales" in simulated_input.columns else "avg_daily_sales"
        demand_label = str(metrics_df["demand_basis_label"].iloc[0]) if "demand_basis_label" in metrics_df.columns else "実績平均"
        order_policy = str(metrics_df["order_policy_label"].iloc[0]) if "order_policy_label" in metrics_df.columns else "都度発注"
        simulated_df = calculate_inventory_metrics(simulated_input, demand_column, demand_label, order_policy)
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
