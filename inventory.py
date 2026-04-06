import math
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


def adjust_order_quantity(
    base_qty: float,
    current_stock: float,
    order_unit: float,
    min_order_qty: float,
    max_stock: float,
) -> Tuple[int, str]:
    """発注単位、最小発注数、在庫上限を考慮して発注量を調整する。"""
    if base_qty <= 0:
        return 0, "target_met"

    if not pd.isna(max_stock) and current_stock >= max_stock:
        return 0, "max_stock_reached"

    rounded_qty = int(math.ceil(base_qty))

    if min_order_qty > 0 and rounded_qty < min_order_qty:
        return 0, "below_min_order_qty"

    rounded_qty = int(math.ceil(rounded_qty / order_unit) * order_unit)

    if not pd.isna(max_stock):
        max_additional = max(int(math.floor(max_stock - current_stock)), 0)
        if max_additional <= 0:
            return 0, "max_stock_reached"
        rounded_qty = min(rounded_qty, max_additional)
        rounded_qty = int(math.floor(rounded_qty / order_unit) * order_unit)
        if rounded_qty <= 0:
            return 0, "max_stock_reached"
        if min_order_qty > 0 and rounded_qty < min_order_qty:
            return 0, "max_stock_reached"

    return rounded_qty, "planned"


def calculate_priority_score(df: pd.DataFrame, demand_column: str = "demand_basis_value") -> pd.Series:
    """欠品リスクと重要度を組み合わせた優先度スコアを計算する。"""
    days_divisor = np.maximum(df["days_left"].replace(np.inf, 9999), 0.5)
    urgency_component = np.maximum(df["planning_target_value"] - df["current_stock"], 0) + df[demand_column]
    lead_time_component = 1 + (df["lead_time_days"] / np.maximum(df["lead_time_days"].max(), 1))
    return (df["priority_weight"] * urgency_component * lead_time_component) / days_divisor


def calculate_inventory_metrics(
    df: pd.DataFrame,
    demand_column: str = "avg_daily_sales",
    demand_label: str = "実績平均",
    order_policy: str = "都度発注",
) -> pd.DataFrame:
    """在庫関連指標と発注条件をまとめて計算する。"""
    df = df.copy()
    if demand_column not in df.columns:
        raise ValueError(f"需要列 '{demand_column}' が見つかりません。")

    df["forecast_daily_sales"] = df.get("forecast_daily_sales", pd.Series(np.nan, index=df.index))
    df["forecast_model_group"] = df.get("forecast_model_group", pd.Series("", index=df.index)).fillna("")
    df["forecast_reason_summary"] = df.get("forecast_reason_summary", pd.Series("", index=df.index)).fillna("")
    df["review_cycle_days"] = df.get("review_cycle_days", pd.Series(0, index=df.index)).fillna(0).clip(lower=0)
    df["max_stock"] = df.get("max_stock", pd.Series(np.inf, index=df.index)).fillna(np.inf)
    df["order_policy_label"] = order_policy
    df["demand_basis_label"] = demand_label
    df["demand_basis_value"] = df[demand_column].fillna(df["avg_daily_sales"]).clip(lower=0)
    df["safety_stock"] = df["demand_basis_value"] * df["safety_days"]
    df["reorder_point"] = df["demand_basis_value"] * df["lead_time_days"] + df["safety_stock"]
    df["target_cover_days"] = df["lead_time_days"] + df["review_cycle_days"] + df["safety_days"]
    df["target_stock"] = df["demand_basis_value"] * df["target_cover_days"]
    if order_policy == "定期発注":
        df["planning_target_label"] = "目標在庫量"
        df["planning_target_value"] = df["target_stock"]
    else:
        df["planning_target_label"] = "発注点"
        df["planning_target_value"] = df["reorder_point"]
    df["base_recommended_order"] = np.ceil(np.maximum(0, df["planning_target_value"] - df["current_stock"])).astype(int)
    adjustment_results = [
        adjust_order_quantity(base_qty, current_stock, order_unit, min_order_qty, max_stock)
        for base_qty, current_stock, order_unit, min_order_qty, max_stock in zip(
            df["base_recommended_order"],
            df["current_stock"],
            df["order_unit"],
            df["min_order_qty"],
            df["max_stock"],
        )
    ]
    df["adjusted_order"] = [qty for qty, _ in adjustment_results]
    df["order_adjustment_reason"] = [reason for _, reason in adjustment_results]
    df["days_left"] = np.where(
        df["demand_basis_value"] == 0,
        np.inf,
        df["current_stock"] / df["demand_basis_value"],
    )
    df["need_order"] = df["adjusted_order"] > 0
    df["estimated_order_cost"] = df["adjusted_order"] * df["unit_cost"]
    df["inventory_value"] = (df["current_stock"] * df["unit_cost"]).round(0)
    df["monthly_holding_cost"] = (df["inventory_value"] * df["holding_cost_rate"]).round(0)
    df["priority_score"] = calculate_priority_score(df, "demand_basis_value").round(2)
    threshold_days = np.where(order_policy == "定期発注", df["lead_time_days"] + df["review_cycle_days"], df["lead_time_days"])
    df["risk_level"] = np.select(
        [
            df["days_left"] <= threshold_days,
            df["days_left"] <= (threshold_days + df["safety_days"]),
        ],
        ["高", "中"],
        default="低",
    )
    df["excess_stock"] = np.maximum(0, df["current_stock"] - df["planning_target_value"] * 1.5).round(1)
    df["excess_stock_cost"] = (df["excess_stock"] * df["unit_cost"]).round(0)
    df["excess_holding_cost"] = (df["excess_stock_cost"] * df["holding_cost_rate"]).round(0)
    df["stockout_risk_cost"] = np.where(
        df["risk_level"] == "高",
        np.maximum(df["planning_target_value"] - df["current_stock"], 0) * df["unit_cost"] * 0.3,
        0,
    ).round(0)
    df["overstock_note"] = np.where(
        (df["days_left"] > (threshold_days + df["safety_days"] + 14))
        & (df["current_stock"] > df["planning_target_value"]),
        "在庫日数が長く、過剰在庫の可能性があります",
        "",
    )
    df["no_order_reason"] = np.select(
        [
            df["need_order"],
            df["order_adjustment_reason"] == "below_min_order_qty",
            df["order_adjustment_reason"] == "max_stock_reached",
            df["demand_basis_value"] == 0,
        ],
        [
            "",
            "最低発注数に満たないため発注見送り",
            "在庫上限を超えるため発注見送り",
            "需要が発生していないため現時点では発注不要",
        ],
        default="現在庫で次回判断タイミングまで十分に持つため発注不要",
    )
    return df


def format_days_left(value: float) -> str:
    """在庫が持つ日数を見やすく整形する。"""
    if value == np.inf:
        return "∞"
    return f"{value:.1f}日"


def prepare_display_df(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """表示用に数値や無限大を整える。"""
    display_df = df[columns].copy()
    if "days_left" in display_df.columns:
        display_df["days_left"] = display_df["days_left"].apply(format_days_left)
    if "estimated_order_cost" in display_df.columns:
        display_df["estimated_order_cost"] = display_df["estimated_order_cost"].map(lambda x: f"{int(x):,}円")
    for currency_column in [
        "inventory_value",
        "monthly_holding_cost",
        "excess_stock_cost",
        "excess_holding_cost",
        "stockout_risk_cost",
    ]:
        if currency_column in display_df.columns:
            display_df[currency_column] = display_df[currency_column].map(lambda x: f"{int(x):,}円")
    for numeric_column in [
        "avg_daily_sales",
        "forecast_daily_sales",
        "forecast_effective_daily_sales",
        "forecast_period_demand",
        "forecast_horizon_days",
        "demand_basis_value",
        "forecast_diff",
        "reorder_point",
        "target_stock",
        "planning_target_value",
        "review_cycle_days",
    ]:
        if numeric_column in display_df.columns:
            display_df[numeric_column] = display_df[numeric_column].apply(
                lambda value: "" if pd.isna(value) else round(float(value), 2)
            )
    if "forecast_change_ratio" in display_df.columns:
        display_df["forecast_change_ratio"] = display_df["forecast_change_ratio"].apply(
            lambda value: "" if pd.isna(value) else f"{float(value) * 100:.0f}%"
        )
    return display_df


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
