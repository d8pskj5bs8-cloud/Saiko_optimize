from typing import Any, Dict, List

import numpy as np

REQUIRED_COLUMNS = [
    "product_id",
    "product_name",
    "current_stock",
    "avg_daily_sales",
    "lead_time_days",
    "safety_days",
]

REQUIRED_TEXT_COLUMNS = [
    "product_id",
    "product_name",
]

OPTIONAL_DEFAULTS: Dict[str, Any] = {
    "order_unit": 1,
    "min_order_qty": 0,
    "unit_cost": 1000,
    "holding_cost_rate": 0.02,
    "priority_weight": 1.0,
    "supplier": "未設定",
    "category": "未分類",
    "location": "default",
    "review_cycle_days": 0,
    "max_stock": np.inf,
}

NUMERIC_COLUMNS = [
    "current_stock",
    "avg_daily_sales",
    "lead_time_days",
    "safety_days",
    "review_cycle_days",
    "order_unit",
    "min_order_qty",
    "max_stock",
    "unit_cost",
    "holding_cost_rate",
    "priority_weight",
]

FORECAST_HISTORY_REQUIRED_COLUMNS = [
    "date",
    "product_id",
    "sales_qty",
]

FORECAST_EXTERNAL_REQUIRED_COLUMNS = [
    "date",
    "temp_avg",
    "rain_mm",
]

FORECAST_HISTORY_NUMERIC_COLUMNS = [
    "sales_qty",
    "promotion_flag",
]

FORECAST_EXTERNAL_NUMERIC_COLUMNS = [
    "temp_avg",
    "rain_mm",
    "is_holiday",
]

FORECAST_FEATURE_COLUMNS = [
    "temp_avg",
    "temp_avg_squared",
    "rain_mm",
    "is_rainy",
    "day_of_week",
    "is_weekend",
    "is_holiday",
    "promotion_flag",
    "lag_1_sales",
    "rolling_mean_7",
]

CSV_ENCODING_CANDIDATES = ["utf-8-sig", "utf-8", "cp932"]
CSV_SEPARATOR_CANDIDATES = [None, ",", ";", "\t"]
NULL_LIKE_VALUES = {
    "",
    "-",
    "--",
    "n/a",
    "na",
    "nan",
    "none",
    "null",
    "nil",
    "なし",
    "未設定",
}

COLUMN_ALIASES: Dict[str, str] = {
    "productid": "product_id",
    "product_id": "product_id",
    "商品id": "product_id",
    "商品コード": "product_id",
    "商品ｺｰﾄﾞ": "product_id",
    "商品番号": "product_id",
    "品番": "product_id",
    "品目コード": "product_id",
    "品目id": "product_id",
    "itemcode": "product_id",
    "item_id": "product_id",
    "sku": "product_id",
    "productname": "product_name",
    "product_name": "product_name",
    "商品名": "product_name",
    "品名": "product_name",
    "currentstock": "current_stock",
    "current_stock": "current_stock",
    "stock": "current_stock",
    "在庫": "current_stock",
    "現在庫": "current_stock",
    "在庫数": "current_stock",
    "available_stock": "current_stock",
    "avgdailysales": "avg_daily_sales",
    "avg_daily_sales": "avg_daily_sales",
    "daily_sales": "avg_daily_sales",
    "daily_avg_sales": "avg_daily_sales",
    "平均販売数": "avg_daily_sales",
    "平均日販": "avg_daily_sales",
    "日販": "avg_daily_sales",
    "leadtimedays": "lead_time_days",
    "lead_time_days": "lead_time_days",
    "leadtime": "lead_time_days",
    "lead_time": "lead_time_days",
    "納期": "lead_time_days",
    "リードタイム": "lead_time_days",
    "発注リードタイム": "lead_time_days",
    "safetydays": "safety_days",
    "safety_days": "safety_days",
    "safetyday": "safety_days",
    "safety_stock_days": "safety_days",
    "安全在庫日数": "safety_days",
    "安全日数": "safety_days",
    "orderunit": "order_unit",
    "order_unit": "order_unit",
    "発注単位": "order_unit",
    "入数": "order_unit",
    "minorderqty": "min_order_qty",
    "min_order_qty": "min_order_qty",
    "最小発注数": "min_order_qty",
    "最低発注数": "min_order_qty",
    "reviewcycledays": "review_cycle_days",
    "review_cycle_days": "review_cycle_days",
    "reviewcycle": "review_cycle_days",
    "発注周期": "review_cycle_days",
    "見直し周期": "review_cycle_days",
    "定期発注周期": "review_cycle_days",
    "orderlot": "order_unit",
    "order_lot": "order_unit",
    "発注ロット": "order_unit",
    "ロット": "order_unit",
    "unitcost": "unit_cost",
    "unit_cost": "unit_cost",
    "cost": "unit_cost",
    "原価": "unit_cost",
    "単価": "unit_cost",
    "holdingcostrate": "holding_cost_rate",
    "holding_cost_rate": "holding_cost_rate",
    "保管コスト率": "holding_cost_rate",
    "在庫保管コスト率": "holding_cost_rate",
    "月次保管コスト率": "holding_cost_rate",
    "priorityweight": "priority_weight",
    "priority_weight": "priority_weight",
    "priority": "priority_weight",
    "importance": "priority_weight",
    "重要度": "priority_weight",
    "supplier": "supplier",
    "supplier_id": "supplier",
    "仕入先": "supplier",
    "仕入れ先": "supplier",
    "サプライヤー": "supplier",
    "仕入先id": "supplier",
    "maxstock": "max_stock",
    "max_stock": "max_stock",
    "在庫上限": "max_stock",
    "最大在庫": "max_stock",
    "category": "category",
    "カテゴリ": "category",
    "category_name": "category",
    "分類": "category",
    "location": "location",
    "拠点": "location",
    "店舗": "location",
    "地域": "location",
    "date": "date",
    "日付": "date",
    "売上日": "date",
    "販売日": "date",
    "取引日": "date",
    "salesqty": "sales_qty",
    "sales_qty": "sales_qty",
    "sales": "sales_qty",
    "qty": "sales_qty",
    "販売数": "sales_qty",
    "売上数": "sales_qty",
    "販売数量": "sales_qty",
    "売上数量": "sales_qty",
    "数量": "sales_qty",
    "出庫数": "sales_qty",
    "出荷数": "sales_qty",
    "実績数": "sales_qty",
    "promotionflag": "promotion_flag",
    "promotion_flag": "promotion_flag",
    "promo": "promotion_flag",
    "販促": "promotion_flag",
    "tempavg": "temp_avg",
    "temp_avg": "temp_avg",
    "temperature": "temp_avg",
    "平均気温": "temp_avg",
    "rainmm": "rain_mm",
    "rain_mm": "rain_mm",
    "rainfall": "rain_mm",
    "降水量": "rain_mm",
    "weathercode": "weather_code",
    "weather_code": "weather_code",
    "天気": "weather_code",
    "isholiday": "is_holiday",
    "is_holiday": "is_holiday",
    "祝日": "is_holiday",
}

TABLE_COLUMNS = [
    "product_name",
    "category",
    "supplier",
    "order_policy_label",
    "current_stock",
    "avg_daily_sales",
    "forecast_daily_sales",
    "demand_basis_label",
    "demand_basis_value",
    "lead_time_days",
    "safety_days",
    "review_cycle_days",
    "planning_target_label",
    "planning_target_value",
    "reorder_point",
    "target_stock",
    "base_recommended_order",
    "adjusted_order",
    "estimated_order_cost",
    "inventory_value",
    "monthly_holding_cost",
    "days_left",
    "priority_score",
    "need_order",
    "no_order_reason",
]

PLAN_COLUMNS = [
    "product_name",
    "supplier",
    "order_policy_label",
    "adjusted_order",
    "estimated_order_cost",
    "days_left",
    "priority_score",
    "selection_reason",
]

ORDER_CANDIDATE_COLUMNS = [
    "product_name",
    "category",
    "supplier",
    "order_policy_label",
    "forecast_daily_sales",
    "demand_basis_label",
    "planning_target_label",
    "planning_target_value",
    "adjusted_order",
    "estimated_order_cost",
    "stockout_risk_cost",
    "days_left",
    "priority_score",
]

NO_ORDER_COLUMNS = [
    "product_name",
    "category",
    "supplier",
    "order_policy_label",
    "current_stock",
    "avg_daily_sales",
    "lead_time_days",
    "review_cycle_days",
    "days_left",
    "no_order_reason",
]

RISK_COLUMNS = [
    "product_name",
    "category",
    "supplier",
    "current_stock",
    "days_left",
    "lead_time_days",
    "review_cycle_days",
    "adjusted_order",
    "priority_score",
    "risk_level",
]

OVERSTOCK_COLUMNS = [
    "product_name",
    "category",
    "supplier",
    "current_stock",
    "avg_daily_sales",
    "forecast_daily_sales",
    "inventory_value",
    "monthly_holding_cost",
    "days_left",
    "planning_target_value",
    "reorder_point",
    "excess_stock",
    "excess_stock_cost",
    "excess_holding_cost",
    "overstock_note",
]

FORECAST_PRODUCT_COLUMNS = [
    "product_name",
    "category",
    "location",
    "avg_daily_sales",
    "forecast_daily_sales",
    "forecast_diff",
    "forecast_change_ratio",
    "forecast_model_group",
    "forecast_reason_summary",
]

FORECAST_COEFFICIENT_COLUMNS = [
    "forecast_model_group",
    "sample_count",
    "feature",
    "coefficient",
]

SAMPLE_CSV = """product_id,product_name,current_stock,avg_daily_sales,lead_time_days,safety_days,review_cycle_days,order_lot,min_order_qty,max_stock,unit_cost,holding_cost_rate,priority_weight,supplier_id,category,location
1,ミネラルウォーター,20,3.5,5,3,7,12,24,60,110,0.02,1.2,飲料仕入先A,飲料,tokyo
2,お茶,8,2.0,7,2,7,24,24,50,95,0.02,1.1,飲料仕入先A,飲料,tokyo
3,コーヒー,50,1.2,10,5,14,10,20,80,380,0.018,1.4,飲料仕入先B,飲料,tokyo
4,カップ麺,5,4.0,3,2,7,12,24,70,180,0.025,1.6,食品仕入先C,食品,tokyo
5,スポーツドリンク,80,1.0,4,2,7,24,24,90,140,0.02,0.8,飲料仕入先A,飲料,tokyo
"""

WELCOME_MESSAGE = """在庫アシスタントです。こんな聞き方ができます。

- 「次回のおすすめ発注を見せて」
- 「予算30000円で発注案を出して」
- 「欠品リスクが高い商品を見せて」
- 「過剰在庫を教えて」
- 「お茶の発注理由は？」
- 「お茶の予測はなぜ上がっているの？」
- 「需要予測を使うと何が変わるの？」
- 「安全在庫を5日にしたらどうなる？」
- 「明日の予測需要が高い商品を見せて」
"""

LLM_SYSTEM_PROMPT = """あなたは在庫分析アシスタントです。
ユーザーへの返答は日本語で簡潔かつ実務的に行ってください。
在庫や発注に関する事実は、必ず提供されたツールの結果に基づいて説明してください。
推測で数値を作らないでください。
必要な情報があれば適切なツールを呼び、足りない場合だけ短く確認してください。
"""

OPENAI_MODEL = "gpt-5"
