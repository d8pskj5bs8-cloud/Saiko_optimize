import io
import math
import json
import os
import re
import unicodedata
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

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
    "商品番号": "product_id",
    "品番": "product_id",
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
    "unitcost": "unit_cost",
    "unit_cost": "unit_cost",
    "cost": "unit_cost",
    "原価": "unit_cost",
    "単価": "unit_cost",
    "priorityweight": "priority_weight",
    "priority_weight": "priority_weight",
    "priority": "priority_weight",
    "importance": "priority_weight",
    "重要度": "priority_weight",
    "supplier": "supplier",
    "仕入先": "supplier",
    "仕入れ先": "supplier",
    "サプライヤー": "supplier",
}

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

LLM_SYSTEM_PROMPT = """あなたは在庫分析アシスタントです。
ユーザーへの返答は日本語で簡潔かつ実務的に行ってください。
在庫や発注に関する事実は、必ず提供されたツールの結果に基づいて説明してください。
推測で数値を作らないでください。
必要な情報があれば適切なツールを呼び、足りない場合だけ短く確認してください。
"""

OPENAI_MODEL = "gpt-5"


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


def canonicalize_column_name(column_name: Any) -> str:
    """列名を比較しやすい形に正規化する。"""
    text = unicodedata.normalize("NFKC", str(column_name)).replace("\ufeff", "").strip().lower()
    text = re.sub(r"[\s/]+", "_", text)
    text = re.sub(r"[^0-9a-zA-Zぁ-んァ-ン一-龥_]", "", text)
    return COLUMN_ALIASES.get(text, text)


def normalize_column_names(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """列名を正規化し、既知の別名を標準列名へ寄せる。"""
    df = df.copy()
    notes: List[str] = []
    renamed_columns: Dict[str, str] = {}

    for original_name in df.columns:
        normalized_name = canonicalize_column_name(original_name)
        renamed_columns[original_name] = normalized_name
        if str(original_name) != normalized_name:
            notes.append(f"列名 '{original_name}' を '{normalized_name}' として扱います。")

    df = df.rename(columns=renamed_columns)
    duplicate_columns = [column for column in df.columns[df.columns.duplicated()].unique()]

    for column in duplicate_columns:
        duplicates = df.loc[:, df.columns == column]
        merged = duplicates.bfill(axis=1).iloc[:, 0]
        df = df.loc[:, df.columns != column]
        df[column] = merged
        notes.append(f"列 '{column}' が複数見つかったため、左から順に値を補完して1列に統合しました。")

    return df, notes


def normalize_text_value(value: Any) -> Any:
    """セル値の余分な空白や表記ゆれを吸収する。"""
    if pd.isna(value):
        return pd.NA
    if not isinstance(value, str):
        return value

    normalized = unicodedata.normalize("NFKC", value).strip()
    normalized = re.sub(r"\s+", " ", normalized)
    if normalized.lower() in NULL_LIKE_VALUES:
        return pd.NA
    return normalized


def normalize_cell_values(df: pd.DataFrame) -> pd.DataFrame:
    """文字列セルを正規化し、空欄表現を欠損に統一する。"""
    df = df.copy()
    for column in df.columns:
        if pd.api.types.is_object_dtype(df[column]) or pd.api.types.is_string_dtype(df[column]):
            df[column] = df[column].map(normalize_text_value)
    return df


def clean_numeric_value(value: Any) -> Any:
    """人間向け表記の数値文字列を数値変換しやすい形に寄せる。"""
    if pd.isna(value):
        return pd.NA
    if isinstance(value, (int, float, np.number)):
        return value

    text = unicodedata.normalize("NFKC", str(value)).strip()
    if not text:
        return pd.NA

    text = text.replace(",", "")
    text = re.sub(r"[¥￥円個台冊本箱kgKGｋｇ]", "", text)
    text = re.sub(r"(days?|日)$", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", "", text)
    return text


def load_csv_with_fallbacks(uploaded_file: Any) -> Tuple[pd.DataFrame, List[str]]:
    """エンコーディングや区切り文字の違いを吸収しながらCSVを読み込む。"""
    raw_bytes = uploaded_file.getvalue()
    errors: List[str] = []
    best_result: Optional[Tuple[pd.DataFrame, str, str]] = None

    for encoding in CSV_ENCODING_CANDIDATES:
        for separator in CSV_SEPARATOR_CANDIDATES:
            try:
                read_options: Dict[str, Any] = {
                    "encoding": encoding,
                    "skipinitialspace": True,
                }
                separator_label = "自動判定"
                if separator is None:
                    read_options["sep"] = None
                    read_options["engine"] = "python"
                else:
                    read_options["sep"] = separator
                    separator_label = separator

                candidate_df = pd.read_csv(io.BytesIO(raw_bytes), **read_options)
                if candidate_df.empty and len(candidate_df.columns) == 0:
                    continue

                best_result = (candidate_df, encoding, separator_label)
                normalized_df, _ = normalize_column_names(candidate_df)
                if not validate_columns(normalized_df):
                    notes = [f"CSVを encoding={encoding}, 区切り文字={separator_label} で読み込みました。"]
                    return candidate_df, notes
            except Exception as exc:
                separator_label = "自動判定" if separator is None else separator
                errors.append(f"encoding={encoding}, sep={separator_label}: {exc}")

    if best_result is not None:
        candidate_df, encoding, separator_label = best_result
        notes = [f"CSVを encoding={encoding}, 区切り文字={separator_label} で読み込みました。"]
        return candidate_df, notes

    raise ValueError("CSVを読み込めませんでした。試した条件: " + " / ".join(errors[:6]))


def validate_columns(df: pd.DataFrame) -> List[str]:
    """必要な列がすべて含まれているかをチェックする。"""
    return [col for col in REQUIRED_COLUMNS if col not in df.columns]


def validate_required_values(df: pd.DataFrame) -> List[str]:
    """必須列の空欄を行単位でチェックする。"""
    errors: List[str] = []

    for column in REQUIRED_COLUMNS:
        if column not in df.columns:
            continue

        missing_rows = df[df[column].isna()].index.tolist()
        for index in missing_rows:
            errors.append(f"行 {index + 2}: 必須列 '{column}' が空欄です。")

    for column in REQUIRED_TEXT_COLUMNS:
        if column not in df.columns:
            continue

        blank_rows = df[df[column].astype(str).str.strip() == ""].index.tolist()
        for index in blank_rows:
            errors.append(f"行 {index + 2}: 必須列 '{column}' が空欄です。")

    return errors


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
        cleaned_values = df[col].map(clean_numeric_value)
        converted = pd.to_numeric(cleaned_values, errors="coerce")
        invalid = cleaned_values.notna() & converted.isna()
        for index in df[invalid].index:
            value = df.at[index, col]
            errors.append(f"行 {index + 2}: 列 '{col}' に数値として扱えない値 '{value}' が含まれています。")
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
        "current_stock": float(row["current_stock"]),
        "avg_daily_sales": float(row["avg_daily_sales"]),
        "lead_time_days": float(row["lead_time_days"]),
        "safety_days": float(row["safety_days"]),
        "safety_stock": round(float(row["safety_stock"]), 1),
        "reorder_point": round(float(row["reorder_point"]), 1),
        "base_recommended_order": int(row["base_recommended_order"]),
        "adjusted_order": int(row["adjusted_order"]),
        "estimated_order_cost": int(row["estimated_order_cost"]),
        "days_left": None if days_left == np.inf else round(days_left, 1),
        "priority_score": round(float(row["priority_score"]), 2),
        "risk_level": str(row["risk_level"]),
        "need_order": bool(row["need_order"]),
        "selection_reason": row.get("selection_reason"),
        "overstock_note": str(row["overstock_note"]),
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
        simulated_df = calculate_inventory_metrics(simulated_input)
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


def inject_pop_ui_styles() -> None:
    """親しみやすく、チャット導線が見えやすい配色と装飾を適用する。"""
    st.markdown(
        """
        <style>
        :root {
            --bg-main: linear-gradient(180deg, #fffef9 0%, #fff6e5 52%, #ffeede 100%);
            --panel: rgba(255, 255, 255, 0.92);
            --panel-strong: #fff7ef;
            --line: rgba(163, 98, 48, 0.22);
            --text-main: #3f2616;
            --text-soft: #63422e;
            --accent: #ea6a3e;
            --accent-strong: #c84f26;
            --accent-pale: #ffe1d8;
            --mint: #dff7ea;
            --yellow: #fff1b8;
        }

        .stApp {
            background: var(--bg-main);
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
            background: linear-gradient(180deg, #fff7ef 0%, #fff1dc 100%);
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
            box-shadow: 0 10px 30px rgba(164, 102, 58, 0.08);
        }

        div[data-testid="stChatMessage"] {
            padding: 0.4rem 0.8rem;
        }

        [data-testid="stFileUploader"] section,
        [data-testid="stFileUploaderDropzone"] {
            background: linear-gradient(180deg, #fffaf1 0%, #fff4e6 100%) !important;
            border: 1px dashed rgba(163, 98, 48, 0.28) !important;
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
            background: #fffaf3 !important;
            color: #4a2b18 !important;
        }

        pre {
            border: 1px solid rgba(163, 98, 48, 0.18) !important;
            border-radius: 18px !important;
        }

        .stCodeBlock code,
        .stCode code,
        pre code {
            color: #4a2b18 !important;
        }

        div[role="radiogroup"] {
            gap: 0.7rem;
            padding: 0.35rem;
            background: rgba(255, 255, 255, 0.72);
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
            background: rgba(255, 255, 255, 0.52);
            border: 1px solid rgba(163, 98, 48, 0.12);
            color: var(--text-main) !important;
            font-weight: 700;
        }

        div[role="radiogroup"] label[data-checked="true"] > div {
            background: linear-gradient(135deg, #ffd68a 0%, #ffbe73 100%);
            color: #4a2b18 !important;
            border-color: rgba(200, 79, 38, 0.28);
            box-shadow: 0 8px 18px rgba(200, 79, 38, 0.16);
        }

        .hero-card {
            background:
                radial-gradient(circle at top left, rgba(255,255,255,0.95) 0%, rgba(255,255,255,0.8) 34%, rgba(255,240,221,0.86) 100%);
            border: 1px solid var(--line);
            border-radius: 28px;
            padding: 1.4rem 1.5rem 1.2rem;
            box-shadow: 0 18px 40px rgba(164, 102, 58, 0.12);
            margin-bottom: 1rem;
        }

        .hero-badge {
            display: inline-block;
            background: var(--yellow);
            color: #6f4300 !important;
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

        .hero-points {
            display: flex;
            flex-wrap: wrap;
            gap: 0.55rem;
            margin-top: 0.35rem;
        }

        .hero-pill {
            background: white;
            border: 1px solid var(--line);
            border-radius: 999px;
            padding: 0.42rem 0.8rem;
            color: var(--text-main) !important;
            font-weight: 600;
            font-size: 0.92rem;
        }

        .chat-shell {
            background: linear-gradient(180deg, rgba(255, 249, 239, 0.92) 0%, rgba(255, 255, 255, 0.94) 100%);
            border: 1px solid rgba(163, 98, 48, 0.16);
            border-radius: 24px;
            padding: 1rem 1rem 0.35rem;
            margin-bottom: 0.85rem;
            box-shadow: 0 14px 30px rgba(164, 102, 58, 0.08);
        }

        .stButton > button, .stDownloadButton > button {
            border-radius: 999px;
            border: none;
            background: linear-gradient(135deg, var(--accent) 0%, #f08d43 100%);
            color: #fff8f2;
            font-weight: 700;
            padding: 0.6rem 1rem;
            box-shadow: 0 10px 20px rgba(200, 79, 38, 0.18);
        }

        .stButton > button:hover, .stDownloadButton > button:hover {
            background: linear-gradient(135deg, var(--accent-strong) 0%, #df7b32 100%);
            color: #fffefb;
        }

        [data-testid="stChatInput"] textarea,
        [data-testid="stTextInput"] input,
        [data-testid="stNumberInput"] input {
            color: var(--text-main) !important;
            background: rgba(255, 255, 255, 0.95) !important;
            caret-color: var(--text-main) !important;
            pointer-events: auto !important;
        }

        [data-testid="stChatInput"] textarea::placeholder,
        [data-testid="stTextInput"] input::placeholder,
        [data-testid="stNumberInput"] input::placeholder {
            color: #8f6e57 !important;
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
                今の在庫状況とおすすめ発注を落ち着いて確認できます。
            </div>
            <div class="hero-points">
                <span class="hero-pill">会話で確認</span>
                <span class="hero-pill">おすすめ発注</span>
                <span class="hero-pill">在庫一覧</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


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
    use_llm_chat: bool,
    openai_api_key: str,
) -> None:
    """チャットUIを表示する。"""
    initialize_chat_state()

    st.divider()
    st.subheader("チャット")
    st.caption("気になる商品名や予算を入れると、その条件で確認できます。")

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


def main() -> None:
    st.set_page_config(page_title="在庫発注最適化アシスタント", layout="wide")
    inject_pop_ui_styles()

    st.title("在庫発注最適化アシスタント")
    st.write(
        "CSVから在庫状況を読み取り、発注単位・最小発注数・原価・重要度・予算を考慮したおすすめ発注案を確認できます。"
    )
    render_pop_hero()

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
        df, load_notes = load_csv_with_fallbacks(uploaded_file)
        df, column_notes = normalize_column_names(df)
        df = normalize_cell_values(df)
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

    required_value_errors = validate_required_values(df)
    if required_value_errors:
        st.error("必須項目の空欄が見つかりました。CSVの該当行を確認してください。")
        for message in required_value_errors:
            st.write(f"- {message}")
        return

    df = sanitize_numeric_values(df)
    normalization_notes = load_notes + column_notes
    if normalization_notes:
        with st.expander("CSV読み込みメモ", expanded=False):
            for note in normalization_notes:
                st.write(f"- {note}")

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
    api_key_default = get_openai_api_key_default()
    openai_api_key = st.sidebar.text_input(
        "OpenAI APIキー",
        value=api_key_default,
        type="password",
        help="設定するとチャットで GPT の関数呼び出しを使えます。未設定時は従来のルールベース回答です。",
    ).strip()
    st.session_state.openai_api_key = openai_api_key
    use_llm_chat = st.sidebar.toggle(
        "GPTチャットを使う",
        value=bool(openai_api_key),
        disabled=not bool(openai_api_key),
        help="APIキー設定後に有効化できます。",
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

    planning_tab, table_tab = st.tabs(["最適化", "一覧"])

    with planning_tab:
        render_planning_tab(optimized_df, skipped_df, risk_df, overstock_df)

    with table_tab:
        render_tables(metrics_df, order_needed_df)

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


if __name__ == "__main__":
    main()
