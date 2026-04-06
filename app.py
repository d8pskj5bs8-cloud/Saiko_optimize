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
    "category": "未分類",
    "location": "default",
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
    "salesqty": "sales_qty",
    "sales_qty": "sales_qty",
    "sales": "sales_qty",
    "qty": "sales_qty",
    "販売数": "sales_qty",
    "売上数": "sales_qty",
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
    "current_stock",
    "avg_daily_sales",
    "forecast_daily_sales",
    "demand_basis_label",
    "demand_basis_value",
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
    "category",
    "supplier",
    "forecast_daily_sales",
    "demand_basis_label",
    "adjusted_order",
    "estimated_order_cost",
    "days_left",
    "priority_score",
]

RISK_COLUMNS = [
    "product_name",
    "category",
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
    "category",
    "supplier",
    "current_stock",
    "avg_daily_sales",
    "forecast_daily_sales",
    "days_left",
    "reorder_point",
    "excess_stock",
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

SAMPLE_CSV = """product_id,product_name,current_stock,avg_daily_sales,lead_time_days,safety_days,order_unit,min_order_qty,unit_cost,priority_weight,supplier,category,location
1,ミネラルウォーター,20,3.5,5,3,12,24,110,1.2,飲料仕入先A,飲料,tokyo
2,お茶,8,2.0,7,2,24,24,95,1.1,飲料仕入先A,飲料,tokyo
3,コーヒー,50,1.2,10,5,10,20,380,1.4,飲料仕入先B,飲料,tokyo
4,カップ麺,5,4.0,3,2,12,24,180,1.6,食品仕入先C,食品,tokyo
5,スポーツドリンク,80,1.0,4,2,24,24,140,0.8,飲料仕入先A,飲料,tokyo
"""

WELCOME_MESSAGE = """在庫アシスタントです。こんな聞き方ができます。

- 「今日のおすすめ発注を見せて」
- 「予算30000円で発注案を出して」
- 「欠品リスクが高い商品を見せて」
- 「過剰在庫を教えて」
- 「お茶の発注理由は？」
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


def convert_columns_to_numeric(df: pd.DataFrame, columns: List[str], label: str) -> Tuple[pd.DataFrame, List[str]]:
    """任意の数値列群を変換し、失敗した行を返す。"""
    df = df.copy()
    errors: List[str] = []

    for col in columns:
        if col not in df.columns:
            continue
        cleaned_values = df[col].map(clean_numeric_value)
        converted = pd.to_numeric(cleaned_values, errors="coerce")
        invalid = cleaned_values.notna() & converted.isna()
        for index in df[invalid].index:
            value = df.at[index, col]
            errors.append(f"{label} 行 {index + 2}: 列 '{col}' に数値として扱えない値 '{value}' が含まれています。")
        df[col] = converted

    return df, errors


def convert_date_column(df: pd.DataFrame, column: str, label: str) -> Tuple[pd.DataFrame, List[str]]:
    """日付列を datetime に変換する。"""
    df = df.copy()
    if column not in df.columns:
        return df, [f"{label}: 必須列 '{column}' が見つかりません。"]

    converted = pd.to_datetime(df[column], errors="coerce")
    invalid = df[column].notna() & converted.isna()
    errors = [f"{label} 行 {index + 2}: 列 '{column}' の日付を解釈できません。" for index in df[invalid].index]
    df[column] = converted
    return df, errors


def validate_duplicate_keys(df: pd.DataFrame, columns: List[str], label: str) -> List[str]:
    """重複キーを検出する。"""
    if any(column not in df.columns for column in columns):
        return []

    duplicated = df[df.duplicated(columns, keep=False)].copy()
    if duplicated.empty:
        return []

    duplicated = duplicated.sort_values(columns).head(5)
    preview = duplicated[columns].astype(str).agg(" / ".join, axis=1).tolist()
    return [f"{label}: キー {', '.join(columns)} の重複があります。例: {item}" for item in preview]


def prepare_forecast_history_df(uploaded_file: Any) -> Tuple[Optional[pd.DataFrame], List[str], List[str]]:
    """販売履歴CSVを読み込み、予測学習向けに整える。"""
    try:
        df, load_notes = load_csv_with_fallbacks(uploaded_file)
        df, column_notes = normalize_column_names(df)
        df = normalize_cell_values(df)
    except Exception as exc:
        return None, [], [f"販売履歴CSVの読み込み中にエラーが発生しました: {exc}"]

    errors: List[str] = []
    missing_columns = [col for col in FORECAST_HISTORY_REQUIRED_COLUMNS if col not in df.columns]
    if missing_columns:
        errors.append("販売履歴CSVに必須列が不足しています: " + ", ".join(missing_columns))
        return None, load_notes + column_notes, errors

    if "promotion_flag" not in df.columns:
        df["promotion_flag"] = 0
    if "location" not in df.columns:
        df["location"] = "default"

    df, numeric_errors = convert_columns_to_numeric(df, FORECAST_HISTORY_NUMERIC_COLUMNS, "販売履歴CSV")
    df, date_errors = convert_date_column(df, "date", "販売履歴CSV")
    errors.extend(numeric_errors)
    errors.extend(date_errors)

    required_value_errors = []
    for column in FORECAST_HISTORY_REQUIRED_COLUMNS:
        missing_rows = df[df[column].isna()].index.tolist()
        for index in missing_rows:
            required_value_errors.append(f"販売履歴CSV 行 {index + 2}: 必須列 '{column}' が空欄です。")
    errors.extend(required_value_errors)
    errors.extend(validate_duplicate_keys(df, ["date", "product_id"], "販売履歴CSV"))

    if errors:
        return None, load_notes + column_notes, errors

    df["product_id"] = df["product_id"].astype(str)
    df["location"] = df["location"].astype(str)
    df["sales_qty"] = df["sales_qty"].clip(lower=0)
    df["promotion_flag"] = df["promotion_flag"].fillna(0).clip(lower=0)
    return df, load_notes + column_notes, []


def prepare_forecast_external_df(uploaded_file: Any) -> Tuple[Optional[pd.DataFrame], List[str], List[str]]:
    """外部要因CSVを読み込み、予測用に整える。"""
    try:
        df, load_notes = load_csv_with_fallbacks(uploaded_file)
        df, column_notes = normalize_column_names(df)
        df = normalize_cell_values(df)
    except Exception as exc:
        return None, [], [f"外部要因CSVの読み込み中にエラーが発生しました: {exc}"]

    errors: List[str] = []
    missing_columns = [col for col in FORECAST_EXTERNAL_REQUIRED_COLUMNS if col not in df.columns]
    if missing_columns:
        errors.append("外部要因CSVに必須列が不足しています: " + ", ".join(missing_columns))
        return None, load_notes + column_notes, errors

    if "location" not in df.columns:
        df["location"] = "default"
    if "is_holiday" not in df.columns:
        df["is_holiday"] = 0

    df, numeric_errors = convert_columns_to_numeric(df, FORECAST_EXTERNAL_NUMERIC_COLUMNS, "外部要因CSV")
    df, date_errors = convert_date_column(df, "date", "外部要因CSV")
    errors.extend(numeric_errors)
    errors.extend(date_errors)
    errors.extend(validate_duplicate_keys(df, ["date", "location"], "外部要因CSV"))

    if errors:
        return None, load_notes + column_notes, errors

    df["location"] = df["location"].astype(str)
    df["temp_avg"] = df["temp_avg"].astype(float)
    df["rain_mm"] = df["rain_mm"].fillna(0).clip(lower=0)
    df["is_holiday"] = df["is_holiday"].fillna(0).clip(lower=0).astype(int)
    return df, load_notes + column_notes, []


def fit_ridge_regression(X: np.ndarray, y: np.ndarray, alpha: float = 1.0) -> Tuple[float, np.ndarray]:
    """シンプルなリッジ回帰を閉形式で解く。"""
    if X.ndim != 2:
        raise ValueError("X must be 2-dimensional")

    x_mean = X.mean(axis=0)
    x_std = X.std(axis=0)
    x_std = np.where(x_std == 0, 1.0, x_std)
    y_mean = float(y.mean())

    X_scaled = (X - x_mean) / x_std
    y_centered = y - y_mean

    identity = np.eye(X_scaled.shape[1])
    coefficients_scaled = np.linalg.solve(X_scaled.T @ X_scaled + alpha * identity, X_scaled.T @ y_centered)
    coefficients = coefficients_scaled / x_std
    intercept = y_mean - float(np.dot(x_mean, coefficients))
    return intercept, coefficients


def build_forecast_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """予測用の説明変数を追加する。"""
    result = df.copy()
    result["day_of_week"] = result["date"].dt.dayofweek
    result["is_weekend"] = (result["day_of_week"] >= 5).astype(int)
    result["is_rainy"] = (result["rain_mm"] > 0).astype(int)
    result["temp_avg_squared"] = result["temp_avg"] ** 2
    if "is_holiday" not in result.columns:
        result["is_holiday"] = 0
    result["is_holiday"] = result["is_holiday"].fillna(0).astype(int)
    if "promotion_flag" not in result.columns:
        result["promotion_flag"] = 0
    result["promotion_flag"] = result["promotion_flag"].fillna(0)
    return result


def prepare_forecast_training_data(
    inventory_df: pd.DataFrame,
    history_df: pd.DataFrame,
    external_df: pd.DataFrame,
) -> pd.DataFrame:
    """販売履歴と外部要因を結合し、学習用データを作る。"""
    inventory_lookup = inventory_df[["product_id", "category", "location"]].drop_duplicates("product_id")
    merged_history = history_df.merge(inventory_lookup, on="product_id", how="left", suffixes=("", "_inventory"))
    if "category_inventory" in merged_history.columns:
        merged_history["category"] = merged_history["category"].fillna(merged_history["category_inventory"])
    if "location_inventory" in merged_history.columns:
        merged_history["location"] = merged_history["location"].fillna(merged_history["location_inventory"])
    merged_history["category"] = merged_history["category"].fillna("未分類")
    merged_history["location"] = merged_history["location"].fillna("default")

    training_df = merged_history.merge(
        external_df,
        on=["date", "location"],
        how="left",
        suffixes=("", "_external"),
    )
    training_df = build_forecast_feature_frame(training_df)
    training_df = training_df.sort_values(["product_id", "date"]).reset_index(drop=True)
    training_df["lag_1_sales"] = training_df.groupby("product_id")["sales_qty"].shift(1)
    training_df["rolling_mean_7"] = (
        training_df.groupby("product_id")["sales_qty"].shift(1).rolling(window=7, min_periods=3).mean().reset_index(level=0, drop=True)
    )
    return training_df


def build_forecast_reason_summary(feature_values: Dict[str, float], coefficients: Dict[str, float]) -> str:
    """寄与の大きい要因を短く説明する。"""
    contributions: List[Tuple[str, float]] = []
    label_map = {
        "temp_avg": "平均気温",
        "temp_avg_squared": "気温の強さ",
        "rain_mm": "降水量",
        "is_rainy": "雨フラグ",
        "day_of_week": "曜日要因",
        "is_weekend": "週末要因",
        "is_holiday": "祝日要因",
        "promotion_flag": "販促要因",
        "lag_1_sales": "前日販売",
        "rolling_mean_7": "直近7日平均",
    }

    for feature_name, coefficient in coefficients.items():
        value = feature_values.get(feature_name)
        if value is None:
            continue
        contributions.append((label_map.get(feature_name, feature_name), coefficient * value))

    if not contributions:
        return "予測要因を十分に説明できませんでした。"

    contributions.sort(key=lambda item: abs(item[1]), reverse=True)
    top_items = contributions[:3]
    parts = []
    for label, contribution in top_items:
        direction = "プラス" if contribution >= 0 else "マイナス"
        parts.append(f"{label}が{direction}")
    return "主な要因: " + "、".join(parts)


def generate_demand_forecast(
    inventory_df: pd.DataFrame,
    history_df: pd.DataFrame,
    external_df: pd.DataFrame,
    forecast_date: pd.Timestamp,
    ridge_alpha: float = 1.0,
) -> Dict[str, Any]:
    """カテゴリ単位の線形モデルで商品別需要を予測する。"""
    training_df = prepare_forecast_training_data(inventory_df, history_df, external_df)
    training_df = training_df.dropna(subset=["temp_avg", "rain_mm"])

    forecast_rows: List[Dict[str, Any]] = []
    coefficient_rows: List[Dict[str, Any]] = []
    notes: List[str] = []

    inventory_products = inventory_df.copy()
    inventory_products["category"] = inventory_products["category"].fillna("未分類")
    inventory_products["location"] = inventory_products["location"].fillna("default")

    latest_history = training_df.sort_values("date").groupby("product_id").tail(7)
    future_external = external_df[external_df["date"] == forecast_date].copy()
    if future_external.empty:
        return {
            "enabled": False,
            "message": f"{forecast_date.date()} の外部要因データが無いため、需要予測を使えません。",
            "forecast_df": pd.DataFrame(),
            "coefficients_df": pd.DataFrame(),
            "notes": notes,
        }

    future_external = build_forecast_feature_frame(future_external)

    for category_name, category_products in inventory_products.groupby("category"):
        category_train = training_df[training_df["category"] == category_name].copy()
        if category_train.empty:
            notes.append(f"{category_name}: 学習データが無いため予測をスキップしました。")
            continue

        category_train = category_train.dropna(subset=FORECAST_FEATURE_COLUMNS + ["sales_qty"])
        if len(category_train) < 20:
            notes.append(f"{category_name}: 学習件数が {len(category_train)} 件のため、予測をスキップしました。")
            continue

        X = category_train[FORECAST_FEATURE_COLUMNS].to_numpy(dtype=float)
        y = category_train["sales_qty"].to_numpy(dtype=float)
        try:
            intercept, coefficients = fit_ridge_regression(X, y, alpha=ridge_alpha)
        except np.linalg.LinAlgError:
            notes.append(f"{category_name}: モデル学習に失敗したため、予測をスキップしました。")
            continue

        coefficient_map = {
            feature_name: float(coefficient)
            for feature_name, coefficient in zip(FORECAST_FEATURE_COLUMNS, coefficients)
        }
        for feature_name, coefficient in coefficient_map.items():
            coefficient_rows.append(
                {
                    "forecast_model_group": category_name,
                    "sample_count": len(category_train),
                    "feature": feature_name,
                    "coefficient": round(coefficient, 4),
                }
            )

        for _, product_row in category_products.iterrows():
            recent_sales = latest_history[latest_history["product_id"] == product_row["product_id"]].sort_values("date")
            if recent_sales.empty:
                notes.append(f"{product_row['product_name']}: 履歴が不足しているため予測をスキップしました。")
                continue

            location = str(product_row["location"])
            external_row = future_external[future_external["location"].astype(str) == location]
            if external_row.empty and future_external["location"].nunique() == 1:
                external_row = future_external.iloc[[0]].copy()
            if external_row.empty:
                notes.append(f"{product_row['product_name']}: 予測日に対応する拠点データが無いため予測をスキップしました。")
                continue

            future_row = external_row.iloc[0].to_dict()
            future_row["promotion_flag"] = 0
            future_row["lag_1_sales"] = float(recent_sales.iloc[-1]["sales_qty"])
            future_row["rolling_mean_7"] = float(recent_sales["sales_qty"].tail(7).mean())
            feature_vector = np.array([float(future_row.get(column, 0.0)) for column in FORECAST_FEATURE_COLUMNS], dtype=float)
            predicted_sales = max(0.0, float(intercept + np.dot(feature_vector, coefficients)))

            forecast_rows.append(
                {
                    "product_id": str(product_row["product_id"]),
                    "forecast_daily_sales": round(predicted_sales, 2),
                    "forecast_model_group": category_name,
                    "forecast_reason_summary": build_forecast_reason_summary(future_row, coefficient_map),
                }
            )

    forecast_df = pd.DataFrame(forecast_rows)
    coefficients_df = pd.DataFrame(coefficient_rows)
    if forecast_df.empty:
        return {
            "enabled": False,
            "message": "需要予測に必要な履歴や外部要因が不足しているため、今回は実績平均日販を使います。",
            "forecast_df": forecast_df,
            "coefficients_df": coefficients_df,
            "notes": notes,
        }

    merged = inventory_products[["product_id", "product_name", "category", "location", "avg_daily_sales"]].merge(
        forecast_df,
        on="product_id",
        how="left",
    )
    merged["forecast_diff"] = (merged["forecast_daily_sales"] - merged["avg_daily_sales"]).round(2)
    merged["forecast_change_ratio"] = np.where(
        merged["avg_daily_sales"] > 0,
        ((merged["forecast_daily_sales"] / merged["avg_daily_sales"]) - 1).round(2),
        np.nan,
    )
    return {
        "enabled": True,
        "message": f"{forecast_date.date()} の需要予測を {forecast_df['product_id'].nunique()} 商品に適用しました。",
        "forecast_df": merged,
        "coefficients_df": coefficients_df,
        "notes": notes,
    }


def round_order_quantity(base_qty: float, order_unit: float, min_order_qty: float) -> int:
    """発注単位と最小発注数を考慮して発注量を丸める。"""
    if base_qty <= 0:
        return 0

    adjusted_qty = max(base_qty, min_order_qty)
    rounded_qty = math.ceil(adjusted_qty / order_unit) * order_unit
    return int(rounded_qty)


def calculate_priority_score(df: pd.DataFrame, demand_column: str = "demand_basis_value") -> pd.Series:
    """欠品リスクと重要度を組み合わせた優先度スコアを計算する。"""
    days_divisor = np.maximum(df["days_left"].replace(np.inf, 9999), 0.5)
    urgency_component = np.maximum(df["reorder_point"] - df["current_stock"], 0) + df[demand_column]
    lead_time_component = 1 + (df["lead_time_days"] / np.maximum(df["lead_time_days"].max(), 1))
    return (df["priority_weight"] * urgency_component * lead_time_component) / days_divisor


def calculate_inventory_metrics(df: pd.DataFrame, demand_column: str = "avg_daily_sales", demand_label: str = "実績平均") -> pd.DataFrame:
    """在庫関連指標と発注条件をまとめて計算する。"""
    df = df.copy()
    if demand_column not in df.columns:
        raise ValueError(f"需要列 '{demand_column}' が見つかりません。")

    df["forecast_daily_sales"] = df.get("forecast_daily_sales", pd.Series(np.nan, index=df.index))
    df["forecast_model_group"] = df.get("forecast_model_group", pd.Series("", index=df.index)).fillna("")
    df["forecast_reason_summary"] = df.get("forecast_reason_summary", pd.Series("", index=df.index)).fillna("")
    df["demand_basis_label"] = demand_label
    df["demand_basis_value"] = df[demand_column].fillna(df["avg_daily_sales"]).clip(lower=0)
    df["safety_stock"] = df["demand_basis_value"] * df["safety_days"]
    df["reorder_point"] = df["demand_basis_value"] * df["lead_time_days"] + df["safety_stock"]
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
        df["demand_basis_value"] == 0,
        np.inf,
        df["current_stock"] / df["demand_basis_value"],
    )
    df["need_order"] = df["adjusted_order"] > 0
    df["estimated_order_cost"] = df["adjusted_order"] * df["unit_cost"]
    df["priority_score"] = calculate_priority_score(df, "demand_basis_value").round(2)
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
    for numeric_column in ["avg_daily_sales", "forecast_daily_sales", "demand_basis_value", "forecast_diff"]:
        if numeric_column in display_df.columns:
            display_df[numeric_column] = display_df[numeric_column].apply(
                lambda value: "" if pd.isna(value) else round(float(value), 2)
            )
    if "forecast_change_ratio" in display_df.columns:
        display_df["forecast_change_ratio"] = display_df["forecast_change_ratio"].apply(
            lambda value: "" if pd.isna(value) else f"{float(value) * 100:.0f}%"
        )
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
    demand_basis_text = f"{row['demand_basis_label']}日販 {row['demand_basis_value']:.2f}"
    forecast_text = ""
    if not pd.isna(row.get("forecast_daily_sales", np.nan)):
        forecast_text = f" 需要予測は {float(row['forecast_daily_sales']):.2f} です。"
    return (
        f"{row['product_name']} は仕入先 {row['supplier']} の商品です。現在在庫は {row['current_stock']}、"
        f" 実績平均日販は {row['avg_daily_sales']}、計算に使っている需要は {demand_basis_text}、"
        f" 発注点は {row['reorder_point']:.1f} です。{forecast_text}"
        f" 基本推奨発注数は {int(row['base_recommended_order'])} 個ですが、発注単位 {int(row['order_unit'])}"
        f" と最小発注数 {int(row['min_order_qty'])} を考慮すると {int(row['adjusted_order'])} 個になります。"
        f" 推定発注金額は {int(row['estimated_order_cost']):,}円、在庫が持つ見込みは"
        f" {format_days_left(float(row['days_left']))} で、{need_order_text}。"
    )


def build_reason_message(row: pd.Series) -> str:
    """発注理由の説明を作る。"""
    return (
        f"{row['product_name']} の安全在庫は {row['demand_basis_label']}日販 {row['demand_basis_value']:.2f} × {row['safety_days']}日 = "
        f"{row['safety_stock']:.1f} です。発注点は 平均販売数 {row['avg_daily_sales']} × "
        f"ではなく、計算対象の需要 {row['demand_basis_value']:.2f} × "
        f"リードタイム {row['lead_time_days']}日 + 安全在庫 {row['safety_stock']:.1f} = "
        f"{row['reorder_point']:.1f} です。そこから基本推奨発注数は {int(row['base_recommended_order'])} 個になり、"
        f" 発注単位 {int(row['order_unit'])} と最小発注数 {int(row['min_order_qty'])} を反映した最終候補は"
        f" {int(row['adjusted_order'])} 個です。優先度スコアは {row['priority_score']:.2f} です。"
        f" {row.get('forecast_reason_summary', '')}"
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
        "forecast_daily_sales": None if pd.isna(row.get("forecast_daily_sales", np.nan)) else round(float(row["forecast_daily_sales"]), 2),
        "lead_time_days": float(row["lead_time_days"]),
        "safety_days": float(row["safety_days"]),
        "safety_stock": round(float(row["safety_stock"]), 1),
        "reorder_point": round(float(row["reorder_point"]), 1),
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
        simulated_df = calculate_inventory_metrics(simulated_input, demand_column, demand_label)
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

    if any(keyword in normalized for keyword in ["予測", "需要", "天気"]):
        forecast_available = metrics_df["forecast_daily_sales"].notna().any()
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
        demand_column = "selected_daily_sales" if "selected_daily_sales" in simulated_input.columns else "avg_daily_sales"
        demand_label = str(metrics_df["demand_basis_label"].iloc[0]) if "demand_basis_label" in metrics_df.columns else "実績平均"
        simulated_df = calculate_inventory_metrics(simulated_input, demand_column, demand_label)
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
    forecast_items = int(metrics_df["forecast_daily_sales"].notna().sum()) if "forecast_daily_sales" in metrics_df.columns else 0

    st.subheader("サマリー")
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("商品数", total_items)
    col2.metric("発注が必要な商品数", total_order_items)
    col3.metric("採用された発注候補", len(optimized_df))
    col4.metric("採用済み発注額", f"{optimized_cost:,}円")
    col5.metric("予測適用商品数", forecast_items)
    st.caption(f"現在の予算設定: {budget_text}")


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
            border-radius: 999px;
            border: none;
            background: linear-gradient(135deg, var(--accent) 0%, #58a6dd 100%);
            color: #f8fcff;
            font-weight: 700;
            padding: 0.6rem 1rem;
            box-shadow: 0 10px 20px rgba(47, 128, 196, 0.18);
        }

        .stButton > button:hover, .stDownloadButton > button:hover {
            background: linear-gradient(135deg, var(--accent-strong) 0%, #438fcb 100%);
            color: #ffffff;
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


def render_forecast_tab(
    forecast_result: Dict[str, Any],
    forecast_mode: str,
    forecast_date: pd.Timestamp,
) -> None:
    """需要予測の概要を表示する。"""
    st.subheader("需要予測")
    st.caption(f"予測対象日: {forecast_date.date()} / 発注計算モード: {forecast_mode}")
    st.write(forecast_result["message"])

    for note in forecast_result.get("notes", [])[:5]:
        st.caption(note)

    forecast_df = forecast_result.get("forecast_df")
    if isinstance(forecast_df, pd.DataFrame) and not forecast_df.empty:
        st.dataframe(prepare_display_df(forecast_df, FORECAST_PRODUCT_COLUMNS), use_container_width=True)
    else:
        st.info("予測一覧はまだありません。")

    coefficients_df = forecast_result.get("coefficients_df")
    st.subheader("モデル係数")
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
    sales_history_file = st.file_uploader("販売履歴CSV（任意）", type="csv")
    external_factors_file = st.file_uploader("外部要因CSV（任意）", type="csv")

    if uploaded_file is None:
        reset_chat_state()
        st.info("まずはCSVをアップロードしてください。")
        st.subheader("サンプルCSV形式")
        st.code(SAMPLE_CSV, language="csv")
        st.caption("任意列として order_unit, min_order_qty, unit_cost, priority_weight, supplier, category, location も使えます。")
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

    df["product_id"] = df["product_id"].astype(str)
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
    forecast_mode = st.sidebar.radio(
        "発注計算に使う需要",
        options=["実績平均", "需要予測", "大きい方"],
        help="需要予測が使えない商品は自動で実績平均にフォールバックします。",
    )
    forecast_date = pd.Timestamp(
        st.sidebar.date_input(
            "予測対象日",
            value=(pd.Timestamp.today().normalize() + pd.Timedelta(days=1)).date(),
        )
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

    forecast_result: Dict[str, Any] = {
        "enabled": False,
        "message": "販売履歴CSVと外部要因CSVをアップロードすると需要予測を利用できます。",
        "forecast_df": pd.DataFrame(),
        "coefficients_df": pd.DataFrame(),
        "notes": [],
    }

    if sales_history_file is not None and external_factors_file is not None:
        history_df, history_notes, history_errors = prepare_forecast_history_df(sales_history_file)
        external_df, external_notes, external_errors = prepare_forecast_external_df(external_factors_file)
        normalization_notes.extend(history_notes + external_notes)

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

        if history_df is not None and external_df is not None:
            forecast_result = generate_demand_forecast(filtered_df, history_df, external_df, forecast_date)
            forecast_df = forecast_result.get("forecast_df")
            if isinstance(forecast_df, pd.DataFrame) and not forecast_df.empty:
                filtered_df = filtered_df.merge(
                    forecast_df[["product_id", "forecast_daily_sales", "forecast_model_group", "forecast_reason_summary"]],
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

    if forecast_mode == "需要予測":
        filtered_df["selected_daily_sales"] = filtered_df["forecast_daily_sales"].fillna(filtered_df["avg_daily_sales"])
    elif forecast_mode == "大きい方":
        filtered_df["selected_daily_sales"] = np.maximum(
            filtered_df["avg_daily_sales"].astype(float),
            filtered_df["forecast_daily_sales"].fillna(0).astype(float),
        )
    else:
        filtered_df["selected_daily_sales"] = filtered_df["avg_daily_sales"].astype(float)

    try:
        metrics_df = calculate_inventory_metrics(filtered_df, "selected_daily_sales", forecast_mode)
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

    planning_tab, table_tab, forecast_tab = st.tabs(["最適化", "一覧", "予測"])

    with planning_tab:
        render_planning_tab(optimized_df, skipped_df, risk_df, overstock_df)

    with table_tab:
        render_tables(metrics_df, order_needed_df)

    with forecast_tab:
        render_forecast_tab(forecast_result, forecast_mode, forecast_date)

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
