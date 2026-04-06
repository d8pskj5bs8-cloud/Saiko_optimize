import io
import re
import unicodedata
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from constants import (
    COLUMN_ALIASES,
    CSV_ENCODING_CANDIDATES,
    CSV_SEPARATOR_CANDIDATES,
    FORECAST_EXTERNAL_NUMERIC_COLUMNS,
    FORECAST_EXTERNAL_REQUIRED_COLUMNS,
    FORECAST_HISTORY_NUMERIC_COLUMNS,
    FORECAST_HISTORY_REQUIRED_COLUMNS,
    NULL_LIKE_VALUES,
    NUMERIC_COLUMNS,
    OPTIONAL_DEFAULTS,
    REQUIRED_COLUMNS,
    REQUIRED_TEXT_COLUMNS,
)


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


def preprocess_csv_text(raw_text: str) -> str:
    """CSV全体の軽微な崩れを補正して読み取りやすくする。"""
    text = raw_text.replace("\r\n", "\n").replace("\r", "\n").replace("\ufeff", "")
    text = text.replace("，", ",").replace("；", ";")

    normalized_lines: List[str] = []
    for line in text.split("\n"):
        stripped = line.strip()
        if len(stripped) >= 2 and stripped[0] == stripped[-1] and stripped[0] in {'"', "'"}:
            inner = stripped[1:-1]
            if any(separator in inner for separator in [",", ";", "\t"]):
                normalized_lines.append(inner)
                continue
        normalized_lines.append(line)
    return "\n".join(normalized_lines)


def load_csv_with_fallbacks(
    uploaded_file: Any,
    expected_columns: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    """エンコーディングや区切り文字の違いを吸収しながらCSVを読み込む。"""
    raw_bytes = uploaded_file.getvalue()
    errors: List[str] = []
    best_result: Optional[Tuple[int, pd.DataFrame, str, str, str]] = None
    expected = expected_columns or REQUIRED_COLUMNS

    for encoding in CSV_ENCODING_CANDIDATES:
        text_variants: List[Tuple[str, Any, str]] = [("bytes", raw_bytes, "そのまま")]
        try:
            decoded_text = raw_bytes.decode(encoding)
            preprocessed_text = preprocess_csv_text(decoded_text)
            if preprocessed_text != decoded_text:
                text_variants.append(("text", preprocessed_text, "前処理"))
        except Exception:
            pass

        for source_type, source_value, source_label in text_variants:
            for separator in CSV_SEPARATOR_CANDIDATES:
                try:
                    read_options: Dict[str, Any] = {
                        "skipinitialspace": True,
                    }
                    if source_type == "bytes":
                        source = io.BytesIO(source_value)
                        read_options["encoding"] = encoding
                    else:
                        source = io.StringIO(source_value)

                    separator_label = "自動判定"
                    if separator is None:
                        read_options["sep"] = None
                        read_options["engine"] = "python"
                    else:
                        read_options["sep"] = separator
                        separator_label = separator

                    candidate_df = pd.read_csv(source, **read_options)
                    if candidate_df.empty and len(candidate_df.columns) == 0:
                        continue

                    normalized_df, _ = normalize_column_names(candidate_df)
                    matched_columns = sum(1 for col in expected if col in normalized_df.columns)
                    score = matched_columns * 10 - len(normalized_df.columns)
                    if best_result is None or score > best_result[0]:
                        best_result = (score, candidate_df, encoding, separator_label, source_label)

                    missing_columns = [col for col in expected if col not in normalized_df.columns]
                    if not missing_columns:
                        notes = [f"CSVを encoding={encoding}, 区切り文字={separator_label} で読み込みました。"]
                        if source_label != "そのまま":
                            notes.append("CSVの全角区切り文字や行全体の引用符を前処理して読み込みました。")
                        return candidate_df, notes
                except Exception as exc:
                    separator_label = "自動判定" if separator is None else separator
                    errors.append(f"encoding={encoding}, sep={separator_label}, source={source_label}: {exc}")

    if best_result is not None:
        _, candidate_df, encoding, separator_label, source_label = best_result
        notes = [f"CSVを encoding={encoding}, 区切り文字={separator_label} で読み込みました。"]
        if source_label != "そのまま":
            notes.append("CSVの全角区切り文字や行全体の引用符を前処理して読み込みました。")
        return candidate_df, notes

    raise ValueError("CSVを読み込めませんでした。試した条件: " + " / ".join(errors[:6]))


def get_required_columns(order_policy: str) -> List[str]:
    """発注方式に応じた必須列を返す。"""
    if order_policy == "定期発注":
        return REQUIRED_COLUMNS + ["review_cycle_days"]
    return REQUIRED_COLUMNS


def validate_columns(df: pd.DataFrame, order_policy: str) -> List[str]:
    """必要な列がすべて含まれているかをチェックする。"""
    return [col for col in get_required_columns(order_policy) if col not in df.columns]


def validate_required_values(df: pd.DataFrame, order_policy: str) -> List[str]:
    """必須列の空欄を行単位でチェックする。"""
    errors: List[str] = []
    required_columns = get_required_columns(order_policy)

    for column in required_columns:
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
    df["review_cycle_days"] = df["review_cycle_days"].clip(lower=0)
    df["order_unit"] = df["order_unit"].clip(lower=1)
    df["min_order_qty"] = df["min_order_qty"].clip(lower=0)
    df["max_stock"] = df["max_stock"].where(df["max_stock"].isna(), df["max_stock"].clip(lower=0))
    df["unit_cost"] = df["unit_cost"].clip(lower=0)
    df["holding_cost_rate"] = df["holding_cost_rate"].clip(lower=0)
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
        df, load_notes = load_csv_with_fallbacks(uploaded_file, FORECAST_HISTORY_REQUIRED_COLUMNS)
        df, column_notes = normalize_column_names(df)
        df = normalize_cell_values(df)
    except Exception as exc:
        return None, [], [f"販売履歴CSVの読み込み中にエラーが発生しました: {exc}"]

    errors: List[str] = []
    missing_columns = [col for col in FORECAST_HISTORY_REQUIRED_COLUMNS if col not in df.columns]
    if missing_columns:
        detected_columns = ", ".join(map(str, df.columns.tolist())) if len(df.columns) > 0 else "なし"
        errors.append("販売履歴CSVに必須列が不足しています: " + ", ".join(missing_columns))
        errors.append("読み取れた列名: " + detected_columns)
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
        df, load_notes = load_csv_with_fallbacks(uploaded_file, FORECAST_EXTERNAL_REQUIRED_COLUMNS)
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
