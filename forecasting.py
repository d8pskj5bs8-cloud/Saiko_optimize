from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from constants import FORECAST_FEATURE_COLUMNS


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


def build_prediction_sequence(
    intercept: float,
    coefficients: np.ndarray,
    recent_sales: pd.Series,
    future_external: pd.DataFrame,
    horizon_days: int,
) -> List[float]:
    """将来日数分の需要を順次予測する。"""
    if horizon_days <= 0:
        return []

    sales_history = [float(value) for value in recent_sales.dropna().tail(7).tolist()]
    if not sales_history:
        return []

    predictions: List[float] = []
    for _, external_row in future_external.head(horizon_days).iterrows():
        feature_row = external_row.to_dict()
        feature_row["promotion_flag"] = 0
        feature_row["lag_1_sales"] = sales_history[-1]
        history_window = sales_history[-7:]
        feature_row["rolling_mean_7"] = float(np.mean(history_window))
        feature_vector = np.array([float(feature_row.get(column, 0.0)) for column in FORECAST_FEATURE_COLUMNS], dtype=float)
        predicted_sales = max(0.0, float(intercept + np.dot(feature_vector, coefficients)))
        predictions.append(predicted_sales)
        sales_history.append(predicted_sales)

    return predictions


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
    inventory_products["forecast_horizon_days"] = (
        inventory_products["lead_time_days"] + inventory_products["review_cycle_days"] + inventory_products["safety_days"]
    ).clip(lower=1).astype(int)

    latest_history = training_df.sort_values("date").groupby("product_id").tail(7)
    future_external = external_df[external_df["date"] >= forecast_date].copy()
    if future_external.empty:
        available_dates = external_df["date"].dropna().sort_values()
        available_range_text = ""
        if not available_dates.empty:
            available_range_text = (
                f" 利用可能な日付は {available_dates.iloc[0].date()} 〜 {available_dates.iloc[-1].date()} です。"
            )
        return {
            "enabled": False,
            "message": f"{forecast_date.date()} の外部要因データが無いため、需要予測を使えません。{available_range_text}",
            "forecast_df": pd.DataFrame(),
            "coefficients_df": pd.DataFrame(),
            "notes": notes,
        }

    future_external = build_forecast_feature_frame(future_external)
    future_external["date"] = pd.to_datetime(future_external["date"])
    max_horizon_days = int(inventory_products["forecast_horizon_days"].max()) if not inventory_products.empty else 1

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
            external_rows = future_external[future_external["location"].astype(str) == location].sort_values("date").copy()
            if external_rows.empty and future_external["location"].nunique() == 1:
                external_rows = future_external.sort_values("date").copy()
            if external_rows.empty:
                notes.append(f"{product_row['product_name']}: 予測日に対応する拠点データが無いため予測をスキップしました。")
                continue

            horizon_days = int(product_row["forecast_horizon_days"])
            available_horizon = min(horizon_days, len(external_rows))
            if available_horizon < horizon_days:
                notes.append(
                    f"{product_row['product_name']}: 予測に使える外部要因は {available_horizon} 日分までのため、"
                    f"{horizon_days} 日必要な累積需要は計算できませんでした。"
                )

            prediction_path = build_prediction_sequence(
                intercept=intercept,
                coefficients=coefficients,
                recent_sales=recent_sales["sales_qty"],
                future_external=external_rows,
                horizon_days=min(max_horizon_days, len(external_rows)),
            )
            if not prediction_path:
                notes.append(f"{product_row['product_name']}: 将来需要の予測列を作れませんでした。")
                continue

            predicted_sales = float(prediction_path[0])
            forecast_period_demand = np.nan
            forecast_effective_daily_sales = np.nan
            if available_horizon >= horizon_days:
                forecast_period_demand = round(float(sum(prediction_path[:horizon_days])), 2)
                forecast_effective_daily_sales = round(float(forecast_period_demand / horizon_days), 2)

            future_row = external_rows.iloc[0].to_dict()
            future_row["promotion_flag"] = 0
            future_row["lag_1_sales"] = float(recent_sales.iloc[-1]["sales_qty"])
            future_row["rolling_mean_7"] = float(recent_sales["sales_qty"].tail(7).mean())

            forecast_rows.append(
                {
                    "product_id": str(product_row["product_id"]),
                    "forecast_daily_sales": round(predicted_sales, 2),
                    "forecast_period_demand": forecast_period_demand,
                    "forecast_horizon_days": horizon_days,
                    "forecast_effective_daily_sales": forecast_effective_daily_sales,
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
