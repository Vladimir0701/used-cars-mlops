import argparse
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
import joblib

import mlflow
import mlflow.sklearn


def parse_args() -> argparse.Namespace:
    """
    Разбор аргументов командной строки.
    Ожидаем только один аргумент: путь к YAML-конфигу.
    """
    parser = argparse.ArgumentParser(description="Train used-cars price model")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to train config .yaml (относительно корня проекта)",
    )
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """
    Загружаем YAML-конфиг в словарь Python.
    """
    config_path = Path(config_path)
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def load_raw_data(data_path: Path) -> pd.DataFrame:
    """
    Загружаем сырые данные из CSV.
    """
    df = pd.read_csv(data_path)
    return df


def add_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Преобразуем текстовые столбцы price и milage в числовые price_num и milage_num.
    Логика такая же, как в ноутбуке.
    """
    # Цена: "$10,300" -> 10300.0
    df["price_num"] = (
        df["price"]
        .str.replace("$", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.strip()
        .astype("float64")
    )

    # Пробег: "51,000 mi." -> 51000.0
    df["milage_num"] = (
        df["milage"]
        .str.replace("mi.", "", regex=False)
        .str.replace("mi", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.strip()
        .astype("float64")
    )

    return df

def add_engine_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Извлекаем числовые признаки из текстового столбца engine:
    - engine_hp: число перед 'HP'
    - engine_liters: число перед 'L' или 'Liter'

    Если распарсить не удалось — оставляем NaN.
    """
    engine_hp = []
    engine_liters = []

    for value in df["engine"].astype(str):
        hp = None
        liters = None

        # Ищем мощность в л.с.: '300.0HP' или '300 HP'
        m_hp = re.search(r"(\d+(?:\.\d+)?)\s*HP", value, flags=re.IGNORECASE)
        if m_hp:
            hp = float(m_hp.group(1))

        # Ищем объём: '3.7L' или '3.5 Liter'
        m_l = re.search(r"(\d+(?:\.\d+)?)\s*(L|Liter)", value, flags=re.IGNORECASE)
        if m_l:
            liters = float(m_l.group(1))

        engine_hp.append(hp)
        engine_liters.append(liters)

    df["engine_hp"] = pd.Series(engine_hp, index=df.index, dtype="float64")
    df["engine_liters"] = pd.Series(engine_liters, index=df.index, dtype="float64")

    return df

def filter_by_quantiles(
    df: pd.DataFrame,
    price_col: str,
    milage_col: str,
    price_q_low: float,
    price_q_high: float,
    milage_q_low: float,
    milage_q_high: float,
) -> pd.DataFrame:
    """
    Обрезаем выбросы по квантилям для цены и пробега.
    """
    p_low, p_high = df[price_col].quantile([price_q_low, price_q_high])
    m_low, m_high = df[milage_col].quantile([milage_q_low, milage_q_high])

    df_filtered = df[
        (df[price_col] >= p_low)
        & (df[price_col] <= p_high)
        & (df[milage_col] >= m_low)
        & (df[milage_col] <= m_high)
    ].reset_index(drop=True)

    return df_filtered


def prepare_datasets(
    df: pd.DataFrame,
    config: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, list[str], list[str]]:

    df = add_numeric_columns(df)

    prep_cfg = config.get("preprocessing", {})
    preprocess_version = prep_cfg.get("version", "v1")

    # При версии v2 добавляем числовые признаки из engine
    if preprocess_version == "v2":
        df = add_engine_features(df)

    target_col = config["target"]["name"]
    numeric_features = config["features"]["numeric"]
    categorical_features = config["features"]["categorical"]

    if prep_cfg.get("filter_by_quantiles", False):
        price_q_low, price_q_high = prep_cfg.get("price_quantiles", [0.01, 0.99])
        milage_q_low, milage_q_high = prep_cfg.get("milage_quantiles", [0.01, 0.99])
        df = filter_by_quantiles(
            df,
            price_col=target_col,
            milage_col="milage_num",
            price_q_low=price_q_low,
            price_q_high=price_q_high,
            milage_q_low=milage_q_low,
            milage_q_high=milage_q_high,
        )

    fill_value_cat = prep_cfg.get("fillna_categorical", "Unknown")
    for col in categorical_features:
        df[col] = df[col].fillna(fill_value_cat)

    df = df.dropna(subset=[target_col] + numeric_features).reset_index(drop=True)

    X = df[numeric_features + categorical_features].copy()
    y = df[target_col].copy()

    test_size = config["data"]["test_size"]
    random_state = config["data"]["random_state"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        shuffle=True,
    )

    return X_train, X_test, y_train, y_test, numeric_features, categorical_features

def build_pipeline(
    numeric_features: list[str],
    categorical_features: list[str],
    model_params: dict,
) -> Pipeline:

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", numeric_features),
            (
                "cat",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
                categorical_features,
            ),
        ]
    )

    rf = RandomForestRegressor(**model_params)

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", rf),
        ]
    )

    return pipeline


def compute_metrics(
    y_train: pd.Series,
    y_train_pred: np.ndarray,
    y_test: pd.Series,
    y_test_pred: np.ndarray,
) -> dict:
    mse_train = mean_squared_error(y_train, y_train_pred)
    mse_test = mean_squared_error(y_test, y_test_pred)

    rmse_train = float(np.sqrt(mse_train))
    rmse_test = float(np.sqrt(mse_test))

    mae_train = float(mean_absolute_error(y_train, y_train_pred))
    r2_train = float(r2_score(y_train, y_train_pred))

    mae_test = float(mean_absolute_error(y_test, y_test_pred))
    r2_test = float(r2_score(y_test, y_test_pred))

    metrics = {
        "mae_train": round(mae_train, 2),
        "rmse_train": round(rmse_train, 2),
        "r2_train": round(r2_train, 4),
        "mae_test": round(mae_test, 2),
        "rmse_test": round(rmse_test, 2),
        "r2_test": round(r2_test, 4),
    }
    return metrics


def train_and_log(config: dict) -> dict:

    data_path = Path(config["data"]["path"])
    if not data_path.is_absolute():
        data_path = Path.cwd() / data_path

    df_raw = load_raw_data(data_path)

    (
        X_train,
        X_test,
        y_train,
        y_test,
        numeric_features,
        categorical_features,
    ) = prepare_datasets(df_raw, config)

    model_params = config["model"]["params"]
    pipeline = build_pipeline(numeric_features, categorical_features, model_params)

    pipeline.fit(X_train, y_train)

    y_train_pred = pipeline.predict(X_train)
    y_test_pred = pipeline.predict(X_test)
    metrics = compute_metrics(y_train, y_train_pred, y_test, y_test_pred)

    print("Metrics:")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))

    model_path = Path(config["output"]["model_path"])
    if not model_path.is_absolute():
        model_path = Path.cwd() / model_path
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_path)
    print(f"Модель сохранена в: {model_path}")

    metrics_path = Path(config["output"]["metrics_path"])
    if not metrics_path.is_absolute():
        metrics_path = Path.cwd() / metrics_path
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"Метрики сохранены в: {metrics_path}")

    # --- 7. Логирование в MLflow (если включено) ---
    mlflow_cfg = config.get("mlflow", {})
    if mlflow_cfg.get("enabled", False):
        # параметры данных
        mlflow.log_param("data_path", str(data_path))
        mlflow.log_param("test_size", config["data"]["test_size"])
        mlflow.log_param("random_state", config["data"]["random_state"])

        # параметры препроцессинга
        prep_cfg = config.get("preprocessing", {})
        mlflow.log_param("preprocess_version", prep_cfg.get("version", "v1"))
        mlflow.log_param("filter_by_quantiles", prep_cfg.get("filter_by_quantiles", False))
        mlflow.log_param("price_quantiles", prep_cfg.get("price_quantiles", [0.01, 0.99]))
        mlflow.log_param("milage_quantiles", prep_cfg.get("milage_quantiles", [0.01, 0.99]))
        mlflow.log_param("fillna_categorical", prep_cfg.get("fillna_categorical", "Unknown"))

        # параметры модели
        mlflow.log_param("model_type", config["model"]["type"])
        for k, v in model_params.items():
            mlflow.log_param(f"model__{k}", v)

        # метрики
        mlflow.log_metrics(metrics)

        # модель (как артефакт MLflow)
        mlflow.sklearn.log_model(pipeline, artifact_path="model")

    return metrics


def main():
    # 1. Читаем аргументы и конфиг
    args = parse_args()
    config = load_config(args.config)

    # 2. Настройка MLflow (эксперимент и трекинг)
    mlflow_cfg = config.get("mlflow", {})
    use_mlflow = mlflow_cfg.get("enabled", False)

    if use_mlflow:
        tracking_uri = mlflow_cfg.get("tracking_uri")
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)

        experiment_name = mlflow_cfg.get("experiment_name", "used_cars_price")
        mlflow.set_experiment(experiment_name)

        run_name = mlflow_cfg.get("run_name", None)

        # Открываем MLflow run
        with mlflow.start_run(run_name=run_name):
            train_and_log(config)
    else:
        # Без MLflow просто тренируем и сохраняем
        train_and_log(config)


if __name__ == "__main__":
    main()