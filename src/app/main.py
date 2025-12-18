from pathlib import Path
import re

import joblib
import numpy as np
import pandas as pd
import yaml
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .schemas import PredictRequest, PredictResponse, PredictResponseItem



CONFIG_PATH = Path("configs") / "config_infer.yaml"


def load_infer_config(config_path: Path) -> dict:
    if not config_path.is_absolute():
        # считаем, что запускаем сервис из корня проекта
        config_path = Path.cwd() / config_path

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def load_model(model_path: Path):
    if not model_path.is_absolute():
        model_path = Path.cwd() / model_path

    if not model_path.exists():
        raise FileNotFoundError(f"Модель не найдена по пути: {model_path}")

    model = joblib.load(model_path)
    return model


# Загружаем конфиг и модель один раз при старте приложения
infer_config = load_infer_config(CONFIG_PATH)
model_path = Path(infer_config["model"]["path"])
preprocess_version = infer_config["model"]["preprocessing_version"]

numeric_features = infer_config["features"]["numeric"]
categorical_features = infer_config["features"]["categorical"]
fillna_categorical = infer_config["preprocessing"].get("fillna_categorical", "Unknown")

model = load_model(model_path)


# ---------- ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ПРЕПРОЦЕССИНГА ----------

def add_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["milage_num"] = (
        df["milage"]
        .astype(str)
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
    """
    df = df.copy()

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


def preprocess_request(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Полный препроцессинг входных данных:
    - milage -> milage_num
    - при версии v2 -> engine_hp и engine_liters
    - заполнение пропусков в категориальных
    - выбор колонок в правильном порядке
    """
    df = df_raw.copy()

    # milage_num
    df = add_numeric_columns(df)

    # engine_hp и engine_liters для v2
    if preprocess_version == "v2":
        df = add_engine_features(df)

    # заполняем пропуски в категориальных
    for col in categorical_features:
        if col in df.columns:
            df[col] = df[col].fillna(fillna_categorical)
        else:
            # если какого-то столбца нет во входе, создадим его
            df[col] = fillna_categorical

    # проверяем, что все числовые признаки присутствуют
    for col in numeric_features:
        if col not in df.columns:
            raise ValueError(f"Ожидаемый числовой признак '{col}' отсутствует во входных данных")

    # формируем X в нужном порядке
    X = df[numeric_features + categorical_features].copy()
    return X


# ---------- СОЗДАЁМ FASTAPI-ПРИЛОЖЕНИЕ ----------

app = FastAPI(
    title="Used Cars Price Prediction API",
    description="Сервис для предсказания цены подержанных автомобилей",
    version="1.0.0",
)



@app.get("/health")
def health():
    """
    Эндпоинт для проверки, что сервис жив.
    """
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    """
    Эндпоинт для предсказания цены.
    Принимает список автомобилей, возвращает список цен.
    """
    # Преобразуем входные данные в DataFrame
    objects = [obj.dict() for obj in request.objects]
    df_raw = pd.DataFrame(objects)

    # Препроцессинг
    X = preprocess_request(df_raw)

    preds = model.predict(X)

    response_items = [
        PredictResponseItem(price=float(p)) for p in preds
    ]

    return PredictResponse(predictions=response_items)
