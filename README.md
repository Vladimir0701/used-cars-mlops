# Курсовая работа по дисциплине MLOps  
Тема: «MLOps-пайплайн для прогнозирования цены подержанных автомобилей»

Выполнил: Баранов Владимир Георгиевич, ШАД-311

---

## Описание проекта

Проект реализует MLOps-пайплайн для регрессии: по характеристикам подержанного автомобиля предсказывается его цена.

Основное:

- датасет: `used_cars.csv` (описание объявлений о продаже авто);
- модель: `RandomForestRegressor` в составе `sklearn.Pipeline`;
- версионирование экспериментов: **MLflow** (3+ эксперимента, 2 версии препроцессинга);
- инференс-сервис: **FastAPI** (`/health`, `/predict`);
- контейнеризация: **Docker** (образ с `python:3.11-slim`).

Доп. документация:

- `MODEL_CARD.md` — описание модели;
- `DATASET_CARD.md` — описание датасета.

---

## Структура репозитория

```text
used-cars-mlops/
│
├── artifacts/
│   └── models/
│       ├── model_v1.joblib
│       ├── metrics_v1.json
│       ├── model_v1_tuned.joblib
│       ├── metrics_v1_tuned.json
│       ├── model_v2.joblib
│       └── metrics_v2.json
│
├── configs/
│   ├── config_train.yaml              # базовый train (v1)
│   ├── config_train_v1_tuned.yaml     # train v1 с другими гиперпараметрами
│   ├── config_train_v2.yaml           # train с препроцессингом v2 (engine_hp / engine_liters)
│   └── config_infer.yaml              # конфиг инференс-сервиса
│
├── data/
│   └── raw/
│       └── used_cars.csv              # исходный датасет
│
├── notebooks/
│   └── 01_eda.ipynb                   # EDA и прототип модели
│
├── src/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py                    # FastAPI-приложение (/health, /predict)
│   │   └── schemas.py                 # Pydantic-схемы запросов/ответов
│   │
│   └── models/
│       ├── __init__.py
│       └── train.py                   # обучение по YAML-конфигу + MLflow
│
├── .gitignore
├── requirements.txt
├── Dockerfile
├── README.md
├── MODEL_CARD.md
└── DATASET_CARD.md
```
---
## Установка окружения
```text
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```
## Обучение моделей
Все запуски выполняются из корня репозитория.


### 1. Базовый эксперимент (v1)

```text
python -m src.models.train --config configs/config_train.yaml
```
Результат:
- модель: artifacts/models/model_v1.joblib;
- метрики: artifacts/models/metrics_v1.json;
- MLflow-run: rf_v1_baseline.

### 2. Эксперимент v1 с другими гиперпараметрами
```text
python -m src.models.train --config configs/config_train_v1_tuned.yaml
```

Результат:
- модель: artifacts/models/model_v1_tuned.joblib;
- метрики: artifacts/models/metrics_v1_tuned.json;
- MLflow-run: rf_v1_tuned_depth12.
### 3. Эксперимент v2 (доп. признаки из engine)
```text
python -m src.models.train --config configs/config_train_v2.yaml
```
Результат:
- модель: artifacts/models/model_v2.joblib;
- метрики: artifacts/models/metrics_v2.json;
- MLflow-run: rf_v2_engine_features.

## MLflow (трекинг экспериментов)

Запуск интерфейса:

```text
mlflow ui
```
Интерфейс будет доступен по адресу:
```text
http://127.0.0.1:5000
```
В MLflow отображаются:
- эксперимент: used_cars_price;
- запуски: rf_v1_baseline, rf_v1_tuned_depth12, rf_v2_engine_features;
- параметры: версия препроцессинга (preprocess_version), гиперпараметры модели (model__n_estimators, model__max_depth и др.), настройки данных;
- метрики: mae_train, rmse_train, r2_train, mae_test, rmse_test, r2_test;
- артефакты: сохранённая модель для каждого запуска.

## Запуск FastAPI-сервиса

Локальный запуск (без Docker):

```text
uvicorn src.app.main:app --host 0.0.0.0 --port 8000 --reload
```
Доступные эндпоинты:
- GET /health — проверка работоспособности сервиса;
- POST /predict — предсказание цены;
- GET /docs — Swagger UI.

Пример тела запроса к /predict:
```text
{
  "objects": [
    {
      "brand": "Ford",
      "model": "F-150",
      "model_year": 2018,
      "milage": "51,000 mi.",
      "fuel_type": "Gasoline",
      "engine": "3.5L V6 24V GDI DOHC Twin Turbo",
      "transmission": "Automatic",
      "ext_col": "White",
      "int_col": "Black",
      "accident": "No accidents reported",
      "clean_title": "Clean"
    }
  ]
}
```

## Docker

### Сборка образа

```text
docker build -t used-cars-service .
```

### Запуск контейнера
```text
docker run --rm -p 8000:8000 used-cars-service
```

### Swagger UI:
```text
http://127.0.0.1:8000/docs
```