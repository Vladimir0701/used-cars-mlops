from pydantic import BaseModel
from typing import List, Optional


class CarFeatures(BaseModel):
    """
    Описание одного автомобиля, как мы его принимаем на вход /predict.
    Формат полей такой же, как в исходном датасете.
    """
    brand: str
    model: str
    model_year: int
    milage: str                 # например, "51,000 mi."
    fuel_type: Optional[str] = None
    engine: str                 # строка с описанием двигателя
    transmission: Optional[str] = None
    ext_col: Optional[str] = None
    int_col: Optional[str] = None
    accident: Optional[str] = None
    clean_title: Optional[str] = None


class PredictRequest(BaseModel):
    """
    Запрос на предсказание: список автомобилей.
    """
    objects: List[CarFeatures]


class PredictResponseItem(BaseModel):
    """
    Ответ по одному автомобилю.
    """
    price: float


class PredictResponse(BaseModel):
    """
    Общий ответ: список предсказаний.
    """
    predictions: List[PredictResponseItem]