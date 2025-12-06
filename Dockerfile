FROM python:3.11-slim

# Переменная окружения: Python не пишет .pyc и вывод не буферизуется
ENV PYTHONDONTWRITEBYTECODE=1\

# Логи сразу летят в консоль
ENV PYTHONUNBUFFERED=1

# Устанавливаем системные зависимости
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Рабочая директория внутри контейнера
WORKDIR /app

COPY requirements.txt .

# Устанавливаем Python-зависимости
RUN pip install --no-cache-dir -r requirements.txt

# Копируем весь проект в контейнер
COPY . .

# 8. Открываем порт 8000
EXPOSE 8000

# 9. Команда по умолчанию: запускаем uvicorn-сервис
CMD ["uvicorn", "src.app.main:app", "--host", "0.0.0.0", "--port", "8000"]