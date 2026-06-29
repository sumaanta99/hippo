FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/

WORKDIR /app/app

ENV HOST=0.0.0.0 \
    RELOAD=false \
    PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["sh", "-c", "uvicorn api.server:create_app --factory --host 0.0.0.0 --port ${PORT:-8000}"]
