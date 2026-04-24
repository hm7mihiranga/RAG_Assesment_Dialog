FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY ui ./ui
COPY data ./data
COPY artifacts ./artifacts

EXPOSE 8080 8501