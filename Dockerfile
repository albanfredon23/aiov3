FROM python:3.12-slim

WORKDIR /app

# Installation des dépendances
COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Copie du code applicatif
COPY aio/ ./aio/
COPY dashboard/ ./dashboard/

ENV PYTHONUNBUFFERED=1
EXPOSE 8080

CMD ["python3", "-m", "uvicorn", "aio.api.main:app", "--host", "0.0.0.0", "--port", "8080"]
