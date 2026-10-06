# Moteur AIOTrade (TAP + SCG + filtre χ²) exposé par FastAPI. Simulation uniquement.
FROM python:3.12-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY aiotrade/ ./aiotrade/

# Étape de test : docker build --target test ./engine
FROM base AS test
COPY requirements-dev.txt pyproject.toml ./
RUN pip install -r requirements-dev.txt
COPY tests/ ./tests/
RUN python -m pytest

# Image d'exécution (cible par défaut)
FROM base AS runtime
RUN useradd --create-home --uid 10001 aiotrade
USER aiotrade
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=4s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=3)" || exit 1
CMD ["uvicorn", "aiotrade.api:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers", "--forwarded-allow-ips", "*"]
