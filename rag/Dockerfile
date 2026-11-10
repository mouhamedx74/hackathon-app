FROM python:3.11-slim

WORKDIR /app

# Dépendances d'abord : cette couche reste en cache tant que requirements.txt ne change pas
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
COPY cli/ ./cli/
COPY data/ ./data/

EXPOSE 8000

# Lancer l'ingestion (idempotente grâce à l'upsert) puis l'API
CMD ["sh", "-c", "python -m cli.ingest && uvicorn app.api:app --host 0.0.0.0 --port 8000"]
