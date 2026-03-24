FROM python:3.11-slim

LABEL maintainer="cognibrew"

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 5000

CMD ["mlflow", "server", \
     "--host", "0.0.0.0", \
     "--port", "5000", \
     "--default-artifact-root", "s3://mlflow-artifacts/", \
     "--backend-store-uri", "${MLFLOW_BACKEND_STORE_URI}"]
