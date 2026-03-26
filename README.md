# CogniBrew Cloud

Cloud-side MLOps platform for the CogniBrew face recognition system. Handles batch vector processing, drift detection, confidence tuning, and edge device synchronisation.

## Services

| Service | Description | Port |
|---------|-------------|------|
| **Edge Gateway** | Receives raw batch uploads from edge devices | 8000 |
| **Vector Operation** | Vector CRUD, drift detection, threshold calibration | — |
| **Edge Sync** | Assembles sync bundles for edge-pull pattern | — |
| **Inference Server** | InsightFace embedding inference | 8010 |
| **MLflow** | Experiment tracking & artifact store | 5001 |
| **Qdrant** | Vector database (512-dim cosine) | 6333 |
| **PostgreSQL** | Metadata & MLflow backend store | 5432 |
| **RustFS** | S3-compatible object store | 9001 (console) |
| **Airflow** | DAG orchestration (CeleryExecutor) | 8080 |

## Related Repositories

| Repository | Description |
|------------|-------------|
| [cognibrew-cloud-edge-gateway](https://github.com/tinnapopd/cognibrew-cloud-edge-gateway) | Edge Gateway service |
| [cognibrew-cloud-vector-operation](https://github.com/tinnapopd/cognibrew-cloud-vector-operation) | Vector Operation service |
| [cognibrew-cloud-edge-sync](https://github.com/tinnapopd/cognibrew-cloud-edge-sync) | Edge Sync service |
| [cognibrew-inference-server](https://github.com/tinnapopd/cognibrew-inference-server) | Inference Server |

## Getting Started

### Prerequisites

- Docker & Docker Compose
- (Optional) Python 3.11+ for local development

### Setup

1. **Clone the repository**

   ```bash
   git clone https://github.com/tinnapopd/cognibrew-cloud.git
   cd cognibrew-cloud
   ```

2. **Create environment file**

   ```bash
   cp .env.example .env
   ```

   Edit `.env` as needed — defaults work for local development.

3. **Start infrastructure**

   ```bash
   docker compose -f compose.infra.yaml up -d
   ```

4. **Start application services**

   ```bash
   docker compose -f compose.cognibrew.yaml up -d
   ```

5. **Start Airflow orchestrator**

   ```bash
   docker compose -f compose.airflow.yaml up -d
   ```

### Verify

- Airflow UI → [http://localhost:8080](http://localhost:8080) (user: `airflow` / pass: `airflow`)
- MLflow UI → [http://localhost:5001](http://localhost:5001)
- RustFS Console → [http://localhost:9001](http://localhost:9001)
- Qdrant Dashboard → [http://localhost:6333/dashboard](http://localhost:6333/dashboard)

## Pipeline

The `cognibrew_pipeline` DAG runs daily at **00:00 UTC**:

```
read_batch --> process_vectors -->  parallel_tasks[get_thresholds, get_vectors] --> edge_sync_update --> edge_sync_healthcheck
```

1. **read_batch** — Lists today's device JSON files from S3 and flattens vectors.
2. **process_vectors** — Groups vectors by device/user and POSTs to Vector Operation to update baselines.
3. **get_thresholds** — Fetches optimal similarity thresholds per device.
4. **get_vectors** — Fetches current vector galleries per device.
5. **edge_sync_update** — Pushes thresholds + vectors to Edge Sync.
6. **edge_sync_healthcheck** — Verifies Edge Sync is reachable.

## Project Structure

```
cognibrew-cloud/
├── airflow-orchestrator/     # Airflow Dockerfile, DAGs, config
│   └── dags/
│       └── pipeline.py       # Main pipeline DAG
├── mlflow-server/            # MLflow Dockerfile
├── docs/                     # Architecture diagrams (PlantUML)
├── compose.airflow.yaml      # Airflow stack (Celery + Redis + Postgres)
├── compose.cognibrew.yaml    # Application services
├── compose.infra.yaml        # Infrastructure (RustFS, Qdrant, Postgres, MLflow)
├── .env.example              # Environment variable template
└── README.md
```

## License

This project is part of a Master's degree coursework (Software Engineering - MLS).
