# Cats vs Dogs — End-to-End MLOps Pipeline

**Course:** MLOps (AIMLCZG523) — Assignment 2
**Author:** Madiha Fathima
**Repository:** https://github.com/MadihaFathima/bits_mlops_assignment_2

An end-to-end MLOps pipeline for a binary Cats vs Dogs image classifier, covering data
versioning, model training with experiment tracking, containerized serving, CI/CD, and
monitoring — using only open-source tools.

For the full step-by-step narrative (every decision, every bug hit and how it was fixed,
in chronological order) see **[RUNBOOK.md](RUNBOOK.md)**. This README is the shorter,
task-oriented reference for setting things up and running each part.

## Pipeline overview

| Stage | Tool | Notes |
|---|---|---|
| Data versioning | Git + DVC | Local DVC remote; raw (25k images) and processed (4k subsample) datasets tracked separately |
| Model | PyTorch (`SimpleCNN`) | Shallow 3-conv-block baseline CNN, CPU-trained |
| Experiment tracking | MLflow | Params, metrics, confusion matrix, loss curves, logged model |
| Serving | FastAPI | `/health`, `/predict`, `/metrics` |
| Containerization | Docker | `python:3.11-slim`, CPU-only PyTorch |
| CI | GitHub Actions | Lint/test → build → smoke-test → push to GHCR |
| CD | GitHub Actions (self-hosted runner) + Kubernetes | Auto-deploys to Docker Desktop's local K8s on every push to `main` |
| Monitoring | Python `logging` + Prometheus | Request/response logs, `/metrics` scraped by a local Prometheus container |

## Project structure

```
src/
  data/
    preprocess.py      # filter corrupt images, subsample, split, resize -> data/processed/
    transforms.py       # shared train/eval image transforms (avoids train/serve skew)
  models/
    model.py             # SimpleCNN architecture
    train.py             # training loop, MLflow logging
  api/
    inference.py          # load_model(), predict() -- testable without a server
    main.py                # FastAPI app: /health, /predict, /metrics
tests/                      # pytest unit tests
scripts/
  post_deploy_eval.py       # sends real requests w/ known labels to the deployed service
docker/
  Dockerfile
  requirements.txt          # lean, inference-only deps for the container image
k8s/
  deployment.yaml            # 2 replicas, probes, resource limits
  service.yaml                # LoadBalancer on port 8080
monitoring/
  prometheus.yml               # scrape config for the deployed service
.github/workflows/ci.yml         # CI (test/build/push) + CD (deploy) pipeline
models/
  cnn_baseline.pt                 # trained model weights (committed to git)
  model_metadata.json              # run id + test metrics for the model above
requirements.txt                    # full dev/training dependencies
```

## Model performance

Baseline `SimpleCNN`, trained on a balanced 4,000-image subsample (3,200 train / 400 val
/ 400 test), 10 epochs, CPU-only:

| Metric | Value |
|---|---|
| Test accuracy | 0.7625 |
| Test precision | 0.7561 |
| Test recall | 0.7750 |
| Test F1 | 0.7654 |

Deliberately a shallow baseline, not a tuned/deep architecture — see RUNBOOK for the
CPU-time-vs-dataset-size tradeoff reasoning.

## Setup

**Prerequisites:** Python 3.11, Docker Desktop (with Kubernetes enabled), Git, a Kaggle
account (for the dataset).

```bash
py -3.11 -m venv venv
venv\Scripts\python.exe -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.13.0 torchvision==0.28.0
venv\Scripts\python.exe -m pip install -r requirements.txt
```

**Get the data** (either download fresh from Kaggle, or `dvc pull` if you have access to
the configured remote):

```bash
kaggle datasets download -d shaunthesheep/microsoft-catsvsdogs-dataset -p data/raw --unzip
# or, if you have the DVC remote configured:
venv\Scripts\dvc.exe pull
```

## M1 — Preprocess & train

```bash
venv\Scripts\python.exe -m src.data.preprocess
venv\Scripts\python.exe -m src.models.train
```

View experiment tracking (MLflow's tracking metadata lives in a SQLite file,
`mlflow.db`, not the `./mlruns` folder — see RUNBOOK for why):

```bash
venv\Scripts\python.exe -m mlflow ui --backend-store-uri sqlite:///mlflow.db
# open http://127.0.0.1:5000
```

## M2 — Run the API

**Locally:**
```bash
venv\Scripts\python.exe -m uvicorn src.api.main:app --reload --port 8000
curl http://localhost:8000/health
curl -X POST -F "file=@data/processed/test/dog/dog_00000.jpg" http://localhost:8000/predict
```

**In Docker:**
```bash
docker build -f docker/Dockerfile -t cats-dogs-api:latest .
docker run -d --name cats-dogs-api -p 8000:8000 cats-dogs-api:latest
curl http://localhost:8000/health
```

## M3 — Tests & CI

```bash
venv\Scripts\python.exe -m pytest tests/ -v
```

`.github/workflows/ci.yml` runs on every push/PR to `main`: checks out, installs deps,
runs the tests above, builds the Docker image, smoke-tests it, and pushes to GitHub
Container Registry (`ghcr.io/madihafathima/bits_mlops_assignment_2`).

## M4 — Deployment

Deploys to Docker Desktop's built-in Kubernetes cluster. The `deploy` job in the same
workflow runs on a **self-hosted GitHub Actions runner** (registered against this repo,
running on the same machine as the cluster), since GitHub's cloud runners can't reach a
local cluster. It applies the manifests, rolls the deployment to the exact commit's image
tag, and runs a post-deploy smoke test that fails the pipeline if the service doesn't
respond correctly.

Manual equivalent, if deploying by hand:
```bash
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl rollout status deployment/cats-dogs-api
curl http://localhost:8080/health
```

(Service is on port **8080**, not 80 — see RUNBOOK for why: an unrelated prior project's
service on this same shared cluster already occupies port 80.)

## M5 — Monitoring

- Every request is logged (method, path, status, latency — no request/response bodies).
- Prometheus-format metrics at `/metrics`: `api_requests_total`,
  `api_request_latency_seconds`, `predictions_total`.
- Scrape it with a local Prometheus container:
  ```bash
  docker run -d --name prometheus-cats-dogs -p 9090:9090 -v "<repo-path>\monitoring\prometheus.yml:/etc/prometheus/prometheus.yml" prom/prometheus
  # http://localhost:9090/targets should show cats-dogs-api as UP
  ```
- Post-deployment performance check (sends real requests with known labels to the live
  deployed service):
  ```bash
  venv\Scripts\python.exe scripts/post_deploy_eval.py
  ```

## Known dataset quirk

The Microsoft Cats vs Dogs dataset contains a small number of corrupt/zero-byte image
files (long-documented in the original release). `src/data/preprocess.py` filters these
out via `is_valid_image()` before training.
