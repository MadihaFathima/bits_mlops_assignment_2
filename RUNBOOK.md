# RUNBOOK — Cats vs Dogs MLOps Pipeline

Course: MLOps (AIMLCZG523) — Assignment 2. This is the detailed engineering log: what
was built, why, and every non-obvious issue hit along the way with its root cause and
fix. For setup/usage instructions, see **[README.md](README.md)**.

## Tooling decisions

| Area | Choice | Why |
|---|---|---|
| Experiment tracking | MLflow (local) | Open-source, no account needed, matches Assignment 1 |
| ML framework | PyTorch | Clean serialization (`.pt`), integrates well with FastAPI |
| Dataset versioning | DVC, local folder remote | No cloud account needed; demonstrates DVC mechanics |
| CI/CD | GitHub Actions | Matches Assignment 1, free, integrates with GHCR |
| Deployment target | Docker Desktop's built-in Kubernetes | Reuses the working `docker-desktop` kubectl context and image-resolution approach from Assignment 1, instead of standalone minikube |
| Local Python | 3.11.9 (via `py -3.11`) | Only Python 3.14 was present on the machine; PyTorch/TensorFlow don't yet ship wheels for 3.14. Docker image also pins `python:3.11-slim` for consistency |

---

## M1 — Model Development & Experiment Tracking

### Data & code versioning

- Git for source, DVC (local folder remote at `D:/Bits-SEM3/dvc-storage/mlops_assignment_2`)
  for datasets.
- Dataset: **Microsoft Cats vs Dogs** (`shaunthesheep/microsoft-catsvsdogs-dataset` on
  Kaggle) — the original dataset behind the classic Kaggle "Dogs vs. Cats" competition.
  Chosen because it's raw/unsplit, matching the assignment's instruction to preprocess
  and split the data ourselves. Downloaded via the `kaggle` CLI, authenticated with
  Kaggle's newer API-token method (`KAGGLE_API_TOKEN` env var).
- Result: 25,000 raw images (`data/raw/PetImages/{Cat,Dog}`), DVC-tracked (`dvc add` +
  `dvc push`). **Known dataset quirk**: contains a small number of corrupt/zero-byte
  files (long-documented in the original release) — filtered out during preprocessing.
- Preprocessed data (resized 224×224 RGB, 80/10/10 split) also DVC-tracked as a separate
  artifact from the raw data, per the assignment's "track pre-processed data" wording.

### Model building

- `src/data/preprocess.py`: filters corrupt images (`is_valid_image`), subsamples a
  **balanced 4,000 images** (2,000/class) from the full 25,000, splits 80/10/10, resizes
  to 224×224 RGB, writes to `data/processed/{train,val,test}/{cat,dog}/`.
  Subsampling (rather than training on all 25,000) was a deliberate engineering
  tradeoff for CPU-only training time, not a shortcut.
- `src/models/model.py`: `SimpleCNN` — 3 conv+ReLU+maxpool blocks (3→32→64→128
  channels) + `AdaptiveAvgPool2d((7,7))` (keeps the FC layer small regardless of input
  size) + a small FC head, single logit output for `BCEWithLogitsLoss`. Deliberately
  shallow given CPU-only training.
- `src/models/train.py`: trains via `ImageFolder` + `DataLoader` (train-only
  augmentation: random flip/rotation), logs everything to MLflow, evaluates on the test
  set, saves the model.
- `src/data/transforms.py` holds the shared train/eval image transforms, reused later by
  the inference API — avoids train/serve skew.

### Experiment tracking

- MLflow experiment `cats-vs-dogs-classification`. Each run logs params (architecture,
  epochs, batch size, lr), per-epoch metrics (train/val loss & accuracy), final test
  metrics, a confusion-matrix plot, a loss-curve plot, the model (via
  `mlflow.pytorch.log_model`, with an `input_example` — this MLflow version needs one to
  trace the model graph), and `models/model_metadata.json`.
- **Result** (run `9a622c5b19bb4e8dab8280c7fb374abf`):

  | Metric | Value |
  |---|---|
  | Test accuracy | 0.7625 |
  | Test precision | 0.7561 |
  | Test recall | 0.7750 |
  | Test F1 | 0.7654 |

  Loss curve shows normal train/val convergence with mild overfitting from epoch ~6-7 —
  expected for a shallow baseline on 3,200 training images. Confusion matrix is
  reasonably balanced (150/50 cat, 45/155 dog).
- `models/cnn_baseline.pt` and `model_metadata.json` are committed **directly to git**
  (not DVC) — small enough (3.5MB) that git handles it fine, and keeps the model
  directly available for Docker/CI without a `dvc pull` step.

**Troubleshooting — MLflow UI shows an empty experiment.** `mlflow ui
--backend-store-uri ./mlruns` initially refused to start at all (MLflow's file-store
backend is in "maintenance mode" in this version; worked around with
`MLFLOW_ALLOW_FILE_STORE=true`), and once it started, showed zero runs despite training
having clearly succeeded. Root cause: `train.py` never sets an explicit tracking URI, and
this MLflow version's implicit default silently *splits* storage — tracking metadata
(runs/params/metrics) goes to an auto-created SQLite file, `mlflow.db`, while artifacts
still go to `./mlruns/.../artifacts/`. Pointing the UI at `./mlruns` only ever found the
artifacts, not the metadata. **Fix**: view experiments via
`mlflow ui --backend-store-uri sqlite:///mlflow.db` instead — confirmed via a direct
`sqlite3` query and then the UI showing all runs, params, metrics, and artifacts
correctly.

---

## M2 — Model Packaging & Containerization

### Inference service

- `src/api/inference.py`: `load_model()` and `predict()`, kept separate from the FastAPI
  layer so they're independently unit-testable.
- `src/api/main.py`: FastAPI app, model loaded once at startup (`lifespan` context
  manager, not per-request).
  - `GET /health` → `{"status": "healthy"}`
  - `POST /predict` → multipart file upload, validates it's an image, returns
    `{label, probability, class_probabilities}`

### Environment specification

- `requirements.txt` — full dev/training dependencies, pinned versions (torch/torchvision
  installed separately from the PyTorch CPU wheel index, since PyPI only hosts CUDA
  builds).
- `docker/requirements.txt` — a leaner, inference-only subset (numpy, pillow, fastapi,
  uvicorn, pydantic, python-multipart, prometheus-client), excluding dvc/mlflow/
  matplotlib/scikit-learn/pytest to keep the container image smaller.

### Containerization

- `docker/Dockerfile`: `python:3.11-slim`, installs CPU torch/torchvision as its own
  layer before copying app code (so code changes don't invalidate that large layer on
  rebuild), copies only `src/` and `models/cnn_baseline.pt`. `HEALTHCHECK` polls
  `/health` every 30s.
- `.dockerignore` excludes `venv/`, `data/` (826MB), `mlruns/`, `.git/`, `.dvc/`,
  `notebooks/`, `outputs/`, `tests/` — otherwise the build context would include the
  entire raw dataset.
- **Verified**: built, ran locally (`docker run -p 8000:8000`), container reported
  `Up ... (healthy)`, `/health` and `/predict` both confirmed correct via curl.

---

## M3 — CI Pipeline for Build, Test & Image Creation

### Automated testing

- `tests/test_preprocess.py` (8 tests) — `is_valid_image()` and `split_paths()`, using
  `tmp_path` fixtures rather than the real dataset.
- `tests/test_inference.py` (4 tests) — `predict()` against synthetic PIL images (no
  dependency on real cat/dog photos): output structure, probabilities sum to 1, handles
  non-square/non-224 images and grayscale input.
- All 12 pass via `python -m pytest tests/ -v`.

### CI setup & artifact publishing

`.github/workflows/ci.yml`, on push/PR to `main`:
- **`test`**: checkout → Python 3.11 → install deps → `python -m pytest`.
- **`build-and-push`** (needs `test`): builds the Docker image, **smoke-tests it locally
  first** (runs the container, polls `/health`, fails with container logs if it never
  comes up healthy), only then logs in to and pushes to **GitHub Container Registry**
  (`ghcr.io/madihafathima/bits_mlops_assignment_2`) using the automatically-provided
  `GITHUB_TOKEN` — no manually-configured secrets needed.

**Design note**: CI does not `dvc pull` or retrain — the DVC remote is a local Windows
folder, unreachable from GitHub's cloud runners. This isn't a gap: the trained model is
committed directly to git, so `actions/checkout` alone gives CI everything it needs.
Training/data-versioning stays a local concern; CI's job is build/test/package/publish.

**Troubleshooting**: first CI run failed test collection with `ModuleNotFoundError: No
module named 'src'`. Cause: the workflow ran bare `pytest`, not `python -m pytest` — only
the latter adds the project root to `sys.path`, which the `from src...` absolute imports
need. Fixed by using `python -m pytest` in the workflow.

---

## M4 — CD Pipeline & Deployment

### Deployment target & manifests

- Target: **Docker Desktop's built-in Kubernetes** cluster on this machine.
- `k8s/deployment.yaml`: 2 replicas, resource requests/limits, `imagePullPolicy: Always`.
- `k8s/service.yaml`: `LoadBalancer`, port **8080** (see troubleshooting below for why
  not 80).
- GHCR package visibility set to **Public** so the cluster can pull the image without
  managing an `imagePullSecret`/PAT.

### CD / GitOps flow

**Architectural constraint**: GitHub's standard cloud runners can't reach a local
Kubernetes cluster. Solved by registering this machine as a **self-hosted GitHub Actions
runner**, rather than settling for a manual deploy script — this gives a genuinely
automatic push-to-deploy flow.

Added a `deploy` job to the same workflow: `needs: build-and-push`, `runs-on:
self-hosted`, gated to `push` events on `main` only (never `pull_request` — a
self-hosted runner shouldn't execute arbitrary PR-triggered code). It applies the
manifests, then `kubectl set image` to the exact `:${{ github.sha }}` tag (guarantees a
real rollout every time, avoiding the classic "K8s doesn't notice `:latest` changed"
trap), waits for rollout completion, then runs a post-deploy smoke test (`/health` + a
real `/predict` call using a fixture image committed to git).

### Smoke test — verified to actually fail the pipeline

Ran a deliberate negative test: pointed the smoke test at a wrong port, pushed, and
confirmed the real rollout succeeded fine (proving the smoke test failure is isolated)
while the smoke test step itself correctly failed, which correctly failed the whole
`deploy` job and workflow run — with the live service completely unaffected throughout.
This is verified evidence, not just code that claims to, that the pipeline fails closed
per the assignment's explicit requirement. Reverted immediately after confirming.

### Troubleshooting notes

- **Pod startup/liveness crash loop**: pods got stuck `ContainerCreating` → briefly
  `Running` → killed and restarted, forever. Root cause: the container takes longer than
  the liveness probe's default tolerance to finish importing PyTorch and loading the
  model (worse under the CPU resource limit), so Kubernetes killed it for being
  "unhealthy" before it had even finished starting. **Fix**: added a `startupProbe`
  (up to 150s grace before liveness/readiness probes even begin evaluating) — the
  K8s-native solution for slow-starting containers. Also added `PYTHONUNBUFFERED=1` to
  the Dockerfile (logs were showing up empty during debugging, since Python buffers
  stdout by default when not attached to a TTY).
- **Service routing conflict**: after the startup fix, `curl http://localhost/health`
  returned a *different* app's response entirely. Root cause: Assignment 1's
  `heart-disease-api-service` is still running on this same shared local cluster and
  already occupied host port 80 as a `LoadBalancer`. **Fix**: moved this project's
  service to port 8080 instead of touching prior work. A related quirk: after editing the
  port, the service stayed `EXTERNAL-IP: <pending>` until deleted and recreated —
  Docker Desktop's LoadBalancer controller apparently only assigns an external IP at
  object *creation*, not on in-place updates.
- **Self-hosted Windows runner shell issues** (several compounding, in order hit):
  a bare `shell: bash` resolved to the WSL bash shim (`C:\Windows\System32\bash.exe`)
  rather than Git Bash, and WSL's isolated network namespace made `127.0.0.1` inside it
  not the same as the Windows host's loopback — causing `kubectl` connection timeouts.
  Pinning bash to its full Git install path then hit an unrelated GitHub Actions runner
  bug parsing absolute Windows paths with spaces in a custom `shell:` value. **Fixed by
  switching to PowerShell** (the unambiguous native shell on Windows) instead of fighting
  bash further — which then surfaced two more environment issues to fix: PowerShell's
  default execution policy blocking the runner's generated `.ps1` step scripts (fixed
  with `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`), and the
  repo's Actions permissions being set to the most restrictive option ("Allow
  MadihaFathima actions only"), which blocked even GitHub's own official actions (fixed
  by whitelisting `actions/checkout@*`, `actions/setup-python@*`, `docker/login-action@*`
  explicitly). Finally, `kubectl rollout status` needed a longer timeout (600s) than the
  default 300s, since a full 2-replica rolling update with the slow-startup grace period
  genuinely takes several minutes.

### Self-hosted runner security

Reviewed GitHub's self-hosted-runner warning for public repos (risk: a malicious fork PR
running code on this machine). The `deploy` job is gated to `push` events on `main` only,
never `pull_request`, so PR-triggered workflows never reach the self-hosted runner
regardless of what a PR's diff contains. Confirmed the repo already has the strongest
available mitigation enabled — "Require approval for all external contributors" for fork
PR workflows — so no outside PR runs anything without a manual approval click first.

---

## M5 — Monitoring, Logs & Final Submission

### Basic monitoring & logging

- `src/api/main.py` gained a middleware (`log_and_measure_requests`) logging method,
  path, status code, and latency for every request via Python's `logging` module —
  metadata only, never request/response bodies (excludes sensitive data).
- Prometheus metrics exposed at `GET /metrics` via `prometheus_client`:
  `api_requests_total` (by method/path/status), `api_request_latency_seconds`
  (histogram, by path), `predictions_total` (by predicted label).
- `monitoring/prometheus.yml` scrapes the deployed service at
  `host.docker.internal:8080`. Verified with a standalone Prometheus container — target
  shows `UP` at `http://localhost:9090/targets`.

### Model performance tracking (post-deployment)

- `scripts/post_deploy_eval.py`: sends a small batch of **real HTTP requests** (test
  images with known true labels) to the live deployed `/predict` endpoint, compares
  predictions against ground truth, reports accuracy, writes a full report to
  `outputs/post_deploy_eval.json`. Verified working against the live service.

**Troubleshooting**: `/metrics` returned 404 against a locally-running dev server despite
the route clearly existing in the source and no import errors on startup — the app's own
`/openapi.json` confirmed the running process's route table didn't match the file on
disk (a stale/orphaned process holding the port). Resolved by a full machine restart, not
pursued further since it was purely a local dev-server hygiene issue, not a code bug —
confirmed by full success on the actually-deployed K8s service throughout.

---

## Status

All 5 modules (M1–M5) implemented and verified against the real running system — not
just written. See [README.md](README.md) for setup, module-by-module commands, and
project structure.
