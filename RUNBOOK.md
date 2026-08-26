# RUNBOOK — Cats vs Dogs MLOps Pipeline

Step-by-step record of what has been done, why, and how to reproduce it.
Course: MLOps (AIMLCZG523) — Assignment 2.

## Tooling decisions

| Area | Choice | Why |
|---|---|---|
| Experiment tracking | MLflow (local) | Open-source, no account needed, matches Assignment 1 |
| ML framework | PyTorch | Clean serialization (`.pt`), integrates well with FastAPI |
| Dataset versioning | DVC, local folder remote | No cloud account needed; demonstrates DVC mechanics |
| CI/CD | GitHub Actions | Matches Assignment 1, free, integrates with GHCR |
| Deployment target | Docker Desktop's built-in Kubernetes | Reuses the working `docker-desktop` kubectl context and image-resolution approach from Assignment 1, instead of standalone minikube |
| Local Python | 3.11.9 (via `py -3.11`) | Only Python 3.14 was present on the machine; PyTorch/TensorFlow don't yet ship wheels for 3.14. Docker image also pins `python:3.11-slim` for consistency (matches Assignment 1) |

## Environment setup

- Verified installed tools: Docker 29.6.1, Git 2.54.0, kubectl v1.36.1, minikube v1.38.1 (not used — see deployment decision above).
- Installed Python 3.11.9 via `winget install --id Python.Python.3.11 -e`, alongside the existing Python 3.14, to get a version compatible with PyTorch/DVC/MLflow.
- Created project folder structure: `src/`, `data/`, `notebooks/`, `tests/`, `docker/`, `k8s/`.
- Created a virtual environment at `venv/` using Python 3.11: `py -3.11 -m venv venv`.

## Git & DVC setup (M1 Task 1 — Data & Code Versioning)

1. `git init` in the project root.
2. Configured global git identity:
   ```
   git config --global user.name "Madiha Fathima"
   git config --global user.email "madihafathima10@gmail.com"
   ```
3. Created root `.gitignore` (`venv/`, `__pycache__/`, `*.pyc`, `.env`, `mlruns/`).
4. Installed DVC into the venv: `pip install dvc` (v3.67.1).
5. `dvc init` — created `.dvc/config`, `.dvc/.gitignore`, `.dvcignore`.
6. Added a **local folder DVC remote** (outside the repo, so it doesn't get committed to git):
   ```
   dvc remote add -d localstorage D:/Bits-SEM3/dvc-storage/mlops_assignment_2
   ```
7. Committed DVC config to git (commit `86108ba`, "Initialize DVC with local storage remote").

Note on `.gitignore` files: DVC auto-generates its own `.gitignore` entries next to
whatever it tracks (`.dvc/.gitignore` for its internal state, `data/raw/.gitignore` once
`dvc add` runs on a folder there). These are left as DVC manages them — merging them into
the root `.gitignore` would just get overwritten/fought on the next `dvc add`. The root
`.gitignore` is reserved for everything DVC doesn't touch (venv, caches, env files).

## Dataset acquisition

- Dataset: **Microsoft Cats vs Dogs** (`shaunthesheep/microsoft-catsvsdogs-dataset` on
  Kaggle) — the original dataset behind the classic Kaggle "Dogs vs. Cats" competition.
  Chosen over pre-split alternatives because it's raw/unsplit, matching the assignment's
  instruction to preprocess and split the data ourselves.
- Kaggle auth: used Kaggle's newer **API token** method (not the legacy `kaggle.json`).
  Token stored as a persistent Windows user environment variable `KAGGLE_API_TOKEN`,
  which the `kaggle` Python package (v2.2.4) reads natively.
- Downloaded via:
  ```
  kaggle datasets download -d shaunthesheep/microsoft-catsvsdogs-dataset -p data/raw --unzip
  ```
- Result: `data/raw/PetImages/Cat/` and `data/raw/PetImages/Dog/`, 25,000 images total
  (12,500 per class), ~826MB. Also included: a Microsoft license doc and a readme, which
  landed outside `PetImages/`.
- **Known issue to handle during preprocessing**: this dataset has a small number of
  corrupt/zero-byte JPEG files (long-documented quirk of the original Microsoft release,
  reflected in the `.dvc` pointer file recording 25,002 files vs the expected 25,000
  images — 2 extra non-image files were bundled inside `PetImages/`). Must filter these
  out before training.

## Dataset versioning with DVC

1. `dvc add data/raw/PetImages` — hashed all files, created `data/raw/PetImages.dvc`
   pointer file, moved actual data into DVC's cache, auto-updated `data/raw/.gitignore`.
2. `dvc push` — copied the cached data to the local remote
   (`D:/Bits-SEM3/dvc-storage/mlops_assignment_2`).
3. Committed the pointer file to git:
   ```
   git add data/raw/PetImages.dvc data/raw/.gitignore
   git commit -m "Track raw Cats vs Dogs dataset with DVC"
   ```
   (commit `5256604`)
4. Committed the license/readme files from the Kaggle zip directly to git (not DVC —
   they're tiny text/docx files, not bulk data):
   ```
   git add "data/raw/MSR-LA - 3467.docx" "data/raw/readme[1].txt"
   git commit -m "Add dataset license and readme from Kaggle source"
   ```
   (commit `d4b5ba8`)
5. Verified: `dvc status` reports "Data and pipelines are up to date."

## Git log so far

```
d4b5ba8 Add dataset license and readme from Kaggle source
5256604 Track raw Cats vs Dogs dataset with DVC
86108ba Initialize DVC with local storage remote
```

## M1 Task 2 design decisions (Model Building)

- **CPU-only training** — no NVIDIA GPU available, so the baseline CNN is deliberately
  shallow (3 conv+pool blocks + small FC head) rather than a deep architecture.
- **Subsampling**: full 25,000-image raw dataset stays DVC-tracked as-is, but training
  uses a balanced subsample of **4,000 images (2,000 cat + 2,000 dog)**, split 80/10/10
  → 3,200 train / 400 val / 400 test. Chosen to keep CPU training time reasonable (an
  engineering tradeoff, not a shortcut — documented here for the report).
- **Preprocessing materializes real files**: resized 224x224 RGB JPEGs are written to
  `data/processed/{train,val,test}/{cat,dog}/`, not just resized on-the-fly — this gives
  a real artifact to `dvc add`, matching the assignment's "track pre-processed data"
  wording.
- **Augmentation** is applied on-the-fly during training (torchvision transforms: random
  flip/rotation) on the train split only — not baked into the saved processed files. This
  is standard practice (avoids inflating stored data, keeps val/test deterministic).
- **Package installs** (run manually by user in `venv`, not via assistant-run commands —
  see workflow note below):
  ```
  venv\Scripts\python.exe -m pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
  venv\Scripts\python.exe -m pip install pillow scikit-learn matplotlib mlflow
  ```

## M1 Task 2 execution — preprocessing

- Created `requirements.txt` with pinned versions for all key libraries (torch/torchvision
  installed separately from the PyTorch CPU wheel index, since PyPI's default index only
  hosts CUDA builds).
- Wrote `src/data/preprocess.py`:
  - `is_valid_image()` — filters corrupt/zero-byte files via Pillow open+verify.
  - `list_valid_images()`, `split_paths()`, `save_resized()`, `process_class()` — pure,
    independently testable helpers (used for M3 unit tests later).
  - Run via `python -m src.data.preprocess` (not as a plain file path) so that
    `src`/`src.data` resolve correctly as packages and the project root is on `sys.path` —
    needed once other modules start importing from `src.data.*`.
  - Added `src/__init__.py`, `src/data/__init__.py` to make them proper packages.
- Ran it: scanned `data/raw/PetImages/{Cat,Dog}` (25,000 raw files), filtered corrupt
  ones, subsampled 2,000 valid images per class (seed=42), split 80/10/10, resized to
  224x224 RGB, wrote to `data/processed/{train,val,test}/{cat,dog}/`.
- Verified output: 1,600/1,600 train, 200/200 val, 200/200 test (cat/dog), total 4,000
  files; spot-checked a sample image is exactly `(224, 224)` in `RGB` mode.
- DVC-tracked the result:
  ```
  dvc add data/processed
  dvc push
  ```
  Pointer file `data/processed.dvc` records 4,000 files, ~37.6MB.
- Committed code + processed-data pointer to git (commit `d3d370a`,
  "Add preprocessing script; DVC-track processed dataset").

## M1 Task 2 execution — baseline model + training script

- Wrote `src/models/model.py` — `SimpleCNN`: 3 conv+ReLU+maxpool blocks (3→32→64→128
  channels), `AdaptiveAvgPool2d((7,7))` to keep the flattened feature size small
  regardless of input resolution, then a small FC head (6272→128→1) outputting a single
  logit for `BCEWithLogitsLoss`. Deliberately shallow given CPU-only training.
- Wrote `src/models/train.py`:
  - `build_dataloaders()` — `torchvision.datasets.ImageFolder` over
    `data/processed/{train,val,test}`; class-to-index mapping is alphabetical, so
    `cat=0`, `dog=1`. Train loader gets `RandomHorizontalFlip` + `RandomRotation(15)`
    augmentation; val/test loaders use a plain (no-augmentation) transform, both use
    ImageNet mean/std normalization.
  - `run_epoch()` — shared train/eval step (acts as trainer when given an optimizer,
    evaluator otherwise), returns avg loss + accuracy.
  - Trains for `--epochs` (default 10), logging train/val loss & accuracy to MLflow every
    epoch; evaluates on the test set at the end (accuracy, precision, recall, F1,
    confusion matrix via scikit-learn).
  - Saves a loss-curve plot and confusion-matrix plot under `outputs/`, logs both as
    MLflow artifacts (satisfies M1 Task 3's "confusion matrix, loss curves" requirement).
  - Saves the trained model to `models/cnn_baseline.pt` (`torch.save(state_dict)`), also
    logs it via `mlflow.pytorch.log_model` for the run's model artifact/registry entry.
  - Writes `models/model_metadata.json` (run ID + test metrics) for later use when
    packaging the model for serving in M2.
  - MLflow experiment name: `cats-vs-dogs-classification` (mirrors Assignment 1's
    `heart-disease-classification` naming convention). Local file-based tracking
    (`./mlruns`, gitignored), viewable via `mlflow ui --backend-store-uri ./mlruns`.
  - Run via `python -m src.models.train` for the same package-resolution reason as
    preprocessing.
- **Bug hit on first run**: `mlflow.pytorch.log_model(model, "model")` failed after all
  10 epochs completed — this MLflow version (3.15.1) defaults to PyTorch's `pt2`
  (torch.export trace-based) serialization format, which requires an example input
  tensor to trace the model graph. Training itself, the loss/confusion-matrix plots, and
  `models/cnn_baseline.pt` had already been saved/logged successfully before this failed
  step; only the MLflow "logged model" entry and `model_metadata.json` were missed on
  that run (recorded as a `FAILED` run in `mlruns/`, harmless, left as-is).
  Fix: pass `input_example=example_input[:1].numpy()` (one sample from the test loader)
  to `log_model`, and switched to the `name=` kwarg (`artifact_path` is deprecated in
  this version).
- **Re-ran successfully.** Results (run `9a622c5b19bb4e8dab8280c7fb374abf`, MLflow
  experiment id `1`):

  | Metric | Value |
  |---|---|
  | Test accuracy | 0.7625 |
  | Test precision | 0.7561 |
  | Test recall | 0.7750 |
  | Test F1 | 0.7654 |

  Loss curve shows normal train/val convergence with mild overfitting starting around
  epoch 6-7 (train loss keeps dropping to ~0.44 while val plateaus ~0.49) — expected and
  acceptable for a shallow baseline on a 3,200-image training set. Confusion matrix is
  reasonably balanced (150/50 cat, 45/155 dog — no severe class bias).
- Model versioning decision: `models/cnn_baseline.pt` (3.5MB) and
  `models/model_metadata.json` are committed **directly to git**, not DVC — small enough
  that git handles it fine, and keeps the model directly visible/downloadable in the repo
  for M2 packaging (Docker can `COPY` it directly) without a `dvc pull` step.
- `mlruns/` (MLflow's local tracking store) stays gitignored — not meant for git (grows
  every run, best viewed via `mlflow ui --backend-store-uri ./mlruns`), but will be
  included in the final zip deliverable so a grader can browse run history without
  retraining.

## MLflow UI — filestore "maintenance mode" + split storage (troubleshooting notes)

**First symptom**: `mlflow ui --backend-store-uri ./mlruns` failed outright:
```
MlflowException: The filesystem tracking backend (e.g., './mlruns') is in maintenance
mode and will not receive further updates. ... set MLFLOW_ALLOW_FILE_STORE=true to opt
out of this exception.
```
Worked around with `$env:MLFLOW_ALLOW_FILE_STORE="true"` — the UI then started, but the
experiment showed as **empty** (no runs), even though training had clearly succeeded.

**Real root cause**: `train.py` never explicitly sets `mlflow.set_tracking_uri(...)`.
MLflow 3.x's implicit defaults turned out to be *split*: tracking metadata (experiments,
runs, params, metrics) went to a local SQLite file, **`mlflow.db`**, auto-created in the
project root — while artifacts (`log_artifact`/`log_model` files) still went to
`./mlruns/<experiment_id>/<run_id>/artifacts/` as before. Pointing the UI at `./mlruns`
only searches that folder for run metadata (`meta.yaml`, `params/`, `metrics/`), finds
none there (confirmed: those run folders contained only an `artifacts/` subfolder), and
reports empty — despite the artifact files genuinely being present.

**Fix**: point the UI at the SQLite store instead, which is where the real metadata is
(confirmed via a direct `sqlite3` query showing the `cats-vs-dogs-classification`
experiment and all 3 runs, including the successful `FINISHED` one):
```
venv\Scripts\python.exe -m mlflow ui --backend-store-uri sqlite:///mlflow.db
```
No `MLFLOW_ALLOW_FILE_STORE` needed for this path — SQLite is the backend MLflow
actually wants. `mlflow.db` is gitignored (regenerated locally by re-running training;
not meant for git). **This is now the canonical command to view experiments for this
project.**

**Verified in the UI**: run `puzzled-colt-687` (`9a622c5b19bb4e8dab8280c7fb374abf`) shows
correct Parameters (architecture, epochs, batch_size, lr, split sizes), Metrics
(per-epoch train/val loss & accuracy, final test accuracy/precision/recall/F1), and
Artifacts (`confusion_matrix.png`, `loss_curve.png`, `cnn_baseline.pt`,
`model_metadata.json`, and a `model/` folder with the MLflow-logged model — MLmodel
manifest, conda/python env specs, input examples). **M1 confirmed fully complete.**

## M1 — complete

All three M1 tasks (Data & Code Versioning, Model Building, Experiment Tracking) are
done and verified: git + DVC versioning in place, baseline CNN trained (76.25% test
accuracy) and saved, MLflow tracking confirmed working end-to-end in the UI.

## Next up (not yet done)

- M2: Inference Service — FastAPI wrapper around `cnn_baseline.pt` with `/health` and
  `/predict` endpoints, then Dockerfile + local build/run verification.

## M2 Task 1 — Inference Service (FastAPI)

- Refactored transforms into `src/data/transforms.py` (`get_train_transform()`,
  `get_eval_transform()`, shared `CLASS_NAMES = ["cat", "dog"]`) so training and
  inference use identical preprocessing — avoids train/serve skew. `get_eval_transform()`
  adds a `Resize(224, 224)` step (a no-op on already-224x224 processed files, but
  necessary for arbitrary-sized images arriving via the API). `train.py` updated to
  import from here instead of defining transforms inline.
- `src/api/inference.py` — `load_model(path)` and `predict(model, image) -> dict`, kept
  separate from the FastAPI layer so they're unit-testable without a running server
  (target for M3's "model utility/inference function" test).
- `src/api/main.py` — FastAPI app, model loaded once at startup via a `lifespan` context
  manager (not per-request):
  - `GET /health` → `{"status": "healthy"}`
  - `POST /predict` → multipart file upload (`UploadFile`), validates content-type and
    that the bytes decode as an image, returns `{label, probability, class_probabilities}`
  - `MODEL_PATH` overridable via env var, defaults to `models/cnn_baseline.pt`
  - Request/response logging and metrics deliberately deferred to M5 (Monitoring), to
    keep M2 scoped to the two required endpoints + packaging.
- Added `python-multipart` to `requirements.txt` (required by FastAPI for file uploads).
- Created `docker/requirements.txt` — leaner, inference-only dependency list (numpy,
  pillow, fastapi, uvicorn, pydantic, python-multipart; torch/torchvision installed
  separately from the CPU wheel index). Excludes dvc/mlflow/matplotlib/scikit-learn/pytest
  to keep the eventual container image smaller.
- **Verified locally**: ran `uvicorn src.api.main:app --reload --port 8000`, tested both
  endpoints with `curl`.
  - `/health` returned `{"status": "healthy"}`.
  - `/predict` on `data/processed/test/cat/cat_00000.jpg` returned `label: "dog"`
    (dog_prob=0.6448) — initially looked like a bug, but cross-checked against the exact
    same evaluation pipeline `train.py` used to measure 76.25% test accuracy (loading the
    same image via `ImageFolder` + `get_eval_transform()`) and got the **identical**
    dog_prob=0.6448. Confirms the API reproduces the model's real behavior with no
    train/serve skew — this image is simply one of the ~50/200 test cats the baseline
    model gets wrong, consistent with the known confusion matrix. Not a bug.
  - `/predict` on other test images (dogs, other cats) returned correct labels with
    reasonable confidence.

## M2 Task 3 — Containerization

- Created `.dockerignore` (excludes `venv/`, `data/` [826MB], `mlruns/`, `.git/`, `.dvc/`,
  `notebooks/`, `outputs/`, `tests/`) — without it, `docker build`'s build context would
  include the entire raw dataset and venv, extremely slow to send to the daemon.
- `docker/Dockerfile`: `python:3.11-slim` base, installs CPU torch/torchvision from the
  PyTorch wheel index as its own layer *before* copying app code (so code changes don't
  invalidate the large PyTorch download layer on rebuild), then `docker/requirements.txt`,
  then copies only `src/` and `models/cnn_baseline.pt` (not training data or scripts'
  other dependencies). `HEALTHCHECK` polls `/health` every 30s. Runs via
  `uvicorn src.api.main:app --host 0.0.0.0 --port 8000`.
- **Troubleshooting**: first build attempt failed —
  `failed to connect to the docker API at npipe:////./pipe/dockerDesktopLinuxEngine` —
  because Docker Desktop wasn't running (the CLI was installed, but its background engine
  needs the Docker Desktop application launched). Started Docker Desktop, waited for
  "Engine running," confirmed via `docker info`, then proceeded.
- **Built and verified successfully**:
  ```
  docker build -f docker\Dockerfile -t cats-dogs-api:latest .
  docker run -d --name cats-dogs-api -p 8000:8000 cats-dogs-api:latest
  ```
  Image: `cats-dogs-api:latest`, 327MB content size. Container status confirmed
  `Up ... (healthy)` — i.e. the Dockerfile's `HEALTHCHECK` itself is passing, not just
  that the process started. `curl http://localhost:8000/health` →
  `{"status":"healthy"}`. `curl -X POST -F "file=@..." http://localhost:8000/predict` on
  a dog test image → correctly returned `label: "dog"` with 66% confidence.

## M2 — complete

All three M2 tasks done and verified: FastAPI inference service with `/health` +
`/predict`, pinned dependencies (root `requirements.txt` for dev/training, leaner
`docker/requirements.txt` for the inference image), Dockerfile built and run locally with
predictions verified via curl against the actual running container.

## M3 Task 1 — Automated Testing

- `tests/test_preprocess.py` (8 tests) — `is_valid_image()` against real/corrupt/empty/
  missing files (using `tmp_path` fixtures, not the real dataset); `split_paths()` for
  correct ratios, no overlap between splits, full coverage, determinism given a seed.
- `tests/test_inference.py` (4 tests) — `predict()` against synthetic PIL images (not
  real cat/dog photos, so tests don't depend on the dataset): output structure/keys,
  probabilities sum to 1, handles non-square/non-224 images and grayscale input. Skips
  gracefully if `models/cnn_baseline.pt` isn't present rather than failing.
- All 12 tests pass via `python -m pytest tests/ -v`.

## GitHub repository

- Remote: `https://github.com/MadihaFathima/bits_mlops_assignment_2.git`
- Renamed local branch `master` → `main` to match GitHub Actions / Assignment 1
  convention. Pushed all commits through M3 Task 1.

## M3 Task 2/3 — CI Pipeline (GitHub Actions) + Registry

- `.github/workflows/ci.yml`, triggered on push/PR to `main`. Two jobs:
  - **`test`**: checkout → Python 3.11 → install CPU torch/torchvision + `requirements.txt`
    → `pytest tests/ -v`.
  - **`build-and-push`** (`needs: test`, only runs if tests pass): checkout → `docker
    build` → **smoke-test the image locally first** (run container, poll `/health` up to
    15x with 2s delay, fail the job with container logs if it never comes up healthy) →
    only then log in to GHCR and push. A broken image never gets published — same
    "test before publish" pattern as Assignment 1's `docker-build-smoke-test` job.
  - Registry: **GitHub Container Registry** (`ghcr.io/madihafathima/bits_mlops_assignment_2`).
    Auth via the automatically-provided `GITHUB_TOKEN` (job-level
    `permissions: packages: write`) — no manually-configured secrets needed, unlike
    Docker Hub.
- **Key design decision**: CI does *not* run `dvc pull` or retrain the model. Our DVC
  remote is a local Windows folder (`D:\Bits-SEM3\dvc-storage\...`), unreachable from
  GitHub's cloud runners. This isn't a gap: `models/cnn_baseline.pt` is committed
  directly to git (per the M1 decision), so `actions/checkout` alone gives CI everything
  it needs — tests use synthetic images, not the real dataset, and the Docker build only
  needs `src/` + the committed model file. Training/data-versioning stays a local
  (DVC-tracked) concern; CI's job is build/test/package/publish.

**Troubleshooting**: first CI run failed both test collection with
`ModuleNotFoundError: No module named 'src'`. Cause: the workflow's test step ran the
bare `pytest tests/ -v` command, not `python -m pytest`. As established earlier in this
project (see the M1 preprocessing notes), only `-m pytest` adds the project root to
`sys.path`, which `from src.data.preprocess import ...`-style absolute imports need —
running `pytest` directly instead only adds the test file's own directory. Same class of
bug as the earlier "why `-m src.data.preprocess` and not the file path" question, just
hitting `pytest` this time instead of a training/preprocessing script. Fixed by changing
the workflow step to `python -m pytest tests/ -v`.

Confirmed working after the fix: both `test` and `build-and-push` jobs pass on GitHub
Actions.

## M3 — complete

All three M3 tasks done: 12 pytest unit tests (preprocessing + inference), CI pipeline on
GitHub Actions (checkout → install → test → build → smoke-test → push), image published
to GHCR automatically on every push to `main`.

## M4 Task 1 — Deployment Target & Manifests

- **Architectural constraint identified up front**: the deployment target is Docker
  Desktop's *local* Kubernetes cluster on this Windows machine. GitHub Actions' standard
  cloud runners are ephemeral VMs with no network path to a local cluster — a normal
  workflow job cannot run `kubectl apply` against it. Resolved by deciding to register
  this machine as a **self-hosted GitHub Actions runner** (see below), rather than
  settling for a manual/documented deploy script — chosen because it gives a genuinely
  automatic push-to-deploy flow, which matters both for meeting the assignment's
  "automatically on main branch changes" wording and for a much better demo video.
- GHCR package visibility set to **Public** (via package settings → Danger Zone → Change
  visibility) — avoids needing to manage an `imagePullSecret`/PAT for the cluster to pull
  the image; acceptable since the image contains no sensitive data (model weights +
  inference code only).
- `k8s/deployment.yaml`: 2 replicas, resource requests/limits (250m/500m CPU,
  256Mi/512Mi memory), `imagePullPolicy: Always` (so redeploys actually re-pull rather
  than reuse a stale cached `:latest`).
- `k8s/service.yaml`: `LoadBalancer` type, originally `port: 80`.
- Confirmed Docker Desktop's Kubernetes was running (`kubectl cluster-info` reachable at
  `docker-desktop` context) before deploying.

**Troubleshooting — pod startup/liveness crash loop**: first deploy attempt showed pods
stuck `ContainerCreating` then `Running` but never `1/1` ready, eventually killed and
restarted by the liveness probe (`kubectl describe pod` showed
`Liveness probe failed: ... connection refused`, then `Killing ... will be restarted`).
Root cause: the container takes longer than the liveness probe's ~50s tolerance
(20s initial delay + 3×10s failures) to finish importing PyTorch and loading the model —
made worse by the 500m CPU limit throttling that import. Kubernetes killed the container
for being "unhealthy" before it had even finished starting, then repeated the cycle
forever. Fixed properly (not just by loosening the existing probes) by adding a
**`startupProbe`** (`periodSeconds: 5, failureThreshold: 30` → up to 150s grace before
liveness/readiness probes even begin evaluating) — the K8s-native solution for
slow-starting containers, since it doesn't permanently weaken liveness responsiveness
once the app is actually up. Also added `PYTHONUNBUFFERED=1` to the Dockerfile — logs
were showing up empty during debugging because Python buffers stdout by default when not
attached to a TTY, which made this issue harder to diagnose than necessary.

**Troubleshooting — service routing conflict**: after the startup fix, pods reached
`1/1 Running`, but `curl http://localhost/health` returned `{"status":"ok"}` (not our
app's actual `{"status":"healthy"}`) and `/predict` errored. Root cause: Assignment 1's
`heart-disease-api-service` is still running on this same shared local cluster and is
*also* a `LoadBalancer` on port 80 — `kubectl get svc -o wide` confirmed it already held
the external IP, while `cats-dogs-api-service` sat at `EXTERNAL-IP: <pending>`, unable to
bind host port 80. Fixed by changing our service to **port 8080** instead (not by
touching/stopping the Assignment 1 service, which is prior graded work). Even after that
edit, the service stayed `<pending>` — `kubectl apply` performed an in-place update, but
Docker Desktop's lightweight LoadBalancer controller apparently only assigns an external
IP at object *creation*, not on updates. Fixed by `kubectl delete -f k8s/service.yaml`
then `kubectl apply -f k8s/service.yaml` (recreate from scratch), which got a real
external IP (`172.18.0.6`) immediately. Verified via
`curl http://localhost:8080/health` → `{"status":"healthy"}` and a correct `/predict`
result against the real service.

## M4 Task 2 — Self-hosted runner + automated deploy job

- Registered this Windows machine as a GitHub Actions self-hosted runner (repo Settings
  → Actions → Runners → New self-hosted runner → downloaded/configured/ran per GitHub's
  generated instructions). Attempted to install as a Windows service (needs admin
  privileges); fell back to running interactively via `./run.cmd` in an open terminal
  ("Listening for Jobs") — sufficient for now, though it only listens while that terminal
  stays open. Revisit installing as a persistent service later if needed.
- Added a `deploy` job to `.github/workflows/ci.yml`: `needs: build-and-push`,
  `runs-on: self-hosted`, gated to `push` events on `main` only (not PRs — a self-hosted
  runner shouldn't execute arbitrary PR-triggered code against local infra). Applies
  `k8s/deployment.yaml`/`service.yaml`, then `kubectl set image` to the exact
  `:${{ github.sha }}` tag (guarantees a real rollout every time, avoiding the classic
  "K8s doesn't notice `:latest` changed" trap), `kubectl rollout status`, then a
  post-deploy smoke test (`/health` + a real `/predict` call using a small fixture image
  committed directly to git at `tests/fixtures/sample_dog.jpg`, since a workflow step
  shouldn't depend on pulling DVC-tracked data) — **fails the job** if either check fails,
  satisfying the assignment's explicit requirement.

**Troubleshooting — stale/uncommitted files caught by CI, not caught locally**:
1. First run failed both jobs with `ModuleNotFoundError: No module named 'src'` in
   `test` — same root cause as the earlier preprocessing `-m` question: the CI step ran
   bare `pytest`, not `python -m pytest`. Fixed.
2. Discovered mid-debugging that `k8s/service.yaml` had a **local uncommitted edit**
   (still committed as `port: 80`, the version that caused the Assignment-1 port
   conflict) — the live cluster had been fixed manually via `kubectl apply`/`delete`
   earlier, but the *file* fix was never actually committed. This would have made the
   `deploy` job re-introduce the exact port-80 conflict on every automated deploy. Caught
   via `git status`/`git show HEAD:... ` comparison before it could bite; committed the
   real fix.
3. A workflow run appeared "not reflecting" on the Actions UI — turned out to be a
   genuine (if confusing) sequence of a hung GitHub-hosted cleanup step
   ("Post Run actions/checkout@v4" stuck `in_progress` for several minutes on the
   `build-and-push` job, unrelated to this project's code — a rare cloud-runner
   infrastructure hiccup) blocking the dependent `deploy` job from starting. Verified via
   GitHub's REST API directly (`/actions/runs`, `/actions/runs/{id}/jobs`) rather than
   trusting the browser UI, which helped distinguish "not triggered" from "triggered but
   stuck."
4. **Real bug, the interesting one**: once `deploy` finally ran, `kubectl apply` failed
   with a raw Windows socket timeout (`Error code: Bash/Service/0x8007274c`) — but
   `kubectl cluster-info` succeeded fine when run manually in both PowerShell and Git
   Bash on the same machine immediately after. Root cause found by expanding the full
   step log: the workflow's `shell: bash` default resolved to
   **`C:\Windows\System32\bash.exe`** (the WSL launcher shim), not real Git Bash —
   visible only in the step's `shell:` metadata line in the full log, not the collapsed
   summary. WSL runs in its own isolated network namespace; `127.0.0.1:<port>` inside WSL
   is not the Windows host's `127.0.0.1` that Docker Desktop's Kubernetes API server
   binds to, so `kubectl` timed out trying to reach it from inside WSL bash. This
   machine has three `bash.exe` on PATH (Git Bash, the WSL shim, and a WindowsApps
   stub) — the WSL one apparently wins when only the bare name `bash` is given. **Fixed**
   by pinning the `deploy` job's shell explicitly to
   `C:\Program Files\Git\bin\bash.exe --noprofile --norc -e -o pipefail {0}` instead of
   the ambiguous `shell: bash`.

5. Pinning bash to its full Git install path (`C:\Program Files\Git\bin\bash.exe`) hit a
   *different*, unrelated bug: `Error: Second path fragment must not be a drive or UNC
   name. (Parameter 'expression')` — a .NET exception from the GitHub Actions runner's
   own internal code (it's a C# process), which doesn't cleanly handle an absolute
   Windows path with spaces given as a custom `shell:` value. Rather than fight bash on
   Windows further, **switched the `deploy` job to PowerShell** (`shell: powershell`) —
   the native, unambiguous default shell on any Windows machine, sidestepping both the
   WSL-shim ambiguity and this path-parsing bug entirely. Rewrote the steps accordingly:
   used `${{ env.IMAGE_NAME }}:${{ github.sha }}` (a GitHub Actions expression, resolved
   before the shell sees it) instead of shell-level env var interpolation, since
   PowerShell treats `$Name:` as drive-qualified variable syntax; used `curl.exe`
   explicitly since PowerShell aliases bare `curl` to `Invoke-WebRequest` (different
   behavior entirely); added explicit `$LASTEXITCODE` checks after each `kubectl`/
   `curl.exe` call, since PowerShell — unlike bash — doesn't auto-fail a script when a
   native command returns non-zero.
6. PowerShell attempt then failed with
   `File ...\_temp\....ps1 cannot be loaded because running scripts is disabled on this
   system` (`PSSecurityException: UnauthorizedAccess`) — a stock Windows PowerShell
   execution-policy restriction (`Restricted` by default), which blocks running *any*
   `.ps1` file, including ones legitimately generated by the Actions runner itself for
   each step. Fixed with `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope
   CurrentUser` (no admin rights needed) — `RemoteSigned` is the standard safe middle
   ground: local/self-generated scripts run freely, downloaded scripts still need a
   trusted signature. Verified via `Get-ExecutionPolicy -List` that `CurrentUser` scope
   now shows `RemoteSigned`.
7. Next attempt failed differently:
   `The actions actions/checkout@v4, actions/setup-python@v5, and docker/login-action@v3
   are not allowed in MadihaFathima/bits_mlops_assignment_2 because all actions must be
   from a repository owned by MadihaFathima`. While reviewing the self-hosted-runner
   security settings (see below), the repo's Actions permissions had been set to "Allow
   MadihaFathima actions and reusable workflows" — the most restrictive option, which
   blocks literally any action not published under the repo owner's own account,
   including GitHub's and Docker's own official actions. Fixed by switching to "Allow
   MadihaFathima, and select non-MadihaFathima, actions and reusable workflows" and
   explicitly whitelisting `actions/checkout@*`, `actions/setup-python@*`,
   `docker/login-action@*` — permits exactly what's needed without opening the door to
   arbitrary third-party actions (relevant given the self-hosted runner's exposure).
8. With execution policy and action permissions both fixed, `kubectl apply`/
   `kubectl set image` finally succeeded and the rollout genuinely started (`image
   updated`, replicas progressing) — but `kubectl rollout status --timeout=300s` still
   timed out waiting on the last old replica to terminate. Consistent with the earlier
   manual test: a full 2-replica rolling update, each pod needing up to 150s for the
   `startupProbe` grace period plus Kubernetes' default one-at-a-time replacement
   behavior, genuinely takes longer than 5 minutes end-to-end. Increased to
   `--timeout=600s`.

## Self-hosted runner security review

While setting this up, reviewed GitHub's self-hosted-runner security warning (public
repos + self-hosted runners = risk of a malicious fork PR running code on this machine).
Findings: the `deploy` job is already gated to `push` events on `main` only (never
`pull_request`), so PR-triggered workflows never reach the self-hosted runner regardless.
Confirmed repo settings already had the strongest available mitigation — "Require
approval for all external contributors" under Actions → General → fork PR workflow
approval — so no PR from an outside contributor runs anything at all without a manual
approval click first. Considered making the repo private instead but kept it public
(zip file remains the primary grading deliverable either way; current settings already
minimize real exposure).

## Next up (not yet done)

- Push the rollout-timeout fix, confirm the `deploy` job succeeds end-to-end (manifests
  applied, image rolled out to the exact commit's tag, post-deploy smoke test passes).
- M4 Task 3 verification: confirm the pipeline actually fails if a smoke test fails
  (already implemented via `exit 1`, worth a deliberate negative test before final
  submission).
- M5: Monitoring, Logs & Final Submission.
