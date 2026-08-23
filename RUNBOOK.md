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

## MLflow UI — filestore "maintenance mode" (troubleshooting note)

Running `mlflow ui --backend-store-uri ./mlruns` with MLflow 3.15.1 (this project's
pinned version) fails with:
```
MlflowException: The filesystem tracking backend (e.g., './mlruns') is in maintenance
mode and will not receive further updates. ... set MLFLOW_ALLOW_FILE_STORE=true to opt
out of this exception.
```
Newer MLflow versions deprecated the plain filesystem backend for the **UI/server**
specifically — this does *not* affect the Python tracking client (`train.py` writes to
`./mlruns` without issue; only starting the UI server against it is restricted). Rather
than migrate to a SQLite backend (`sqlite:///mlflow.db`, MLflow's recommended path, but
requires changing `train.py`'s tracking URI and re-running training), we opted out via
the documented escape hatch, keeping the same local file-based setup as Assignment 1:
```
$env:MLFLOW_ALLOW_FILE_STORE="true"; venv\Scripts\python.exe -m mlflow ui --backend-store-uri ./mlruns
```
UI confirmed working at http://127.0.0.1:5000 after this.

## Next up (not yet done)

- M2: Inference Service — FastAPI wrapper around `cnn_baseline.pt` with `/health` and
  `/predict` endpoints, then Dockerfile + local build/run verification.
