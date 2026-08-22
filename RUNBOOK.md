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

## Next up (not yet done)

- M1 Task 1 (cont.): DVC-track the *preprocessed* data as a separate artifact once
  preprocessing exists (per assignment wording: "track pre-processed data").
- M1 Task 2: preprocessing script (resize to 224x224 RGB, filter corrupt files, 80/10/10
  split, train-only augmentation) + baseline PyTorch CNN + training script.
- M1 Task 3: MLflow experiment tracking (params, metrics, confusion matrix, loss curves).
