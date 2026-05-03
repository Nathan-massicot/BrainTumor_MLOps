# Brain Tumor MLOps — Detection & Localization

End-to-end MLOps pipeline for **brain tumor detection and localization** from MRI images. The project covers the full ML lifecycle: data ingestion, training, inference API, frontend, monitoring, drift detection, and continuous retraining.

> ⚠️ **Disclaimer**: this is an **academic / educational project**. It is **not intended for clinical use**. No medical validation, no CE/FDA certification.

---

## Context

- **Course**: MSc-level MLOps module — graded project requiring at least 3 MLOps tools.
- **Team**: Gabriel Gillmann · Helena Martínez Río · Nathan Massicot · Jahnavi Patil.
- **Dataset**: [LGG MRI Segmentation](https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation) (Kaggle).
- **Task**: binary classification (tumor / no tumor) on MRI slices, with localization as a stretch goal.
- **Model**: CNN with **transfer learning** (ResNet, EfficientNet, etc.).

---

## Tech Stack

| Layer | Tool |
|-------|------|
| Env & dependency management | **uv** (Astral) |
| ML framework | **PyTorch** + torchvision |
| Experiment tracking | **Weights & Biases** |
| Configs | **Hydra** / OmegaConf |
| Data versioning | **DVC** |
| API | **FastAPI** |
| Frontend | **Streamlit** (or React, TBD) |
| Containerization | **Docker** + docker-compose |
| CI/CD | **GitHub Actions** |
| Monitoring | **Prometheus** + **Grafana** |
| Drift detection | **Evidently AI** |
| Orchestration / retraining | **Prefect** (or Airflow) |
| Model Registry | **W&B Model Registry** |
| Testing | **pytest** + httpx |
| Lint / format | **Ruff** |
| Pre-commit | **pre-commit** + Conventional Commits |

---

## Project Structure

```
brain-tumor-mlops/
├── .github/workflows/        # CI/CD pipelines
├── configs/                  # Hydra configs (model, training, data, ...)
├── data/                     # DVC-tracked, never commit raw data
│   ├── raw/
│   └── processed/
├── docker/                   # per-service Dockerfiles
├── docs/                     # MkDocs documentation
├── frontend/                 # Streamlit / React UI
├── monitoring/               # Grafana dashboards, Prometheus config
├── notebooks/                # exploration / EDA — not for production code
├── pipelines/                # Prefect flows (training, retraining, drift)
├── src/brain_tumor_mlops/
│   ├── api/                  # FastAPI app, routes, schemas
│   ├── data/                 # datasets, transforms, loaders
│   ├── models/               # CNN architectures
│   ├── training/             # train/val loops
│   ├── inference/            # prediction logic
│   ├── monitoring/           # drift, metrics
│   └── utils/                # logging, helpers
├── tests/                    # pytest (unit + integration)
├── pyproject.toml
├── docker-compose.yml
├── dvc.yaml
├── .pre-commit-config.yaml
├── .env.example
├── CLAUDE.md                 # AI-assistant guidelines
└── README.md
```

**Rule**: all production code lives in `src/brain_tumor_mlops/`. Notebooks are for exploration only — never import from a notebook.

---

## Prerequisites

- **Python 3.12**
- **[uv](https://docs.astral.sh/uv/)**: `curl -LsSf https://astral.sh/uv/install.sh | sh`
- **Git**
- **Docker** + **docker-compose**
- A **Kaggle** account + a **Weights & Biases** account
- **GPU** recommended (CUDA / MPS) for training, optional for inference

---

## Setup (first time)

### 1. Clone the repo

```bash
git clone <repo-url>
cd brain-tumor-mlops
```

### 2. Install dependencies

```bash
uv sync
```

`uv` creates `.venv/` and installs everything (deps + dev) from `pyproject.toml` / `uv.lock`.

### 3. Configure environment variables

```bash
cp .env.example .env
```

Open `.env` and fill in:

- `WANDB_API_KEY` — get yours at https://wandb.ai/authorize
- `KAGGLE_USERNAME` + `KAGGLE_KEY` — at https://www.kaggle.com/settings → *Create New API Token*

### 4. Enable pre-commit hooks

```bash
pre-commit install
pre-commit install --hook-type commit-msg
```

### 5. Pull data via DVC

```bash
uv run dvc pull
```

If this is the **first init** (maintainer only):

```bash
uv run dvc init
uv run dvc remote add -d storage <remote-url>   # e.g. gdrive://..., s3://..., etc.
```

### 6. Configure Kaggle (alternative to `.env`)

```bash
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

---

## Common Commands

### Development

```bash
uv run pytest                                # all tests
uv run pytest tests/test_models.py -v        # one test file
uv run ruff check .                          # lint
uv run ruff format .                         # format
pre-commit run --all-files                   # run all hooks manually
```

### Training

```bash
# default config
uv run python -m brain_tumor_mlops.training.train

# Hydra overrides
uv run python -m brain_tumor_mlops.training.train model=resnet50 training.lr=1e-4

# hyperparameter sweep (W&B)
uv run wandb sweep configs/sweeps/lr_sweep.yaml
```

### API & Frontend

```bash
# FastAPI (dev mode)
uv run uvicorn brain_tumor_mlops.api.main:app --reload
# -> http://localhost:8000/docs

# Streamlit frontend
uv run streamlit run frontend/app.py
# -> http://localhost:8501

# Full stack
docker-compose up
```

### Data & Models

```bash
dvc add data/raw/lgg-mri              # version a new dataset
dvc push                              # push to remote storage
dvc repro                             # rerun pipeline if inputs changed
```

### Monitoring

```bash
docker-compose up prometheus grafana                          # monitoring stack
uv run python -m brain_tumor_mlops.monitoring.drift_check     # manual drift report
```

---

## Git Workflow

### Branches

- `main` — protected, production-ready, deployable at any time.
- `dev` — integration branch.
- `feature/<short-description>` — new features.
- `fix/<short-description>` — bug fixes.
- `docs/<short-description>` — documentation only.

### Conventional Commits

Format: `type(scope): description`

Types: `feat`, `fix`, `docs`, `test`, `refactor`, `perf`, `chore`, `style`.

```
feat(models): add ResNet50 transfer learning architecture
fix(api): handle empty image upload with 422 response
docs: add monitoring setup to README
test(data): add tests for patient-level train/val split
chore(deps): bump pytorch to 2.4.0
```

### Pull Requests

- Open PR against `dev`.
- Require ≥ 1 review from another team member.
- All CI checks must pass.
- **Squash merge** to keep history clean.
- Tasks are tracked via the **GitHub Project** linked to the repo.

---

## Critical Rules

### Data integrity

- **Patient-level splits only.** Never split by image — same patient in train and val = data leakage.
- **No raw data in Git.** Always use DVC.
- **No PHI** (protected health information) in commits, logs, or W&B.

### Models

- Models are versioned in **W&B Model Registry**, not Git.
- Promoting a model to production requires **human validation** (medical context — non-negotiable).
- Tag production releases with semver: `v1.0.0`, `v1.1.0`, etc.

### Secrets

- Never commit `.env`, API keys, tokens, passwords.
- `pre-commit` runs `detect-private-key` to catch accidents.
- If a secret leaks: rotate it immediately, then clean Git history.

### File size

- `pre-commit` blocks files > 500 KB.
- Large files (data, models, images) → DVC.

---

## Medical Metrics

Plain accuracy is misleading. Always report:

- **Sensitivity (recall)** — a false negative = missed tumor, the worst case.
- **Specificity** — avoid unnecessary false positives.
- **AUC-ROC** — overall classifier performance.
- **Confusion matrix** — always include in reports.

---

## Documentation for AI Assistants

The `CLAUDE.md` file at the repo root contains detailed guidelines for Claude (and other LLMs):
code conventions, MLOps rules, Git workflow, critical rules. Any assistant touching the repo should read it first.

---

## Useful Links

- **Dataset**: https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation
- **W&B project**: _to be added_
- **Deployed app**: _to be added_
- **Internal docs**: _to be added_
