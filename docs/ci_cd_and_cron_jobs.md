# TuneTrace Backend — CI/CD Pipelines & Scheduled Cron Jobs Reference Guide

This comprehensive reference manual documents all continuous delivery pipelines and automated cron jobs running on the TuneTrace backend repository, their architecture, execution schedules, required credentials, failure modes, and local execution runbooks.

---

## 1. System Architecture & Automation Topology

The TuneTrace backend utilizes GitHub Actions for continuous deployment to Azure App Service and scheduled catalog maintenance (semantic backfilling and trending catalog ingestion).

```mermaid
flowchart TD
    subgraph Triggers["Trigger Events"]
        T1["Push to main (excluding docs/csv/sqlite)"]
        T_PR["Pull Request to main (excluding docs/csv/sqlite)"]
        T2["Cron: 0 2 * * * (Daily @ 02:00 UTC)"]
        T3["Cron: 0 0 * * 0 (Weekly Sun @ 00:00 UTC)"]
        T4["workflow_dispatch (Manual Run via CLI / UI)"]
    end

    subgraph CI_CD["deploy_to_azure.yml (CI/CD Pipeline)"]
        subgraph Test_Gate["Job: test (Automated Quality Gate)"]
            T_CO["Checkout code & Setup Python 3.11 with pip cache"]
            T_DEP["Install dependencies (requirements.txt + pytest)"]
            T_RUN["Run Test Suite (pytest -v)"]
            T_CO --> T_DEP --> T_RUN
        end

        subgraph Deploy_Stage["Job: build-and-deploy (needs: test, if != PR)"]
            CD1["Checkout & Setup Python 3.11"]
            CD2["Cache / Pre-download ML Weights (all-MiniLM-L6-v2)"]
            CD3["Run Alembic DB Migrations (alembic upgrade head)"]
            CD4["OIDC Azure Login & ACR Authentication"]
            CD5["Docker Build with Baked Models & Push to ACR"]
            CD6["Deploy Container to Azure App Service (song-suggest-fastapi)"]
            CD1 --> CD2 --> CD3 --> CD4 --> CD5 --> CD6
        end

        T_RUN -->|Passes| CD1
    end

    subgraph Cron_Janitor["v3_janitor.yml (Daily Daemon)"]
        J1["Checkout & Setup Python 3.11"]
        J2["Cache HuggingFace Cache (~/.cache/huggingface)"]
        J3["Install dependencies (requirements.txt: psycopg + pgvector)"]
        J4["Run scripts/v3_janitor.py"]
        J5["Batch vectorize NULL embedding rows into 384-d vectors"]
    end

    subgraph Cron_Seeder["seed_trending_music.yml (Weekly Seeder)"]
        S1["Checkout & Setup Python 3.11"]
        S2["Cache HuggingFace Cache (~/.cache/huggingface)"]
        S3["Install dependencies (requirements.txt)"]
        S4["Run utils/seed_trending_music.py"]
        S5["Fetch 1000 YouTube trending tracks across 10 genres"]
        S6["Enrich metadata, generate 384-d vectors & bulk insert"]
    end

    subgraph Storage["Persistent Storage & Cloud Target"]
        DB[("PostgreSQL / Supabase with pgvector")]
        ACR["Azure Container Registry (songsouggest.azurecr.io)"]
        APP["Azure Web App (song-suggest-fastapi)"]
    end

    T1 --> T_CO
    T_PR --> T_CO
    T4 -.-> T_CO
    CD3 --> DB
    CD5 --> ACR
    CD6 --> APP

    T2 --> J1
    T4 -.-> J1
    J1 --> J2 --> J3 --> J4 --> J5 --> DB

    T3 --> S1
    T4 -.-> S1
    S1 --> S2 --> S3 --> S4 --> S5 --> S6 --> DB
```

---

## 2. Inventory of Automated Workflows

### 2.1. `v3_janitor.yml` — V3 Semantic Embedding Backfill

* **File Location**: [`.github/workflows/v3_janitor.yml`](../.github/workflows/v3_janitor.yml)
* **Execution Schedule**: `0 2 * * *` (Every night at 02:00 UTC / 07:30 IST)
* **Manual Trigger**: Supported (`workflow_dispatch`)
* **Target Script**: [`scripts/v3_janitor.py`](../scripts/v3_janitor.py)
* **Primary Function**:
  - Scans `song_metadata` in the PostgreSQL database for legacy V2 rows where `embedding IS NULL`.
  - Loads the `all-MiniLM-L6-v2` transformer model (384 dimensions).
  - Processes un-vectorized tracks in batches of `100` tracks.
  - Builds rich semantic context: `"{title}. Artist: {artist}. Genre: {genre}. Tags: {tags}"`.
  - Updates rows with dense normalized float arrays (`embedding = :vec, enriched = 'V3'`).
* **Operational Isolation**:
  - Designed specifically to run **outside** the FastAPI web process to prevent Python Global Interpreter Lock (GIL) contention and CPU starvation on production Azure Web App nodes.
* **Hugging Face Model Caching**:
  - Uses `actions/cache@v4` on `~/.cache/huggingface` with cache key `hf-all-MiniLM-L6-v2-${{ runner.os }}` to prevent redundant 90 MB downloads from HuggingFace Hub on every run.
* **Required Secrets**:
  - `POSTGRES_DATABASE_URL`: PostgreSQL connection string with `pgvector` enabled.

---

### 2.2. `seed_trending_music.yml` — Trending Music Catalog Seeder

* **File Location**: [`.github/workflows/seed_trending_music.yml`](../.github/workflows/seed_trending_music.yml)
* **Execution Schedule**: `0 0 * * 0` (Every Sunday at 00:00 UTC / 05:30 IST)
* **Manual Trigger**: Supported (`workflow_dispatch`)
* **Target Script**: [`utils/seed_trending_music.py`](../utils/seed_trending_music.py)
* **Primary Function**:
  - Periodically refreshes and enriches candidate songs in the catalog so recommendation algorithms do not experience catalog decay.
  - Queries YouTube Data API v3 across 10 diverse music verticals:
    1. Pop (`top pop official music video`)
    2. Hip-Hop / Rap (`hip hop rap official music video`)
    3. Rock / Alternative (`rock alternative official music video`)
    4. EDM / Electronic (`edm electronic dance official music video`)
    5. R&B / Soul (`r&b soul official music video`)
    6. Latin Music (`latin music hits official music video`)
    7. Country (`country songs official music video`)
    8. Indie Alternative (`indie alternative official music video`)
    9. K-Pop (`k-pop official music video`)
    10. Afrobeats (`afrobeats official music video`)
  - Target intake: up to `1,000` new candidate songs per run.
  - Checks for track existence (`SELECT id FROM song_metadata WHERE id = :id`) to guarantee idempotency.
  - Computes 384-dimensional vector embeddings on ingestion.
  - Features exponential backoff (`utils/enrichment.py`) with jitter to handle YouTube API transient 429/5xx errors safely.
* **Required Secrets**:
  - `POSTGRES_DATABASE_URL`: Database connection string.
  - `YOUTUBE_API_KEY`: API key with YouTube Data API v3 enabled.

---

### 2.3. `deploy_to_azure.yml` — Continuous Integration & Deployment (CI/CD) to Azure

* **File Location**: [`.github/workflows/deploy_to_azure.yml`](../.github/workflows/deploy_to_azure.yml)
* **Execution Trigger**:
  - Push to `main` branch (path-filtered to ignore `**/*.md`, `**/*.sqlite`, `**/*.csv`)
  - Pull Request targeting `main` (path-filtered to ignore `**/*.md`, `**/*.sqlite`, `**/*.csv`)
  - Manual execution via `workflow_dispatch`
* **Job Architecture & Deployment Gating**:
  The workflow operates in two serialized phases to guarantee that faulty code never deploys to production:

  1. **Phase 1: Automated Test Gate (`job: test`)**:
     - **Checkout & Environment**: Checks out code and provisions Python 3.11 with pip caching.
     - **Dependency Installation**: Installs all project dependencies from `requirements.txt` (including `pytest` and `pytest-cov`).
     - **Test Execution**: Runs `pytest -v` in a completely self-contained in-memory SQLite sandbox with mocked ML engine instances.
     - **Zero Secret Dependency**: Runs without requiring cloud secrets, making it safe and reliable for all PR evaluations.
     - **PR Check**: When triggered by a pull request, only this `test` job executes, validating proposed changes without attempting deployment.

  2. **Phase 2: Build, Migrate & Deploy (`job: build-and-deploy`)**:
     - **Gate Condition**: Requires `needs: test` and only runs when `github.event_name != 'pull_request'` (i.e. pushes to `main` and manual triggers). If any test fails in Phase 1, deployment is strictly aborted.
     - **Model Cache**: Checks GitHub Actions cache for `./models` (`model-all-MiniLM-L6-v2-v1`). If missing, invokes [`utils/download_model.py`](../utils/download_model.py) to pull weights directly into `./models/all-MiniLM-L6-v2`.
     - **Database Migrations**: Installs migration dependencies (`alembic`, `sqlalchemy`, `psycopg[binary]`, `psycopg2-binary`, `pgvector`, `python-dotenv`) and runs `alembic upgrade head`.
     - **Azure Authentication**: Authenticates with Azure via OIDC using client/tenant/subscription credentials.
     - **Docker Image Build & Push**: Builds container with local `./models` context to ensure zero runtime model download overhead in production. Pushes image tagged with commit SHA to Azure Container Registry (`songsouggest.azurecr.io/backend:<sha>`).
     - **App Service Rollout**: Updates Azure App Service (`song-suggest-fastapi`) to the newly built container image.
* **Required Secrets**:
  - `POSTGRES_DATABASE_URL`
  - `AZUREAPPSERVICE_CLIENTID_4AED1299D2C04CAA9208F16E2D318BEA`
  - `AZUREAPPSERVICE_TENANTID_614CA175BDB04D7DB7A9C5C4A9543192`
  - `AZUREAPPSERVICE_SUBSCRIPTIONID_D43E42F06B844B6B9D8BBE1F39EA3069`
  - `ACR_USERNAME`
  - `ACR_PASSWORD`

---

## 3. Database Driver & Dialect Resolution Runbook

### Background on `ModuleNotFoundError: No module named 'psycopg'`

In SQLAlchemy 2.0+, PostgreSQL URLs are parsed into dialects:
* `postgresql+psycopg://...` specifies the modern **Psycopg 3** driver (`import psycopg`).
* `postgresql+psycopg2://...` or default `postgresql://...` specifies the legacy **Psycopg 2** driver (`import psycopg2`).

If a production secret or cloud provider (e.g., Supabase / Neon / Azure) specifies `postgresql+psycopg://...`, the environment **must** have `psycopg[binary]` installed. If only `psycopg2-binary` is installed, SQLAlchemy attempts `import psycopg` and crashes with:
```
ModuleNotFoundError: No module named 'psycopg'
```

### Architectural Mitigation Implemented in TuneTrace

1. **Dual Driver Support in `requirements.txt`**:
   Both `psycopg[binary]` (Psycopg 3) and `psycopg2-binary` (Psycopg 2) are pinned in `requirements.txt`.
2. **Resilient Driver Normalization in [`db.py`](../db.py) and [`alembic/env.py`](../alembic/env.py)**:
   The `normalize_database_url` helper dynamically inspects installed DBAPI modules and translates the URL scheme:
   - If `postgresql+psycopg://` is configured but only `psycopg2` is available $\to$ falls back to `postgresql+psycopg2://`.
   - If `postgresql+psycopg2://` is configured but only `psycopg` is available $\to$ translates to `postgresql+psycopg://`.
   - If legacy `postgres://` is provided $\to$ rewrites to standard `postgresql://`.
3. **Migration Step in `deploy_to_azure.yml`**:
   The migration step explicitly installs `"psycopg[binary]"` alongside `psycopg2-binary` to guarantee migration compatibility with both connection string formats.

---

## 4. Runbook: Triggering & Monitoring via GitHub CLI

You can trigger, watch, and diagnose any workflow directly from the command line using `gh`:

```bash
# -------------------------------------------------------------
# 1. Trigger Scheduled Workflows Manually
# -------------------------------------------------------------
# Run V3 embedding backfill
gh workflow run v3_janitor.yml --ref main

# Run weekly trending music seeder
gh workflow run seed_trending_music.yml --ref main

# Trigger full Azure deployment
gh workflow run deploy_to_azure.yml --ref main

# -------------------------------------------------------------
# 2. View Status and Live Logs
# -------------------------------------------------------------
# List recent runs for a workflow
gh run list --workflow=v3_janitor.yml --limit 5

# View live progress of the latest run
gh run watch

# Inspect failures and view step logs
gh run view <run-id> --log-failed
```

---

## 5. Runbook: Local Script Execution

All automated scripts can be executed locally for testing, debugging, or ad-hoc backfilling.

### Prerequisites

Create and activate a virtual environment and install requirements:
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

### Local Environment Variables

Create or update `.env` in the repository root:
```env
POSTGRES_DATABASE_URL=postgresql+psycopg://postgres:yourpassword@localhost:5432/tunetrace
YOUTUBE_API_KEY=AIzaSy...your_key_here
MODEL_NAME=all-MiniLM-L6-v2
```

### Running the V3 Embedding Backfill

```bash
python scripts/v3_janitor.py
```
* Expected log output:
  ```
  INFO     [v3_janitor] Loading SentenceTransformer model from 'all-MiniLM-L6-v2'...
  INFO     [v3_janitor] Model loaded in 4.2s.
  INFO     [v3_janitor] Batch complete: 100 rows vectorized (total: 100).
  INFO     [v3_janitor] V3 backfill finished. Total rows updated: 100.
  ```

### Running the Trending Music Seeder

```bash
python utils/seed_trending_music.py
```
* Expected log output:
  ```
  INFO     [seed_trending] Starting trending music seeding run (Target: 1000 songs)...
  INFO     [seed_trending] Processing category: top pop official music video
  INFO     [seed_trending] Inserted 85 new songs. Running total: 85/1000
  INFO     [seed_trending] Seeding run completed successfully! Total added: 1000
  ```

### Running Database Migrations Locally

```bash
alembic upgrade head
```

---

## 6. Common Issues & Troubleshooting Matrix

| Issue / Symptom | Root Cause | Resolution |
| :--- | :--- | :--- |
| **`ModuleNotFoundError: No module named 'psycopg'`** | Secret `POSTGRES_DATABASE_URL` uses `postgresql+psycopg://` but `psycopg[binary]` is missing. | Verify `requirements.txt` contains `psycopg[binary]`. Ensure `db.py` contains `normalize_database_url`. |
| **`QuotaExceeded` (403 from YouTube API)** | Daily quota limit (10,000 units) exhausted on the YouTube Data API project. | Wait until 00:00 PST for Google quota reset, or provision a secondary API key under `YOUTUBE_API_KEY`. The script implements exponential backoff and saves progress safely. |
| **Hugging Face 429 / Slow Download** | GitHub Actions downloading model weights from HuggingFace on every job run without cache. | Check `actions/cache@v4` step in the workflow. Ensure cache key is valid. For local builds, run `python utils/download_model.py` to pre-bake `./models`. |
| **`OperationalError: no such column: embedding`** | Database schema has not been updated with the V3 pgvector column. | Execute `alembic upgrade head` to run migration `002_add_v3_embedding_columns.py`. |
| **Azure ACR Login Failure (`unauthorized: authentication required`)** | ACR admin password expired or rotated in Azure portal. | In Azure Portal, navigate to Container Registries $\to$ `songsouggest` $\to$ Access Keys. Update `ACR_USERNAME` and `ACR_PASSWORD` in GitHub Secrets. |
| **Azure OIDC Token Exchange Failure** | GitHub Actions Federated Credential mismatch on Azure App Registration. | Check `AZUREAPPSERVICE_CLIENTID_*`, `AZUREAPPSERVICE_TENANTID_*`, and federated credential branch pattern (`repo:Agrannya-Singh/Tune_Trace_backend:ref:refs/heads/main`). |
