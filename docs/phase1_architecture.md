---

# SAMSUNG PRISM: PHASE 1 DELIVERABLE
## Worklet Title: Semantic Music Retrieval & Behavioral Alignment Layer
**Milestone Focus:** Tech Stack Validation, System Architecture Blueprinting, and Custom Ingestion Design

### 1. Technical Stack Selection & Validation
* **Backend Framework:** The application leverages **FastAPI** as its primary REST framework. As evidenced by `main.py` and the `routers/` layer, FastAPI provides native asynchronous capacity, enabling non-blocking I/O operations necessary for concurrent downstream requests (e.g., parallel YouTube Data API lookups). It heavily utilizes dependency injection (e.g., `Depends(get_repo)`, `Depends(get_current_user)`) to cleanly supply database repositories and service instances, alongside `BackgroundTasks` for asynchronous cache updates without blocking client responses.
* **Database Layer:** The system employs **PostgreSQL** managed via **SQLAlchemy ORM**. Crucially, as defined in `models.py` (`SongMetadata`), the architecture is explicitly validated to integrate the `pgvector` extension. The embedding column is fixed at 384 dimensions (`Vector(384)`), which directly aligns with the output size of the chosen `all-MiniLM-L6-v2` SentenceTransformer model for efficient, high-dimensional semantic similarity lookups.
* **Dependency & State Management:** The project utilizes **Poetry** (`pyproject.toml`) for deterministic dependency resolution and state management, exporting to a standard `requirements.txt`. The core stack is anchored by `fastapi`/`uvicorn` for serving, `sqlalchemy`/`psycopg2-binary`/`alembic` for state persistence and migrations, `redis` for fast-access caching (e.g., user likes, anti-repetition), and `sentence-transformers`/`torch` for local tensor operations.

### 2. Core Architectural Design & System Boundaries
* **Layered Layout:** The architecture enforces a strict separation of concerns:
    * **Presentation/API Layer (`routers/`):** Routers (`suggestions.py`, `discovery.py`, `users.py`) manage the HTTP lifecycle, input validation against Pydantic schemas (`api_models.py`), authentication enforcement, and background task scheduling.
    * **Service/Domain Layer (`services.py`, `ml_engine.py`):** Contains the core business logic, orchestrating calls between external APIs, the recommendation engine, and caching layers.
    * **Data Access Layer (`repository.py`, `models.py`):** Abstracted through a Repository Pattern, ensuring the business logic remains decoupled from raw SQL execution and SQLAlchemy object mappings.
* **System Flow Diagram:** 
    1. **Client Initialization:** The client submits an authenticated request (e.g., `/suggestions` or `/discover`) accompanied by a JWT token.
    2. **Authentication & Routing:** FastAPI middleware validates the identity (`get_current_user`). The router processes the payload using strongly-typed models (e.g., `LikedSongsRequest`).
    3. **Data Retrieval & Hydration:** The Repository layer is queried. If songs are unrecognized, parallel asynchronous calls are dispatched to external APIs to hydrate missing metadata.
    4. **Asynchronous State Updates:** User histories and anti-repetition filters are written to PostgreSQL and synchronized with Redis cache via background tasks.
    5. **Semantic Processing:** The ML Engine consumes the request and user history, computing L2/Cosine similarity against the `pgvector` index.
    6. **Response Delivery:** Transformed JSON representations (`SuggestionResponse`, `DiscoverResponse`) are streamed back to the client.

### 3. Data Schema & Gap Analysis (Yambda Mismatch)
* **The Behavioral Limitation:** The prescribed Yambda dataset schema relies completely on anonymized relational sequence matrices (user IDs, item IDs, playback percentages, and the `is_organic` recommendation flag). This purely behavioral structure entirely lacks natural language descriptions, category classifications, or textual metadata.
* **The Semantic Gap:** High-level LLM retrieval intent states operate on rich semantics, emitting text-heavy token blocks (e.g., "chill lo-fi vibes for studying"). These token vectors have zero direct mapping targets within the flat, numeric behavioral tables provided by the Yambda dataset, creating a fundamental gap between natural language search intents and behavioral item IDs.

### 4. Custom Enrichment & Ingestion Engine Design
* **The YouTube API Bridge:** To bridge the semantic gap, the system conceptually integrates the YouTube Data API v3 as a secondary metadata scaffold. The system uses raw video IDs from the behavioral sequence to cross-reference and fetch textual descriptors (`snippet.tags`, titles, descriptions) from YouTube. This effectively maps rich text attributes onto otherwise sterile behavioral item sequences.
* **Ingestion Logic:** Based on `utils/enrichment.py` and `print_tag_extraction.py`, the ingestion engine operates a robust background pipeline:
    1. **Batch Harvesting:** Un-enriched records are pooled in batches of 50 and queried against the YouTube API using exponential backoff to handle rate limits.
    2. **Cleaning & Normalization:** The raw `tags` array is harvested and scanned against a predefined ontology of `KNOWN_MUSIC_GENRES` (e.g., matching "synthwave" or "indie pop").
    3. **Structural Binding:** The exact genre is extracted, and the top 20 raw tags are concatenated into a string. These are bound back to the database entity (`enriched="V2"`).
    4. **Feature Construction:** Finally, the ML Engine compiles these fields into a dense textual feature block, generating the ultimate TF-IDF or embedding vector that enables pure semantic retrieval against the originally barren IDs.

---
