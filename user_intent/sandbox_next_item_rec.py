"""
user_intent/sandbox_next_item_rec.py

Sandboxed Intent State Modeling & Sequential Next-Item Recommender.
Designed for the Samsung PRISM (SRI-B) Phase 2 Sandbox.
Scaled to handle the full YAMDA-50M dataset safely within a 6-hour window.

ETL & Modeling Stack:
    1. Kaggle Local Parquet Search — Scans '/kaggle/input' for local YAMDA parquets.
    2. Hugging Face Datasets      — Fallback streaming of YAMDA-50M parquets.
    3. DuckDB (Disk-Backed)       — Streaming chunk ingestion with 8 GB memory cap.
    4. FAISS (IVFPQ)              — Product-quantization compressed index (~90 % RAM
                                     reduction).
    5. NumPy Memmap               — Virtual-memory paging for track embedding matrix.
    6. PyTorch GRU + InfoNCE      — Contrastive learning to pull target items and push
                                     in-batch negatives.
    7. Multi-Head Softmax Intent  — Calibrates ranking scores by predicted user intent.
    8. Artifact Exports:
       - recommendations.db        (SQLite)
       - faiss_track_index.bin     (FAISS)
       - yamda_gru_weights.pt      (PyTorch state dict)
"""

# ---------------------------------------------------------------------------
# Standard-library imports
# ---------------------------------------------------------------------------
import logging
import os
import random
import sqlite3
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Third-party imports
# ---------------------------------------------------------------------------
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# ---------------------------------------------------------------------------
# Logging — forced reconfiguration so output is visible in all runtimes
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    force=True,
)
logger = logging.getLogger(__name__)


def log_info(msg: str) -> None:
    """Log an informational message to both the logger and stdout."""
    logger.info(msg)
    print(f"[INFO] {msg}", flush=True)


def log_warn(msg: str) -> None:
    """Log a warning message to both the logger and stdout."""
    logger.warning(msg)
    print(f"[WARN] {msg}", flush=True)


def log_err(msg: str) -> None:
    """Log an error message to both the logger and stdout."""
    logger.error(msg)
    print(f"[ERROR] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Global constants
# ---------------------------------------------------------------------------

# Device selection — prefer CUDA when available
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EMBEDDING_DIM: int = 256   # YAMDA CNN audio embedding dimensionality
SEQUENCE_LEN: int = 20     # Number of past interactions used per training sample
DECAY_LAMBDA: float = 1.0  # Exponential-decay rate for the heuristic profile engine

# ---------------------------------------------------------------------------
# File & output-path configuration
# ---------------------------------------------------------------------------
try:
    # Works when running as a script; __file__ is defined.
    _current_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    # Falls back to cwd inside notebooks / interactive interpreters.
    _current_dir = os.getcwd()

# Required output artifacts
DB_FILE_PATH: str = os.path.join(_current_dir, "recommendations.db")
FAISS_INDEX_PATH: str = os.path.join(_current_dir, "faiss_track_index.bin")
WEIGHTS_PATH: str = os.path.join(_current_dir, "yamda_gru_weights.pt")
CHECKPOINT_PATH: str = os.path.join(_current_dir, "yamda_checkpoint.pt")


# =====================================================================
# Database Setup (SQLite)
# =====================================================================

class YambdaSandboxDatabase:
    """Manages SQLite relations for track catalogues, histories, and
    recommendations.

    On instantiation the old database file (if any) is removed so every
    run starts from a clean slate.
    """

    def __init__(self, db_path: str) -> None:
        # Remove a stale database file if it exists (skip for in-memory DBs).
        if os.path.exists(db_path) and db_path != ":memory:":
            try:
                os.remove(db_path)
            except OSError as exc:
                log_warn(f"Could not remove old DB file: {exc}")

        self.conn: sqlite3.Connection = sqlite3.connect(db_path)
        self.cursor: sqlite3.Cursor = self.conn.cursor()
        self._create_tables()

    # ------------------------------------------------------------------
    # Schema helpers
    # ------------------------------------------------------------------

    def _create_tables(self) -> None:
        """Create the three core tables idempotently."""
        self.cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS tracks (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                yamda_song_id   INTEGER UNIQUE
            )
            """
        )
        self.cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS user_histories (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id         INTEGER,
                track_id        INTEGER,
                played_ratio_pct REAL,
                intent_state    TEXT,
                intent_weight   REAL,
                timestamp       TEXT,
                FOREIGN KEY (track_id) REFERENCES tracks (id)
            )
            """
        )
        self.cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS recommendations (
                id                          INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id                     INTEGER,
                rank                        INTEGER,
                model_type                  TEXT,
                recommended_yamda_song_id   INTEGER,
                score                       REAL
            )
            """
        )
        self.conn.commit()

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def get_yamda_id(self, sqlite_id: int) -> int:
        """Map an internal SQLite ``tracks.id`` back to the original YAMDA
        song identifier.  Returns ``-1`` when the mapping is absent."""
        self.cursor.execute(
            "SELECT yamda_song_id FROM tracks WHERE id = ?",
            (sqlite_id,),
        )
        row = self.cursor.fetchone()
        return row[0] if row else -1

    # ------------------------------------------------------------------
    # Resource management
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Commit any pending changes and close the connection."""
        try:
            self.conn.commit()
        except sqlite3.Error:
            pass
        self.conn.close()


# =====================================================================
# Local Parquet Search Helper
# =====================================================================

def locate_local_parquet(filename: str) -> Optional[str]:
    """Scan typical Kaggle and local directories for a parquet whose name
    contains *filename*.

    Returns
    -------
    str or None
        Absolute path to the first matching ``.parquet`` file, or ``None``
        if nothing is found.
    """
    search_dirs = [
        "/kaggle/input",
        "./",
        "../",
    ]
    for base in search_dirs:
        if not os.path.exists(base):
            continue
        for root, _dirs, files in os.walk(base):
            for fname in files:
                if filename in fname and fname.endswith(".parquet"):
                    path = os.path.abspath(os.path.join(root, fname))
                    log_info(
                        f"Found local parquet file for '{filename}' at: {path}"
                    )
                    return path
    return None


# =====================================================================
# Real Data Ingestion (Kaggle Parquet / HF Streaming + DuckDB ETL)
# =====================================================================

def _resolve_column(
    candidates: List[str],
    available: List[str],
    fallback: str,
) -> str:
    """Return the first candidate name that exists in *available*, else
    *fallback*."""
    return next((c for c in candidates if c in available), fallback)


def _is_array_type(type_str: str) -> bool:
    """Heuristic check whether a DuckDB type string represents an array /
    list column."""
    upper = type_str.upper()
    return "[]" in type_str or "LIST" in upper


def fetch_yamda_duckdb_data(
    db: YambdaSandboxDatabase,
    num_songs: int = 1_000_000,
    num_listens: int = 50_000_000,
) -> Tuple[np.ndarray, Dict[int, List[Tuple[int, float, str, float]]], int]:
    """Load YAMDA from local Kaggle parquets or fall back to Hugging Face
    streaming.

    Parameters
    ----------
    db : YambdaSandboxDatabase
        Target database for persisting track and history rows.
    num_songs : int
        Maximum number of song embeddings to ingest.
    num_listens : int
        Maximum number of listen rows to ingest.

    Returns
    -------
    aligned_embeddings : np.ndarray
        L2-normalised embedding matrix indexed by ``(sqlite_id - 1)``.
    user_histories : dict
        ``{user_id: [(sqlite_track_id, ratio, state, weight), …]}``.
    dim : int
        Embedding dimensionality.
    """
    import duckdb  # Lazy import — not needed when using mock data

    local_embeddings = locate_local_parquet("embeddings.parquet")
    local_listens = locate_local_parquet("listens.parquet")
    use_local_parquets = (
        local_embeddings is not None and local_listens is not None
    )

    if use_local_parquets:
        return _ingest_local_parquets(
            db, local_embeddings, local_listens, num_listens, duckdb
        )

    # ------------------------------------------------------------------
    # Fallback: Hugging Face streaming
    # ------------------------------------------------------------------
    return _ingest_huggingface_streaming(db, num_songs, num_listens, duckdb)


# ------------------------------------------------------------------
# LOCAL-PARQUET INGESTION
# ------------------------------------------------------------------

def _ingest_local_parquets(
    db: YambdaSandboxDatabase,
    local_embeddings: str,
    local_listens: str,
    num_listens: int,
    duckdb: Any,
) -> Tuple[np.ndarray, Dict[int, List[Tuple[int, float, str, float]]], int]:
    """Ingest YAMDA data from locally available parquet files via DuckDB."""
    log_info("Local YAMDA parquet files detected. Verifying schemas…")

    duckdb_scratch = os.path.join(_current_dir, "duckdb_scratch.db")
    con = duckdb.connect(duckdb_scratch)
    try:
        con.execute("PRAGMA memory_limit='8GB'")

        # --- Dynamic schema resolution: Embeddings ---
        embed_cols = [
            col[0]
            for col in con.execute(
                f"SELECT * FROM read_parquet('{local_embeddings}') LIMIT 0"
            ).description
        ]
        log_info(f"Embeddings file columns: {embed_cols}")

        vec_col = _resolve_column(
            ["normalized_embed", "embed", "embedding",
             "features", "audio_embedding", "vector"],
            embed_cols,
            fallback="",
        )
        if not vec_col:
            raise ValueError(
                "Could not find a valid embedding column in local parquet. "
                f"Found: {embed_cols}"
            )

        embed_id_col = _resolve_column(
            ["item_id", "item", "track_id"], embed_cols, "item_id"
        )

        # --- Dynamic schema resolution: Listens ---
        listens_cols = [
            col[0]
            for col in con.execute(
                f"SELECT * FROM read_parquet('{local_listens}') LIMIT 0"
            ).description
        ]
        log_info(f"Listens file columns: {listens_cols}")

        uid_col = _resolve_column(
            ["uid", "user_id", "user"], listens_cols, "uid"
        )
        listens_item_col = _resolve_column(
            ["item_id", "item", "track_id"], listens_cols, "item_id"
        )
        ratio_col = _resolve_column(
            ["played_ratio_pct", "played_ratio", "ratio"],
            listens_cols,
            fallback="",
        )
        ts_col = _resolve_column(
            ["timestamp", "ts", "time"], listens_cols, "timestamp"
        )

        # Determine per-column DuckDB types so we know which need UNNEST
        describe_rows = con.execute(
            f"DESCRIBE SELECT * FROM read_parquet('{local_listens}')"
        ).fetchall()
        # Each row is a tuple: (name, type, …) — extract only the first two.
        listens_types: Dict[str, str] = {
            str(row[0]): str(row[1]) for row in describe_rows
        }

        uid_is_array = _is_array_type(listens_types.get(uid_col, ""))
        item_is_array = _is_array_type(
            listens_types.get(listens_item_col, "")
        )
        ratio_is_array = bool(ratio_col) and _is_array_type(
            listens_types.get(ratio_col, "")
        )
        ts_is_array = _is_array_type(listens_types.get(ts_col, ""))

        # Build column expressions, applying UNNEST for array columns
        uid_expr = (
            f"UNNEST(l.{uid_col}) AS uid"
            if uid_is_array
            else f"l.{uid_col} AS uid"
        )
        item_expr = (
            f"UNNEST(l.{listens_item_col}) AS item_id"
            if item_is_array
            else f"l.{listens_item_col} AS item_id"
        )
        if ratio_col:
            ratio_expr = (
                f"UNNEST(l.{ratio_col}) AS played_ratio_pct"
                if ratio_is_array
                else f"l.{ratio_col} AS played_ratio_pct"
            )
        else:
            # No ratio column → default everything to 100 %
            ratio_expr = "100.0 AS played_ratio_pct"
        ts_expr = (
            f"UNNEST(l.{ts_col}) AS timestamp"
            if ts_is_array
            else f"l.{ts_col} AS timestamp"
        )

        log_info(
            f"Executing ETL on local parquets: uid={uid_col}, "
            f"item={listens_item_col}, ratio={ratio_col}, vec={vec_col}…"
        )

        query_etl = f"""
            WITH unnested_listens AS (
                SELECT
                    {uid_expr},
                    {item_expr},
                    {ratio_expr},
                    {ts_expr}
                FROM read_parquet('{local_listens}') l
                LIMIT {num_listens}
            ),
            joined_data AS (
                SELECT
                    u.uid,
                    u.item_id,
                    CAST(u.played_ratio_pct AS REAL) AS played_ratio_pct,
                    u.timestamp,
                    CASE
                        WHEN u.played_ratio_pct >= 50.0 THEN 'PLAY_COMPLETE'
                        WHEN u.played_ratio_pct <  30.0 THEN 'SKIP'
                        ELSE 'PARTIAL_PLAY'
                    END AS intent_state,
                    CASE
                        WHEN u.played_ratio_pct >= 50.0 THEN  0.7
                        WHEN u.played_ratio_pct <  30.0 THEN -0.5
                        ELSE  0.3
                    END AS intent_weight
                FROM unnested_listens u
                INNER JOIN read_parquet('{local_embeddings}') e
                    ON u.item_id = e.{embed_id_col}
            ),
            user_counts AS (
                SELECT uid, count(*) AS session_len
                FROM joined_data
                GROUP BY uid
                HAVING session_len > {SEQUENCE_LEN}
            )
            SELECT j.uid, j.item_id, j.played_ratio_pct,
                   j.intent_state, j.intent_weight, j.timestamp
            FROM joined_data j
            INNER JOIN user_counts c ON j.uid = c.uid
            ORDER BY j.uid, j.timestamp
        """

        # Stream ETL results in 100 k-row chunks into SQLite
        cursor_result = con.execute(query_etl)
        user_histories: Dict[int, deque] = {}
        yamda_to_sqlite: Dict[int, int] = {}
        next_sqlite_id = 1
        total_rows_ingested = 0
        chunk_num = 0

        while True:
            chunk = cursor_result.fetch_df_chunk(vectors_per_chunk=100_000)
            if chunk.empty:
                break
            chunk_num += 1

            # Register new tracks with minimal DB round-trips
            new_tracks: List[Tuple[int, int]] = []
            for song_id in chunk["item_id"].unique():
                sid = int(song_id)
                if sid not in yamda_to_sqlite:
                    yamda_to_sqlite[sid] = next_sqlite_id
                    new_tracks.append((next_sqlite_id, sid))
                    next_sqlite_id += 1

            if new_tracks:
                db.cursor.execute("BEGIN TRANSACTION;")
                db.cursor.executemany(
                    "INSERT OR IGNORE INTO tracks (id, yamda_song_id) "
                    "VALUES (?, ?)",
                    new_tracks,
                )
                db.cursor.execute("COMMIT;")

            # Accumulate history rows for a single batch INSERT
            history_rows: List[Tuple[int, int, float, str, float, str]] = []
            for row in chunk.itertuples(index=False):
                uid = int(row.uid)
                sid = int(row.item_id)
                sqlite_id = yamda_to_sqlite[sid]
                ratio = float(row.played_ratio_pct)
                state = str(row.intent_state)
                weight = float(row.intent_weight)
                ts = str(row.timestamp)

                history_rows.append(
                    (uid, sqlite_id, ratio, state, weight, ts)
                )

                # Bounded rolling deque — caps per-user RAM even across 50 M
                # rows.
                if uid not in user_histories:
                    user_histories[uid] = deque(maxlen=300)
                user_histories[uid].append((sqlite_id, ratio, state, weight))

            db.cursor.execute("BEGIN TRANSACTION;")
            db.cursor.executemany(
                """INSERT INTO user_histories
                   (user_id, track_id, played_ratio_pct,
                    intent_state, intent_weight, timestamp)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                history_rows,
            )
            db.cursor.execute("COMMIT;")

            total_rows_ingested += len(chunk)
            if chunk_num % 10 == 0:
                log_info(
                    f"  Streamed chunk {chunk_num}: "
                    f"{total_rows_ingested} rows ingested into SQLite…"
                )

        if total_rows_ingested == 0:
            raise ValueError(
                "Local ETL returned zero records. Check data filters."
            )

        # Convert bounded deques to plain lists for downstream training
        user_histories_list: Dict[int, List[Tuple[int, float, str, float]]] = {
            u: list(h) for u, h in user_histories.items()
        }
        num_tracks_in_db = len(yamda_to_sqlite)
        log_info(
            f"Streaming ETL complete: {total_rows_ingested} rows, "
            f"{num_tracks_in_db} unique tracks."
        )

        # Determine embedding dimensionality from a single sample row
        sample_row = con.execute(
            f"SELECT {vec_col} FROM read_parquet('{local_embeddings}') LIMIT 1"
        ).fetchone()
        if sample_row is None or sample_row[0] is None:
            raise ValueError("Could not read a sample embedding vector.")
        dim = len(sample_row[0])

        # Stream active embeddings into a disk-backed memmap (zero RAM
        # overhead beyond the OS page cache).
        aligned_embeddings = _build_aligned_embeddings_memmap(
            con, local_embeddings, embed_id_col, vec_col,
            yamda_to_sqlite, num_tracks_in_db, dim,
        )

        return np.asarray(aligned_embeddings), user_histories_list, dim

    finally:
        # Always close DuckDB and clean up the scratch file
        con.close()
        try:
            os.remove(duckdb_scratch)
        except OSError:
            pass


# ------------------------------------------------------------------
# HUGGING FACE STREAMING INGESTION
# ------------------------------------------------------------------

def _ingest_huggingface_streaming(
    db: YambdaSandboxDatabase,
    num_songs: int,
    num_listens: int,
    duckdb: Any,
) -> Tuple[np.ndarray, Dict[int, List[Tuple[int, float, str, float]]], int]:
    """Fall back to Hugging Face streaming when local parquets are absent."""
    log_info(
        "Local YAMDA parquets not found. "
        "Falling back to Hugging Face streaming loader…"
    )
    try:
        from datasets import load_dataset  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError("Run: pip install datasets") from exc

    # --- Stream embeddings ---
    embeddings_ds = load_dataset(
        "yandex/yambda",
        data_files="embeddings.parquet",
        streaming=True,
    )["train"]
    embeddings_chunk = list(embeddings_ds.take(num_songs))

    embeddings_list: List[Dict[str, Any]] = []
    vec_candidates = [
        "normalized_embed", "embed", "embedding",
        "features", "audio_embedding", "vector",
    ]
    for row in embeddings_chunk:
        vec_key = next((c for c in vec_candidates if c in row), None)
        item_id = row.get("item_id", row.get("item"))
        if vec_key and row[vec_key] is not None and item_id is not None:
            embeddings_list.append(
                {"item_id": int(item_id), "embedding": row[vec_key]}
            )

    df_raw_embeddings = pd.DataFrame(embeddings_list)

    # --- Stream listens ---
    listens_ds = load_dataset(
        "yandex/yambda",
        data_dir="sequential/50m",
        data_files="listens.parquet",
        streaming=True,
    )["train"]
    listens_chunk = list(listens_ds.take(min(num_listens, 200_000)))

    listens_list: List[Dict[str, Any]] = []
    for row in listens_chunk:
        uid = row.get("uid", row.get("user_id"))
        item_id = row.get("item_id", row.get("item"))
        ratio = row.get("played_ratio_pct", 100.0)
        ts = row.get("timestamp", "")
        if uid is not None and item_id is not None:
            listens_list.append(
                {
                    "uid": int(uid),
                    "item_id": int(item_id),
                    "played_ratio_pct": (
                        float(ratio) if ratio is not None else 100.0
                    ),
                    "timestamp": str(ts),
                }
            )

    df_raw_listens = pd.DataFrame(listens_list)

    # --- DuckDB ETL on in-memory DataFrames ---
    con = duckdb.connect()
    try:
        query_etl = f"""
            WITH joined_data AS (
                SELECT
                    l.uid,
                    l.item_id,
                    l.played_ratio_pct,
                    l.timestamp,
                    CASE
                        WHEN l.played_ratio_pct >= 50.0 THEN 'PLAY_COMPLETE'
                        WHEN l.played_ratio_pct <  30.0 THEN 'SKIP'
                        ELSE 'PARTIAL_PLAY'
                    END AS intent_state,
                    CASE
                        WHEN l.played_ratio_pct >= 50.0 THEN  0.7
                        WHEN l.played_ratio_pct <  30.0 THEN -0.5
                        ELSE  0.3
                    END AS intent_weight
                FROM df_raw_listens l
                INNER JOIN df_raw_embeddings e ON l.item_id = e.item_id
            ),
            user_counts AS (
                SELECT uid, count(*) AS session_len
                FROM joined_data
                GROUP BY uid
                HAVING session_len > {SEQUENCE_LEN}
            )
            SELECT j.uid, j.item_id, j.played_ratio_pct,
                   j.intent_state, j.intent_weight, j.timestamp
            FROM joined_data j
            INNER JOIN user_counts c ON j.uid = c.uid
            ORDER BY j.uid, j.timestamp
        """
        df_logs = con.execute(query_etl).df()

        active_song_ids = df_logs["item_id"].unique().tolist()
        query_active_embeddings = (
            "SELECT item_id, embedding FROM df_raw_embeddings "
            "WHERE item_id IN (SELECT UNNEST(?))"
        )
        df_active_embeddings = con.execute(
            query_active_embeddings, [active_song_ids]
        ).df()
    finally:
        con.close()

    # Persist tracks to SQLite
    db.cursor.execute("BEGIN TRANSACTION;")
    for song_id in active_song_ids:
        db.cursor.execute(
            "INSERT OR IGNORE INTO tracks (yamda_song_id) VALUES (?)",
            (song_id,),
        )
    db.cursor.execute("COMMIT;")

    db.cursor.execute("SELECT id, yamda_song_id FROM tracks")
    yamda_to_sqlite: Dict[int, int] = {
        row[1]: row[0] for row in db.cursor.fetchall()
    }

    # Persist user histories
    db.cursor.execute("BEGIN TRANSACTION;")
    user_histories: Dict[int, List[Tuple[int, float, str, float]]] = {}
    for _, row in df_logs.iterrows():
        uid = int(row["uid"])
        sid = int(row["item_id"])
        if sid not in yamda_to_sqlite:
            continue
        sqlite_id = yamda_to_sqlite[sid]
        ratio = float(row["played_ratio_pct"])
        state = str(row["intent_state"])
        weight = float(row["intent_weight"])
        ts = str(row["timestamp"])

        db.cursor.execute(
            """INSERT INTO user_histories
               (user_id, track_id, played_ratio_pct,
                intent_state, intent_weight, timestamp)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (uid, sqlite_id, ratio, state, weight, ts),
        )

        if uid not in user_histories:
            user_histories[uid] = []
        user_histories[uid].append((sqlite_id, ratio, state, weight))
    db.cursor.execute("COMMIT;")

    # Build the aligned, L2-normalised embedding matrix via memmap
    num_tracks_in_db = len(yamda_to_sqlite)
    if df_active_embeddings.empty:
        raise ValueError("No active embeddings matched the listen log.")
    dim = len(df_active_embeddings["embedding"].iloc[0])

    memmap_path = os.path.join(_current_dir, "aligned_embeddings.mmap")
    aligned_embeddings = np.memmap(
        memmap_path, dtype="float32", mode="w+",
        shape=(num_tracks_in_db, dim),
    )
    aligned_embeddings[:] = 0.0

    for _, row in df_active_embeddings.iterrows():
        sid = int(row["item_id"])
        if sid in yamda_to_sqlite:
            sqlite_idx = yamda_to_sqlite[sid] - 1
            aligned_embeddings[sqlite_idx] = row["embedding"]

    # Batched L2 normalisation (avoids loading the entire matrix at once)
    _normalise_memmap_batched(aligned_embeddings, num_tracks_in_db)
    aligned_embeddings.flush()

    log_info(
        f"DuckDB ETL Complete: Loaded {len(user_histories)} users and "
        f"{num_tracks_in_db} tracks into recommendations.db."
    )
    return np.asarray(aligned_embeddings), user_histories, dim


# ------------------------------------------------------------------
# Shared embedding helpers
# ------------------------------------------------------------------

def _build_aligned_embeddings_memmap(
    con: Any,
    local_embeddings: str,
    embed_id_col: str,
    vec_col: str,
    yamda_to_sqlite: Dict[int, int],
    num_tracks: int,
    dim: int,
) -> np.ndarray:
    """Stream embeddings from a parquet into a disk-backed memmap, then
    L2-normalise in batches.  Returns the memmap array."""
    memmap_path = os.path.join(_current_dir, "aligned_embeddings.mmap")
    aligned = np.memmap(
        memmap_path, dtype="float32", mode="w+", shape=(num_tracks, dim)
    )
    aligned[:] = 0.0

    log_info(
        f"Streaming embeddings into disk memmap "
        f"({num_tracks} tracks, dim={dim})…"
    )
    cursor_embed = con.execute(
        f"SELECT {embed_id_col} AS item_id, {vec_col} AS embedding "
        f"FROM read_parquet('{local_embeddings}')"
    )

    while True:
        chunk_embed = cursor_embed.fetch_df_chunk(vectors_per_chunk=50_000)
        if chunk_embed.empty:
            break
        for row in chunk_embed.itertuples(index=False):
            sid = int(row.item_id)
            if sid in yamda_to_sqlite:
                sqlite_idx = yamda_to_sqlite[sid] - 1
                aligned[sqlite_idx] = row.embedding

    _normalise_memmap_batched(aligned, num_tracks)
    aligned.flush()
    return aligned


def _normalise_memmap_batched(
    embeddings: np.ndarray,
    n_vectors: int,
    batch_size: int = 100_000,
) -> None:
    """L2-normalise *embeddings* **in-place** in fixed-size batches so
    only one batch resides in RAM at any time."""
    for start in range(0, n_vectors, batch_size):
        end = min(start + batch_size, n_vectors)
        batch = np.array(embeddings[start:end])
        norms = np.linalg.norm(batch, axis=1, keepdims=True)
        batch /= norms + 1e-9
        embeddings[start:end] = batch


# =====================================================================
# Synthetic Fallback Ingestion
# =====================================================================

def populate_mock_data(
    db: YambdaSandboxDatabase,
    num_songs: int = 500,
    num_users: int = 50,
) -> Tuple[np.ndarray, Dict[int, List[Tuple[int, float, str, float]]], int]:
    """Generate synthetic listening histories and random embeddings when
    the real YAMDA data is unavailable.

    Returns the same triple as :func:`fetch_yamda_duckdb_data`.
    """
    log_warn("Falling back to synthetic mock data generation…")

    yamda_song_ids = [200_000 + i for i in range(num_songs)]
    db.cursor.execute("BEGIN TRANSACTION;")
    for song_id in yamda_song_ids:
        db.cursor.execute(
            "INSERT OR IGNORE INTO tracks (yamda_song_id) VALUES (?)",
            (song_id,),
        )
    db.cursor.execute("COMMIT;")

    db.cursor.execute("SELECT id, yamda_song_id FROM tracks")
    sqlite_track_ids = [row[0] for row in db.cursor.fetchall()]

    user_histories: Dict[int, List[Tuple[int, float, str, float]]] = {}
    db.cursor.execute("BEGIN TRANSACTION;")
    for user_id in range(num_users):
        seq_len = random.randint(8, 12)
        user_histories[user_id] = []
        for _ in range(seq_len):
            track_id = random.choice(sqlite_track_ids)
            ratio = random.choice([10.0, 45.0, 90.0])
            if ratio < 30.0:
                state = "SKIP"
            elif ratio >= 50.0:
                state = "PLAY_COMPLETE"
            else:
                state = "PARTIAL_PLAY"
            weight = (
                -0.5 if state == "SKIP"
                else 0.7 if state == "PLAY_COMPLETE"
                else 0.3
            )

            db.cursor.execute(
                """INSERT INTO user_histories
                   (user_id, track_id, played_ratio_pct,
                    intent_state, intent_weight, timestamp)
                   VALUES (?, ?, ?, ?, ?, datetime('now'))""",
                (user_id, track_id, ratio, state, weight),
            )
            user_histories[user_id].append(
                (track_id, ratio, state, weight)
            )
    db.cursor.execute("COMMIT;")

    # Random unit-norm embeddings
    embeddings = np.random.randn(num_songs, EMBEDDING_DIM).astype(np.float32)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings /= norms + 1e-9
    return embeddings, user_histories, EMBEDDING_DIM


# =====================================================================
# FAISS Index Setup (IVFPQ for 50 M scale, HNSWFlat fallback)
# =====================================================================

# Below this count we use the simpler HNSWFlat; above it we switch to
# the compressed IVFPQ index to keep memory tractable.
IVFPQ_THRESHOLD: int = 50_000


def build_faiss_index(embeddings: np.ndarray, dim: int) -> Any:
    """Build a FAISS inner-product index over *embeddings*.

    Strategy
    --------
    * < 50 k vectors → HNSWFlat (exact, fast for small catalogues).
    * ≥ 50 k vectors → IVFPQ   (compressed, ~90 % RAM reduction).

    Falls back to :class:`PythonCosineIndex` when FAISS cannot be
    installed.
    """
    try:
        import faiss  # type: ignore[import-untyped]
    except ImportError:
        log_warn(
            "FAISS is not installed. "
            "Attempting to install 'faiss-cpu' dynamically…"
        )
        try:
            import subprocess
            import sys

            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "faiss-cpu"]
            )
            import faiss  # type: ignore[import-untyped]

            log_info(
                "Successfully installed FAISS in the current environment."
            )
        except Exception as exc:
            log_warn(
                f"Failed to install FAISS: {exc}. "
                "Defaulting to Python cosine index."
            )
            return PythonCosineIndex(embeddings)

    # Ensure contiguous float32 — necessary when *embeddings* is a memmap
    normalized_embeds = np.ascontiguousarray(
        embeddings, dtype=np.float32
    ).copy()
    faiss.normalize_L2(normalized_embeds)
    n_vectors = normalized_embeds.shape[0]

    if n_vectors < IVFPQ_THRESHOLD:
        log_info(f"Building FAISS HNSWFlat index ({n_vectors} vectors)…")
        index = faiss.IndexHNSWFlat(dim, 32, faiss.METRIC_INNER_PRODUCT)
        index.add(normalized_embeds)
        return index

    # ----- Large dataset: IVFPQ -----
    nlist = min(int(np.sqrt(n_vectors)), 8192)
    m = 32  # Number of sub-quantizers
    nbits = 8

    # *m* must divide *dim*; try smaller factors if it doesn't
    if dim % m != 0:
        for candidate_m in [16, 8, 4]:
            if dim % candidate_m == 0:
                m = candidate_m
                break

    log_info(
        f"Building FAISS IVFPQ index: {n_vectors} vectors, "
        f"nlist={nlist}, m={m}…"
    )
    quantizer = faiss.IndexFlatIP(dim)
    index = faiss.IndexIVFPQ(
        quantizer, dim, nlist, m, nbits, faiss.METRIC_INNER_PRODUCT
    )

    # Train on a random subset capped at 100 k
    train_size = min(n_vectors, 100_000)
    rng = np.random.default_rng(42)
    train_indices = rng.choice(n_vectors, train_size, replace=False)
    train_sample = normalized_embeds[train_indices]
    log_info(f"Training IVFPQ quantiser on {train_size} sample vectors…")
    index.train(train_sample)

    # Add in batches to control peak memory
    add_batch_size = 500_000
    for start in range(0, n_vectors, add_batch_size):
        end = min(start + add_batch_size, n_vectors)
        index.add(normalized_embeds[start:end])

    index.nprobe = min(64, nlist)
    log_info(f"IVFPQ index built successfully. Total vectors: {index.ntotal}")
    return index


class PythonCosineIndex:
    """Pure-Python fallback when FAISS is unavailable.

    Performs brute-force cosine similarity via NumPy.  Adequate for small
    catalogues (< 50 k tracks) but **not** suitable for production at
    scale.
    """

    def __init__(self, embeddings: np.ndarray) -> None:
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        self.embeddings: np.ndarray = embeddings / (norms + 1e-9)
        self.ntotal: int = len(embeddings)

    def search(
        self, query: np.ndarray, k: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return the *k* nearest neighbours by cosine similarity.

        Returns
        -------
        distances : np.ndarray, shape ``(n_queries, k)``
        indices   : np.ndarray, shape ``(n_queries, k)``
        """
        q_norm = query / (np.linalg.norm(query, axis=1, keepdims=True) + 1e-9)
        similarities = np.dot(self.embeddings, q_norm.T).T
        indices = np.argsort(-similarities, axis=1)[:, :k]
        distances = np.take_along_axis(similarities, indices, axis=1)
        return distances, indices


# =====================================================================
# PyTorch GRU Sequential Model & InfoNCE Contrastive Dataset
# =====================================================================

class SequentialRecDataset(Dataset):
    """Build sequential training pairs with multi-state intent labels.

    Intent classes (mapped per Yambda Listen+/Like/Skip specs):

    ====  ===========================  ================================
    Code  Label                        Condition
    ====  ===========================  ================================
    0     LIKE / PLAY_COMPLETE         played_ratio ≥ 50 % or explicit
    1     PARTIAL_PLAY                 30 % ≤ played_ratio < 50 %
    2     SKIP                         played_ratio < 30 %
    3     DISLIKE                      explicit negative signal
    ====  ===========================  ================================
    """

    def __init__(
        self,
        user_histories: Dict[int, List[Tuple[int, float, str, float]]],
        embeddings: np.ndarray,
        seq_len: int = SEQUENCE_LEN,
        max_history_per_user: int = 300,
    ) -> None:
        self.sequences: List[List[int]] = []
        self.targets: List[int] = []
        self.target_intents: List[int] = []
        self.embeddings = embeddings

        for _user_id, history in user_histories.items():
            if len(history) <= seq_len:
                continue

            # Cap per-user history to bound memory
            user_slice = (
                history[-max_history_per_user:]
                if len(history) > max_history_per_user
                else history
            )

            for i in range(len(user_slice) - seq_len):
                # IDs are 1-based SQLite ids → subtract 1 for array index
                seq_ids = [
                    item[0] - 1
                    for item in user_slice[i : i + seq_len]
                ]
                target_item = user_slice[i + seq_len]
                target_id = target_item[0] - 1
                ratio = float(target_item[1])
                state = str(target_item[2])

                # Map state + ratio to an intent class
                if state == "DISLIKE":
                    intent_cls = 3
                elif state == "SKIP" or ratio < 30.0:
                    intent_cls = 2
                elif 30.0 <= ratio < 50.0 or state == "PARTIAL_PLAY":
                    intent_cls = 1
                else:
                    intent_cls = 0

                self.sequences.append(seq_ids)
                self.targets.append(target_id)
                self.target_intents.append(intent_cls)

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        seq_embeds = self.embeddings[self.sequences[idx]]
        target_embed = self.embeddings[self.targets[idx]]
        target_intent = self.target_intents[idx]
        return (
            torch.tensor(seq_embeds, dtype=torch.float32),
            torch.tensor(target_embed, dtype=torch.float32),
            torch.tensor(target_intent, dtype=torch.long),
        )


class InfoNCERecLoss(nn.Module):
    """InfoNCE contrastive loss with in-batch negative sampling and
    **learnable temperature**.

    A learnable temperature lets the model self-calibrate gradient
    magnitude: early in training a warmer τ produces softer gradients
    that prevent collapse, while late training cools τ for sharper
    discrimination.
    """

    def __init__(self, init_temperature: float = 0.07) -> None:
        super().__init__()
        # Store log(τ) so τ stays positive after exp()
        self.log_temperature = nn.Parameter(
            torch.tensor(np.log(init_temperature), dtype=torch.float32)
        )

    @property
    def temperature(self) -> torch.Tensor:
        """Effective temperature, clamped to [0.01, 1.0] for stability."""
        return self.log_temperature.exp().clamp(min=0.01, max=1.0)

    def forward(
        self,
        pred_vectors: torch.Tensor,
        target_vectors: torch.Tensor,
    ) -> torch.Tensor:
        # Scaled cosine-similarity matrix (B × B)
        sim_matrix = (
            torch.matmul(pred_vectors, target_vectors.T) / self.temperature
        )
        # Positive pairs lie on the diagonal
        labels = torch.arange(
            pred_vectors.size(0), device=pred_vectors.device
        )
        return nn.functional.cross_entropy(sim_matrix, labels)


class GRUIntentRecModel(nn.Module):
    """Dual-head GRU sequential recommender (v3 — heavy).

    Architecture (v3 — designed for maximum fitting capacity):
    ----------------------------------------------------------
    * **hidden_dim 256 → 512** — 4× the original backbone width; matches
      the scale needed for 50 M interactions and 934 k tracks.
    * **4 GRU layers** — deeper temporal abstraction.
    * **Multi-head self-attention pooling** — instead of just taking the
      last GRU time-step, we attend over all time-steps.  This lets the
      model weight *which* items in the sequence matter most (e.g. a
      LIKE 3 items ago may be more informative than a SKIP just now).
    * **4-layer projection MLP** with GELU + LayerNorm + residual skip.
    * **Wider intent head** — 512 → 256 → 128 → num_classes.
    * **Dropout 0.05** — minimal regularisation (priority: reduce loss).
    """

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int = 512,
        num_layers: int = 4,
        num_intent_classes: int = 4,
        num_attn_heads: int = 4,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim

        # ── Input projection (embedding_dim → hidden_dim) ──
        # Needed because GRU input_size must match hidden_size for the
        # residual connections inside GRU cells to work optimally.
        self.input_proj = nn.Linear(embedding_dim, hidden_dim)

        # ── 4-layer GRU backbone ──
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.gru_norm = nn.LayerNorm(hidden_dim)

        # ── Self-attention pooling over all GRU time-steps ──
        # Multi-head attention: queries = last step, keys/values = all steps
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_attn_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(hidden_dim)

        # ── Projection head (hidden → embedding space) ──
        # 4-layer MLP: 512 → 512 → 512 → 256
        self.fc_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, embedding_dim),
        )
        # Residual shortcut for gradient flow
        self.proj_residual = nn.Linear(hidden_dim, embedding_dim, bias=False)

        # ── Intent-squeezing head (3-layer) ──
        self.fc_intent = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.LayerNorm(hidden_dim // 2),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, num_intent_classes),
        )

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x : Tensor, shape ``(B, seq_len, embedding_dim)``

        Returns
        -------
        pred_embed    : Tensor, shape ``(B, embedding_dim)``
            L2-normalised predicted next-item embedding.
        intent_logits : Tensor, shape ``(B, num_intent_classes)``
            Raw logits for the intent classifier.
        """
        # Project input to hidden dim
        h = self.input_proj(x)                    # (B, T, 512)

        gru_out, _ = self.gru(h)                  # (B, T, 512)
        gru_out = self.gru_norm(gru_out)

        # Self-attention pooling: query = last step, KV = full sequence
        query = gru_out[:, -1:, :]                # (B, 1, 512)
        attn_out, _ = self.attn(
            query, gru_out, gru_out,              # Q, K, V
            need_weights=False,
        )                                         # (B, 1, 512)
        # Residual: last_step + attention-weighted context
        pooled = self.attn_norm(
            attn_out.squeeze(1) + gru_out[:, -1, :]
        )                                         # (B, 512)

        # Projection with residual skip
        proj = self.fc_proj(pooled) + self.proj_residual(pooled)
        pred_embed = nn.functional.normalize(proj, p=2, dim=1)

        intent_logits = self.fc_intent(pooled)
        return pred_embed, intent_logits


# =====================================================================
# Heuristic Intent Profile Engine
# =====================================================================

class HeuristicIntentEngine:
    """Computes a user-preference profile as an exponentially-decayed,
    intent-weighted average of recent track embeddings."""

    def __init__(self, decay_lambda: float = DECAY_LAMBDA) -> None:
        self.decay_lambda = decay_lambda

    def compute_heuristic_profile(
        self,
        history_slice: List[Tuple[int, float, str, float]],
        embeddings: np.ndarray,
    ) -> np.ndarray:
        """Return a unit-norm profile vector for the given history slice.

        Parameters
        ----------
        history_slice
            List of ``(sqlite_track_id, ratio, state, weight)`` tuples,
            ordered chronologically.
        embeddings
            The full aligned-embedding matrix (0-indexed).

        Returns
        -------
        np.ndarray, shape ``(dim,)``
            L2-normalised profile vector, or the zero vector when the
            history is empty or the weighted sum collapses.
        """
        n = len(history_slice)
        dim = embeddings.shape[1]
        if n == 0:
            return np.zeros(dim, dtype=np.float32)

        slice_embeddings = np.array(
            [embeddings[item[0] - 1] for item in history_slice],
            dtype=np.float32,
        )
        signal_weights = np.array(
            [item[3] for item in history_slice], dtype=np.float32
        )

        # Exponential recency decay: most-recent item gets decay=1
        steps_from_recent = np.arange(n - 1, -1, -1, dtype=np.float32)
        decay = np.exp(-self.decay_lambda * steps_from_recent)

        combined_weights = signal_weights * decay
        raw_profile = (combined_weights[:, np.newaxis] * slice_embeddings).sum(
            axis=0
        )

        norm = np.linalg.norm(raw_profile)
        if norm < 1e-8:
            return np.zeros(dim, dtype=np.float32)
        return (raw_profile / norm).astype(np.float32)


# =====================================================================
# Main Execution Pipeline
# =====================================================================

def run_sandbox_training() -> None:  # noqa: C901  (complexity accepted)
    """End-to-end pipeline: ingest → index → train → recommend → export."""
    log_info("Initialising Large-Scale Sandbox Ingestion…")
    log_info(f"Target SQLite Database File : {DB_FILE_PATH}")
    log_info(f"Target FAISS Index File     : {FAISS_INDEX_PATH}")
    log_info(f"Target GRU Weights File     : {WEIGHTS_PATH}")

    db = YambdaSandboxDatabase(DB_FILE_PATH)

    try:
        # ==============================================================
        # STAGE 1 — Data Ingestion
        # ==============================================================
        try:
            raw_embeddings, user_histories, actual_dim = (
                fetch_yamda_duckdb_data(
                    db,
                    num_songs=1_000_000,    # All ~934 k tracks in YAMDA-50M
                    num_listens=50_000_000, # Full 46.5 M interaction stream
                )
            )
            log_info("Successfully loaded real YAMDA dataset via DuckDB.")
        except Exception as exc:
            log_err(f"Failed to load real data: {exc}.")
            raw_embeddings, user_histories, actual_dim = populate_mock_data(
                db, num_songs=1000, num_users=200
            )

        if not user_histories:
            log_err("No active user sequences were parsed. Exiting.")
            return

        # ==============================================================
        # STAGE 2 — Build & persist FAISS index  (artifact #2)
        # ==============================================================
        faiss_index = build_faiss_index(raw_embeddings, actual_dim)
        _persist_faiss_index(faiss_index)

        # ==============================================================
        # STAGE 3 — Train GRU sequence model     (artifact #3)
        # ==============================================================
        heuristic_engine = HeuristicIntentEngine(decay_lambda=DECAY_LAMBDA)

        dataset = SequentialRecDataset(
            user_histories, raw_embeddings,
            seq_len=SEQUENCE_LEN, max_history_per_user=300,
        )
        if len(dataset) < 2:
            log_err(
                "Dataset size is too small for PyTorch sequence batching."
            )
            return

        model = _train_gru_model(dataset, actual_dim)

        # Save trained weights
        torch.save(model.state_dict(), WEIGHTS_PATH)
        log_info(
            f"✅ Output #3: GRU model weights saved to {WEIGHTS_PATH}"
        )

        # ==============================================================
        # STAGE 4 — Generate recommendations      (artifact #1)
        # ==============================================================
        _generate_recommendations(
            db, model, faiss_index, heuristic_engine,
            user_histories, raw_embeddings,
        )

        log_info(
            f"✅ Output #1: All recommendations saved to {DB_FILE_PATH}"
        )

        # Print a sample for the first 5 users
        _print_sample_recommendations(db, user_histories)

    finally:
        # Always close the database, even on error
        db.close()


# ------------------------------------------------------------------
# Pipeline sub-routines (extracted for readability)
# ------------------------------------------------------------------

def _persist_faiss_index(faiss_index: Any) -> None:
    """Write the FAISS index to disk (no-op for PythonCosineIndex)."""
    try:
        import faiss  # type: ignore[import-untyped]

        if hasattr(faiss, "write_index") and not isinstance(
            faiss_index, PythonCosineIndex
        ):
            faiss.write_index(faiss_index, FAISS_INDEX_PATH)
            log_info(
                f"✅ Output #2: FAISS index saved to {FAISS_INDEX_PATH}"
            )
    except Exception as exc:
        log_warn(f"Could not persist FAISS index to disk: {exc}")


def _train_gru_model(
    dataset: SequentialRecDataset,
    embedding_dim: int,
    target_epochs: Optional[int] = None,
) -> GRUIntentRecModel:
    """Configure hyper-parameters, build the model, and run the training
    loop with checkpointing and auto-resume.

    v3 changes (Heavy Fitting + Checkpoint Persistence):
    - Model v3: hidden_dim=512, num_layers=4, multi-head self-attention pooling,
      4-layer projection MLP with residual connection (3.5M+ parameters).
    - InfoNCE: learnable temperature registered into AdamW optimizer.
    - Schedulers: Linear warmup (5 epochs) + Cosine Annealing.
    - Regularization: minimal dropout (0.05) + gradient clipping (max_norm=1.0).
    - Checkpoint persistence: saves checkpoint dict at every 15th epoch (15, 30, 45, 60...)
      and on every new best loss.
    - Seamless auto-resume: if 'yamda_checkpoint.pt' exists, automatically restores
      weights, optimizer state, scheduler, and starts from checkpoint['epoch'].
    """

    is_cuda = torch.cuda.is_available()

    # Hardware-adjusted training targets
    if is_cuda:
        batch_size = min(2048, max(32, len(dataset)))
        epochs = target_epochs if target_epochs is not None else 60
        use_amp = True
        log_info(
            f"CUDA accelerator active: batch_size={batch_size}, "
            f"target_epochs={epochs}, AMP enabled."
        )
    else:
        torch.set_num_threads(4)
        batch_size = min(512, max(32, len(dataset)))
        epochs = target_epochs if target_epochs is not None else 30
        use_amp = False
        log_info(
            f"CPU runtime detected: 4 BLAS threads, batch_size={batch_size}, "
            f"target_epochs={epochs}."
        )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=(len(dataset) > batch_size),
        pin_memory=is_cuda,
    )

    # ── Model v3 (Heavy: 512 hidden, 4 layers, self-attention, 4-layer head) ──
    model = GRUIntentRecModel(
        embedding_dim=embedding_dim,
        hidden_dim=512,
        num_layers=4,
        num_intent_classes=4,
        num_attn_heads=4,
        dropout=0.05,
    ).to(DEVICE)

    # Learnable temperature module
    contrastive_criterion = InfoNCERecLoss(init_temperature=0.07).to(DEVICE)
    intent_criterion = nn.CrossEntropyLoss()

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log_info(f"Model v3 heavy parameter count: {param_count:,} trainable weights.")

    # ── Optimiser + Schedulers ──
    base_lr = 3e-3  # Aggressive base LR for high-capacity fitting
    # IMPORTANT: Include contrastive_criterion.parameters() so temperature is learned!
    trainable_params = list(model.parameters()) + list(contrastive_criterion.parameters())
    optimizer = optim.AdamW(
        trainable_params, lr=base_lr, weight_decay=1e-4
    )

    warmup_epochs = 5
    total_steps = epochs * len(loader)
    warmup_steps = warmup_epochs * len(loader)

    def lr_lambda(step: int) -> float:
        """Linear warmup for first 5 epochs, then cosine annealing to zero."""
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = torch.amp.GradScaler(device="cuda", enabled=use_amp)

    # ── Checkpoint Auto-Resume Logic ──
    start_epoch = 0
    best_loss = float("inf")
    best_weights_path = os.path.join(_current_dir, "yamda_best_model.pt")

    if os.path.exists(CHECKPOINT_PATH):
        try:
            log_info(f"🔄 Checkpoint found at {CHECKPOINT_PATH}. Resuming training session...")
            ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            if "scheduler_state_dict" in ckpt and ckpt["scheduler_state_dict"] is not None:
                scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            if use_amp and ckpt.get("scaler_state_dict") is not None:
                scaler.load_state_dict(ckpt["scaler_state_dict"])
            if "temperature_param" in ckpt:
                contrastive_criterion.log_temperature.data.copy_(ckpt["temperature_param"])

            start_epoch = ckpt.get("epoch", 0)
            best_loss = ckpt.get("best_loss", float("inf"))
            log_info(
                f"✅ Resumed successfully from epoch {start_epoch}/{epochs} "
                f"(previous best loss: {best_loss:.4f})."
            )
        except Exception as exc:
            log_warn(f"Failed to restore checkpoint ({exc}). Starting fresh from epoch 0.")
            start_epoch = 0

    if start_epoch >= epochs:
        log_info(
            f"Model has already completed {start_epoch} epochs (target: {epochs}). "
            f"Skipping training loop."
        )
        model.eval()
        return model

    model.train()
    log_info(
        f"🚀 Starting training from epoch {start_epoch + 1} to {epochs} "
        f"on {DEVICE} | base_lr={base_lr}, checkpoints saved every 15 epochs…"
    )

    for epoch in range(start_epoch, epochs):
        total_loss = 0.0
        contrastive_sum = 0.0
        intent_sum = 0.0

        for batch_idx, (batch_seq, batch_target, batch_intent) in enumerate(loader):
            batch_seq = batch_seq.to(DEVICE, non_blocking=True)
            batch_target = batch_target.to(DEVICE, non_blocking=True)
            batch_intent = batch_intent.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type=DEVICE.type, enabled=use_amp):
                pred_embed, intent_logits = model(batch_seq)
                loss_nce = contrastive_criterion(pred_embed, batch_target)
                loss_intent = intent_criterion(intent_logits, batch_intent)
                loss = loss_nce + 0.3 * loss_intent

            if use_amp:
                scaler.scale(loss).backward()
                # Unscale before clipping gradients
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                optimizer.step()

            scheduler.step()

            total_loss += loss.item()
            contrastive_sum += loss_nce.item()
            intent_sum += loss_intent.item()

            if (batch_idx + 1) % 100 == 0 or (batch_idx + 1) == len(loader):
                current_lr = scheduler.get_last_lr()[0]
                current_tau = contrastive_criterion.temperature.item()
                print(
                    f"  [Epoch {epoch + 1}/{epochs}] "
                    f"Batch {batch_idx + 1}/{len(loader)} | "
                    f"Joint: {loss.item():.4f} "
                    f"(InfoNCE: {loss_nce.item():.4f}, Intent: {loss_intent.item():.4f}) "
                    f"lr={current_lr:.5f} τ={current_tau:.4f}",
                    flush=True,
                )

        n_batches = max(1, len(loader))
        avg_loss = total_loss / n_batches
        avg_nce = contrastive_sum / n_batches
        avg_intent = intent_sum / n_batches
        log_info(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Avg Loss: {avg_loss:.4f} "
            f"(InfoNCE: {avg_nce:.4f}, Intent: {avg_intent:.4f})"
        )

        # ── Best Model Tracking ──
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), best_weights_path)
            log_info(f"  🏆 New best model saved! Loss: {best_loss:.4f} -> {best_weights_path}")

        # ── Milestone Checkpointing (Every 15th epoch: 15, 30, 45, 60...) ──
        current_epoch_num = epoch + 1
        if current_epoch_num % 15 == 0 or current_epoch_num == epochs:
            milestone_path = os.path.join(
                _current_dir, f"yamda_checkpoint_epoch_{current_epoch_num}.pt"
            )
            checkpoint_payload = {
                "epoch": current_epoch_num,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict() if use_amp else None,
                "temperature_param": contrastive_criterion.log_temperature.data,
                "loss": avg_loss,
                "best_loss": best_loss,
            }
            # Save periodic milestone snapshot
            torch.save(checkpoint_payload, milestone_path)
            # Save latest checkpoint for automated resumption
            torch.save(checkpoint_payload, CHECKPOINT_PATH)
            log_info(
                f"💾 Checkpoint milestone reached! Saved epoch {current_epoch_num} "
                f"state to {milestone_path} and {CHECKPOINT_PATH}"
            )

    # If best weights exist, restore them for inference
    if os.path.exists(best_weights_path):
        try:
            model.load_state_dict(torch.load(best_weights_path, map_location=DEVICE, weights_only=True))
            log_info(f"Loaded best checkpoint weights from {best_weights_path} for final evaluation.")
        except Exception:
            pass

    model.eval()
    return model


def _generate_recommendations(
    db: YambdaSandboxDatabase,
    model: GRUIntentRecModel,
    faiss_index: Any,
    heuristic_engine: HeuristicIntentEngine,
    user_histories: Dict[int, List[Tuple[int, float, str, float]]],
    raw_embeddings: np.ndarray,
) -> None:
    """Score every user with both the GRU model and the heuristic engine,
    then persist the top-5 recommendations per model into SQLite."""
    log_info("=" * 66)
    log_info(
        f"Generating recommendations for {len(user_histories)} users…"
    )
    log_info("=" * 66)

    db.cursor.execute("BEGIN TRANSACTION;")

    for uid, history in user_histories.items():
        last_slice = history[-SEQUENCE_LEN:]
        last_seq_ids = [item[0] - 1 for item in last_slice]

        # Ensure we have a valid-length context window
        if len(last_seq_ids) < SEQUENCE_LEN:
            continue

        seq_embeds = np.asarray(
            raw_embeddings[last_seq_ids], dtype=np.float32
        )

        # ----- Model 1: GRU + Softmax Intent Squeezing -----
        input_tensor = (
            torch.tensor(seq_embeds, dtype=torch.float32)
            .unsqueeze(0)
            .to(DEVICE)
        )
        with torch.no_grad():
            gru_vec, intent_logits = model(input_tensor)
            gru_vec_np = gru_vec.cpu().numpy()
            intent_probs = (
                torch.softmax(intent_logits, dim=-1).cpu().numpy()[0]
            )
            # Confidence that the user will *like* the next item
            pos_intent_conf = float(intent_probs[0])

        # Guard: skip FAISS query if the predicted vector is degenerate
        if np.linalg.norm(gru_vec_np) < 1e-8:
            continue

        distances_gru, indices_gru = faiss_index.search(gru_vec_np, k=5)
        for rank, (idx, dist) in enumerate(
            zip(indices_gru[0], distances_gru[0])
        ):
            sqlite_id = int(idx) + 1
            yamda_song_id = db.get_yamda_id(sqlite_id)
            # Calibrate raw similarity by the model's like-confidence
            calibrated = float(dist) * (0.5 + 0.5 * pos_intent_conf)
            db.cursor.execute(
                """INSERT INTO recommendations
                   (user_id, rank, model_type,
                    recommended_yamda_song_id, score)
                   VALUES (?, ?, 'GRU_Seq', ?, ?)""",
                (uid, rank + 1, yamda_song_id, calibrated),
            )

        # ----- Model 2: Heuristic Intent Engine -----
        heuristic_profile = heuristic_engine.compute_heuristic_profile(
            last_slice, raw_embeddings
        )

        # Guard: skip if profile is the zero vector (no usable signal)
        if np.linalg.norm(heuristic_profile) < 1e-8:
            continue

        query_vec = np.expand_dims(heuristic_profile, axis=0).astype(
            np.float32
        )
        distances_heur, indices_heur = faiss_index.search(query_vec, k=5)
        for rank, (idx, dist) in enumerate(
            zip(indices_heur[0], distances_heur[0])
        ):
            sqlite_id = int(idx) + 1
            yamda_song_id = db.get_yamda_id(sqlite_id)
            db.cursor.execute(
                """INSERT INTO recommendations
                   (user_id, rank, model_type,
                    recommended_yamda_song_id, score)
                   VALUES (?, ?, 'Heuristic_IntentEngine', ?, ?)""",
                (uid, rank + 1, yamda_song_id, float(dist)),
            )

    db.conn.commit()


def _print_sample_recommendations(
    db: YambdaSandboxDatabase,
    user_histories: Dict[int, List[Tuple[int, float, str, float]]],
    num_users: int = 5,
) -> None:
    """Log the recommendations for the first *num_users* users as a quick
    sanity check."""
    test_users = list(user_histories.keys())[:num_users]
    for uid in test_users:
        recent_yamda_ids = [
            db.get_yamda_id(item[0])
            for item in user_histories[uid][-5:]
        ]
        log_info(f"User {uid} history (last 5): {recent_yamda_ids}")

        db.cursor.execute(
            "SELECT recommended_yamda_song_id, score "
            "FROM recommendations "
            "WHERE user_id = ? AND model_type = 'GRU_Seq' "
            "ORDER BY rank",
            (uid,),
        )
        gru_recs = db.cursor.fetchall()
        log_info(f" ├─ GRU Top 5: {[r[0] for r in gru_recs]}")

        db.cursor.execute(
            "SELECT recommended_yamda_song_id, score "
            "FROM recommendations "
            "WHERE user_id = ? AND model_type = 'Heuristic_IntentEngine' "
            "ORDER BY rank",
            (uid,),
        )
        heur_recs = db.cursor.fetchall()
        log_info(f" └─ Heuristic Top 5: {[r[0] for r in heur_recs]}")
        log_info("-" * 66)


# =====================================================================
# Entry point
# =====================================================================

if __name__ == "__main__":
    run_sandbox_training()
