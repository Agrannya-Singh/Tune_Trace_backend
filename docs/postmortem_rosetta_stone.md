# ADR-001: YAMBDA Multimodal Integration — Feasibility Assessment & Decision Record

**Project:** TuneTrace — Semantic Music Recommendation Engine  
**Team:** Samsung PRISM (SRI-B)  
**Date:** 2026-07-03  
**Author:** Agrannya Singh  
**Status:** Decided — Approach Pivoted  
**Decision:** Retain text-only semantic pipeline; defer multimodal integration pending a suitable paired dataset.

---

## 1. Context

The TuneTrace recommendation engine operates on a text-semantic pipeline: song metadata (title, artist, genre, tags) is encoded into 384-dimensional dense vectors using `all-MiniLM-L6-v2` (a distilled Sentence Transformer), stored in a pgvector-enabled PostgreSQL column, and retrieved via HNSW cosine-distance nearest-neighbor search.

As part of the Samsung PRISM worklet, the team was directed to explore **multimodal recommendation** — augmenting the existing text-based retrieval with audio signal features. The **Yandex YAMBDA** (Yandex Music Billion-Interactions Dataset) was identified as a candidate data source, as it ships with pre-computed audio embeddings for ~7.72 million tracks alongside billions of user interaction events.

This document records the investigation, the data constraints discovered, the architectural decision taken, and the engineering outcomes of the spike.

---

## 2. Approach: Rosetta Stone Projection Architecture

### 2.1 Hypothesis

If we can learn a mapping function from YAMBDA's native audio embedding space into our existing `all-MiniLM-L6-v2` 384-d text space, we can ingest audio-derived vectors into the same `song_metadata.embedding` column — enabling the recommendation engine to surface results based on acoustic similarity without modifying the retrieval infrastructure.

### 2.2 Architecture

A lightweight MLP ("Rosetta Stone Mapper") was implemented to learn the projection:

$$f: \mathbb{R}^{d_{\text{audio}}} \rightarrow \mathbb{R}^{384}$$

| Component | Location | Role |
|-----------|----------|------|
| `rosetta_stone.py` | Project root | PyTorch MLP mapper, dataset class, training loop |
| `scripts/janitor_yamda_backfill.py` | `scripts/` | GitHub Actions janitor: streams YAMBDA via HuggingFace, trains mapper on a 5,000-track calibration subset, projects 100K embeddings, bulk-inserts into Supabase |
| `.github/workflows/yamda.yaml` | `.github/workflows/` | `workflow_dispatch` Action to orchestrate the backfill |
| `multi_modal.mmd` | Project root | Architecture diagram for the dual-branch pipeline |

**Network specification:**
```
RosettaStoneMapper(
    Linear(d_audio → 512) → BatchNorm1d → ReLU → Dropout(0.2)
    Linear(512 → 512)     → BatchNorm1d → ReLU
    Linear(512 → 384)     → L2 Normalize
)

Loss: CosineEmbeddingLoss (angular alignment)
Optimizer: Adam (lr=1e-3)
Training: 3–5 epochs on 5,000 calibration pairs
```

### 2.3 Ingestion Pipeline

The janitor script was designed to run as a one-shot GitHub Action:

1. Stream 100K audio embeddings from the YAMBDA-50M variant via HuggingFace (`streaming=True`).
2. Train the Rosetta Stone MLP on 5,000 paired vectors (audio → text target).
3. Project the remaining 95,000 audio embeddings through the trained mapper.
4. Bulk-insert the 384-d projected vectors into `song_metadata` via `psycopg2.extras.execute_values`.

---

## 3. Constraints Discovered

During implementation and integration testing, two fundamental data constraints were identified that make the YAMBDA dataset incompatible with cross-modal projection into our text-semantic space.

### 3.1 Proprietary CNN Embedding Origin

The YAMBDA audio embeddings are produced by a **custom-trained CNN** built by Yandex Research, trained on raw audio spectrograms. The model architecture, training objective, loss function, and hyperparameters are proprietary — only the output vectors are published.

**Implication for cross-modal projection:**

A projection network learns a *structural correspondence* between two vector spaces. This requires that both spaces encode semantically overlapping information in a way that admits a smooth, learnable mapping. The YAMBDA CNN space is organized by **perceptual audio features** (timbre, rhythm, spectral energy), while `all-MiniLM-L6-v2` is organized by **linguistic semantics** (word meaning, genre labels, artist context). These organizational axes are fundamentally disjoint.

Without access to the CNN's architecture or an intermediate representation that bridges the two modalities, a shallow MLP will overfit to spurious correlations in the calibration set and produce semantically meaningless projections at scale.

### 3.2 Full Data Anonymization — No Audio–Text Pairing Possible

The YAMBDA dataset is **fully anonymized**. All track identifiers are opaque integer IDs (`item_id`) with zero mapping to external metadata — no song titles, no artist names, no genre labels, no ISRCs, no MusicBrainz links.

**Why this is the decisive constraint:**

The Rosetta Stone MLP requires **paired training data**: matched tuples of `(audio_embedding, text_embedding)` for the *same song*. To generate the text-side target, we need to encode a rich descriptive string like `"Bohemian Rhapsody. Artist: Queen. Genre: Rock. Tags: classic, progressive"` through `all-MiniLM-L6-v2`.

Because the data is scrubbed, the ingestion script had no choice but to fabricate synthetic text targets:

```python
# From the ingestion janitor (now removed)
mock_titles.append(f"Yambda Track {item_id}")    # ← Not a real title
mock_artists.append(f"Yambda Artist")             # ← Not a real artist
```

These synthetic strings are near-identical across 100K tracks, producing text embeddings that cluster in a tiny region of the 384-d space. Training the MLP against these targets causes it to converge to a **constant function** — outputting roughly the same vector for every input, destroying all discriminative information from the audio space. Retrieval on these vectors would return effectively random results.

> **Core Issue:** The YAMBDA dataset is designed for collaborative filtering research (user×item interaction modeling), not for content-based multimodal retrieval. Its anonymization is intentional and by design — the embeddings are meant to serve as opaque content features within recommendation models, not as inputs to cross-space projection.

---

## 4. Engineering Outcomes

Although the multimodal approach was not viable for the reasons above, the investigation was not without value. The spike produced tangible engineering outcomes for the project:

### 4.1 Pipeline Infrastructure Hardened

The YAMBDA integration required building and stress-testing a full GitHub Actions ingestion pipeline — streaming large HuggingFace datasets, training PyTorch models on CI runners, and bulk-inserting into Supabase. Several latent issues in the CI/CD infrastructure were discovered and resolved as a direct result:

| Issue Discovered | Impact | Resolution |
|-------|--------|------------|
| `v3_janitor.yml` referenced `python v3_janitor.py` (root) instead of `python scripts/v3_janitor.py` | **Nightly cron job failing silently since deployment** | Fixed path; added model caching and standardized dependency install |
| `sys.path` in all `scripts/*.py` pointed to wrong directory | All scripts broken when invoked from outside project root | Fixed to use `os.path.join(os.path.dirname(__file__), "..")` |
| `print_tag_extraction.py` called non-existent method `_build_feature_text()` | Script unusable since V3 rewrite | Updated to current API (`MLEngine.build_text_context()`) |
| Workflow dependency installs split across `pip` and `poetry` | Non-reproducible environments on runners | Standardized to single `pip install -r requirements.txt` |
| `streaming=False` on 50GB dataset vs. 14GB Actions disk | Would crash runner before processing a single record | Enforced `streaming=True` pattern |

### 4.2 Bulk Insert Performance Characterized

The Supabase Nano instance's behavior under bulk pgvector writes was profiled. Key finding: `psycopg2.extras.execute_values` with 1,000-row chunks outperforms SQLAlchemy ORM inserts by ~10× on the constrained instance. This knowledge directly benefits any future large-scale ingestion work.

### 4.3 Dataset Evaluation Framework Established

The `YambdaDataset` wrapper class (using `datasets.load_dataset` with `streaming=True`) demonstrated a pattern for evaluating large HuggingFace datasets within the resource constraints of GitHub Actions runners — a reusable pattern for future dataset evaluations.

---

## 5. Decision

**Retain the text-only semantic pipeline** as the production recommendation architecture. The `all-MiniLM-L6-v2` encoder with pgvector HNSW cosine retrieval:

- Operates on **real, rich metadata** (titles, artists, genres, user-generated tags) — genuine semantic signal.
- Requires no cross-modal translation — encoder and retrieval index share the same native space.
- Is proven in production with sub-second query latency on the current catalog.

**Defer multimodal integration** until a dataset with the following properties is available:
- (a) Matched audio files/embeddings **and** human-readable metadata for the same tracks, OR
- (b) A publicly documented audio encoder (e.g., CLAP, MERT, MusicGen encoder) whose latent space organization is well-characterized, enabling principled cross-modal alignment.

---

## 6. Cleanup Record

All Rosetta Stone and YAMBDA-specific code was removed from the repository on 2026-07-03:

| File | Type | Action |
|------|------|--------|
| `rosetta_stone.py` | PyTorch MLP module | **Removed** |
| `scripts/janitor_yamda_backfill.py` | Backfill janitor script | **Removed** |
| `.github/workflows/yamda.yaml` | GitHub Actions workflow | **Removed** |
| `multi_modal.mmd` | Architecture diagram | **Removed** |

Verification: zero remaining references to `rosetta`, `yamda`, `yambda`, or `multimodal` across all `.py`, `.yml`, `.yaml`, and `.toml` files.

---

## 7. Key Takeaways

1. **Cross-modal projection requires paired data with shared semantic grounding.** The Rosetta Stone metaphor is instructive: the actual Rosetta Stone succeeded because it contained the *same text* in three scripts. Our scenario had real audio on one side and synthetic placeholder text on the other — no semantic bridge for the network to learn.

2. **The YAMBDA dataset is purpose-built for collaborative filtering**, not content-based multimodal retrieval. Its anonymization is a deliberate design choice by Yandex to protect user/artist privacy while enabling interaction-pattern research. This is clearly documented but easy to overlook when focusing on the audio embedding component in isolation.

3. **Spike-driven investigation has compounding returns.** Even though the multimodal approach was ruled out, the infrastructure stress-testing it required exposed a nightly cron failure, multiple broken script paths, and stale method references — all of which are now resolved.

---

*Filed as ADR-001 under `docs/postmortem_rosetta_stone.md` in the TuneTrace backend repository.*  
*Filename retained as `postmortem_rosetta_stone.md` for git history continuity.*
