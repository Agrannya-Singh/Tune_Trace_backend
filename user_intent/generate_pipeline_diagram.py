"""
generate_pipeline_diagram.py

Generates a pydot/Graphviz diagram of the YAMDA-50M Sandbox Pipeline.
Run on Kaggle (Graphviz pre-installed) or locally after `apt install graphviz`.

Usage:
    python generate_pipeline_diagram.py
    # → writes yamda_pipeline_diagram.png to the current directory
"""

import pydot


def build_pipeline_graph() -> pydot.Dot:
    """Build and return the full pipeline flowchart as a pydot.Dot graph."""

    graph = pydot.Dot(
        "YAMDA_Pipeline",
        graph_type="digraph",
        rankdir="TB",
        bgcolor="white",
        fontname="Helvetica",
        label="YAMDA-50M Sandbox Pipeline",
        labelloc="t",
        labeljust="c",
        fontsize="22",
        fontcolor="#1a237e",
        pad="0.5",
    )
    graph.set_node_defaults(
        shape="box",
        style="rounded,filled",
        fontname="Helvetica",
        fontsize="10",
    )
    graph.set_edge_defaults(color="#757575", arrowsize="0.8", fontsize="9")

    # ── STAGE 1: Data Ingestion ──────────────────────────────────────
    stage1 = pydot.Cluster(
        "ingestion",
        label="STAGE 1 — DATA INGESTION",
        style="rounded",
        color="#bbdefb",
        bgcolor="#e3f2fd",
        fontname="Helvetica Bold",
        fontsize="12",
    )
    stage1.add_node(pydot.Node("kaggle", label="Kaggle Local\nParquets\n(embeddings + listens)", fillcolor="#bbdefb"))
    stage1.add_node(pydot.Node("hf", label="Hugging Face\nStreaming\n(fallback)", fillcolor="#bbdefb"))
    stage1.add_node(pydot.Node("duckdb", label="DuckDB ETL\n8 GB cap · dynamic schema\nstreaming 100k chunks", fillcolor="#90caf9"))
    stage1.add_node(pydot.Node("sqlite", label="SQLite DB\n(tracks + histories)", fillcolor="#e8eaf6"))
    stage1.add_node(pydot.Node("memmap", label="NumPy Memmap\n(L2-normalised embeddings)", fillcolor="#e8eaf6"))
    stage1.add_node(pydot.Node("mock", label="Synthetic Mock\nData (fallback)", fillcolor="#fff9c4", style="rounded,filled,dashed"))

    stage1.add_edge(pydot.Edge("kaggle", "duckdb"))
    stage1.add_edge(pydot.Edge("hf", "duckdb"))
    stage1.add_edge(pydot.Edge("duckdb", "sqlite"))
    stage1.add_edge(pydot.Edge("duckdb", "memmap"))
    stage1.add_edge(pydot.Edge("mock", "sqlite", style="dashed", label="fallback"))
    graph.add_subgraph(stage1)

    # ── STAGE 2: Indexing ────────────────────────────────────────────
    stage2 = pydot.Cluster(
        "indexing",
        label="STAGE 2 — FAISS INDEXING",
        style="rounded",
        color="#c8e6c9",
        bgcolor="#e8f5e9",
        fontname="Helvetica Bold",
        fontsize="12",
    )
    stage2.add_node(pydot.Node("faiss_builder", label="FAISS Index Builder", fillcolor="#a5d6a7"))
    stage2.add_node(pydot.Node("hnsw", label="< 50 k vectors\nHNSWFlat (exact)", fillcolor="#c8e6c9"))
    stage2.add_node(pydot.Node("ivfpq", label="≥ 50 k vectors\nIVFPQ (≈ 90 % RAM ↓)", fillcolor="#c8e6c9"))
    stage2.add_node(pydot.Node("faiss_bin", label="faiss_track_index.bin", fillcolor="#dcedc8", shape="note"))

    stage2.add_edge(pydot.Edge("faiss_builder", "hnsw"))
    stage2.add_edge(pydot.Edge("faiss_builder", "ivfpq"))
    stage2.add_edge(pydot.Edge("hnsw", "faiss_bin"))
    stage2.add_edge(pydot.Edge("ivfpq", "faiss_bin"))
    graph.add_subgraph(stage2)

    # ── STAGE 3: Training ────────────────────────────────────────────
    stage3 = pydot.Cluster(
        "training",
        label="STAGE 3 — HEAVY GRU TRAINING (v3: 512-dim + Attn + InfoNCE + Checkpoints)",
        style="rounded",
        color="#ffe0b2",
        bgcolor="#fff3e0",
        fontname="Helvetica Bold",
        fontsize="12",
    )
    stage3.add_node(pydot.Node("dataset", label="SequentialRecDataset\nwindow=20 · 4 intent classes", fillcolor="#ffe0b2"))
    stage3.add_node(pydot.Node("loader", label="DataLoader\nbatch_size 2048 / 512", fillcolor="#ffe0b2"))
    stage3.add_node(pydot.Node("gru", label="4-Layer GRU Backbone\n(hidden_dim=512)", fillcolor="#ffcc80"))
    stage3.add_node(pydot.Node("attn", label="Multi-Head Self-Attention\n(4 heads · sequence pooling)", fillcolor="#ffe082"))
    stage3.add_node(pydot.Node("proj_head", label="4-Layer Projection MLP\n(residual skip → 256d)", fillcolor="#fff9c4"))
    stage3.add_node(pydot.Node("intent_head", label="3-Layer Intent Head\n(LayerNorm → 4 classes)", fillcolor="#fff9c4"))
    stage3.add_node(pydot.Node("info_nce", label="InfoNCE Loss\n(learnable temperature τ)", fillcolor="#ffecb3"))
    stage3.add_node(pydot.Node("ce_loss", label="CrossEntropy\nIntent Loss × 0.3", fillcolor="#ffecb3"))
    stage3.add_node(pydot.Node("optim", label="AdamW + Cosine LR\nwarmup=5 ep · grad clip=1.0", fillcolor="#ffe0b2"))
    stage3.add_node(pydot.Node("ckpt", label="Checkpoints (every 15 eps)\nyamda_checkpoint_epoch_*.pt\nauto-resume supported", fillcolor="#b2dfdb", shape="note"))
    stage3.add_node(pydot.Node("weights", label="yamda_best_model.pt\nyamda_gru_weights.pt", fillcolor="#dcedc8", shape="note"))

    stage3.add_edge(pydot.Edge("dataset", "loader"))
    stage3.add_edge(pydot.Edge("loader", "gru"))
    stage3.add_edge(pydot.Edge("gru", "attn"))
    stage3.add_edge(pydot.Edge("attn", "proj_head"))
    stage3.add_edge(pydot.Edge("attn", "intent_head"))
    stage3.add_edge(pydot.Edge("proj_head", "info_nce"))
    stage3.add_edge(pydot.Edge("intent_head", "ce_loss"))
    stage3.add_edge(pydot.Edge("info_nce", "optim", label="sum"))
    stage3.add_edge(pydot.Edge("ce_loss", "optim", label="sum"))
    stage3.add_edge(pydot.Edge("optim", "ckpt"))
    stage3.add_edge(pydot.Edge("optim", "weights"))
    graph.add_subgraph(stage3)

    # ── STAGE 4: Recommendation ──────────────────────────────────────
    stage4 = pydot.Cluster(
        "recommendation",
        label="STAGE 4 — RECOMMENDATION GENERATION",
        style="rounded",
        color="#ce93d8",
        bgcolor="#f3e5f5",
        fontname="Helvetica Bold",
        fontsize="12",
    )
    stage4.add_node(pydot.Node("gru_infer", label="GRU Seq Model\n(inference)", fillcolor="#e1bee7"))
    stage4.add_node(pydot.Node("heuristic", label="Heuristic Intent\nEngine\n(decay-weighted avg)", fillcolor="#e1bee7"))
    stage4.add_node(pydot.Node("faiss_query", label="FAISS Index\ntop-5 query", fillcolor="#ce93d8"))
    stage4.add_node(pydot.Node("calibrate", label="Intent Calibration\nscore × (0.5 + 0.5·P(like))", fillcolor="#f8bbd0"))
    stage4.add_node(pydot.Node("recs_db", label="recommendations.db\n(SQLite)", fillcolor="#dcedc8", shape="note"))

    stage4.add_edge(pydot.Edge("gru_infer", "faiss_query", label="query"))
    stage4.add_edge(pydot.Edge("heuristic", "faiss_query", label="query"))
    stage4.add_edge(pydot.Edge("faiss_query", "calibrate", label="GRU scores"))
    stage4.add_edge(pydot.Edge("faiss_query", "recs_db", label="heuristic\nscores"))
    stage4.add_edge(pydot.Edge("calibrate", "recs_db"))
    graph.add_subgraph(stage4)

    # ── Cross-stage edges ────────────────────────────────────────────
    graph.add_edge(pydot.Edge("memmap", "faiss_builder", color="#43a047"))
    graph.add_edge(pydot.Edge("memmap", "dataset", color="#ef6c00", style="dashed"))
    graph.add_edge(pydot.Edge("faiss_bin", "faiss_query", color="#7b1fa2", style="bold"))
    graph.add_edge(pydot.Edge("weights", "gru_infer", color="#7b1fa2", style="bold"))

    return graph


if __name__ == "__main__":
    g = build_pipeline_graph()
    output_path = "yamda_pipeline_diagram.png"
    g.write_png(output_path)
    print(f"Pipeline diagram saved to {output_path}")
