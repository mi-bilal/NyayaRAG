# NyayaRAG

Streamlit legal RAG app for the `CaseText + Statutes` pipeline from `notebooks/NyayaRAG_(2).ipynb`.

## Stack

- Streamlit UI
- Groq `openai/gpt-oss-120b` generation
- Qwen3 embeddings (`Qwen/Qwen3-Embedding-0.6B`)
- Qdrant vector store
- BM25 sparse retrieval
- Hybrid RRF ranking
- `uv` for dependency management

## Setup

```bash
uv sync
uv run streamlit run app/streamlit_app.py
```

The app needs prebuilt artifacts under `artifacts/qdrant` and `artifacts/bm25/statutes_bm25.pkl`.

For a quick UI smoke test without Colab artifacts:

```bash
uv run python scripts/build_sample_artifacts.py
uv run streamlit run app/streamlit_app.py
```

## Build Artifacts

Recommended: use the one-shot Colab guide in `COLAB_ARTIFACTS.md`.

1. Prepare statute chunks from the NyayaRAG corpus JSON:

```bash
uv run python scripts/prepare_corpus.py --input data/raw/updated_SCI_56k_single.json
```

2. Build embeddings and Qdrant on Colab GPU:

```bash
uv run python scripts/colab_build_qwen_qdrant.py
```

3. Download or unzip the generated artifact zip into `artifacts/` locally.

## Local BM25 Only

```bash
uv run python scripts/build_bm25.py
```

## Test

```bash
uv run pytest
```

## Notes

This is a legal research assistant, not legal advice. V1 intentionally implements only `CaseText + Statutes`.
