# Colab Artifact Build

Use Colab only for the GPU-heavy embedding/index build. The local Streamlit app only needs the generated `artifacts/` directory.

## Files You Need In Colab

Upload or clone the project repo in Colab. The important files are:

- `pyproject.toml`
- `uv.lock`
- `.env` or Colab secrets for `HF_TOKEN`
- `src/nyayarag/**`
- `scripts/colab_full_build.py`

You do not need to run the notebook for the app path.

## Colab Commands

```bash
!curl -LsSf https://astral.sh/uv/install.sh | sh
import os
os.environ["PATH"] = os.environ["HOME"] + "/.local/bin:" + os.environ["PATH"]
```

```bash
from google.colab import drive
drive.mount('/content/drive')
```

If you cloned the repo:

```bash
%cd /content/NyayaRAG
!uv sync --dev
!uv run python scripts/colab_full_build.py --batch-size 32
```

For a quick small test before the full run:

```bash
!uv run python scripts/colab_full_build.py --batch-size 32 --limit 500
```

## Output To Bring Back Locally

Colab writes:

```text
/content/drive/MyDrive/nyayarag_artifacts/nyayarag_artifacts.zip
```

Unzip that file into the local project root. After unzip, these must exist:

```text
artifacts/qdrant/
artifacts/bm25/statutes_bm25.pkl
artifacts/manifests/index_manifest.json
```

Then run locally:

```bash
uv sync --dev
uv run streamlit run app/streamlit_app.py
```

## Expected Flow

1. Colab downloads `L-NLProc/NyayaRAG` dataset archive.
2. It extracts statute-like text from the `sections` field.
3. It embeds statute chunks using `Qwen/Qwen3-Embedding-0.6B` on GPU.
4. It builds Qdrant local vector artifacts.
5. It builds BM25 sparse retrieval artifacts.
6. It zips `artifacts/` to Google Drive.
7. You unzip that locally and run Streamlit.
