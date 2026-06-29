# Colab Artifact Build And Local Run Guide

Use Colab only for the GPU-heavy embedding/index build. The local Streamlit app only needs the generated `artifacts/` directory.

The intended workflow is:

```text
GitHub repo -> Colab GPU indexing -> artifact zip in Google Drive -> unzip locally -> run Streamlit locally
```

You do not need to run `notebooks/NyayaRAG_(2).ipynb` for the app path.

## 0. What Colab Builds

Colab will build these files:

```text
artifacts/qdrant/
artifacts/bm25/statutes_bm25.pkl
artifacts/manifests/index_manifest.json
```

Those are the only heavyweight runtime artifacts the local app needs.

## 1. Start Colab

1. Open <https://colab.research.google.com/>.
2. Create a new notebook.
3. Go to `Runtime -> Change runtime type`.
4. Select a GPU runtime, ideally `T4`, `L4`, or `A100` if available.
5. Save the notebook if you want to reuse it.

Quick GPU check:

```python
!nvidia-smi
```

If this says no GPU, switch runtime type before continuing.

## 2. Install uv In Colab

```python
!curl -LsSf https://astral.sh/uv/install.sh | sh
import os
os.environ["PATH"] = os.environ["HOME"] + "/.local/bin:" + os.environ["PATH"]
!uv --version
```

## 3. Clone The Repo

Public clone:

```python
%cd /content
!git clone https://github.com/mi-bilal/NyayaRAG.git
%cd /content/NyayaRAG
```

If you already cloned it and want the latest version:

```python
%cd /content/NyayaRAG
!git pull origin main
```

## 4. Add Secrets In Colab

Preferred: use Colab secrets.

1. Click the key icon in the left sidebar.
2. Add `HF_TOKEN`.
3. Add `GROQ_API_KEY` only if you want to test generation on Colab too.

Then create a local `.env` inside Colab from secrets:

```python
from google.colab import userdata

hf = userdata.get("HF_TOKEN") or ""
groq = userdata.get("GROQ_API_KEY") or ""

env = f"""HF_TOKEN={hf}
GROQ_API_KEY={groq}
GROQ_MODEL=openai/gpt-oss-120b
EMBEDDING_MODEL=Qwen/Qwen3-Embedding-0.6B
EMBEDDING_DIM=1024
QDRANT_PATH=artifacts/qdrant
QDRANT_COLLECTION=nyayarag_statutes
BM25_PATH=artifacts/bm25/statutes_bm25.pkl
DATA_DIR=data
ARTIFACT_DIR=artifacts
TOP_K=5
CANDIDATE_POOL=80
RRF_K=60
MAX_CASE_TOKENS=3500
MAX_CONTEXT_TOKENS=5000
"""

open(".env", "w").write(env)
```

Shortcut if you do not want Colab secrets:

```python
%%writefile .env
HF_TOKEN=your_hf_token_here
GROQ_API_KEY=your_groq_key_here
GROQ_MODEL=openai/gpt-oss-120b
EMBEDDING_MODEL=Qwen/Qwen3-Embedding-0.6B
EMBEDDING_DIM=1024
QDRANT_PATH=artifacts/qdrant
QDRANT_COLLECTION=nyayarag_statutes
BM25_PATH=artifacts/bm25/statutes_bm25.pkl
DATA_DIR=data
ARTIFACT_DIR=artifacts
TOP_K=5
CANDIDATE_POOL=80
RRF_K=60
MAX_CASE_TOKENS=3500
MAX_CONTEXT_TOKENS=5000
```

Do not commit `.env`.

## 5. Mount Google Drive

The artifact zip will be saved to Drive so it survives Colab disconnects.

```python
from google.colab import drive
drive.mount('/content/drive')
```

## 6. Install Dependencies

```python
%cd /content/NyayaRAG
!uv sync --dev
```

This may take a few minutes the first time.

This installs `accelerate` from `pyproject.toml`. The Colab scripts no longer use
`device_map="auto"` or deprecated `tokenizer_kwargs`; they load Qwen3 directly on
`cuda` when Colab exposes a GPU.

If you previously cloned before this fix, update first:

```python
%cd /content/NyayaRAG
!git pull origin main
!uv sync --dev
```

## 7. Do A Small Test Build First

Run a small 500-record build to confirm the full workflow works before spending GPU time on the full corpus.

```python
!uv run python scripts/colab_full_build.py --batch-size 16 --limit 500
```

Expected output ends with something like:

```text
Artifact zip: /content/drive/MyDrive/nyayarag_artifacts/nyayarag_artifacts.zip
```

## 8. Run The Full GPU Build

After the small test succeeds:

```python
!rm -rf artifacts
!uv run python scripts/colab_full_build.py --batch-size 16
```

If Colab runs out of memory, retry with a smaller batch size:

```python
!rm -rf artifacts
!uv run python scripts/colab_full_build.py --batch-size 8
```

If you get a better GPU like A100/L4, you can try:

```python
!rm -rf artifacts
!uv run python scripts/colab_full_build.py --batch-size 64
```

## 9. Verify Artifact Zip In Drive

```python
!ls -lh /content/drive/MyDrive/nyayarag_artifacts/
```

You should see:

```text
nyayarag_artifacts.zip
```

## 10. Download Artifacts To Local Machine

Download this file from Google Drive:

```text
MyDrive/nyayarag_artifacts/nyayarag_artifacts.zip
```

Put it in your local project root:

```text
D:\Work\KICS\NyayaRAG\nyayarag_artifacts.zip
```

Then unzip it so the local repo has:

```text
D:\Work\KICS\NyayaRAG\artifacts\qdrant\
D:\Work\KICS\NyayaRAG\artifacts\bm25\statutes_bm25.pkl
D:\Work\KICS\NyayaRAG\artifacts\manifests\index_manifest.json
```

PowerShell unzip from local project root:

```powershell
Expand-Archive -LiteralPath "nyayarag_artifacts.zip" -DestinationPath "artifacts" -Force
```

Important: if the zip extracts as `artifacts/artifacts/...`, move the inner contents up so the path is exactly `artifacts/qdrant` and `artifacts/bm25`.

## 11. Local Setup

In the local repo:

```powershell
uv sync --dev
```

Create local `.env` from `.env.example` if it does not exist:

```powershell
Copy-Item -LiteralPath ".env.example" -Destination ".env"
```

Then edit `.env` and set:

```env
GROQ_API_KEY=your_groq_key_here
HF_TOKEN=your_hf_token_here
GROQ_MODEL=openai/gpt-oss-120b
```

For local app runtime, `HF_TOKEN` is not usually needed unless the app has to download the embedding model for query embedding. Keep it set anyway.

## 12. Run Locally

```powershell
uv run streamlit run app/streamlit_app.py
```

The browser should open automatically. If not, Streamlit prints a local URL like:

```text
http://localhost:8501
```

## 13. Local Smoke Test Without Full Artifacts

If you only want to confirm the UI works before downloading full Colab artifacts:

```powershell
uv run python scripts/build_sample_artifacts.py
uv run streamlit run app/streamlit_app.py
```

This uses tiny fake sample vectors and is only for app testing, not real retrieval quality.

## 14. Troubleshooting

If the app says `BM25 artifact not found`:

```text
artifacts/bm25/statutes_bm25.pkl
```

is missing or nested incorrectly after unzip.

If the app says `Qdrant artifact directory not found`:

```text
artifacts/qdrant/
```

is missing or nested incorrectly after unzip.

If local query embedding is slow on first run, that is normal. The app downloads/loads `Qwen/Qwen3-Embedding-0.6B` once.

If Colab disconnects, rerun from step 6. The final output is only useful once the artifact zip has been written to Drive.

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
