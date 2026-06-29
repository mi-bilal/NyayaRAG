from __future__ import annotations

from pathlib import Path

import streamlit as st

from nyayarag.config import get_settings
from nyayarag.generation.answer import generate_legal_analysis
from nyayarag.groq_client import GroqGenerator
from nyayarag.retrieval.hybrid import HybridRetriever, RetrievalNotReadyError

st.set_page_config(page_title="NyayaRAG", page_icon="⚖", layout="wide")


@st.cache_resource(show_spinner=False)
def load_retriever() -> HybridRetriever:
    settings = get_settings()
    return HybridRetriever.from_settings(settings)


@st.cache_resource(show_spinner=False)
def load_generator() -> GroqGenerator:
    return GroqGenerator(get_settings())


def render_result(result: dict) -> None:
    outcome = str(result.get("outcome", "uncertain")).title()
    confidence = result.get("confidence")
    if outcome.lower() in {"allowed", "accepted"}:
        st.success(f"Likely outcome: {outcome}")
    elif outcome.lower() in {"dismissed", "rejected"}:
        st.error(f"Likely outcome: {outcome}")
    else:
        st.warning(f"Likely outcome: {outcome}")
    if isinstance(confidence, int | float):
        st.progress(max(0.0, min(1.0, float(confidence))), text=f"Confidence: {confidence:.2f}")

    for title, key in [
        ("Issues", "issues"),
        ("Applicable Statutes", "applicable_statutes"),
        ("Application To Facts", "application_to_facts"),
        ("Caveats", "caveats"),
    ]:
        values = result.get(key) or []
        if values:
            st.subheader(title)
            for value in values:
                if isinstance(value, dict):
                    citation = value.get("citation", "Citation")
                    reason = value.get("why_relevant", "")
                    st.markdown(f"- **{citation}**: {reason}")
                else:
                    st.markdown(f"- {value}")

    if result.get("final_reasoning"):
        st.subheader("Final Reasoning")
        st.write(result["final_reasoning"])

    with st.expander("Raw JSON"):
        st.json(result)


def main() -> None:
    settings = get_settings()

    st.title("NyayaRAG")
    st.caption("CaseText + Statutes retrieval with Qdrant, Qwen3 embeddings, and Groq generation.")

    with st.sidebar:
        st.header("Runtime")
        st.write(f"Model: `{settings.groq_model}`")
        st.write(f"Embedding: `{settings.embedding_model}`")
        st.write(f"Vector DB: `{settings.qdrant_path}`")
        top_k = st.slider("Top statutes", 3, 12, settings.top_k)
        candidate_pool = st.slider("Candidate pool", 20, 150, settings.candidate_pool, step=10)
        show_debug = st.toggle("Show retrieval debug", value=True)

    sample_path = Path(settings.data_dir) / "samples" / "case_input.txt"
    default_text = sample_path.read_text(encoding="utf-8") if sample_path.exists() else ""

    case_text = st.text_area(
        "Case text",
        value=default_text,
        height=300,
        placeholder="Paste case facts, proceedings, or judgment summary here...",
    )
    known_sections = st.text_area(
        "Known sections or articles (optional)",
        height=100,
        placeholder="Example: Article 14, Article 21, Section 45 PMLA...",
    )

    col_a, col_b = st.columns([1, 1])
    retrieve_clicked = col_a.button("Retrieve Statutes", type="secondary", use_container_width=True)
    analyze_clicked = col_b.button("Generate Analysis", type="primary", use_container_width=True)

    if not case_text.strip():
        st.info("Paste case text to begin.")
        return

    if retrieve_clicked or analyze_clicked:
        try:
            retriever = load_retriever()
            with st.spinner("Retrieving statutes..."):
                retrieved = retriever.retrieve(
                    case_text=case_text,
                    known_sections=known_sections,
                    top_k=top_k,
                    candidate_pool=candidate_pool,
                )
        except RetrievalNotReadyError as exc:
            st.error(str(exc))
            st.markdown(
                "Run the Colab indexing script first, then place artifacts under `artifacts/`."
            )
            return

        st.subheader("Retrieved Statutes")
        for item in retrieved:
            with st.container(border=True):
                st.markdown(f"**[{item.rank}] {item.title or item.statute_id}**")
                st.write(item.text)
                if show_debug:
                    st.caption(
                        f"dense_rank={item.dense_rank} bm25_rank={item.bm25_rank} "
                        f"rrf={item.rrf_score:.5f} id={item.statute_id}"
                    )

        if analyze_clicked:
            generator = load_generator()
            with st.spinner("Generating legal analysis with Groq..."):
                result = generate_legal_analysis(
                    generator=generator,
                    case_text=case_text,
                    known_sections=known_sections,
                    retrieved_statutes=retrieved,
                    settings=settings,
                )
            render_result(result)

    with st.expander("About"):
        st.write(
            "This app is a research assistant. It retrieves statute context and produces "
            "structured legal analysis. It is not legal advice."
        )
        st.code("uv run streamlit run app/streamlit_app.py")


if __name__ == "__main__":
    main()
