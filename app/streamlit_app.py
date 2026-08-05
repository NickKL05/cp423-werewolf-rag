"""Interactive demo and corpus browser for the Werewolf: the Apocalypse RAG system.

Two tabs:

  Ask          Run a question through any retriever, see the retrieved chunks
               beside the generated answer, with citations resolved back to the
               passage and the source page.
  Corpus       Search the chunk index directly. This is the tool used to find
               gold chunk IDs when writing the evaluation set.

Run:
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wolfrag import config, generate, retrieval  # noqa: E402

st.set_page_config(
    page_title="Werewolf: the Apocalypse RAG", page_icon="C", layout="wide"
)


@st.cache_resource(show_spinner="Building retrievers ...")
def load_retrievers():
    return retrieval.build_all()


@st.cache_data(show_spinner=False)
def corpus_summary():
    chunks = retrieval.load_chunks()
    titles = sorted({c["title"] for c in chunks})
    return len(chunks), titles


def render_chunk(chunk: dict, show_score: bool = True) -> None:
    score = f" | score {chunk['score']:.4f}" if show_score and "score" in chunk else ""
    header = (
        f"**{chunk['chunk_id']}** | {chunk['title']} | *{chunk['section']}*"
        f" | {chunk['n_tokens']} tokens{score}"
    )
    st.markdown(header)
    st.caption(chunk["url"])
    st.write(chunk["text"])
    st.divider()


st.title("Werewolf: the Apocalypse RAG")
st.caption(
    f"{config.GENERATION_MODEL} via local Ollama | "
    f"embeddings {config.EMBEDDING_MODEL} | corpus from whitewolf.fandom.com (CC BY-SA)"
)

tab_ask, tab_corpus = st.tabs(["Ask", "Corpus browser"])

with tab_ask:
    col_controls, col_main = st.columns([1, 3])

    with col_controls:
        system = st.radio(
            "Retrieval system",
            options=["hybrid", "bm25", "dense", "closed_book"],
            help="closed_book runs the same model with no retrieved context.",
        )
        top_k = st.slider("Chunks in context (k)", 1, 10, config.TOP_K)
        st.caption(
            f"temperature {config.GENERATION_TEMPERATURE}, seed {config.GENERATION_SEED}"
        )

    with col_main:
        question = st.text_input(
            "Question",
            placeholder="Which tribe was destroyed after descending into the Black Spiral?",
        )
        run = st.button("Ask", type="primary")

    if run and question.strip():
        closed_book = system == "closed_book"
        chunks = []

        if not closed_book:
            retrievers = load_retrievers()
            with st.spinner(f"Retrieving with {system} ..."):
                chunks = retrievers[system].retrieve(question, top_k)

        if not generate.check_server():
            st.error(
                f"Ollama is not reachable at {config.OLLAMA_HOST}, or the model "
                f"{config.GENERATION_MODEL} is not installed."
            )
        else:
            with st.spinner("Generating ..."):
                result = generate.answer(question, chunks, closed_book=closed_book)

            answer_col, context_col = st.columns([1, 1])

            with answer_col:
                st.subheader("Answer")
                if result["refused"]:
                    st.warning(result["answer"])
                else:
                    st.success(result["answer"])

                if result["cited_chunk_ids"]:
                    st.markdown("**Cited passages**")
                    by_id = {c["chunk_id"]: c for c in chunks}
                    for chunk_id in result["cited_chunk_ids"]:
                        cited = by_id.get(chunk_id)
                        if cited:
                            with st.expander(
                                f"{chunk_id} | {cited['title']} | {cited['section']}"
                            ):
                                st.write(cited["text"])
                                st.caption(cited["url"])
                        else:
                            st.error(
                                f"{chunk_id} was cited but never retrieved. "
                                "This is a fabricated citation."
                            )
                elif not closed_book:
                    st.info("The model produced no inline citation.")

            with context_col:
                st.subheader(f"Retrieved context ({len(chunks)})")
                for chunk in chunks:
                    render_chunk(chunk)

with tab_corpus:
    n_chunks, titles = corpus_summary()
    st.write(
        f"{len(titles)} pages, {n_chunks} chunks. "
        "Use this tab to find gold chunk IDs when writing evaluation questions."
    )

    mode = st.radio(
        "Find chunks by", ["Search", "Page title"], horizontal=True
    )

    if mode == "Search":
        query = st.text_input("Search the corpus", key="corpus_query")
        n_results = st.slider("Results", 5, 40, 10, key="corpus_n")
        if query.strip():
            retrievers = load_retrievers()
            method = st.selectbox(
                "Ranking", ["hybrid", "bm25", "dense"], key="corpus_method"
            )
            for chunk in retrievers[method].retrieve(query, n_results):
                render_chunk(chunk)
    else:
        title = st.selectbox("Page", titles, key="corpus_title")
        chunks = [c for c in retrieval.load_chunks() if c["title"] == title]
        st.write(f"{len(chunks)} chunks on this page.")
        for chunk in chunks:
            render_chunk(chunk, show_score=False)
