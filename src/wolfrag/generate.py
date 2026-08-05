"""Answer generation against a locally deployed Llama model through Ollama.

The prompt enforces the three behaviours the assignment requires: answer only
from the retrieved context, cite the supporting passage inline by its chunk ID,
and return "I don't know" when the context is insufficient.

Every system in the comparison uses the same model and the same decoding
settings. Temperature is zero and the sampling seed is fixed, so a rerun
reproduces the reported answers.
"""

from __future__ import annotations

import json
import re
from typing import Any

import requests

from wolfrag import config

CITATION_RE = re.compile(r"\[(C\d{5})\]")

SYSTEM_PROMPT = """You are a careful research assistant answering questions \
about the tabletop roleplaying game Werewolf: the Apocalypse.

Follow these rules exactly:
1. Answer using ONLY the information in the context passages provided. Never \
use prior knowledge, even if you believe you know the answer.
2. Cite the supporting passage inline using its ID in square brackets, for \
example [C00123]. Every factual claim needs a citation.
3. If the context passages do not contain enough information to answer the \
question, reply with exactly: I don't know
4. Do not speculate, and do not pad the answer. Three sentences at most."""

CLOSED_BOOK_SYSTEM_PROMPT = """You are a careful research assistant answering \
questions about the tabletop roleplaying game Werewolf: the Apocalypse.

Follow these rules exactly:
1. Answer from your own knowledge. No reference material is provided.
2. If you do not know the answer, reply with exactly: I don't know
3. Do not speculate, and do not pad the answer. Three sentences at most."""

DIAGNOSTIC_SYSTEM_PROMPT = """You are answering trivia questions about the \
tabletop roleplaying game Werewolf: the Apocalypse from your own knowledge.

Answer each question as accurately and specifically as you can. If you are \
uncertain, still give your single best answer rather than declining. Reply in \
one or two sentences."""


def format_context(chunks: list[dict[str, Any]]) -> str:
    """Render retrieved chunks into the numbered block the model reads."""
    blocks = []
    for chunk in chunks:
        header = (
            f"[{chunk['chunk_id']}] Page: {chunk['title']} | "
            f"Section: {chunk['section']}"
        )
        blocks.append(f"{header}\n{chunk['text']}")
    return "\n\n".join(blocks)


def build_prompt(question: str, chunks: list[dict[str, Any]] | None) -> str:
    if not chunks:
        return f"Question: {question}\n\nAnswer:"
    return (
        "Context passages:\n\n"
        f"{format_context(chunks)}\n\n"
        f"Question: {question}\n\nAnswer:"
    )


def ollama_generate(
    prompt: str,
    system: str,
    model: str = config.GENERATION_MODEL,
    temperature: float = config.GENERATION_TEMPERATURE,
) -> str:
    """Call the local Ollama server and return the generated text."""
    payload = {
        "model": model,
        "prompt": prompt,
        "system": system,
        "stream": False,
        "keep_alive": config.GENERATION_KEEP_ALIVE,
        "options": {
            "temperature": temperature,
            "top_p": config.GENERATION_TOP_P,
            "seed": config.GENERATION_SEED,
            "num_predict": config.GENERATION_NUM_PREDICT,
            "num_ctx": config.GENERATION_NUM_CTX,
        },
    }
    response = requests.post(
        f"{config.OLLAMA_HOST}/api/generate",
        json=payload,
        timeout=config.GENERATION_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return response.json().get("response", "").strip()


def answer(
    question: str,
    chunks: list[dict[str, Any]] | None,
    closed_book: bool = False,
) -> dict[str, Any]:
    """Generate one answer and report which chunks it cited."""
    system = CLOSED_BOOK_SYSTEM_PROMPT if closed_book else SYSTEM_PROMPT
    prompt = build_prompt(question, None if closed_book else chunks)
    text = ollama_generate(prompt, system)

    cited = sorted(set(CITATION_RE.findall(text)))
    retrieved_ids = [c["chunk_id"] for c in (chunks or [])]

    return {
        "answer": text,
        "cited_chunk_ids": cited,
        "retrieved_chunk_ids": retrieved_ids,
        # A citation the model invented rather than took from its context.
        "hallucinated_citations": [c for c in cited if c not in retrieved_ids],
        "refused": is_refusal(text),
        "prompt": prompt,
        "system_prompt": system,
    }


def is_refusal(text: str) -> bool:
    """True when the model declined for lack of context."""
    normalised = re.sub(r"[^a-z ]", "", text.lower()).strip()
    return normalised.startswith("i dont know") or normalised == "i dont know"


def model_metadata() -> dict[str, Any]:
    """Capture model identity and decoding settings for the report."""
    details: dict[str, Any] = {}
    try:
        response = requests.post(
            f"{config.OLLAMA_HOST}/api/show",
            json={"model": config.GENERATION_MODEL},
            timeout=30,
        )
        response.raise_for_status()
        payload = response.json()
        details = {
            "parameter_size": payload.get("details", {}).get("parameter_size"),
            "quantization_level": payload.get("details", {}).get(
                "quantization_level"
            ),
            "family": payload.get("details", {}).get("family"),
            "digest": payload.get("digest"),
        }
    except Exception as exc:  # noqa: BLE001 - metadata is best effort
        details = {"error": str(exc)}

    return {
        "model_name": config.GENERATION_MODEL,
        "access_method": f"local Ollama HTTP API at {config.OLLAMA_HOST}",
        "temperature": config.GENERATION_TEMPERATURE,
        "top_p": config.GENERATION_TOP_P,
        "seed": config.GENERATION_SEED,
        "num_predict": config.GENERATION_NUM_PREDICT,
        "num_ctx": config.GENERATION_NUM_CTX,
        "embedding_model": config.EMBEDDING_MODEL,
        "top_k_chunks": config.TOP_K,
        "model_details": details,
        "system_prompt_rag": SYSTEM_PROMPT,
        "system_prompt_closed_book": CLOSED_BOOK_SYSTEM_PROMPT,
        "system_prompt_diagnostic": DIAGNOSTIC_SYSTEM_PROMPT,
    }


def check_server() -> bool:
    """True when Ollama is reachable and the configured model is present."""
    try:
        response = requests.get(f"{config.OLLAMA_HOST}/api/tags", timeout=10)
        response.raise_for_status()
        names = {m.get("name") for m in response.json().get("models", [])}
        return config.GENERATION_MODEL in names
    except Exception:  # noqa: BLE001
        return False


if __name__ == "__main__":
    print(json.dumps(model_metadata(), indent=2)[:2000])
