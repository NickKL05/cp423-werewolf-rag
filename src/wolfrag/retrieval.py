"""Three retrievers over the chunked corpus: BM25, dense, and their fusion.

BM25 is the classical lexical baseline. The dense retriever embeds chunks and
queries with a pretrained sentence-transformers model and ranks by cosine
similarity. The hybrid retriever fuses the two ranked lists with reciprocal
rank fusion, which needs no score normalisation because it consumes ranks
rather than raw scores.

Run to build and cache the dense index:
    python -m wolfrag.retrieval
"""

from __future__ import annotations

import json
import re
from typing import Any

import numpy as np
from nltk.stem.porter import PorterStemmer
from rank_bm25 import BM25Okapi

from wolfrag import config

# A fixed stopword list is used instead of the nltk download so that a clone of
# this repository runs without fetching any corpus data at import time.
STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an",
    "and", "any", "are", "as", "at", "be", "because", "been", "before",
    "being", "below", "between", "both", "but", "by", "can", "did", "do",
    "does", "doing", "don", "down", "during", "each", "few", "for", "from",
    "further", "had", "has", "have", "having", "he", "her", "here", "hers",
    "herself", "him", "himself", "his", "how", "i", "if", "in", "into", "is",
    "it", "its", "itself", "just", "me", "more", "most", "my", "myself", "no",
    "nor", "not", "now", "of", "off", "on", "once", "only", "or", "other",
    "our", "ours", "ourselves", "out", "over", "own", "s", "same", "she",
    "should", "so", "some", "such", "t", "than", "that", "the", "their",
    "theirs", "them", "themselves", "then", "there", "these", "they", "this",
    "those", "through", "to", "too", "under", "until", "up", "very", "was",
    "we", "were", "what", "when", "where", "which", "while", "who", "whom",
    "why", "will", "with", "you", "your", "yours", "yourself", "yourselves",
}

TOKEN_RE = re.compile(r"[a-z0-9]+")
_STEMMER = PorterStemmer()
_STEM_CACHE: dict[str, str] = {}


def tokenize(text: str) -> list[str]:
    """Lowercase, strip punctuation, drop stopwords, then Porter stem."""
    tokens = TOKEN_RE.findall(text.lower())
    output = []
    for token in tokens:
        if token in STOPWORDS or len(token) < 2:
            continue
        stem = _STEM_CACHE.get(token)
        if stem is None:
            stem = _STEMMER.stem(token)
            _STEM_CACHE[token] = stem
        output.append(stem)
    return output


def load_chunks() -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in config.CHUNKS_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


# ---------------------------------------------------------------------------
# Retrievers
# ---------------------------------------------------------------------------


class BaseRetriever:
    name = "base"

    def __init__(self, chunks: list[dict[str, Any]]):
        self.chunks = chunks
        self.chunk_ids = [c["chunk_id"] for c in chunks]

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        raise NotImplementedError

    def retrieve(self, query: str, k: int) -> list[dict[str, Any]]:
        """Return full chunk records with their scores attached."""
        by_id = {c["chunk_id"]: c for c in self.chunks}
        results = []
        for chunk_id, score in self.search(query, k):
            record = dict(by_id[chunk_id])
            record["score"] = float(score)
            results.append(record)
        return results


class BM25Retriever(BaseRetriever):
    name = "bm25"

    def __init__(self, chunks: list[dict[str, Any]]):
        super().__init__(chunks)
        corpus = [tokenize(c["retrieval_text"]) for c in chunks]
        self.model = BM25Okapi(corpus, k1=config.BM25_K1, b=config.BM25_B)

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        scores = self.model.get_scores(tokenize(query))
        # Ties are broken by chunk index so results are deterministic.
        order = np.lexsort((np.arange(len(scores)), -scores))[:k]
        return [(self.chunk_ids[i], float(scores[i])) for i in order]


class DenseRetriever(BaseRetriever):
    name = "dense"

    def __init__(self, chunks: list[dict[str, Any]], embeddings: np.ndarray | None = None):
        super().__init__(chunks)
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(
            config.EMBEDDING_MODEL, device=config.EMBEDDING_DEVICE
        )
        if embeddings is None:
            embeddings = load_or_build_embeddings(chunks, self.model)
        self.embeddings = embeddings

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        query_vector = self.model.encode(
            [query], normalize_embeddings=True, show_progress_bar=False
        )[0]
        # Embeddings are L2 normalised, so the dot product is cosine similarity.
        scores = self.embeddings @ query_vector
        order = np.lexsort((np.arange(len(scores)), -scores))[:k]
        return [(self.chunk_ids[i], float(scores[i])) for i in order]


class HybridRetriever(BaseRetriever):
    name = "hybrid"

    def __init__(self, bm25: BM25Retriever, dense: DenseRetriever):
        super().__init__(bm25.chunks)
        self.bm25 = bm25
        self.dense = dense

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        depth = max(k, config.MAX_RETRIEVAL_DEPTH) * 2
        fused: dict[str, float] = {}
        for retriever in (self.bm25, self.dense):
            for rank, (chunk_id, _score) in enumerate(
                retriever.search(query, depth), start=1
            ):
                fused[chunk_id] = fused.get(chunk_id, 0.0) + 1.0 / (
                    config.RRF_K + rank
                )
        ranked = sorted(fused.items(), key=lambda item: (-item[1], item[0]))
        return ranked[:k]


# ---------------------------------------------------------------------------
# Index building
# ---------------------------------------------------------------------------


def load_or_build_embeddings(chunks: list[dict[str, Any]], model=None) -> np.ndarray:
    """Load cached chunk embeddings, computing and caching them if absent."""
    if config.EMBEDDINGS_PATH.exists():
        embeddings = np.load(config.EMBEDDINGS_PATH)
        if embeddings.shape[0] == len(chunks):
            return embeddings
        print("Cached embeddings are stale, rebuilding.")

    if model is None:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(
            config.EMBEDDING_MODEL, device=config.EMBEDDING_DEVICE
        )

    print(f"Embedding {len(chunks)} chunks with {config.EMBEDDING_MODEL} ...")
    embeddings = model.encode(
        [c["retrieval_text"] for c in chunks],
        batch_size=config.EMBEDDING_BATCH_SIZE,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    ).astype(np.float32)
    np.save(config.EMBEDDINGS_PATH, embeddings)
    print(f"Wrote {config.EMBEDDINGS_PATH}")
    return embeddings


def build_all() -> dict[str, BaseRetriever]:
    """Construct every retriever once, sharing the chunk list and index."""
    config.set_seeds()
    chunks = load_chunks()
    bm25 = BM25Retriever(chunks)
    dense = DenseRetriever(chunks)
    hybrid = HybridRetriever(bm25, dense)
    return {"bm25": bm25, "dense": dense, "hybrid": hybrid}


def main() -> None:
    config.set_seeds()
    chunks = load_chunks()
    print(f"Loaded {len(chunks)} chunks.")
    load_or_build_embeddings(chunks)
    print("Dense index ready.")


if __name__ == "__main__":
    main()
