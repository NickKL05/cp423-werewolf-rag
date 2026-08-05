"""Clean the crawled wikitext and split it into retrievable chunks.

Two stages, both deterministic and runnable offline from the committed snapshot:

  1. Document construction. Filter the raw pages down to the Werewolf line,
     convert wikitext to plain prose, flatten infoboxes into "key: value" lines,
     drop stubs, and assign a stable document ID.

  2. Chunking. Split on wiki section headings first so that a chunk never spans
     two unrelated topics, then apply a sliding token window inside any section
     that is still too long. Every chunk keeps the document ID, title, section
     and source URL so a retrieved passage traces back to its page.

Run:
    python -m wolfrag.preprocess
"""

from __future__ import annotations

import json
import re
from typing import Any

import mwparserfromhell

from wolfrag import config

HEADING_RE = re.compile(r"^\s*(={2,6})\s*(.+?)\s*\1\s*$", re.MULTILINE)
FILE_LINK_RE = re.compile(r"\[\[(?:File|Image):[^\]]*\]\]", re.IGNORECASE)
HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
REF_RE = re.compile(r"<ref[^>]*>.*?</ref>|<ref[^>]*/>", re.DOTALL | re.IGNORECASE)
GALLERY_RE = re.compile(r"<gallery[^>]*>.*?</gallery>", re.DOTALL | re.IGNORECASE)
TABLE_RE = re.compile(r"\{\|.*?\|\}", re.DOTALL)
CATEGORY_LINE_RE = re.compile(r"\[\[Category:[^\]]*\]\]", re.IGNORECASE)
MULTI_BLANK_RE = re.compile(r"\n{3,}")
MULTI_SPACE_RE = re.compile(r"[ \t]{2,}")
# Line breaks carry real structure inside stat blocks. Stripping the tag without
# putting a separator back produces welded text such as "HomidAuspice: Theurge".
BREAK_TAG_RE = re.compile(r"<\s*br\s*/?\s*>", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------


def keep_page(categories: list[str]) -> bool:
    """True when the page belongs to Werewolf: the Apocalypse and nothing else."""
    lowered = [c.lower() for c in categories]
    has_wta = any(
        pattern in cat
        for cat in lowered
        for pattern in config.WTA_CATEGORY_PATTERNS
    )
    has_other = any(
        pattern in cat
        for cat in lowered
        for pattern in config.OTHER_LINE_CATEGORY_PATTERNS
    )
    is_publication = any(
        pattern in cat
        for cat in lowered
        for pattern in config.PUBLICATION_CATEGORY_PATTERNS
    )
    return has_wta and not has_other and not is_publication


# ---------------------------------------------------------------------------
# Wikitext cleaning
# ---------------------------------------------------------------------------


def extract_infobox(wikitext: str) -> tuple[str, str]:
    """Pull infobox parameters out as prose lines.

    Returns the flattened infobox text and the wikitext with infoboxes removed.
    Infoboxes hold the densest factual content on a wiki page (tribe, auspice,
    breed, totem), so they are kept rather than stripped, but converted to
    "Key: value" lines that a text retriever can actually match against.
    """
    code = mwparserfromhell.parse(BREAK_TAG_RE.sub("\n", wikitext))
    lines: list[str] = []

    for template in code.filter_templates(recursive=False):
        name = str(template.name).strip().lower()
        if not name.startswith("infobox"):
            continue
        for param in template.params:
            key = str(param.name).strip()
            value = mwparserfromhell.parse(str(param.value)).strip_code().strip()
            value = FILE_LINK_RE.sub("", value).strip()
            if not value or key.lower() in {"image", "caption", "imagewidth"}:
                continue
            key = key.replace("_", " ").strip().title()
            lines.append(f"{key}: {value}")
        try:
            code.remove(template)
        except ValueError:
            pass

    return "\n".join(lines), str(code)


def clean_fragment(wikitext: str) -> str:
    """Convert a wikitext fragment to plain prose."""
    text = wikitext
    text = BREAK_TAG_RE.sub("\n", text)
    text = HTML_COMMENT_RE.sub("", text)
    text = REF_RE.sub("", text)
    text = GALLERY_RE.sub("", text)
    text = FILE_LINK_RE.sub("", text)
    text = TABLE_RE.sub("", text)
    text = CATEGORY_LINE_RE.sub("", text)

    code = mwparserfromhell.parse(text)
    for template in code.filter_templates(recursive=True):
        try:
            code.remove(template)
        except ValueError:
            pass

    text = code.strip_code(normalize=True, collapse=True)

    cleaned_lines = []
    for line in text.split("\n"):
        stripped = line.strip()
        if not stripped:
            cleaned_lines.append("")
            continue
        # Leftover list and table markup.
        stripped = re.sub(r"^[\*\#:;]+\s*", "", stripped)
        stripped = re.sub(r"^\|.*$", "", stripped)
        stripped = re.sub(r"^!.*$", "", stripped)
        if stripped:
            cleaned_lines.append(stripped)

    text = "\n".join(cleaned_lines)
    text = MULTI_SPACE_RE.sub(" ", text)
    text = MULTI_BLANK_RE.sub("\n\n", text)
    return text.strip()


def split_sections(wikitext: str) -> list[tuple[str, str]]:
    """Split wikitext into (section title, section wikitext) pairs."""
    matches = list(HEADING_RE.finditer(wikitext))
    sections: list[tuple[str, str]] = []

    lead = wikitext[: matches[0].start()] if matches else wikitext
    if lead.strip():
        sections.append(("Introduction", lead))

    for index, match in enumerate(matches):
        # Headings can carry their own markup, for example <u>'''Second
        # Edition'''</u>, so they need the same cleaning as body text.
        title = mwparserfromhell.parse(match.group(2)).strip_code().strip()
        title = re.sub(r"<[^>]+>", "", title).strip() or "Section"
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(wikitext)
        sections.append((title, wikitext[start:end]))

    return sections


def build_document(record: dict[str, Any]) -> dict[str, Any] | None:
    """Turn one raw crawl record into a cleaned document, or None if filtered."""
    infobox_text, remainder = extract_infobox(record["wikitext"])
    sections: list[dict[str, str]] = []

    if infobox_text:
        sections.append({"section": "Infobox", "text": infobox_text})

    for title, body in split_sections(remainder):
        if title.lower().strip() in config.DROPPED_SECTIONS:
            continue
        cleaned = clean_fragment(body)
        if cleaned:
            sections.append({"section": title, "text": cleaned})

    if not sections:
        return None

    full_text = "\n\n".join(f"{s['section']}\n{s['text']}" for s in sections)
    word_count = len(full_text.split())

    return {
        "pageid": record["pageid"],
        "title": record["title"],
        "url": record["url"],
        "categories": record["categories"],
        "revision_id": record.get("revision_id"),
        "revision_timestamp": record.get("revision_timestamp"),
        "crawled_at": record.get("crawled_at"),
        "sections": sections,
        "text": full_text,
        "word_count": word_count,
    }


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------


def window_tokens(tokens: list[str], size: int, overlap: int) -> list[list[str]]:
    """Sliding window over a token list."""
    if len(tokens) <= size:
        return [tokens]
    stride = max(1, size - overlap)
    windows = []
    for start in range(0, len(tokens), stride):
        window = tokens[start : start + size]
        if not window:
            break
        windows.append(window)
        if start + size >= len(tokens):
            break
    return windows


def chunk_document(doc: dict[str, Any]) -> list[dict[str, Any]]:
    """Split one document into overlapping, section-aware chunks.

    Infobox chunks are held to a much lower length floor than prose. They are
    short by nature but carry the densest facts on the page (tribe, auspice,
    breed, totem), so discarding them for being brief would throw away the best
    material for factoid questions.
    """
    chunks: list[dict[str, Any]] = []

    for section in doc["sections"]:
        tokens = section["text"].split()
        if not tokens:
            continue
        is_infobox = section["section"] == "Infobox"
        floor = config.MIN_INFOBOX_CHUNK_TOKENS if is_infobox else config.MIN_CHUNK_TOKENS

        for window in window_tokens(
            tokens, config.CHUNK_TARGET_TOKENS, config.CHUNK_OVERLAP_TOKENS
        ):
            if len(window) < floor:
                continue
            chunks.append(
                {
                    "doc_id": doc["doc_id"],
                    "title": doc["title"],
                    "section": section["section"],
                    "url": doc["url"],
                    "text": " ".join(window),
                    "n_tokens": len(window),
                }
            )

    # A document that produced nothing (every section below the floor) still
    # belongs in the index, so fall back to a single whole-document chunk.
    if not chunks:
        tokens = doc["text"].split()[: config.CHUNK_TARGET_TOKENS]
        if tokens:
            chunks.append(
                {
                    "doc_id": doc["doc_id"],
                    "title": doc["title"],
                    "section": "Full page",
                    "url": doc["url"],
                    "text": " ".join(tokens),
                    "n_tokens": len(tokens),
                }
            )

    return chunks


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    config.set_seeds()

    raw_records = [
        json.loads(line)
        for line in config.RAW_CORPUS_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    print(f"Loaded {len(raw_records)} raw pages.")

    line_filtered = [r for r in raw_records if keep_page(r.get("categories", []))]
    print(
        f"Kept {len(line_filtered)} pages after the Werewolf line filter "
        f"(dropped {len(raw_records) - len(line_filtered)})."
    )

    documents = []
    for record in line_filtered:
        doc = build_document(record)
        if doc is not None:
            documents.append(doc)
    print(f"Built {len(documents)} cleaned documents.")

    documents = [d for d in documents if d["word_count"] >= config.MIN_DOC_WORDS]
    print(
        f"Kept {len(documents)} documents with at least "
        f"{config.MIN_DOC_WORDS} words."
    )

    # Deterministic cap: keep the richest pages, then restore title order so
    # document IDs are stable across runs.
    if len(documents) > config.MAX_DOCS:
        documents.sort(key=lambda d: (-d["word_count"], d["title"]))
        documents = documents[: config.MAX_DOCS]
        print(f"Capped to the {config.MAX_DOCS} longest documents.")

    documents.sort(key=lambda d: d["title"])
    for index, doc in enumerate(documents, start=1):
        doc["doc_id"] = f"WTA{index:04d}"

    all_chunks: list[dict[str, Any]] = []
    for doc in documents:
        all_chunks.extend(chunk_document(doc))

    for index, chunk in enumerate(all_chunks, start=1):
        chunk["chunk_id"] = f"C{index:05d}"
        # Indexed text carries the page and section title so that a retriever
        # can match on them, which matters for short infobox chunks.
        chunk["retrieval_text"] = (
            f"{chunk['title']}. {chunk['section']}. {chunk['text']}"
        )

    with config.DOCS_PATH.open("w", encoding="utf-8") as handle:
        for doc in documents:
            handle.write(json.dumps(doc, ensure_ascii=False) + "\n")

    with config.CHUNKS_PATH.open("w", encoding="utf-8") as handle:
        for chunk in all_chunks:
            handle.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    token_counts = [c["n_tokens"] for c in all_chunks]
    stats = {
        "documents": len(documents),
        "chunks": len(all_chunks),
        "mean_chunk_tokens": round(sum(token_counts) / len(token_counts), 2),
        "min_chunk_tokens": min(token_counts),
        "max_chunk_tokens": max(token_counts),
        "mean_doc_words": round(
            sum(d["word_count"] for d in documents) / len(documents), 2
        ),
        "min_doc_words": min(d["word_count"] for d in documents),
        "max_doc_words": max(d["word_count"] for d in documents),
        "chunk_target_tokens": config.CHUNK_TARGET_TOKENS,
        "chunk_overlap_tokens": config.CHUNK_OVERLAP_TOKENS,
        "min_doc_words_filter": config.MIN_DOC_WORDS,
    }
    (config.PROCESSED_DIR / "corpus_stats.json").write_text(
        json.dumps(stats, indent=2), encoding="utf-8"
    )

    print(json.dumps(stats, indent=2))
    print(f"Wrote {config.DOCS_PATH}")
    print(f"Wrote {config.CHUNKS_PATH}")


if __name__ == "__main__":
    main()
