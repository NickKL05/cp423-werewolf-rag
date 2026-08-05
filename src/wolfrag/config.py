"""Central configuration for the Werewolf: the Apocalypse RAG system.

Every tunable constant lives here so that the values reported in the write-up
can be traced back to a single file. Random seeds are fixed for reproducibility.
"""

from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
EVAL_DIR = PROJECT_ROOT / "eval"
RESULTS_DIR = PROJECT_ROOT / "results"

RAW_CORPUS_PATH = RAW_DIR / "wta_corpus_raw.jsonl"
CRAWL_MANIFEST_PATH = RAW_DIR / "crawl_manifest.json"
DOCS_PATH = PROCESSED_DIR / "documents.jsonl"
CHUNKS_PATH = PROCESSED_DIR / "chunks.jsonl"
EMBEDDINGS_PATH = PROCESSED_DIR / "chunk_embeddings.npy"

GOLD_QUESTIONS_PATH = EVAL_DIR / "gold_questions.csv"
DIAGNOSTIC_QUESTIONS_PATH = EVAL_DIR / "diagnostic_questions.json"

for _directory in (RAW_DIR, PROCESSED_DIR, EVAL_DIR, RESULTS_DIR):
    _directory.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

SEED = 42


def set_seeds(seed: int = SEED) -> None:
    """Fix every random source the pipeline touches."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# Corpus construction
# ---------------------------------------------------------------------------

WIKI_API = "https://whitewolf.fandom.com/api.php"
WIKI_BASE = "https://whitewolf.fandom.com/wiki/"
USER_AGENT = (
    "CP423-RAG-coursework/1.0 (Wilfrid Laurier University student project; "
    "contact nkl992023@gmail.com)"
)

# Politeness. The wiki is a volunteer-run Fandom host, so we stay well under
# any plausible rate limit and identify ourselves in the User-Agent.
REQUEST_DELAY_SECONDS = 0.34
BATCH_SIZE = 20
MAX_RETRIES = 4

# Curated seeds. These cover the lore spine of the line: the metaphysics
# (Triat), the shapeshifters (Garou and Fera), their society (tribes, auspices,
# septs, caerns), their powers (Gifts), and their opposition (Wyrm, Pentex,
# Black Spiral Dancers).
SEED_CATEGORIES = [
    "Werewolf: The Apocalypse",
    "Werewolf: The Apocalypse glossary",
    "Werewolf: The Apocalypse geography",
    "Werewolf: The Apocalypse character",
    "Garou",
    "Garou tribes",
    "Fera",
    "Gifts",
    "Wyrm",
    "Weaver",
    "Wyld",
    "Caerns",
    "Black Spiral Dancers",
    "Pentex",
    "Auspices",
]

MAX_CATEGORY_DEPTH = 2

# Categories dropped during traversal. Two groups:
#   1. Publication metadata (books, media, merchandise, creators, community).
#      These describe products rather than in-world lore, so they answer no
#      interesting question and dilute retrieval.
#   2. Adjacent settings and editions. Dark Ages, Wild West, 5th Edition and
#      the video games are separate continuities whose lore contradicts the
#      classic line, which would make gold answers ambiguous.
EXCLUDED_CATEGORY_SUBSTRINGS = [
    "book",
    "media",
    "merchandise",
    "creator",
    "community",
    "stub",
    "fiction",
    "novel",
    "sourcebook",
    "gallery",
    "image",
    "template",
    "disambiguation",
    "dark ages",
    "wild west",
    "5th edition",
    "earthblood",
    "rage across series",
    "rage characters",
    "video game",
    "computer game",
    "card game",
]

# Titles carrying another game line's disambiguation suffix, or which are
# navigational rather than substantive.
EXCLUDED_TITLE_SUBSTRINGS = [
    "(CTD)",
    "(VTM)",
    "(MTA)",
    "(WTO)",
    "(KOE)",
    "(DTF)",
    "(HTR)",
    "(MTR)",
    "(CofD)",
    "(WTF)",
    "(VTR)",
    "(MTAw)",
    "(PTC)",
    "(EX)",
    "List of",
    "Timeline of",
    "Index of",
]

# Page level line filter, applied during preprocessing.
#
# Category traversal alone is not enough. A page such as '"Lulu" Hagen' sits in
# both "Vampire: The Masquerade character" and "Werewolf: The Apocalypse
# character", so walking the Werewolf tree pulls in Vampire lore. A page is kept
# only when it carries at least one Werewolf category and no category belonging
# to another game line, edition or setting.
WTA_CATEGORY_PATTERNS = [
    "werewolf: the apocalypse",
    "garou",
    "fera",
    "wyrm",
    "weaver",
    "wyld",
    "caern",
    "sept",
    "black spiral",
    "pentex",
    "gift",
    "rite",
    "auspice",
    "metis",
    "lupus",
    "homid",
    "bastet",
    "corax",
    "gurahl",
    "ananasi",
    "mokol",
    "nagah",
    "nuwisha",
    "ratkin",
    "rokea",
    "ajaba",
    "kitsune",
    "kinfolk",
    "totem",
    "umbra",
    "bane",
    "fomori",
    "spirit",
]

OTHER_LINE_CATEGORY_PATTERNS = [
    "vampire: the masquerade",
    "vampire: the requiem",
    "mage: the ascension",
    "mage: the awakening",
    "wraith: the oblivion",
    "wraith: the great war",
    "changeling: the dreaming",
    "changeling: the lost",
    "demon: the fallen",
    "demon: the descent",
    "hunter: the reckoning",
    "hunter: the vigil",
    "mummy:",
    "kindred of the east",
    "werewolf: the forsaken",
    "werewolf: the wild west",
    "dark ages",
    "victorian age",
    "promethean",
    "geist:",
    "beast:",
    "deviant:",
    "exalted",
    "scion",
    "trinity",
    "aberrant",
    "adventure!",
    "orpheus",
    "street fighter",
    "5th edition",
    "earthblood",
]

# Publication metadata. A sourcebook page describes a product rather than the
# game world, so it answers no lore question while still matching Werewolf
# vocabulary strongly enough to pollute retrieval.
PUBLICATION_CATEGORY_PATTERNS = [
    " books",
    "sourcebook",
    "novels",
    "fiction",
    "comics",
    "magazines",
    "merchandise",
    "publications",
    "products",
    "video games",
    "card games",
    "computer games",
    "soundtrack",
]

# Sections that carry no answerable prose.
DROPPED_SECTIONS = [
    "references",
    "external links",
    "see also",
    "gallery",
    "sources",
    "further reading",
    "appearances",
    "notes and references",
    "bibliography",
    "navigation",
    "credits",
]

# A page must survive cleaning with at least this many words to enter the
# corpus. Filters out stubs, which carry too little text to answer anything.
MIN_DOC_WORDS = 200

# Hard ceiling so the corpus stays inside the 200 to 2000 document range the
# assignment recommends. Selection is deterministic: richest pages first.
MAX_DOCS = 1200

# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

CHUNK_TARGET_TOKENS = 250
CHUNK_OVERLAP_TOKENS = 50
MIN_CHUNK_TOKENS = 40
# Infoboxes are short by design and factually dense, so they get their own floor.
MIN_INFOBOX_CHUNK_TOKENS = 5

# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

# all-MiniLM-L6-v2 is the sentence-transformers model used in the course
# lecture material, so the dense retriever stays comparable to what was taught.
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_BATCH_SIZE = 64
EMBEDDING_DEVICE = "cpu"

BM25_K1 = 1.5
BM25_B = 0.75

# Reciprocal rank fusion constant, following Cormack et al. 2009.
RRF_K = 60

TOP_K = 5
EVAL_K_VALUES = [1, 3, 5, 10]
MAX_RETRIEVAL_DEPTH = 10

# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
GENERATION_MODEL = "llama3.1:8b"
GENERATION_TEMPERATURE = 0.0
GENERATION_TOP_P = 1.0
GENERATION_SEED = SEED
GENERATION_NUM_PREDICT = 512
GENERATION_NUM_CTX = 8192
GENERATION_TIMEOUT_SECONDS = 300

REFUSAL_STRING = "I don't know"

SYSTEMS = ["closed_book", "bm25", "dense", "hybrid"]
RETRIEVAL_SYSTEMS = ["bm25", "dense", "hybrid"]
