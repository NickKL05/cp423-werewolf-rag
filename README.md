# Retrieval-Augmented Generation over the Werewolf: the Apocalypse wiki

CP423 course project.

**Nick Kunde-Lenny, 169056417**

**Demo video:** https://youtu.be/wOOc_iAEFt4
**Report:** [report/REPORT.pdf](report/REPORT.pdf)

A RAG system built over a 1,200 document corpus crawled from the Werewolf: the
Apocalypse section of the White Wolf Wiki. Three retrievers (BM25, dense, and
their fusion) are compared against a closed-book ablation, all sharing one
locally hosted generation model.

## Why this corpus

The assignment requires a corpus that a modern LLM has not memorised, otherwise
retrieval cannot be shown to contribute anything. Werewolf: the Apocalypse is a
tabletop game line whose deep lore, named characters, septs and camps are
documented almost nowhere outside this wiki.

The closed-book diagnostic confirms it. Ten factual questions drawn from the
corpus were put to `llama3.1:8b` with no retrieved context and no permission to
decline:

| Measure | Score |
| --- | --- |
| Human verified correct | **0 / 10** |
| Automatic keyword match | 1 / 10 |
| Mean token F1 | 0.103 |

The single automatic match is a false positive. Asked which tribe held the Caern
of the Sentinel, the model answered "the Ahroun, specifically the Uktena sept of
the Uktena tribe". Ahroun is an auspice, not a tribe, and the phrase is
incoherent, so it was judged incorrect on review. The model did not decline: it
confidently invented the Rocky Mountains, a Raven totem, the year 1995, a Theurge
named Eshu and a Shadow Lord camp called the Umbra Collective. Full verdicts are
in `eval/diagnostic_human_judgements.csv`.

## System design

```
crawl.py        MediaWiki API -> raw wikitext snapshot (data/raw/)
preprocess.py   clean, filter, chunk -> documents.jsonl + chunks.jsonl
retrieval.py    BM25 | dense | hybrid (reciprocal rank fusion)
generate.py     Ollama, enforced citations and refusals
evaluate.py     retrieval + generation metrics -> results/
```

### Corpus construction

Pages are collected from 15 seed categories, traversed two levels deep through
the MediaWiki API with a fixed politeness delay and an identifying User-Agent.
4,994 candidate pages are reduced to 1,200 documents by three filters:

1. **Line filter.** A page is kept only if it carries at least one Werewolf
   category and no category from another game line. Category traversal alone is
   not sufficient: pages such as `"Lulu" Hagen` sit in both the Vampire and
   Werewolf character categories, and would otherwise import Vampire lore.
2. **Publication filter.** Sourcebook and product pages are dropped. They
   describe products rather than the game world, but match Werewolf vocabulary
   strongly enough to pollute retrieval.
3. **Length filter.** Pages under 200 words are dropped as stubs. The remainder
   is capped at the 1,200 longest, keeping the corpus inside the recommended
   200 to 2,000 range.

Infoboxes are flattened into `Key: value` lines rather than stripped, since they
carry the densest facts on a page (tribe, auspice, breed, sept, totem).

### Chunking

Documents are split on wiki section headings first, so a chunk never spans two
unrelated topics, then a 250 token sliding window with 50 token overlap is
applied inside any section still too long. Every chunk retains its document ID,
page title, section name and source URL. Infobox chunks are held to a lower
length floor since they are short by nature and factually dense.

Result: 8,988 chunks, mean 146 tokens.

### Retrieval

| System | Method |
| --- | --- |
| `bm25` | `rank_bm25` Okapi BM25, k1=1.5, b=0.75, over lowercased, stopword filtered, Porter stemmed tokens |
| `dense` | `all-MiniLM-L6-v2` embeddings, cosine similarity over L2 normalised vectors |
| `hybrid` | Reciprocal rank fusion of the two lists, k=60 |
| `closed_book` | No retrieval. The ablation that isolates retrieval's contribution |

Ties are broken by index so rankings are deterministic.

### Generation

`llama3.1:8b` served locally by Ollama. Every system uses the same model and the
same decoding settings, so any difference between them comes from retrieval
alone.

| Setting | Value |
| --- | --- |
| Model | `llama3.1:8b` |
| Access | Local Ollama HTTP API |
| Temperature | 0.0 |
| top_p | 1.0 |
| Seed | 42 |
| num_ctx | 8192 |
| num_predict | 512 |
| Chunks in context | 5 |

The prompt instructs the model to answer only from the retrieved context, cite
each claim inline with its chunk ID (`[C07264]`), and reply exactly `I don't
know` when the context is insufficient. Citations are parsed back out of the
answer, which allows citation precision to be measured and fabricated citations
to be detected. The exact prompts are recorded in `results/run_metadata.json`.

## Setup

Requires Python 3.11 and [Ollama](https://ollama.com).

```bash
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
ollama pull llama3.1:8b
```

Every command below calls `.venv\Scripts\python.exe` explicitly rather than a
bare `python`. This is deliberate. A bare `python` silently resolves to whatever
is first on PATH, and a global interpreter carrying a partial PyTorch install
produces confusing failures such as `ModuleNotFoundError: No module named
'torchvision'` that have nothing to do with this project. Activating the venv
first and using bare `python` works equally well, but the explicit path cannot
be got wrong.

## Reproducing the results

One command reproduces every table in the report:

```bash
.venv\Scripts\python.exe run_all.py
```

This preprocesses the committed snapshot, builds the dense index, runs the
closed-book diagnostic, and evaluates all four systems. Output lands in
`results/`:

| File | Contents |
| --- | --- |
| `tables.md` | The three tables reported in the write-up |
| `summary_metrics.csv` | Per system retrieval and generation metrics |
| `metrics_by_question_type.csv` | Broken down by factoid, multi-hop, unanswerable |
| `per_question_results.json` | Every retrieved ranking and generated answer |
| `diagnostic_closed_book.csv` | The corpus suitability diagnostic |
| `human_eval_template.csv` | Sheet for manual judging of every answer |
| `run_metadata.json` | Model version, settings and full prompts |

The network crawl is deliberately excluded from `run_all.py`. The snapshot in
`data/raw/` is committed so results reproduce even as the wiki changes. To
refetch it:

```bash
.venv\Scripts\python.exe run_all.py --crawl
```

Note that recrawling produces a different snapshot, which reassigns chunk IDs and
invalidates the gold chunk IDs in the evaluation set.

## Demo application

```bash
.venv\Scripts\python.exe -m streamlit run app/streamlit_app.py
```

Two tabs. **Ask** runs a question through any retriever and shows the retrieved
chunks beside the generated answer, with citations resolved back to their
passage and flagged in red if fabricated. **Corpus browser** searches the chunk
index directly, which is how gold chunk IDs were found when writing the
evaluation set.

## Evaluation set

`eval/gold_questions.csv` holds 25 questions written by hand after reading the
corpus, covering factoid, multi-hop and unanswerable types. Writing guidance is
in `eval/HOW_TO_WRITE_QUESTIONS.md`.

The gold set is validated before any experiment runs. The run aborts if there
are fewer than 10 questions, fewer than 2 multi-hop, fewer than 2 unanswerable,
if a multi-hop question has fewer than 2 gold chunks, if an unanswerable
question has any gold chunk, or if a gold chunk ID is not in the corpus.

Retrieval metrics are computed over answerable questions only, since an
unanswerable question has no chunk to find. Unanswerable questions are scored
instead on whether the system correctly refused.

**Disclosure.** The ten closed-book *diagnostic* questions in
`eval/diagnostic_questions.json` were drafted with LLM assistance and then
verified by hand against the corpus text, with each model answer judged
manually. The 25 *gold evaluation* questions were written by the author after
reading the corpus, as the assignment requires.

## Metrics

**Retrieval**, at chunk level and again at document level:
Precision@k (k = 1, 3, 5, 10), Recall@k, MAP, nDCG@10, MRR.

**Generation**: token F1 and ROUGE-L against the reference answer, exact match,
citation precision and recall against the gold chunks, rate of answers carrying
any citation, rate of fabricated citations, refusal accuracy on unanswerable
questions, and false refusal rate on answerable ones.

Automatic answer scoring is supplemented by human judgement, since token overlap
penalises a correct answer phrased differently from the reference.

## Reproducibility

Seeds are fixed in `src/wolfrag/config.py` (`SEED = 42`) and applied to Python,
NumPy and torch. Decoding is greedy at temperature 0 with a fixed sampling seed.
Retrieval tie-breaking is deterministic. The corpus snapshot, the chunk file and
the evaluation set are all committed.

## Attribution and licensing

Corpus text comes from the [White Wolf Wiki](https://whitewolf.fandom.com), used
under [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/). It was
collected through the MediaWiki API with rate limiting and an identifying
User-Agent. Werewolf: the Apocalypse and the World of Darkness are trademarks of
Paradox Interactive. This is non-commercial academic coursework.

## Layout

```
run_all.py                  reproduce every result
requirements.txt            pinned dependencies
app/streamlit_app.py        demo and corpus browser
src/wolfrag/
    config.py               all constants and seeds
    crawl.py                MediaWiki API crawler
    preprocess.py           cleaning and chunking
    retrieval.py            BM25, dense, hybrid
    generate.py             Ollama client and prompts
    metrics.py              retrieval and generation metrics
    evaluate.py             experiment runner
    diagnostic.py           closed-book corpus check
    make_gold_template.py   creates the empty evaluation set
data/raw/                   committed corpus snapshot
data/processed/             cleaned documents and chunks
eval/                       evaluation set and diagnostic questions
results/                    generated tables and per question output
```
