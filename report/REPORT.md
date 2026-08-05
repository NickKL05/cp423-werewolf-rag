# Build and Evaluate a Retrieval-Augmented Generation System

**Nick Kunde-Lenny, 169056417**
CP423 Course Project

**Repository:** https://github.com/NickKL05/cp423-werewolf-rag
**Demo video:** https://youtu.be/wOOc_iAEFt4

---

## 1. Corpus construction

The corpus is drawn from the Werewolf: the Apocalypse section of the White Wolf
Wiki (whitewolf.fandom.com), a fan-maintained encyclopedia of the World of
Darkness tabletop setting. Text is used under CC BY-SA 3.0.

Collection ran through the MediaWiki API rather than HTML scraping, since the
API is the sanctioned interface, returns original wikitext, and permits a fixed
politeness delay (0.34s) and an identifying User-Agent. Fifteen seed categories
covering the metaphysics, shapeshifter breeds, social structures, powers and
antagonists of the line were traversed two levels deep, yielding 4,994 candidate
pages.

Three filters reduced this to 1,200 documents:

1. **Game line filter.** A page is kept only if it carries at least one Werewolf
   category and no category belonging to another game line. Category traversal
   alone proved insufficient: pages such as `"Lulu" Hagen` sit simultaneously in
   the Vampire and Werewolf character categories, and would have imported
   Vampire lore into a Werewolf corpus. This filter removed 2,189 pages.
2. **Publication filter.** Sourcebook and product pages were dropped. They
   describe commercial products rather than the game world, so they answer no
   lore question, yet they match Werewolf vocabulary strongly enough to compete
   in retrieval.
3. **Length filter.** Pages under 200 words were dropped as stubs, leaving 1,686.
   The remainder was capped at the 1,200 longest documents to stay inside the
   recommended 200 to 2,000 range. Selection is deterministic, and document IDs
   are assigned in title order so they remain stable across runs.

Each document retains its page title, source URL, categories, revision ID,
revision timestamp and crawl timestamp, and is assigned an identifier of the
form `WTA0123`.

Cleaning converts wikitext to prose using `mwparserfromhell`, removing
references, galleries, file links, comments and templates. Two decisions
deserve mention. First, infoboxes are not discarded but flattened into
`Key: value` lines, because they carry the densest facts on a page (tribe,
auspice, breed, sept, totem) in a form a text retriever can match. Second,
`<br>` tags are replaced with newlines before stripping, without which character
stat blocks collapse into unreadable strings such as `HomidAuspice: Theurge`.

**Chunking.** Documents are split on wiki section headings first, so that a
chunk never spans two unrelated topics, and a 250 token sliding window with 50
token overlap is then applied inside any section still too long. Every chunk
carries its document ID, page title, section name and source URL, so any
retrieved passage traces back to its source. Infobox chunks are held to a lower
length floor than prose, since they are short by nature and factually dense.
The result is **8,988 chunks, mean 146 tokens**.

## 2. Corpus suitability diagnostic

The assignment requires evidence that the corpus is not already memorised by the
generation model. Ten factual questions with answers in the corpus were put to
`llama3.1:8b` with no retrieved context. The prompt deliberately pushed for a
best guess rather than permitting a refusal, since a model declining because it
was told it may proves nothing about what it knows.

| Measure | Score |
| --- | --- |
| **Human verified correct** | **0 / 10** |
| Automatic keyword match | 1 / 10 |
| Mean token F1 | 0.103 |

The single automatic match is a false positive and was judged incorrect on
review. Asked which tribe held the Caern of the Sentinel, the model answered
"the Ahroun, specifically the Uktena sept of the Uktena tribe". Ahroun is an
auspice rather than a tribe, and the phrase is incoherent, so containing the
correct word is not evidence of knowing the fact.

More telling than the score is the failure mode. The model did not decline. It
confidently produced the Rocky Mountains for a caern in the Wrangell Mountains,
a Raven totem for one named Tijus-keha, the year 1995 for a death in 1998, a
Theurge called Eshu instead of Akosha's Eye, and a Shadow Lord camp called the
Umbra Collective, which does not exist. The corpus is therefore suitable:
performance of the full system can be attributed to retrieval rather than
parametric knowledge. Per-question verdicts are in
`eval/diagnostic_human_judgements.csv`.

## 3. System design

**Retrieval.** Three systems are compared against a no-retrieval ablation.

| System | Method |
| --- | --- |
| `closed_book` | No retrieval. Isolates retrieval's contribution |
| `bm25` | Okapi BM25, k1 = 1.5, b = 0.75, over lowercased, stopword filtered, Porter stemmed tokens |
| `dense` | `all-MiniLM-L6-v2` embeddings, cosine similarity over L2 normalised vectors |
| `hybrid` | Reciprocal rank fusion of the above, k = 60 |

Reciprocal rank fusion consumes ranks rather than raw scores, so it needs no
score normalisation between two retrievers whose scores are not comparable.
Ranking ties are broken by index so results are deterministic.

**Generation.** All four systems share one model and one decoding configuration,
so any difference between them is attributable to retrieval alone.

| Setting | Value |
| --- | --- |
| Model | `llama3.1:8b` (Meta Llama 3.1, 8B, Q4_K_M) |
| Access | Local Ollama HTTP API |
| Temperature / top_p | 0.0 / 1.0 |
| Seed | 42 |
| num_ctx / num_predict | 8192 / 512 |
| Chunks in context | 5 |

The prompt instructs the model to answer only from the retrieved context, to
cite each claim inline with its chunk ID such as `[C07264]`, and to reply
exactly `I don't know` when the context is insufficient. Citations are parsed
back out of the generated text, which makes citation precision measurable and
allows fabricated citations to be detected automatically. Full prompts are
recorded in `results/run_metadata.json`.

## 4. Evaluation set construction

The evaluation set contains **25 questions written by hand after reading the
corpus**: 15 factoid, 5 multi-hop and 5 unanswerable. Each carries a reference
answer, the ground-truth chunk IDs and document IDs, and a note on why it is
difficult. A validator refuses to run the experiment unless the set satisfies
the assignment's constraints, checking question counts by type, that multi-hop
questions have at least two gold chunks, that unanswerable questions have none,
and that every gold chunk ID exists in the corpus.

Multi-hop questions were written to require genuinely separate pages. For
example, Q16 needs the White Howlers page for the Great Pit and the Black Spiral
Dancers page for the Spiral Labyrinth, and neither page alone supplies both
halves.

Unanswerable questions were written to be hard rather than trivially off-topic.
They include a false premise (a fourth Pure Lands tribe, where the corpus states
there were three), an invented tribe whose phrasing mimics real tribe questions,
and three cases where the surrounding context is highly retrievable but the
specific fact is simply never stated, such as the auspice of a named figure the
wiki mentions without describing.

**Sources of bias.** Several are worth acknowledging. The questions were written
by the same person who built the system, so they may unconsciously favour
content the pipeline handles well, and no second annotator was available to
measure agreement. Question topics skew toward pages encountered while
inspecting the corpus during development, which over-represents well-developed
pages relative to the many short character entries. Gold chunk annotation is
itself a judgement call: with 50 token overlap between windows, a fact often
appears in two adjacent chunks, and labelling only one as gold slightly
understates retrieval performance. Finally, 25 questions is a small sample, so
per-type figures computed over 5 multi-hop and 5 unanswerable questions carry
wide uncertainty and should be read as indicative rather than precise.

## 5. Results

**Table 1. Retrieval quality** (answerable questions, chunk level)

| system | P@1 | P@3 | P@5 | P@10 | MAP | nDCG@10 | MRR | Recall@5 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bm25 | 0.350 | 0.167 | 0.140 | 0.085 | 0.431 | 0.505 | 0.458 | 0.600 |
| dense | 0.450 | 0.233 | 0.170 | 0.105 | 0.571 | 0.649 | 0.599 | 0.725 |
| **hybrid** | **0.450** | **0.267** | **0.190** | **0.110** | **0.582** | **0.672** | **0.618** | **0.800** |

**Table 2. Generation quality**

| system | token F1 | ROUGE-L | citation precision | citation recall | fabricated citations | refusal accuracy | false refusal rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| closed_book | 0.185 | 0.178 | n/a | n/a | n/a | 0.800 | 0.200 |
| bm25 | 0.361 | 0.350 | 0.417 | 0.500 | 0.000 | 1.000 | 0.050 |
| dense | 0.385 | 0.367 | 0.542 | 0.600 | 0.000 | 1.000 | 0.100 |
| **hybrid** | **0.419** | **0.380** | **0.642** | **0.725** | **0.000** | **1.000** | **0.050** |

**Table 3. Breakdown by question type**

| system | type | n | token F1 | nDCG@10 | Recall@5 | refusal accuracy |
| --- | --- | --- | --- | --- | --- | --- |
| closed_book | factoid | 15 | 0.180 | n/a | n/a | n/a |
| closed_book | multi-hop | 5 | 0.198 | n/a | n/a | n/a |
| closed_book | unanswerable | 5 | n/a | n/a | n/a | 0.800 |
| bm25 | factoid | 15 | 0.364 | 0.603 | 0.700 | n/a |
| bm25 | multi-hop | 5 | 0.351 | 0.211 | 0.300 | n/a |
| bm25 | unanswerable | 5 | n/a | n/a | n/a | 1.000 |
| dense | factoid | 15 | 0.431 | 0.755 | 0.833 | n/a |
| dense | multi-hop | 5 | 0.247 | 0.331 | 0.400 | n/a |
| dense | unanswerable | 5 | n/a | n/a | n/a | 1.000 |
| hybrid | factoid | 15 | 0.434 | 0.768 | 0.933 | n/a |
| hybrid | multi-hop | 5 | 0.376 | 0.383 | 0.400 | n/a |
| hybrid | unanswerable | 5 | n/a | n/a | n/a | 1.000 |

**Retrieval does the work.** Answer quality more than doubles from closed-book
to hybrid (token F1 0.185 to 0.419). Since the model and decoding settings are
identical across all four systems, the difference is attributable to retrieved
context alone. This is the project's central claim and the diagnostic in section
2 rules out the competing explanation that the model already knew the material.

**Hybrid fusion beats both parents.** Hybrid leads on MAP, nDCG@10, MRR,
Precision@3, Precision@5 and Recall@5, and produces the best generation scores
on every measure. The two retrievers fail differently: BM25 matches surface
vocabulary and does well when a question reuses wiki phrasing, while the dense
retriever matches meaning and copes with paraphrase. Fusion recovers documents
that only one of them ranked highly.

**Dense clearly outperforms BM25** here (MAP 0.571 against 0.431), which is
expected given that the questions were written in natural language rather than
by copying wiki phrasing, so lexical overlap with the source passage is often
low.

**Refusal behaviour is essentially solved by retrieval.** All three retrieval
systems refused all five unanswerable questions, while the closed-book system
refused only four of five. Its single failure is instructive: asked which totem
the Ashen Hunters follow, a tribe that does not exist anywhere in the corpus, it
answered "the Totem of the Stag... associated with qualities like perseverance
and tenacity". False refusal on answerable questions stayed low (0.050 for BM25
and hybrid).

**No system fabricated a citation.** Every chunk ID appearing in every generated
answer across all 75 retrieval-system answers corresponded to a chunk actually
supplied in that prompt.

## 6. Error analysis

**Multi-hop retrieval is the system's clear weakness.** For hybrid, nDCG@10
falls from 0.768 on factoid questions to 0.383 on multi-hop, and Recall@5 from
0.933 to 0.400. Inspecting which gold chunks reached the top 5, no system
retrieved both required chunks for more than one of the five multi-hop
questions, and Q16 was missed entirely by all three.

Q16 shows the mechanism plainly. It asks which tribe became the Black Spiral
Dancers and what they found that corrupted them, which needs the White Howlers
page for the Great Pit and the Black Spiral Dancers page for the Spiral
Labyrinth. All five retrieved chunks came from the Black Spiral Dancers page,
and neither "Great Pit" nor "Spiral Labyrinth" appears anywhere in the retrieved
context.

The generated answer is worth examining carefully, because it is not wrong. It
names the White Howlers correctly and says they descended into Malfeas and were
corrupted, which is true: the Spiral Labyrinth lies within Malfeas. What it never
does is answer the question that was asked, namely what they found. A reader
using this system to learn the setting would come away without the Great Pit or
the Spiral Labyrinth and with no indication that anything was missing. The
failure is one of specificity rather than accuracy, and it is the more dangerous
kind for a system whose purpose is to teach a corpus, because a plausible and
technically true answer offers the user no signal to go and check. Deleting the
second clause and asking only which tribe became the Black Spiral Dancers is
answered correctly and precisely in a single sentence, which isolates the added
hop as the thing that breaks retrieval.

The cause is structural: a multi-hop question is encoded as a single query, so
whichever hop dominates lexically or semantically pulls the entire ranking toward
one page and the second page never surfaces. Query decomposition, or an
iterative retrieve-read loop that issues a follow-up query using the first hop's
result, is the standard remedy and would be the first thing to add.

**Generation can fail even when retrieval succeeds, but unstably.** Q11 asks how
much starting Gnosis each Garou breed receives. The gold chunk was retrieved and
ranked inside the top 5, and it states the answer explicitly: "Homid characters
start out with the least Gnosis (1), Lupus Characters with the most (5) and
Metis characters start out in the middle (3)." In the recorded evaluation run
the model reported only the Homid value and claimed no information was available
about the other breeds, then supplied Gnosis values for Gurahl, Mokole, Nuwisha
and Rokea, which are not Garou breeds at all but separate Changing Breed species.

Repeating the query with byte-identical retrieved context did not reproduce
consistently. Across sessions it sometimes reproduced that failure and sometimes
extracted all three values correctly. The variation tracks Ollama model reloads
rather than anything in the pipeline, and is discussed in section 8. Two
conclusions follow. Values bound to superlatives rather than stated as a list sit
close to this model's extraction limit, so tiny numerical differences in the
forward pass are enough to flip the outcome; the wiki's overloading of the word
"breed" compounds it. More importantly, any single generated answer is weak
evidence about this system. Conclusions should rest on aggregate metrics and on
retrieval-level diagnostics, which are exact and reproducible, rather than on an
individual output. Preserving wiki tables during preprocessing instead of
stripping them, and adding a worked extraction example to the prompt, would both
reduce the difficulty of this particular case.

**Automatic answer metrics understate quality.** Exact match is 0.000 for every
system, which reflects the fact that reference answers are full sentences rather
than short spans and carries no real information. Token F1 similarly penalises
correct answers phrased differently from the reference. On Q05, the model
correctly reported that a klaive is made of silver and cited a genuine mechanical
consequence, but scored 0.254 because it chose a different consequence than the
reference did. Reported automatic figures should therefore be read as a lower
bound on correctness, which is why the repository also emits a human judgement
sheet covering every generated answer.

## 7. Limitations

The evaluation set is small at 25 questions, and per-type conclusions rest on
five questions each. The corpus deliberately excludes Dark Ages, Wild West,
Fifth Edition and video game material to keep the lore internally consistent, so
results do not generalise to the full line. The 1,200 document cap keeps the
richest pages and discards shorter ones, which biases the corpus toward
well-developed topics. Retrieval is evaluated against gold chunks chosen by one
annotator, and chunk overlap makes that labelling partly arbitrary. The
generation model is an 8B parameter quantised model, so the generation-side
results reflect its capability rather than the ceiling of the approach. Finally,
no reranking stage was implemented, and a cross-encoder reranker over the top 10
would be the obvious next improvement alongside multi-hop query decomposition.

## 8. Reproducibility

Seeds are fixed at 42 and applied to Python, NumPy and torch, and retrieval
tie-breaking is deterministic. The corpus snapshot, the chunk file and the
evaluation set are all committed, so the network crawl is not on the
reproduction path and results do not drift as the wiki is edited. Dependencies
are pinned in `requirements.txt`.

**The retrieval pipeline is bit-reproducible and this was verified.** Cloning
the repository into a clean directory and rerunning preprocessing produces a
`chunks.jsonl` identical by SHA256 to the committed one, which matters because
the gold chunk IDs would silently break otherwise. Repeated retrieval for a
fixed query returns an identical ranking.

**Generation is not bit-reproducible, and this should be stated plainly.**
Decoding is greedy at temperature 0 with a fixed sampling seed. Within a single
model load, repeated generation against a fixed context was byte-identical
across five runs. Across model loads it was not: the Q11 answer reproduced its
failure in one session and produced the fully correct answer in another, with
identical retrieved context both times. This is consistent with llama.cpp
backends, where GPU kernel selection and floating point reduction order can vary
between loads, and neither temperature 0 nor a fixed seed defends against it.

Two consequences. The generation figures in Tables 2 and 3 carry run-to-run
variance and should not be read as exact constants, whereas the retrieval
figures in Table 1 are exact. And any claim resting on a single generated answer
is fragile, which is why the error analysis above is anchored on which chunks
were retrieved rather than on what the model said about them.

A single command reproduces every table above:

```
python run_all.py
```

## Disclosure

The ten closed-book diagnostic questions in `eval/diagnostic_questions.json`
were drafted with LLM assistance and then verified by hand against the corpus
text, with each model answer judged manually. The 25 gold evaluation questions
in `eval/gold_questions.csv` were written by the author after reading the
corpus, as the assignment requires.
