"""Run the full experiment: every system against the gold evaluation set.

Four systems share one generation model and one decoding configuration, so any
difference between them comes from retrieval alone:

    closed_book  no retrieval, the ablation that isolates retrieval's benefit
    bm25         classical lexical retrieval
    dense        sentence-transformers embeddings, cosine similarity
    hybrid       reciprocal rank fusion of the two above

Retrieval metrics are reported over answerable questions only, since an
unanswerable question has no gold chunk to find. Unanswerable questions are
instead scored on whether the system correctly refused.

Run:
    python -m wolfrag.evaluate
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from wolfrag import config, generate, metrics, retrieval

QUESTION_TYPES = ["factoid", "multi_hop", "unanswerable"]


# ---------------------------------------------------------------------------
# Gold set
# ---------------------------------------------------------------------------


def _split_ids(value: str) -> list[str]:
    if not value:
        return []
    return [part.strip() for part in value.replace(",", ";").split(";") if part.strip()]


def load_gold(path=None) -> list[dict[str, Any]]:
    """Read the human written evaluation set."""
    path = path or config.GOLD_QUESTIONS_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Gold question set not found at {path}. "
            "Run 'python -m wolfrag.make_gold_template' and fill it in."
        )

    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            question = (row.get("question") or "").strip()
            if not question:
                continue
            qtype = (row.get("question_type") or "factoid").strip().lower()
            if qtype not in QUESTION_TYPES:
                raise ValueError(
                    f"Question {row.get('question_id')!r} has unknown type {qtype!r}. "
                    f"Expected one of {QUESTION_TYPES}."
                )
            rows.append(
                {
                    "question_id": (row.get("question_id") or "").strip(),
                    "question": question,
                    "question_type": qtype,
                    "reference_answer": (row.get("reference_answer") or "").strip(),
                    "gold_chunk_ids": _split_ids(row.get("gold_chunk_ids") or ""),
                    "gold_doc_ids": _split_ids(row.get("gold_doc_ids") or ""),
                    "notes": (row.get("notes") or "").strip(),
                }
            )
    return rows


def validate_gold(gold: list[dict[str, Any]], chunk_ids: set[str]) -> list[str]:
    """Check the gold set against the assignment's requirements."""
    problems: list[str] = []
    counts = {t: sum(1 for g in gold if g["question_type"] == t) for t in QUESTION_TYPES}

    if len(gold) < 10:
        problems.append(f"Only {len(gold)} questions, the assignment requires at least 10.")
    if counts["multi_hop"] < 2:
        problems.append(
            f"Only {counts['multi_hop']} multi-hop questions, at least 2 are required."
        )
    if counts["unanswerable"] < 2:
        problems.append(
            f"Only {counts['unanswerable']} unanswerable questions, at least 2 are required."
        )

    for row in gold:
        qid = row["question_id"] or row["question"][:40]
        if row["question_type"] == "unanswerable":
            if row["gold_chunk_ids"]:
                problems.append(f"{qid}: unanswerable questions must have no gold chunks.")
            continue
        if not row["gold_chunk_ids"]:
            problems.append(f"{qid}: missing gold_chunk_ids.")
        if row["question_type"] == "multi_hop" and len(row["gold_chunk_ids"]) < 2:
            problems.append(
                f"{qid}: multi-hop questions need at least 2 gold chunks, "
                f"found {len(row['gold_chunk_ids'])}."
            )
        for chunk_id in row["gold_chunk_ids"]:
            if chunk_id not in chunk_ids:
                problems.append(f"{qid}: gold chunk {chunk_id} is not in the corpus.")
        if not row["reference_answer"]:
            problems.append(f"{qid}: missing reference_answer.")

    return problems


# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------


def run_system(
    system: str,
    gold: list[dict[str, Any]],
    retrievers: dict[str, retrieval.BaseRetriever],
    chunk_lookup: dict[str, dict[str, Any]],
    verbose: bool = True,
) -> list[dict[str, Any]]:
    """Retrieve and generate for every question under one system."""
    records: list[dict[str, Any]] = []
    closed_book = system == "closed_book"

    for index, row in enumerate(gold, start=1):
        if closed_book:
            retrieved: list[dict[str, Any]] = []
        else:
            retrieved = retrievers[system].retrieve(
                row["question"], config.MAX_RETRIEVAL_DEPTH
            )

        top_chunks = retrieved[: config.TOP_K]
        result = generate.answer(row["question"], top_chunks, closed_book=closed_book)

        ranked_chunks = [c["chunk_id"] for c in retrieved]
        ranked_docs: list[str] = []
        for chunk in retrieved:
            if chunk["doc_id"] not in ranked_docs:
                ranked_docs.append(chunk["doc_id"])

        records.append(
            {
                "system": system,
                "question_id": row["question_id"],
                "question": row["question"],
                "question_type": row["question_type"],
                "reference_answer": row["reference_answer"],
                "gold_chunk_ids": row["gold_chunk_ids"],
                "gold_doc_ids": row["gold_doc_ids"],
                "ranked_chunk_ids": ranked_chunks,
                "ranked_doc_ids": ranked_docs,
                "context_chunk_ids": [c["chunk_id"] for c in top_chunks],
                "answer": result["answer"],
                "cited_chunk_ids": result["cited_chunk_ids"],
                "hallucinated_citations": result["hallucinated_citations"],
                "refused": result["refused"],
            }
        )

        if verbose:
            print(
                f"  [{system}] {index}/{len(gold)} {row['question_id']}",
                flush=True,
            )

    return records


def score_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate one system's per-question records into reported metrics."""
    answerable = [r for r in records if r["question_type"] != "unanswerable"]
    unanswerable = [r for r in records if r["question_type"] == "unanswerable"]
    has_retrieval = any(r["ranked_chunk_ids"] for r in records)

    row: dict[str, Any] = {"system": records[0]["system"], "n_questions": len(records)}

    if has_retrieval and answerable:
        for k in config.EVAL_K_VALUES:
            row[f"P@{k}"] = metrics.mean(
                [
                    metrics.precision_at_k(r["ranked_chunk_ids"], r["gold_chunk_ids"], k)
                    for r in answerable
                ]
            )
            row[f"Recall@{k}"] = metrics.mean(
                [
                    metrics.recall_at_k(r["ranked_chunk_ids"], r["gold_chunk_ids"], k)
                    for r in answerable
                ]
            )
        row["MAP"] = metrics.mean(
            [
                metrics.average_precision(r["ranked_chunk_ids"], r["gold_chunk_ids"])
                for r in answerable
            ]
        )
        row["nDCG@10"] = metrics.mean(
            [
                metrics.ndcg_at_k(r["ranked_chunk_ids"], r["gold_chunk_ids"], 10)
                for r in answerable
            ]
        )
        row["MRR"] = metrics.mean(
            [
                metrics.reciprocal_rank(r["ranked_chunk_ids"], r["gold_chunk_ids"])
                for r in answerable
            ]
        )
        row["doc_MAP"] = metrics.mean(
            [
                metrics.average_precision(r["ranked_doc_ids"], r["gold_doc_ids"])
                for r in answerable
                if r["gold_doc_ids"]
            ]
        )
        row["doc_MRR"] = metrics.mean(
            [
                metrics.reciprocal_rank(r["ranked_doc_ids"], r["gold_doc_ids"])
                for r in answerable
                if r["gold_doc_ids"]
            ]
        )

    if answerable:
        row["token_F1"] = metrics.mean(
            [metrics.token_f1(r["answer"], r["reference_answer"]) for r in answerable]
        )
        row["ROUGE_L"] = metrics.mean(
            [metrics.rouge_l(r["answer"], r["reference_answer"]) for r in answerable]
        )
        row["exact_match"] = metrics.mean(
            [metrics.exact_match(r["answer"], r["reference_answer"]) for r in answerable]
        )
        row["false_refusal_rate"] = metrics.mean(
            [float(r["refused"]) for r in answerable]
        )

    if unanswerable:
        row["refusal_accuracy"] = metrics.mean(
            [float(r["refused"]) for r in unanswerable]
        )

    if has_retrieval and answerable:
        row["citation_precision"] = metrics.mean(
            [
                metrics.citation_precision(r["cited_chunk_ids"], r["gold_chunk_ids"])
                for r in answerable
            ]
        )
        row["citation_recall"] = metrics.mean(
            [
                metrics.citation_recall(r["cited_chunk_ids"], r["gold_chunk_ids"])
                for r in answerable
            ]
        )
        row["answers_with_citation"] = metrics.mean(
            [float(bool(r["cited_chunk_ids"])) for r in answerable]
        )
        row["hallucinated_citation_rate"] = metrics.mean(
            [float(bool(r["hallucinated_citations"])) for r in records]
        )

    return row


def score_by_type(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Break a system's results down by question type."""
    rows = []
    for qtype in QUESTION_TYPES:
        subset = [r for r in records if r["question_type"] == qtype]
        if not subset:
            continue
        row: dict[str, Any] = {
            "system": records[0]["system"],
            "question_type": qtype,
            "n": len(subset),
        }
        if qtype == "unanswerable":
            row["refusal_accuracy"] = metrics.mean([float(r["refused"]) for r in subset])
        else:
            row["token_F1"] = metrics.mean(
                [metrics.token_f1(r["answer"], r["reference_answer"]) for r in subset]
            )
            row["ROUGE_L"] = metrics.mean(
                [metrics.rouge_l(r["answer"], r["reference_answer"]) for r in subset]
            )
            if subset[0]["ranked_chunk_ids"]:
                row["nDCG@10"] = metrics.mean(
                    [
                        metrics.ndcg_at_k(r["ranked_chunk_ids"], r["gold_chunk_ids"], 10)
                        for r in subset
                    ]
                )
                row["Recall@5"] = metrics.mean(
                    [
                        metrics.recall_at_k(r["ranked_chunk_ids"], r["gold_chunk_ids"], 5)
                        for r in subset
                    ]
                )
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def write_csv(path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    k: (round(v, 4) if isinstance(v, float) else v)
                    for k, v in row.items()
                }
            )


def markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    present = [c for c in columns if any(c in row for row in rows)]
    lines = ["| " + " | ".join(present) + " |"]
    lines.append("| " + " | ".join("---" for _ in present) + " |")
    for row in rows:
        cells = []
        for column in present:
            value = row.get(column, "")
            cells.append(f"{value:.3f}" if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_human_eval_template(all_records: list[dict[str, Any]]) -> None:
    """Emit the sheet for manual judging of every generated answer."""
    path = config.RESULTS_DIR / "human_eval_template.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "system",
                "question_id",
                "question_type",
                "question",
                "reference_answer",
                "generated_answer",
                "cited_chunk_ids",
                "context_chunk_ids",
                "correct_1_0",
                "supported_by_citations_1_0",
                "correctly_refused_1_0",
                "judge_notes",
            ]
        )
        for record in all_records:
            writer.writerow(
                [
                    record["system"],
                    record["question_id"],
                    record["question_type"],
                    record["question"],
                    record["reference_answer"],
                    record["answer"],
                    ";".join(record["cited_chunk_ids"]),
                    ";".join(record["context_chunk_ids"]),
                    "",
                    "",
                    "",
                    "",
                ]
            )
    print(f"Wrote {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--systems",
        nargs="*",
        default=config.SYSTEMS,
        help="Subset of systems to run.",
    )
    parser.add_argument(
        "--gold",
        type=Path,
        default=None,
        help="Alternative gold question CSV. Used for testing the harness.",
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    config.set_seeds()

    if not generate.check_server():
        raise SystemExit(
            f"Ollama is not reachable at {config.OLLAMA_HOST} or the model "
            f"{config.GENERATION_MODEL} is missing. Start Ollama and run "
            f"'ollama pull {config.GENERATION_MODEL}'."
        )

    print("Building retrievers ...")
    retrievers = retrieval.build_all()
    chunks = retrievers["bm25"].chunks
    chunk_lookup = {c["chunk_id"]: c for c in chunks}

    gold = load_gold(args.gold)
    problems = validate_gold(gold, set(chunk_lookup))
    if problems:
        print("\nGold set validation problems:")
        for problem in problems:
            print(f"  - {problem}")
        raise SystemExit(
            "\nFix the gold question set before running the experiment."
        )
    print(f"Gold set: {len(gold)} questions, all checks passed.")

    all_records: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    by_type_rows: list[dict[str, Any]] = []

    for system in args.systems:
        print(f"\nRunning system: {system}")
        records = run_system(
            system, gold, retrievers, chunk_lookup, verbose=not args.quiet
        )
        all_records.extend(records)
        summary_rows.append(score_records(records))
        by_type_rows.extend(score_by_type(records))

    (config.RESULTS_DIR / "per_question_results.json").write_text(
        json.dumps(all_records, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_csv(config.RESULTS_DIR / "summary_metrics.csv", summary_rows)
    write_csv(config.RESULTS_DIR / "metrics_by_question_type.csv", by_type_rows)
    write_human_eval_template(all_records)

    metadata = generate.model_metadata()
    metadata["systems"] = args.systems
    metadata["n_questions"] = len(gold)
    metadata["n_chunks"] = len(chunks)
    (config.RESULTS_DIR / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    retrieval_columns = (
        ["system"]
        + [f"P@{k}" for k in config.EVAL_K_VALUES]
        + ["MAP", "nDCG@10", "MRR"]
        + [f"Recall@{k}" for k in config.EVAL_K_VALUES]
        + ["doc_MAP", "doc_MRR"]
    )
    generation_columns = [
        "system",
        "token_F1",
        "ROUGE_L",
        "exact_match",
        "citation_precision",
        "citation_recall",
        "answers_with_citation",
        "hallucinated_citation_rate",
        "refusal_accuracy",
        "false_refusal_rate",
    ]

    tables = [
        "# Experimental results\n",
        "## Table 1. Retrieval quality (answerable questions, chunk level)\n",
        markdown_table(
            [r for r in summary_rows if "MAP" in r], retrieval_columns
        ),
        "\n## Table 2. Generation quality\n",
        markdown_table(summary_rows, generation_columns),
        "\n## Table 3. Breakdown by question type\n",
        markdown_table(
            by_type_rows,
            [
                "system",
                "question_type",
                "n",
                "token_F1",
                "ROUGE_L",
                "nDCG@10",
                "Recall@5",
                "refusal_accuracy",
            ],
        ),
        "\n",
    ]
    (config.RESULTS_DIR / "tables.md").write_text("\n".join(tables), encoding="utf-8")

    print("\n" + "\n".join(tables))
    print(f"Wrote results to {config.RESULTS_DIR}")


if __name__ == "__main__":
    main()
