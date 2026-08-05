"""Closed-book corpus suitability diagnostic (assignment section 2).

Ten factual questions whose answers live in the corpus are put to the
generation model with no retrieved context and no permission to decline. If the
model answers most of them correctly from parametric knowledge alone, the corpus
is too well known for retrieval to demonstrate any benefit.

The prompt deliberately pushes for a best guess rather than a refusal. A model
that says "I don't know" because it was told it may would prove nothing about
what it actually knows.

An automatic keyword check gives a first pass over the answers. It is a
convenience, not the verdict: the graded result is the human judgement column.

Run:
    python -m wolfrag.diagnostic
"""

from __future__ import annotations

import csv
import json

from wolfrag import config, generate, metrics


def load_questions() -> list[dict]:
    if not config.DIAGNOSTIC_QUESTIONS_PATH.exists():
        raise FileNotFoundError(
            f"No diagnostic questions at {config.DIAGNOSTIC_QUESTIONS_PATH}."
        )
    return json.loads(config.DIAGNOSTIC_QUESTIONS_PATH.read_text(encoding="utf-8"))


def keyword_hit(answer: str, keywords: list[str]) -> bool:
    """True when every required keyword appears in the answer."""
    normalised = metrics.normalise_answer(answer)
    return all(metrics.normalise_answer(k) in normalised for k in keywords)


def load_human_judgements() -> dict[str, dict[str, str]]:
    """Read the committed human verdicts, which override the keyword check.

    Human judgement is an input to this script, not an output of it. Keeping the
    verdicts in their own file means rerunning the diagnostic never overwrites
    them, and the reported figure stays reproducible.
    """
    path = config.EVAL_DIR / "diagnostic_human_judgements.csv"
    if not path.exists():
        return {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return {row["id"]: row for row in csv.DictReader(handle)}


def main() -> None:
    config.set_seeds()

    if not generate.check_server():
        raise SystemExit(
            f"Ollama is not reachable at {config.OLLAMA_HOST} or the model "
            f"{config.GENERATION_MODEL} is missing."
        )

    questions = load_questions()
    judgements = load_human_judgements()
    rows = []

    print(
        f"Closed-book diagnostic: {len(questions)} questions, "
        f"model {config.GENERATION_MODEL}, no retrieved context.\n"
    )

    for index, item in enumerate(questions, start=1):
        text = generate.ollama_generate(
            prompt=f"Question: {item['question']}\n\nAnswer:",
            system=generate.DIAGNOSTIC_SYSTEM_PROMPT,
        )
        auto_correct = keyword_hit(text, item.get("answer_keywords", []))
        qid = item.get("id", f"D{index:02d}")
        verdict = judgements.get(qid, {})
        rows.append(
            {
                "id": qid,
                "question": item["question"],
                "reference_answer": item["reference_answer"],
                "source_page": item.get("source_page", ""),
                "model_answer": text,
                "auto_keyword_correct": int(auto_correct),
                "token_F1": round(metrics.token_f1(text, item["reference_answer"]), 4),
                "human_correct_1_0": verdict.get("human_correct_1_0", ""),
                "judge_note": verdict.get("judge_note", ""),
            }
        )
        mark = "HIT " if auto_correct else "miss"
        print(f"{mark} {item.get('id', index)}: {item['question']}")
        print(f"      model: {text[:160]}")
        print(f"      gold : {item['reference_answer'][:160]}\n")

    path = config.RESULTS_DIR / "diagnostic_closed_book.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    auto_score = sum(r["auto_keyword_correct"] for r in rows)
    mean_f1 = metrics.mean([r["token_F1"] for r in rows])
    judged = [r for r in rows if r["human_correct_1_0"] != ""]
    human_score = sum(int(r["human_correct_1_0"]) for r in judged)

    summary = {
        "model": config.GENERATION_MODEL,
        "n_questions": len(rows),
        "auto_keyword_correct": auto_score,
        "auto_keyword_accuracy": round(auto_score / len(rows), 4),
        "mean_token_f1": round(mean_f1, 4),
        "human_judged": len(judged),
        "human_correct": human_score,
        "human_accuracy": (
            round(human_score / len(judged), 4) if judged else None
        ),
        "note": (
            "Automatic keyword matching is a first pass only and overcounts: it "
            "credits any answer containing the keyword, including incoherent "
            "ones. The figure quoted in the report is the human accuracy."
        ),
    }
    (config.RESULTS_DIR / "diagnostic_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print("-" * 70)
    print(
        f"Automatic keyword score: {auto_score}/{len(rows)} "
        f"({auto_score / len(rows):.0%}), mean token F1 {mean_f1:.3f}"
    )
    if judged:
        print(
            f"Human verified score:    {human_score}/{len(judged)} "
            f"({human_score / len(judged):.0%})  <- report this figure"
        )
    else:
        print("No human judgements found. Fill in diagnostic_human_judgements.csv.")
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
