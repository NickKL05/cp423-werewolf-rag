"""Create the empty gold evaluation set for a human to fill in.

The assignment requires the questions to be written by a team member reading the
corpus, so this script deliberately produces empty question rows. It only lays
out the required structure: 25 slots split across the three question types, with
the minimums the assignment demands already satisfied by the layout.

Existing files are never overwritten.

Run:
    python -m wolfrag.make_gold_template
"""

from __future__ import annotations

import csv

from wolfrag import config

FIELDNAMES = [
    "question_id",
    "question_type",
    "question",
    "reference_answer",
    "gold_chunk_ids",
    "gold_doc_ids",
    "notes",
]

# 15 factoid, 5 multi-hop, 5 unanswerable. The assignment floor is 2 multi-hop
# and 2 unanswerable, so this leaves room to discard a few during review.
LAYOUT = [("factoid", 15), ("multi_hop", 5), ("unanswerable", 5)]

GUIDANCE = """\
How to fill this in
===================

One row per question. Delete any rows you do not use, but keep at least 10
questions overall, at least 2 multi_hop and at least 2 unanswerable.

question_id       Leave as generated, for example Q01.
question_type     factoid, multi_hop, or unanswerable. Do not invent others.
question          Your question, written after reading the page.
reference_answer  The correct answer in your own words, one or two sentences.
                  For unanswerable questions write exactly: I don't know
gold_chunk_ids    Semicolon separated chunk IDs that contain the answer, for
                  example C01234;C01235. Find these in the Corpus browser tab
                  of the Streamlit app. Leave EMPTY for unanswerable questions.
gold_doc_ids      Semicolon separated document IDs, for example WTA0123.
                  Leave EMPTY for unanswerable questions.
notes             Anything worth recording, such as why a question is hard.

Rules that the validator enforces before the experiment will run:
  - multi_hop questions need at least 2 gold chunk IDs, and those chunks should
    come from different sections or different pages. The question must genuinely
    require both, not merely be answerable from one.
  - unanswerable questions must have no gold chunks. Write questions that sound
    like they belong to this corpus but whose answer is genuinely absent, for
    example asking about a tribe that does not exist, or asking for a detail the
    wiki never states. Do not use questions from another game line, since those
    are too easy to refuse.
  - every gold chunk ID must exist in data/processed/chunks.jsonl.
"""


def main() -> None:
    if config.GOLD_QUESTIONS_PATH.exists():
        print(
            f"{config.GOLD_QUESTIONS_PATH} already exists, leaving it untouched."
        )
        return

    rows = []
    index = 1
    for question_type, count in LAYOUT:
        for _ in range(count):
            rows.append(
                {
                    "question_id": f"Q{index:02d}",
                    "question_type": question_type,
                    "question": "",
                    "reference_answer": (
                        "I don't know" if question_type == "unanswerable" else ""
                    ),
                    "gold_chunk_ids": "",
                    "gold_doc_ids": "",
                    "notes": "",
                }
            )
            index += 1

    with config.GOLD_QUESTIONS_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    guide_path = config.EVAL_DIR / "HOW_TO_WRITE_QUESTIONS.md"
    guide_path.write_text(GUIDANCE, encoding="utf-8")

    print(f"Wrote {config.GOLD_QUESTIONS_PATH} with {len(rows)} empty rows.")
    print(f"Wrote {guide_path}")


if __name__ == "__main__":
    main()
