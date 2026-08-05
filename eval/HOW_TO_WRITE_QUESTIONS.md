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
