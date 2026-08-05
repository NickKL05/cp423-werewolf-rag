# Experimental results

## Table 1. Retrieval quality (answerable questions, chunk level)

| system | P@1 | P@3 | P@5 | P@10 | MAP | nDCG@10 | MRR | Recall@1 | Recall@3 | Recall@5 | Recall@10 | doc_MAP | doc_MRR |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bm25 | 0.350 | 0.167 | 0.140 | 0.085 | 0.431 | 0.505 | 0.458 | 0.325 | 0.450 | 0.600 | 0.700 | 0.628 | 0.690 |
| dense | 0.450 | 0.233 | 0.170 | 0.105 | 0.571 | 0.649 | 0.599 | 0.425 | 0.625 | 0.725 | 0.850 | 0.850 | 0.875 |
| hybrid | 0.450 | 0.267 | 0.190 | 0.110 | 0.582 | 0.672 | 0.618 | 0.400 | 0.700 | 0.800 | 0.900 | 0.793 | 0.833 |

## Table 2. Generation quality

| system | token_F1 | ROUGE_L | exact_match | citation_precision | citation_recall | answers_with_citation | hallucinated_citation_rate | refusal_accuracy | false_refusal_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| closed_book | 0.185 | 0.178 | 0.000 |  |  |  |  | 0.800 | 0.200 |
| bm25 | 0.361 | 0.350 | 0.000 | 0.417 | 0.500 | 0.950 | 0.000 | 1.000 | 0.050 |
| dense | 0.385 | 0.367 | 0.000 | 0.542 | 0.600 | 0.900 | 0.000 | 1.000 | 0.100 |
| hybrid | 0.419 | 0.380 | 0.000 | 0.642 | 0.725 | 0.950 | 0.000 | 1.000 | 0.050 |

## Table 3. Breakdown by question type

| system | question_type | n | token_F1 | ROUGE_L | nDCG@10 | Recall@5 | refusal_accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- |
| closed_book | factoid | 15 | 0.180 | 0.170 |  |  |  |
| closed_book | multi_hop | 5 | 0.198 | 0.204 |  |  |  |
| closed_book | unanswerable | 5 |  |  |  |  | 0.800 |
| bm25 | factoid | 15 | 0.364 | 0.357 | 0.603 | 0.700 |  |
| bm25 | multi_hop | 5 | 0.351 | 0.326 | 0.211 | 0.300 |  |
| bm25 | unanswerable | 5 |  |  |  |  | 1.000 |
| dense | factoid | 15 | 0.431 | 0.415 | 0.755 | 0.833 |  |
| dense | multi_hop | 5 | 0.247 | 0.222 | 0.331 | 0.400 |  |
| dense | unanswerable | 5 |  |  |  |  | 1.000 |
| hybrid | factoid | 15 | 0.434 | 0.409 | 0.768 | 0.933 |  |
| hybrid | multi_hop | 5 | 0.376 | 0.292 | 0.383 | 0.400 |  |
| hybrid | unanswerable | 5 |  |  |  |  | 1.000 |

