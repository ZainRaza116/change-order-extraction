Evaluated 9 documents with gpt-4o-mini.

| Metric | Result |
|---|---|
| Field accuracy (16 structured fields) | 99% |
| Required-field accuracy | 100% |
| Hallucinated values (value where truth is empty) | 0 of 27 |
| Line-item amounts recovered | 26 of 26 |
| Markup amounts recovered | 7 of 7 |
| Wrong fields flagged (confidence < 0.7) | 1 of 2 |
| Documents with an error that were flagged for review | 2 of 2 |
| Clean documents passed without review | 5 of 7 |
| Cost per document (mean) | $0.0010 |
| LLM latency per document (mean) | 11.1 s |

**Calibration** (does higher confidence mean more often right?)

| Confidence | Values | Accuracy |
|---|---|---|
| 0.0-0.5 | 0 | - |
| 0.5-0.8 | 4 | 75% (too few to judge) |
| 0.8-1.0 | 150 | 97% |

**Free text** (description, reason_text; answers are paraphrased, so these are similarity scores, not pass/fail): 8 of 17 at or above 80, mean 77.

**Errors**

| Document | Field | Outcome | Confidence | Got | Expected |
|---|---|---|---|---|---|
| s02_scanned_messy | project_number | wrong | 0.97 | 24 18 | 24-118 |
| s03_letter_format | status | wrong | 0.60 | pending | proposed |

