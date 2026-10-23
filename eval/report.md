Evaluated 4 documents with gpt-4o-mini.

| Metric | Result |
|---|---|
| Field accuracy (16 structured fields) | 97% |
| Required-field accuracy | 100% |
| Hallucinated values (value where truth is empty) | 0 of 5 |
| Line-item amounts recovered | 15 of 15 |
| Markup amounts recovered | 2 of 2 |
| Wrong fields flagged (confidence < 0.7) | 1 of 2 |
| Documents with an error that were flagged for review | 2 of 2 |
| Clean documents passed without review | 2 of 2 |
| Cost per document (mean) | $0.0010 |
| LLM latency per document (mean) | 12.0 s |

**Calibration** (does higher confidence mean more often right?)

| Confidence | Values | Accuracy |
|---|---|---|
| 0.0-0.5 | 0 | - |
| 0.5-0.8 | 3 | 67% (too few to judge) |
| 0.8-1.0 | 73 | 99% |

**Free text** (description, reason_text; answers are paraphrased, so these are similarity scores, not pass/fail): 5 of 8 at or above 80, mean 81.

**Errors**

| Document | Field | Outcome | Confidence | Got | Expected |
|---|---|---|---|---|---|
| co_001_email | status | wrong | 0.60 | pending | proposed |
| co_004_real_scan | project_number | wrong | 0.97 | 24 18 | 24-118 |

