Evaluated 4 documents with gpt-4o-mini.

| Metric | Result |
|---|---|
| Field accuracy (all 18 fields) | 93% |
| Required-field accuracy | 93% |
| Hallucinated values (value where truth is empty) | 0 of 5 |
| Line-item amounts recovered | 14 of 15 |
| Markup amounts recovered | 2 of 2 |
| Wrong fields flagged (confidence < 0.7) | 2 of 5 |
| Documents flagged for review | 2 of 4 |
| Cost per document (mean) | $0.0015 |
| LLM latency per document (mean) | 26.0 s |

**Calibration** (does higher confidence mean more often right?)

| Confidence | Values | Accuracy |
|---|---|---|
| 0.0-0.5 | 2 | 50% |
| 0.5-0.8 | 7 | 86% |
| 0.8-1.0 | 74 | 96% |

**Errors**

| Document | Field | Outcome | Confidence | Got | Expected |
|---|---|---|---|---|---|
| co_001_email | description | missed | 0.00 | None | Add 2-hr rated GWB at Stair 2, relocate  |
| co_004_real_scan | project_number | wrong | 0.97 | 24 18 | 24-118 |
| co_004_real_scan | description | wrong | 0.94 | Per Owner request and Architect's Supple | Per Owner request and Architect's Supple |
| co_004_real_scan | reason_text | wrong | 0.97 | Per Owner request and Architect's Supple | Owner-requested design revision. |
| co_004_real_scan | status | wrong | 0.60 | approved | pending |

