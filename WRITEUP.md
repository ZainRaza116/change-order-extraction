# Change-order extraction: approach and failure modes

## Approach

**Deterministic code around a single LLM call.** Text comes out of the document first:
PyMuPDF for the PDF text layer, pdfplumber for tables, and for scanned pages (detected by
having almost no extractable text) an OpenCV clean-up — median denoise, deskew, removal of
ruled lines — followed by Tesseract, keeping its word confidences. One schema-constrained
LLM call per document then fills a strict JSON schema; if the JSON fails validation, the
errors go back to the model once. There is no agent framework: the problem is one structured
call plus deterministic code, and a framework would add weight without adding value.

**The model copies, code computes.** The model returns every value exactly as printed
(`"($2,250.00)"`, `"Sept. 18, 2026"`, `"Six (6) calendar days"`) together with a verbatim
evidence quote and page number. Tested parsers turn those strings into decimals, ISO dates
and day counts, and return nothing rather than guess (`"f/u Mon"` is not a date). The model
never does arithmetic, so every number traces back to the page.

**Validation checks that the numbers agree with each other.** Eleven rules run on the typed
result: original + previous changes + this change = revised contract sum; line items +
markups (overhead & profit, bond) = change amount; quantity × unit cost = amount; credits are
negative; nobody signs before the issue date; "approved" requires every signature. A failed
rule never changes a value. It lowers confidence on the fields involved and explains why.

**Confidence is measured, not asked for.** Each field's score combines five signals: does the
evidence quote actually occur in the source text (fuzzy match, 35%), does the value appear
inside its own quote (20%), did it parse and pass every rule (20%), OCR quality of its page
(15%), and whether competing values were seen, such as a superseded proposal (10%). Caps apply
on top: evidence not found in the source → at most 0.3 (likely hallucination); a failed rule
→ at most 0.6. A document needs review if any required field is below 0.7 or any rule fails.

## Results

Four documents with hand-labelled ground truth: a plain-text email, a digital G701-style PDF,
a synthetic scan of that PDF, and the real faxed scan supplied with the task (stamp across the
line items, handwriting, a struck-out superseded figure). Model: `gpt-4o-mini`.

| Metric | Result |
|---|---|
| Field accuracy (18 fields × 4 documents) | 93% |
| Hallucinated values (value where the truth is empty) | 0 of 5 |
| Line-item amounts recovered | 14 of 15 (the missing one is under the stamp) |
| Documents containing an error that were flagged for review | **2 of 2** |
| Clean documents passed without review | 2 of 2 |
| Cost / LLM latency per document | $0.0015 / 26 s |

| Confidence | Values | Accuracy |
|---|---|---|
| 0.0–0.5 | 2 | 50% |
| 0.5–0.8 | 7 | 86% |
| 0.8–1.0 | 74 | 96% |

Accuracy rises with confidence, which is the property the scores need. **Four documents are a
smoke test, not a benchmark**: two values in the lowest bucket prove nothing on their own, and
I labelled the ground truth myself.

## Failure modes

| Failure mode | Observed? | Mitigation | Residual risk |
|---|---|---|---|
| **Stamp or mark over values** | Yes: the real scan's RECEIVED stamp hid line item 1, half the description and the stated reason | Missing amount → confidence 0; line-item total rule fails → document flagged | Text fields lost under the stamp are not flagged (see next row) |
| **Self-consistent OCR error** | Yes: project number `24-118` read as `24 18`, scored 0.97 | Contract-math rules catch this for amounts | **Not caught for text fields.** The model faithfully quotes the misread text, so every evidence check passes. The main weakness. |
| Model substitutes a nearby field | Yes: with the reason hidden, the model copied the description into `reason_text` (0.97) | Evidence must contain the value | Wrong-but-grounded values pass |
| Status contradicted by a note | Yes: "approved", but the owner's line says "waiting on CFO sig" | `status_consistency` rule capped it at 0.60 → flagged | — |
| Hallucinated values | None in this set | Evidence fuzzy-matched against the source; null if absent | A real quote that belongs to another field |
| Wrong one of several amounts | Prompt v1 put the total into `subtotal` on 3 documents | Prompt rule; ambiguity signal; totals rules | Two plausible amounts that both satisfy the math |
| Credits and signs | No errors | Parentheses kept verbatim, parsed as negative; `credit_signs` rule | Unusual credit wording |
| Handwriting | Partly: "OK per mtg" read, "waiting on CFO sig" garbled | Low OCR confidence lowers the score | Not reliably extracted |
| Multiple COs in one file, non-USD, ambiguous dates | Not tested | One CO per file assumed; `02/03/26` read as US month/day | Out of scope for this version |
| Prompt injection in document text | Not tested | Document wrapped as data; output is schema-constrained | Low: the model can only fill fields |

## What I would do next

1. **Vision re-read when a rule fails on a scanned page.** Three of the four real errors come
   from the stamp. The vision fallback exists but triggers only on low average OCR confidence,
   and this scan averaged 0.82 because the damage is local.
2. **Word-level OCR confidence for each evidence quote** instead of the page average, which
   might catch errors like `24 18`.
3. **More and harder documents** (real change orders, multi-page tables, credits-only, missing
   fields) before trusting the calibration, and per-field review thresholds tuned on them.
4. A review UI that highlights each field's evidence on the page image.
