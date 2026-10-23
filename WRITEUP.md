# Change-order extraction: approach and failure modes

## Approach

**Deterministic code around a single LLM call.** Text comes out of the document first:
PyMuPDF for the PDF text layer, pdfplumber for tables, and for scanned pages (detected by
having almost no extractable text) an OpenCV clean-up (median denoise, deskew, removal of
ruled lines) followed by Tesseract, keeping its word confidences. One schema-constrained call
to `gpt-4o-mini` per document then fills a strict JSON schema; if the JSON fails validation,
the errors go back to the model once. There is no agent framework: the problem is one
structured call plus deterministic code.

**The model copies, code computes.** The model returns every value exactly as printed
(`"($2,250.00)"`, `"Sept. 18, 2026"`, `"Six (6) calendar days"`) with a verbatim evidence
quote and page. Tested parsers turn those strings into decimals, ISO dates and day counts,
and return nothing rather than guess. Gaps that are normalisation rather than reading are
filled by code and labelled as derived: currency from the `$` on the amounts, 0 days from
"no time extension", a subtotal printed under another label ("Net of items"), and one
unreadable line item from the total less the others.

**Validation checks that the numbers agree.** Eleven rules run on the typed result: contract
sums add up, line items + markups = change amount, quantity × unit cost = amount, credits are
negative, nobody signs before the issue date, and status matches the signatures ("approved"
needs every signature; "pending" needs at least one approver, otherwise it is a proposal). A
failed rule never changes a value; it lowers confidence on the fields involved.

**Confidence is measured, not asked for.** Each field combines five signals: the evidence
quote occurs in the source (35%), the value occurs in its quote (20%), it parsed and passed
the rules (20%), OCR quality of its page (15%), and competing values (10%). Caps apply on
top: evidence not found → at most 0.3; a failed rule → 0.6; an issue date with a competing
labelled date in the document → 0.65; a derived amount → 0.6. A document needs review if any
required field is below 0.7, any rule fails, or a value had to be inferred.

## Results

The headline set is the 8 documents of `co_test_set` (9 inputs: the email comes as PDF and
text), **labelled independently of this pipeline**: a clean G701, the real faxed scan, a
free-form letter, a net deduct, a table split across two pages, an email thread, a sparse
proposal and a revision that voids an earlier amount. One fresh live run, `gpt-4o-mini`:

| Metric | Result |
|---|---|
| Accuracy, 16 structured fields | **99%** (required fields 100%) |
| Invented values (a value where the answer is empty) | **0 of 27** |
| Line items / markups recovered | 26 of 26 / 7 of 7 |
| Documents containing an error that were flagged for review | **2 of 2** |
| Clean documents passed without review | 5 of 7 (the email, flagged for its competing dates) |
| Cost / LLM time per document | $0.0010 / 11 s |

| Confidence | Values | Accuracy |
|---|---|---|
| 0.8–1.0 | 150 | 97% |
| 0.5–0.8 | 4 | 75%, too few values to judge |
| 0.0–0.5 | 0 | — |

The 4 development documents I labelled myself give 97% on structured fields, 0 invented
values, and both documents with an error flagged.

**How free text is measured.** `description` and `reason_text` are paraphrased in the answer
files (the test set's README says so), so a string comparison scores the wording, not the
extraction. I moved them out of the headline after the first evaluation and report them as
similarity scores instead: 8 of 17 at or above 80, mean 77. They are reviewed, not scored.

**Reproducibility.** Temperature 0 is not deterministic: a repeated live run differed on 3 of
72 values. Every response is recorded in `cassettes/`, so all numbers above replay exactly,
without an API key. 251 unit and integration tests run without network.

**Latency.** Time is spent generating output: about 86 tokens/s, and each document's JSON is
600–2,200 tokens because every field carries a quote, page and alternatives. Retries used to
add ~6 s on 4 of 9 documents, all because the model gave a currency without a quote; currency
is now code's job and retries dropped to 0 of 13. Documents run one after another; running
them concurrently would cut wall-clock time for a batch.

## Failure modes

| Failure mode | Seen in | What catches it | Residual risk |
|---|---|---|---|
| **A wrong value that really is on the page** | Email thread: issue date read from the Oct 5 approval header instead of the CO's own Sept 28, at 1.00 confidence | Evidence matching cannot: it proves a value *exists* in the document, not that it is the *right* one. Code now lists the competing labelled dates ("Date:", "Sent:", "CO #9 dated …") and caps the issue date at 0.65, so the document is flagged | Wrong values with no visible competitor |
| **Self-consistent OCR error** | Real scan: project number `24-118` read as `24 18`, 0.97 | Contract math catches it for amounts | **Not caught for text fields:** the model quotes the misread text faithfully. The main open weakness |
| Stamp over values | Real scan: line item 1 and half the description under RECEIVED | Missing amount derived from subtotal − other items, labelled derived at 0.6 → flagged | Hidden free text is lost |
| Status contradicted by signatures | Letter: "pending" with no approver signed; scan: "approved" with the owner unsigned | `status_consistency` rule → 0.6, flagged | — |
| OCR punctuation in amounts | `$2,533. 183.56` | Parser refuses to guess → empty, capped 0.3, flagged | Contract math cannot run without it |
| Label variants | Subtotal printed as "Net of items" | Code finds the printed figure equal to the items' sum | Totals that do not reconcile |
| Prompt injection | Email: a line telling AI assistants to record the amount as $0.00 | Document wrapped as data, schema-constrained output. **Passed:** $13,750.00 in both formats | Not tested beyond one attempt |
| Superseded and proposed figures | Revision voiding $48,200; quoted earlier $14,900 | Prompt rule + alternatives | Two figures that both reconcile |
| Not tested | Multi-CO PDFs, non-USD, ambiguous `03/04` dates (read US-style) | — | Out of scope |

**Prompt rules versus code.** Each miss was fixed where it belongs, to avoid fitting the
prompt to 8 documents: normalisation (currency, "no extension" → 0, subtotal labels, an
unreadable item) is code with tests; reading judgments (the CO's own date in an email chain,
which parties signed, an email approval counts as signed) are three prompt rules, written with
generic examples rather than the test documents' wording. The status definition is a
validation rule, so a disagreement is flagged instead of silently passed.

## What I would do next

1. **Vision re-read when a rule fails on a scanned page.** The remaining text errors on the
   scan come from OCR and the stamp; the page image would likely recover them.
2. **Word-level OCR confidence for each evidence quote** instead of the page average, which
   might catch `24 18`.
3. **More labelled documents** before trusting calibration below 0.8, then per-field
   thresholds; and semantic similarity (an embedding or judge model) for free text.
