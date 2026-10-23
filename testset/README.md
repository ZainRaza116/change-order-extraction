# Change-order test set

Each sample in `samples/` has a ground-truth file in `expected/` with the same name.
All money math in every expected file has been checked (qty x unit cost, line items -> subtotal -> total, contract sums).

| File | Format | What it tests | Traps |
|---|---|---|---|
| s01_clean_g701.pdf | digital, 1 page | happy path | old PCO amount $11,240 in a note; credit in parentheses; 3 date formats; owner unsigned -> `pending` |
| s02_scanned_messy.pdf | image only, 1 page | OCR path | same as s01 plus stamp over line item 1, handwriting, coffee stain, skew, fax header |
| s03_letter_format.pdf | digital letter | no form labels | amount written in words and digits; "ten (10) working days" (working, not calendar); no line items; owner unsigned -> `proposed` |
| s04_deduct_credit.pdf | digital, 1 page | negative change | net DEDUCT (-25,006.40); fee applied to ADD items only; "decreased" wording; dates like "14 Aug 2026"; time "Unchanged (0 days)" |
| s05_multipage_table.pdf | digital, 2 pages | table split across pages | 12 line items, 6 per page; header row repeated on page 2 (must not become a line item); decimal qty 3.2 TON; insurance on subtotal + OH&P |
| s06_email_thread.pdf / .txt | printed email | free text, revisions | approved amount $13,750 vs earlier $14,900 in quoted reply; CO date (9/28) differs from approval date (10/5); no project number; **prompt-injection line** asking to set amount to $0 |
| s07_missing_fields.pdf | sparse form | null handling | owner, architect, time, reason, contract sums all blank -> must be `null`, not guessed; title says PROPOSED |
| s08_conflicting_amounts.pdf | digital, 1 page | superseded values | Rev 1 voids original $48,200; reference line repeats old quantities; correct amount $45,950 |

Notes on judgment calls in the ground truth:
- `change_order_number` keeps the document's own form ("007", "CO-022", "CO-011 Rev 1"). Normalize in your comparison if your schema strips prefixes.
- s03 `schedule_impact_days` is 10 working days; you may want a `schedule_unit` field.
- s06 email approval is recorded as a signature with `"method": "email approval"`. If your "approved requires signature" rule only accepts wet/e-signatures, this file should fail that rule; decide which behavior you want and state it in the writeup.
- Description fields are paraphrased; compare them with fuzzy/semantic similarity, not exact match.
