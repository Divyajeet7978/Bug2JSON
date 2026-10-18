# Bug report labeling guide

Use these rules before marking a row `reviewed: true`. Labels describe only the
report, not assumptions about an unknown production system. Keep the source and
reviewer notes for every real issue.

## Output fields

Each `output` object must contain exactly:

`category`, `severity`, `priority`, `environment`, `reproducible`,
`probable_cause`, and `suggested_action`.

- **category:** choose one of the schema values. Use `unknown` if the report does
  not identify a subsystem; use `other` only when the area is clear but not listed.
- **severity:** rate impact described in the report, not presumed business impact.
  Critical means broad outage, data loss, or an exploitable security issue. High
  means an important workflow is blocked. Medium means functionality is impaired
  but scope is limited or a workaround exists. Low is minor or cosmetic. Use
  `unknown` when impact cannot be judged.
- **priority:** P1 is immediate, P2 urgent, P3 normal, P4 low. Apply severity and
  stated scope together. Do not mechanically equate every `high` with P1. Use
  `unknown` if available information cannot support prioritization.
- **environment:** include only named browsers, operating systems, devices, or
  runtimes. Otherwise use `unknown`.
- **reproducible:** `true` only if repeated behavior is stated or clearly implied;
  `false` only if the report says it cannot be reproduced; otherwise `"unknown"`.
- **probable_cause:** a short, explicitly tentative hypothesis supported by the
  report. Prefer `"unknown"` to a detailed guess.
- **suggested_action:** one concise investigation step. Avoid prescribing a patch
  before the cause is verified.

## Dataset hygiene

- Mark an example `reviewed: true` only after a human checked all seven labels.
- Record a stable `group_id` for the originating issue, incident, or synthetic
  scenario. Keep all paraphrases from the same group in one data split.
- Keep an immutable test set. Do not tune prompts or labels against test examples.
- Prepare splits from at least 30 reviewed examples across at least three distinct
  `group_id` values so train, validation, and test are all non-empty.
- Do not include secrets, personal data, or private customer reports.
- The bundled JSONL rows are illustrative starter examples and are deliberately
  marked unreviewed. They are excluded from training and evaluation.
