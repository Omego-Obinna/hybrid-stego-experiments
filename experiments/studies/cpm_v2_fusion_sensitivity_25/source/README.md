# Paired validation-only fusion-weight sensitivity

This phase addresses Reviewer 4's request to explain the choice lambda=0.75.
It measures sensitivity of recovery, fidelity, pixel changes and computation;
it does not select a weight for detector resistance or claim an optimal weight.
No manuscript or old experiment source/output is changed.

## Design fixed before outcomes

Twenty-five covers are selected from the already frozen 1,000-cover validation
partition using SHA256 ordering of `cpm-fusion-validation-20261017:ID`.
Duplicate-content covers are excluded by the prepared manifest. No final test
cover or pilot-training cover participates. Three net application payloads
(0.10/0.20/0.40 bpp) and five weights (0/0.25/0.50/0.75/1) give 375 attempts.
This is a small validation sensitivity panel, not a population benchmark.

For each cover/payload, all five weights share the exact plaintext, public
parameter objects, session, epoch, nonces, masked payload and STC seed. The
weights are counterfactual variants with separate receiver states, not five
epochs intended for one live session. Other QIM/CF/SLIC/STC arguments remain
unchanged. Weight changes propagate through the existing fused residual
feature map and its cost modulation; this is sensitivity of the executable
pipeline, not an artificial interpolation holding those derived maps fixed.

The endpoints retain CPM's modulation/complex features and the fusion code
path. They are internal pipeline variants, not published standalone MiPOD or
S-UNIWARD baselines. Existing public parameter streams are reused, with one
pair per selected cover; this phase does not test parameter naturalness.

Weight order is a seeded permutation rotated by cover/payload index, balanced
within every payload: each weight appears in each position on five covers.
Case order is shuffled with a separate fixed seed. Three concurrent cases use
one CPU thread per sender/receiver process. Sender/receiver wall measurements
include process startup and share workstation contention; CPU time for cost
construction plus embedding is recorded separately. Results support descriptive
cost sensitivity, not an isolated competitive speed benchmark. Endpoints still
compute both constituent costs, as the original fused path does.

## Source and correctness

The frozen pilot protocol, original image worker and legacy embedder are copied
unchanged. A separate `image_worker_weighted.py` adds only a bounded weight
argument and CPU timing; `source_adaptation.diff` records its complete change.
Before the batch, all five weights pass independent native receiver extraction
on one prior training-only fixture. At 0.75 the new worker reproduces the old
fixture's exact stego bytes and authenticated transcript hash. Development
outputs remain separate from this recorded validation batch.

Each attempt runs a fresh sender and independent original receiver. The receiver
receives the authenticated frames, session/key and its state; no original cover,
sender mask or expected plaintext is supplied. The evaluator then compares all
plaintext bytes, records BER, and checks paired context/masked-payload equality.
All trials, including failures, are retained. No retries, replacement covers or
proxy embedder are allowed. Accepted wrong plaintext or repeated operational
failures stop further work for investigation.

## Analysis

Report all 25 attempts per weight/payload and recovery failures explicitly.
For the common valid paired covers, report PSNR, SSIM, modified-pixel fraction,
cost/embedding CPU and wall time, and actual serialized frame bytes. Differences
against 0.75 use cover-paired summaries. Fixed-seed paired bootstrap intervals
are descriptive validation-sample uncertainty; label them unadjusted and avoid
choosing a winner from multiple comparisons. No causal runtime claim follows
from a single timing observation per variant on a shared workstation.

Higher PSNR or fewer changes alone does not establish lower detectability.
The result may support retaining 0.75 as a fixed design choice or expose a
trade-off; it cannot establish that 0.75 is security-optimal. Final test data
and manuscript decisions remain reserved for the later detector benchmark.

## Artifacts

`run_config.json`, `cover_manifest.csv` and `case_manifest.json` freeze sources,
selection, execution order and pairing. `trial_results.jsonl` records every
attempt. `trials/case_*/lambda_*` retains frames, metrics, stdout/stderr, receiver
state and recovered plaintext. `inputs/` holds shared case fixtures. Research
keys stay in the private local `secrets/` directory and are never dashboard data.
The dashboard reads only the aggregate status file.
