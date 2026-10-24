# Timed-window active-attack results

All **110 recorded traces** passed the receiver/evaluator checks. The batch reused **30 authenticated real-image epochs from ten development covers** at 0.10/0.20/0.40 bpp. Two window/timeout profiles and five seeds produce repeated, correlated traces.

Across 3,300 offered epoch events, **2,920** were recovered exactly and **380** were erased. There were **zero unauthorised transcript acceptances, zero incorrect plaintexts and zero repeated epoch acceptances**. The erasures arise from the explicit burst disruption and persistent suppression scenarios; the aggregate delivery rate should not be treated as a network-population performance estimate.

| Scenario | W / timeout | Traces | Exact recovery / offered | Erasures | Full MAC rejects |
|---|---|---:|---:|---:|---:|
| honest | W3_T200 | 5 | 150/150 | 0 | 0 |
| honest | W5_T400 | 5 | 150/150 | 0 | 0 |
| bounded reordering | W3_T200 | 5 | 150/150 | 0 | 0 |
| bounded reordering | W5_T400 | 5 | 150/150 | 0 | 0 |
| burst delay | W3_T200 | 5 | 130/150 | 20 | 0 |
| burst delay | W5_T400 | 5 | 130/150 | 20 | 0 |
| burst suppression | W3_T200 | 5 | 130/150 | 20 | 0 |
| burst suppression | W5_T400 | 5 | 130/150 | 20 | 0 |
| persistent C2 suppression | W3_T200 | 5 | 0/150 | 150 | 0 |
| persistent C2 suppression | W5_T400 | 5 | 0/150 | 150 | 0 |
| accepted replay | W3_T200 | 5 | 150/150 | 0 | 0 |
| accepted replay | W5_T400 | 5 | 150/150 | 0 | 0 |
| tamper and splice | W3_T200 | 5 | 150/150 | 0 | 80 |
| tamper and splice | W5_T400 | 5 | 150/150 | 0 | 80 |
| bounded injection flood | W3_T200 | 5 | 150/150 | 0 | 3740 |
| bounded injection flood | W5_T400 | 5 | 150/150 | 0 | 3740 |
| adaptive injection | W3_T200 | 5 | 150/150 | 0 | 182 |
| adaptive injection | W5_T400 | 5 | 150/150 | 0 | 182 |
| malformed frames | W3_T200 | 5 | 150/150 | 0 | 0 |
| malformed frames | W5_T400 | 5 | 150/150 | 0 | 0 |
| restart then replay | W3_T200 | 5 | 150/150 | 0 | 0 |
| restart then replay | W5_T400 | 5 | 150/150 | 0 | 0 |

Both profiles have the same per-scenario delivery counts. The delayed C2 frames violate the 100 ms cross-channel skew limit even under the longer 400 ms timeout; a larger window or timeout alone does not make those triples eligible. This is the declared policy behaviour, not evidence that the profiles have equal performance on other traffic.

## Recovery, rejection and resources

All 20 burst-disruption traces recovered every one of the 18 clean-tail epochs (360/360). The first clean-tail epoch completed 29–35 ms after its scheduled send. This is a finite observation under the declared clock/skip policy, not a heartbeat resynchronisation bound.

The receiver made 10,924 complete authentication attempts and rejected 8,004 incorrect complete MACs. It invoked native image extraction 2,920 times, exactly once for each accepted epoch. Wrong-MAC attempts never decoded or changed replay state. Front-end, freshness, skew and timeout decisions are separate counters in the saved summary.

Peak retained serialized-frame storage was 1,401,055 bytes, and peak candidate count was 9. These remained within the configured caps. They exclude Python object overhead, transient input parsing, the evaluator and protected replay state. Maximum protected-state file size was 4,426 bytes; freshness history grows with accepted epochs.

The state-reload scenario restored the saved accepted-epoch and nonce sets in the same receiver process, cleared transient buffers, then rejected recorded accepted traffic. This checks state restoration; it is not a process-kill/restart or power-loss test and does not establish resilience to storage rollback, concurrent receivers or endpoint key compromise.

## What this contributes to the revision

- Reviewer 5, comment 3: adds concrete timed-receiver simulations of delay, out-of-order epochs, correlated suppression, replay, mutation/splicing, bounded flooding, feedback-driven injection and malformed traffic. It addresses these tested active behaviours; it does not cover every MC-ATTACK adversary or message-recovery attack.
- Reviewer 4, comment 4: provides complete new simulator settings, seed list, fixture/source hashes and event logs. It does not complete the separate detector-training/baseline details.
- Empirical/theory connection: the checks support tested authentication, correctness, freshness and availability behaviour. Counts and zero observed incorrect accepts do not estimate epsilon_MAC, epsilon_mask, epsilon_emb, epsilon_unl or epsilon_sched.

## Scope and remaining experiments

The simulator assumes a known shared epoch origin. The two profiles change W and timeout together, so their effects cannot be isolated. Virtual network arrival latency excludes CPU service and queueing. Flooding has a fixed finite budget and precedes the target legitimate frames; successful delivery in that scenario is conditional on those facts. Complete one-channel suppression prevents delivery, as expected. No heartbeat, ECC, real traffic scheduler, image recompression/noise recovery, key-compromise response or imperfect entropy evaluation is claimed.

The next evidence gap is source-consistent revised steganalysis and cross-channel linkage evaluation with the clean fixed split. Matched comparisons against suitable published multichannel/generative-hybrid methods remain required. This batch alone does not close all reviewer criticisms or establish submission readiness.

## Reproduction and audit

`source/README.md` states the exact receiver/window/injection policy. `run_config.json` freezes the design and source hashes. `fixture_manifest.json` binds the real-image inputs. `trace_results.jsonl` and `traces/*/events.jsonl` preserve every completed trace and decision. `verified_summary.json` and `scenario_summary.csv` give descriptive aggregates. The independent post-run check verified every event log, saved state, plaintext hash, transcript identity, source hash, counter relationship and buffer limit without rerunning the batch. No original manuscript or prior experiment output was changed.
