# Full manuscript and supplementary consistency audit

Completed 24 October 2026. The working manuscript and supplementary sources were reviewed throughout, including all model/protocol interfaces, security games, claims and proofs, experimental sections, applications, abstract, conclusion, captions, tables, configuration records and cross-references. The resulting PDFs have 68 and 40 pages. This is a manual mathematical and scientific-consistency review supported by saved-record and document checks; it is not machine-checked proof verification or a new external-literature/retraction audit.

## Outcome

The document checks pass after the corrections below. The manuscript now consistently distinguishes conditional security reductions from finite empirical observations, and image-carrier detectability from full-public-view CP-UNL and authenticated acceptance. No new experiment, embedding, training, selection, inference or simulator execution was performed. Experimental artifacts and synced project sources were not edited.

## Material corrections

1. **Publication language.** Removed revision-stage narrative such as “revised”, “historical” and “legacy” from manuscript/supplement prose, titles and captions. Replaced it with functional configuration names and exact source identifiers. Internal cross-reference labels and immutable run identifiers remain unchanged.
2. **Abstract and conclusion.** Rewrote the abstract to distinguish recovery, linkage and detection and include unfavourable evidence. It now reports the fusion failure, HSMNet pointwise intervals excluding chance and strong component-detector results at 0.40 bpp. Narrowed an overly general conclusion about secure deployment. Relative HSMNet advantages, unresolved contrasts and mixed fidelity/cost outcomes remain explicit.
3. **Timed receiver scope.** Removed the obsolete assertion that the active evaluation covers only assembled epochs. The protocol section now distinguishes the complete-image worker from the tested bounded timed-receiver adapter and controlled virtual arrivals. Heartbeat/ECC and deployed-network performance are not inferred.
4. **Cost analysis.** Added parameter-sampling cost to sender complexity. The aggregate receiver bound counts every authenticated candidate reaching extraction, including subsequent length/commit failures. Supplementary runtime equations now also include sender preflight and protected receiver-state work. Persistent replay-state storage is included in the supplementary memory expression.
5. **Ternary interface versus observed output.** Replaced the unconditional one-level native pixel-change assertion with an explicitly scoped directional-cost interface. The disclosed two-level recovered-image deviation and main-panel HILL deviation remain retained. Geometric-mean normalisation is described correctly before clipping, which can change the final mean.
6. **Naturalness interpretation.** RoBERTa AUCs of 0.5543–0.6207 are described as residual separability, alongside high MAUVE and small POS effects. They are no longer collectively described as near-chance evidence. No universal sampler upper bound is inferred.
7. **HILL recovery failures.** The four known-message HILL failures returned mismatching native payload bytes. The supplementary prose and column explanation now state this explicitly. The single-carrier baseline has no CPM authenticated-acceptance interface; the absence of incorrect CPM accepts must not be transferred to that baseline. All numerical failure counts are preserved.
8. **Scheduling denominator.** The saved scheduling summary counted 30,000 seed/epoch pairs over one overlapping simulated time span, producing approximately 3 epochs/s. Recalculation within each of the three 10,000-epoch seeds gives approximately **1 epoch/s** for cover, shaped and unshaped conditions. The table now reports per-run emission rate, not authenticated receiver throughput. Raw timing records and analysis sources are unchanged.
9. **Duplicate transport records.** The loss/jitter summary contains **26,191 rows but 25,920 unique parameter/seed scenarios**. The 271 repeats have identical simulated outcomes. Supplementary Tables XIX and XX now count each scenario once, with recomputed descriptive means/sample SDs. All raw rows are preserved. The 480 representative figure records contain no repeats; the plotted data are unchanged. Its intervals are described as summaries over the selected settings/seeds, rather than population reliability bounds.
10. **Runtime denominators and schema.** Independently streamed the 153-column fused-v3.2 partition: 30,000 attempts, 10,000 per principal payload. Its runtime and memory means/SDs match the retained table. Extraction times have **9,999 / 9,996 / 9,996** finite observations; the table note now discloses those denominators. The appended 178-column 0.30-bpp partition is excluded. Binary-layout overhead is explicitly an analytical budget, distinct from measured JSON/base64 wire cost.
11. **Exact configuration attribution.** Explicitly identified the candidate-ranking encoder as `ours_qim_embed_SAFE_v1.py`, with S-UNIWARD-like/local-variance costs, distinct from the fused-v3.2 CNN cohort and `cpm-epoch-v2` linkage. Clarified the local MiPOD approximation and fixed fusion weight; no optimality or causal detector benefit is inferred.
12. **Application and presentation consistency.** Removed the blockchain application's alternative acceptance path that appeared to waive the fixed skew condition. Re-expressed its conditional correctness identity to fit the column. Expanded first-use terminology, replaced the running-title placeholder, repaired long bibliography-URL wrapping and separated the transport-table note from its rule.

## Theory-to-evidence review

| Area | Checked interpretation |
|---|---|
| Mask reconstruction and correctness | Received authenticated fields and secret keys reproduce the mask; source-image features remain sender-only. Correctness is conditional on carrier inversion and usable delivery. Failed preflights stay in attempted denominators. |
| CP-IND | Sampler PRF replacement and ideal-sampler mismatch remain distinct from masking. Message independence and public parameter entropy do not establish benign sampling or secrecy. Corpus probes cover selected observers and sources only. |
| Composition | Full-history key, mask, linkage/tag, sampler, carrier and scheduling transitions remain explicit conditional premises. Marginal tests do not prove the joint transitions. Per-query losses require aggregation. |
| CP-UNL | The wrong-candidate diagnostic differs from independent-content sampling. The full-view experiment retains diagonal combinations, matched metadata, valid native tags and four-query history at h=0. Neither estimates the H2 transition loss after key/mask replacement. |
| Message confidentiality | Prior guessing, message/key independence, fresh masking contexts and permitted feedback remain explicit. Public parameters and the 31-bit coding seed are not credited as secret-key entropy. |
| Replay/MitM | Freshness rejects accepted-epoch replays; complete-transcript MAC security addresses unauthorised new transcripts. Decoder error and state failure remain separate. The maximum of task advantages is not an attack-union probability. |
| Re-synchronisation | The opportunity bound uses a conditional success lower bound and an actual opportunity count. Heartbeat cadence, window size and timeout alone do not establish delivery. |
| Image detection | Fixed-observer AUC is distinct from a fixed-rule acceptance gap and optimal error. The population inequality for a fixed detector has the lower-bound direction for TV; empirical scores do not instantiate a cryptographic upper bound. |
| Fidelity and sensitivity | Valid-image fidelity and all-attempt recovery have distinct denominators. High PSNR/SSIM and the fusion sensitivity panel do not imply lower detectability or an optimal weight. |
| Native comparisons | Application bytes are matched while native benign sources, formats, services and object sizes remain different. Undefined baseline CP-UNL or protocol-wire metrics are not zero scores. |
| Active traces | Reused images/traces are correlated. Virtual arrivals, bounded flooding and same-process state reload do not qualify real networks, crashes, rollback or concurrent receivers. |
| Applications | Banking, telemetry and ledger examples are proposed interfaces, not measured deployments. Blind decoding, semantic constraints, exact authentication, additional commitments and finality need their own premises. |

## Verification

- No unresolved manuscript/supplement citations, cross-references or duplicate-label warnings.
- All **235** remote PDF destinations resolve across the manuscript, supplement and existing response bundle.
- All main-table numerical bodies are preserved.
- All **14 HSMNet numerical tables** are preserved. Fifteen AUC/pointwise-interval rows, twelve adjusted contrast rows and matched fidelity values agree with the saved independently checked CSVs.
- Only three supplementary numerical tables change: scheduling emission rate and the two deduplicated synthetic transport summaries.
- All manuscript/supplement figure paths and figure inputs are preserved; representative transport figure data are unaffected by deduplication. The requested HSMNet subsection retains one explanatory paragraph and one figure.
- Completed-study receipt/report hashes match the preceding audit. The bibliography and combined reviewer-response PDF are unchanged.
- All 108 pages were rendered and reviewed in page contact sheets; detailed pages included the protocol diagram, complexity, HSMNet figure, transport summaries, HILL mismatch accounting and supplementary bibliography. No clipped text or overlapping table notes remain. A small 0.66039-point supplementary bibliography line-width warning remains without visible clipping.

`FINAL_QA.json` records checks, hashes and links. `before/` preserves the incoming sources/PDFs. Exact source diffs and the saved-accounting, runtime-partition and transport-deduplication receipts are retained in this directory. Original experiment logs, models, images, predictions and source snapshots remain unchanged.

## Limits that remain disclosed

These are evidence boundaries, not silently repaired claims: the original SLIC launch/package state and original STC executed-binary hash were not fully saved; native baselines do not implement an identical four-metric full-public-view interface; the detector intervals condition on the saved models and cohorts; public-view studies cover finite banks/feature families/query histories; real benign traffic, byte-changing carrier transformations and crash/rollback/concurrent-state behaviour are not deployment-qualified. The matched SRM continuation remains paused.

The completed audit supports consistency of the present bounded claims. It does not certify universal security, negligible instantiated losses, population-perfect recovery, or editorial acceptance.
