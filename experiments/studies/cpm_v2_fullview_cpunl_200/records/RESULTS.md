# CP-UNL full-public-view extension

All 9,600 native counterfactual/history attempts are accounted for: 9,572 exact recoveries and 28 retained preflight failures. No incorrect accepted plaintext was observed. No attempted case was retried or replaced.

The real-world h=0 comparison uses 200 new BOSSBase covers (100 train / 40 validation / 60 test), twelve fixed equal-length parameter pairs, fresh keys/sessions per cover/payload, and four adaptive chosen-message queries before a known-zero challenge. All challenge contexts and visible lengths are matched between worlds. Both worlds carry valid native HMAC tags. The independent world includes all 144 pair/carrier combinations, including its 1/12 diagonal overlap.

## Primary held-out full-history results

| Payload | Family | Test covers | AUC | Adjusted AUC interval | Balanced accuracy | Signed gap | Absolute gap |
|---:|---|---:|---:|---:|---:|---:|---:|
| 0.10 | logistic | 60 | 0.5023 | [0.4824, 0.5210] | 51.43% | 0.0286 | 0.0286 |
| 0.20 | extra_trees | 60 | 0.5119 | [0.4926, 0.5311] | 50.00% | 0.0000 | 0.0000 |
| 0.40 | extra_trees | 59 | 0.4905 | [0.4715, 0.5096] | 50.00% | 0.0000 | 0.0000 |

AUC chance is 0.5; balanced-accuracy chance is 50%. A fixed-rule gap is its observed TPR minus FPR (or absolute value), not a universal epsilon_unl bound. Selection/orientation were frozen using validation only before any held-out scoring. Primary AUC intervals use 3,000 test-cover cluster bootstrap draws and Bonferroni adjustment over three payloads; other intervals are pointwise 95%. They condition on the fixed models, bank and query policy.

![Full-view CP-UNL results](${ARTIFACT_ROOT}/revision_20261017/runs/cpm_v2_fullview_cpunl_200/fullview_cpunl.png)

## Complete-block accounting

| Payload | Partition | Planned blocks | Complete | Excluded | Linked views | Independent views |
|---:|---|---:|---:|---:|---:|---:|
| 0.10 | train | 100 | 100 | 0 | 1200 | 14400 |
| 0.10 | val | 40 | 39 | 1 | 468 | 5616 |
| 0.10 | test | 60 | 60 | 0 | 720 | 8640 |
| 0.20 | train | 100 | 100 | 0 | 1200 | 14400 |
| 0.20 | val | 40 | 40 | 0 | 480 | 5760 |
| 0.20 | test | 60 | 60 | 0 | 720 | 8640 |
| 0.40 | train | 100 | 98 | 2 | 1176 | 14112 |
| 0.40 | val | 40 | 40 | 0 | 480 | 5760 |
| 0.40 | test | 60 | 59 | 1 | 708 | 8496 |

There are 596 complete blocks and 92,976 full public views. 36 otherwise valid native outputs belong to incomplete blocks; all remain in raw accounting. Both worlds exclude incomplete blocks symmetrically. Inference is conditional on complete valid blocks; failure/availability linkage is not estimated.

## Every predefined view and family

| Payload | Observer features | Family | AUC | Pointwise interval | Balanced accuracy |
|---:|---|---|---:|---:|---:|
| 0.10 | metadata_only | logistic | 0.5000 | [0.5000, 0.5000] | 50.00% |
| 0.10 | metadata_only | extra_trees | 0.5000 | [0.5000, 0.5000] | 50.00% |
| 0.10 | content_only | logistic | 0.4979 | [0.4959, 0.4996] | 50.06% |
| 0.10 | content_only | extra_trees | 0.4986 | [0.4949, 0.5025] | 50.00% |
| 0.10 | authentication_only | logistic | 0.5010 | [0.4846, 0.5177] | 50.45% |
| 0.10 | authentication_only | extra_trees | 0.5143 | [0.4942, 0.5341] | 50.00% |
| 0.10 | full_challenge | logistic | 0.5012 | [0.4850, 0.5181] | 50.31% |
| 0.10 | full_challenge | extra_trees | 0.5172 | [0.4992, 0.5353] | 50.00% |
| 0.10 | full_history | logistic | 0.5023 | [0.4864, 0.5184] | 51.43% |
| 0.10 | full_history | extra_trees | 0.5124 | [0.4965, 0.5283] | 50.00% |
| 0.20 | metadata_only | logistic | 0.5000 | [0.5000, 0.5000] | 50.00% |
| 0.20 | metadata_only | extra_trees | 0.5000 | [0.5000, 0.5000] | 50.00% |
| 0.20 | content_only | logistic | 0.4987 | [0.4951, 0.5015] | 50.15% |
| 0.20 | content_only | extra_trees | 0.5013 | [0.4968, 0.5060] | 50.00% |
| 0.20 | authentication_only | logistic | 0.5034 | [0.4832, 0.5238] | 51.38% |
| 0.20 | authentication_only | extra_trees | 0.5255 | [0.5080, 0.5427] | 50.00% |
| 0.20 | full_challenge | logistic | 0.4986 | [0.4842, 0.5137] | 49.30% |
| 0.20 | full_challenge | extra_trees | 0.4941 | [0.4734, 0.5163] | 50.00% |
| 0.20 | full_history | logistic | 0.5067 | [0.4863, 0.5270] | 51.26% |
| 0.20 | full_history | extra_trees | 0.5119 | [0.4957, 0.5280] | 50.00% |
| 0.40 | metadata_only | logistic | 0.5000 | [0.5000, 0.5000] | 50.00% |
| 0.40 | metadata_only | extra_trees | 0.5000 | [0.5000, 0.5000] | 50.00% |
| 0.40 | content_only | logistic | 0.4997 | [0.4942, 0.5047] | 49.44% |
| 0.40 | content_only | extra_trees | 0.4956 | [0.4898, 0.5010] | 50.00% |
| 0.40 | authentication_only | logistic | 0.4933 | [0.4776, 0.5076] | 48.33% |
| 0.40 | authentication_only | extra_trees | 0.4999 | [0.4818, 0.5185] | 50.00% |
| 0.40 | full_challenge | logistic | 0.4988 | [0.4877, 0.5085] | 50.09% |
| 0.40 | full_challenge | extra_trees | 0.5082 | [0.4892, 0.5280] | 50.00% |
| 0.40 | full_history | logistic | 0.5018 | [0.4870, 0.5177] | 50.29% |
| 0.40 | full_history | extra_trees | 0.4905 | [0.4751, 0.5057] | 50.00% |

## What the observer receives

The full public record contains every challenge frame byte, public session/epoch/header/nonces, lengths/codec/version, actual HMAC tags, and four earlier public query messages and complete responses. The classifier inputs are fixed finite maps of those bytes, including ordered history features and cross-history comparisons. Keys, sender masks/seeds/cost maps, original covers, hidden world/pair labels and file identifiers never enter feature inputs. Full raw bytes can be reconstructed with source/export_view.py using the saved view manifest and native artifacts.

This implements the manuscript's matched-metadata linked versus independent-content comparison for the declared real-world finite source. Counterfactual challenge branches share context because they are alternative worlds; they are never emitted together as nonce-reusing live traffic. Independent-content views are laboratory-challenger hybrids, retagged consistently, not adversarial forgeries or claimed correct encodings of the known challenge message.

## Controls and retained anomalies

Metadata-only chance scores follow from exact metadata balance and are not an independent secrecy result. A separate deliberately implanted public digest-to-image coupling is processed by the same public feature maps and classifiers. Its valid tags and diagonal overlap are retained; its theoretical maximum AUC is 23/24, not 1. This control checks sensitivity to the implanted signal, not every possible leak.

Synthetic known-leak logistic AUC: 0.9583.
Synthetic known-leak extra_trees AUC: 0.9583.

Recovered native cases with pixel changes beyond one level: 0; total such pixels: 0. Exact cases and change histograms remain in pixel_deviations.json. These deviations are retained, not repaired or suppressed.

## Interpretation and reproducibility

Report all observed scores, including above-chance or unfavourable panels. Near-chance results support only the tested full-public-view features, fresh-key source, finite bank and four-query policy; they do not prove security against every PPT warden or estimate the compositional epsilon_unl. No H2 uniform-mask hybrid premise is experimentally certified. Public headers intentionally associate frames. Network scheduling/timing and active decryption queries are outside this CP-UNL experiment. Fixed-length parameter prefixes define a controlled conditional length stratum, not a naturalness or unseen-bank result.

The completed earlier content-only study remains unchanged and uses different covers/parameter lengths; do not pool its results. source/DESIGN.md and run_config.json freeze the design/source/model choices; generation_results.jsonl preserves every native attempt; public_views.jsonl binds reconstructible complete views and their hashes; world_audit.json records world marginals/failures; feature_schema.json binds feature arrays; evaluation/ retains all 90 checkpoints, validation selection, predictions, seed metrics and 30 panels. Independent verification reproduced every held-out prediction and verified every world tag, wire/observer hash, conditional marginal and source split. No manuscript, synced source, old experiment or deferred CNN description was edited.
