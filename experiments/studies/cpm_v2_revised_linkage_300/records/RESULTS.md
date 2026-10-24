# Revised CPM content-linkage evaluation

**10,794/10,800 attempted epochs recovered exactly**; 6 sender preflight failures were retained without retries or replacement covers. No incorrect accepted plaintext was observed. The study uses 300 distinct BOSSBase covers: 150 train, 50 validation and 100 test, drawn from the prepared clean partitions. Twelve fixed public parameter pairs and three application payloads are evaluated with the unchanged CPM-v2/fused-0.75 sender.

## Held-out primary results

| Payload | Selected classifier | Test covers | Test images | AUC | Familywise-95% AUC interval | Balanced accuracy | Top-1 / 12 |
|---:|---|---:|---:|---:|---:|---:|---:|
| 0.10 | extra_trees | 100 | 1200 | 0.4986 | [0.4963, 0.5007] | 50.00% | 8.33% |
| 0.20 | extra_trees | 99 | 1188 | 0.5012 | [0.4983, 0.5041] | 50.00% | 7.83% |
| 0.40 | extra_trees | 99 | 1188 | 0.4999 | [0.4967, 0.5031] | 50.00% | 8.42% |

The primary classifier family and score orientation were chosen using validation scores and frozen before test scoring. Each score averages three fixed model seeds. Binary AUC and balanced-accuracy chance references are 0.5 and 50%; random top-1 among twelve candidates is 8.33%. AUC is a ranking metric, not a cryptographic advantage bound.

![Primary linkage results](${ARTIFACT_ROOT}/revision_20261017/runs/cpm_v2_revised_linkage_300/primary_linkage.png)

AUC intervals in the primary table use Bonferroni adjustment over the three selected payload tests. Other diagnostic intervals and the top-1 figure use pointwise 95% cover-bootstrap intervals. The bootstrap jointly resamples the 100 attempted test covers across all their payloads and variants; each payload uses its available complete blocks. It quantifies test-cover sampling conditional on the fixed trained ensembles and twelve-pair bank, not uncertainty over arbitrary future models/parameter banks.

## Complete-block cohort and all-attempt accounting

| Payload | Partition | Attempts | Recovered | Preflight failures | Complete covers | Images in linkage test |
|---:|---|---:|---:|---:|---:|---:|
| 0.10 | train | 1800 | 1800 | 0 | 150 | 1800 |
| 0.10 | val | 600 | 600 | 0 | 50 | 600 |
| 0.10 | test | 1200 | 1200 | 0 | 100 | 1200 |
| 0.20 | train | 1800 | 1799 | 1 | 149 | 1788 |
| 0.20 | val | 600 | 600 | 0 | 50 | 600 |
| 0.20 | test | 1200 | 1199 | 1 | 99 | 1188 |
| 0.40 | train | 1800 | 1797 | 3 | 147 | 1764 |
| 0.40 | val | 600 | 600 | 0 | 50 | 600 |
| 0.40 | test | 1200 | 1199 | 1 | 99 | 1188 |

In addition to 6 failed attempts, 66 valid images belong to incomplete twelve-pair blocks and are excluded from the primary content-linkage cohort. This preserves candidate/label balance; it makes the linkage inference conditional on complete valid blocks. Every failure and excluded epoch remains in the generation audit/raw logs. No claim is made about leakage through delivery failures.

## All predefined feature views and classifier families

| Payload | View | Classifier | AUC | Pointwise 95% interval | Balanced accuracy | Top-1 |
|---:|---|---|---:|---:|---:|---:|
| 0.10 | image_only | logistic | 0.5000 | [0.5000, 0.5000] | 50.00% | 8.33% |
| 0.10 | image_only | extra_trees | 0.5000 | [0.5000, 0.5000] | 50.00% | 8.33% |
| 0.10 | text_only | logistic | 0.5000 | [0.5000, 0.5000] | 50.00% | 8.33% |
| 0.10 | text_only | extra_trees | 0.5000 | [0.5000, 0.5000] | 50.00% | 8.33% |
| 0.10 | cross_only | logistic | 0.5000 | [0.4998, 0.5003] | 49.97% | 8.25% |
| 0.10 | cross_only | extra_trees | 0.5003 | [0.4984, 0.5022] | 50.00% | 8.58% |
| 0.10 | text_image | logistic | 0.5001 | [0.4995, 0.5008] | 49.92% | 8.17% |
| 0.10 | text_image | extra_trees | 0.4986 | [0.4968, 0.5003] | 50.00% | 8.33% |
| 0.20 | image_only | logistic | 0.5000 | [0.5000, 0.5000] | 50.00% | 8.33% |
| 0.20 | image_only | extra_trees | 0.5000 | [0.5000, 0.5000] | 50.00% | 8.33% |
| 0.20 | text_only | logistic | 0.5000 | [0.5000, 0.5000] | 50.00% | 8.33% |
| 0.20 | text_only | extra_trees | 0.5000 | [0.5000, 0.5000] | 50.00% | 8.33% |
| 0.20 | cross_only | logistic | 0.5000 | [0.5000, 0.5000] | 50.00% | 8.33% |
| 0.20 | cross_only | extra_trees | 0.4991 | [0.4961, 0.5020] | 50.00% | 8.33% |
| 0.20 | text_image | logistic | 0.5000 | [0.5000, 0.5000] | 50.00% | 8.33% |
| 0.20 | text_image | extra_trees | 0.5012 | [0.4988, 0.5036] | 50.00% | 7.83% |
| 0.40 | image_only | logistic | 0.5000 | [0.5000, 0.5000] | 50.00% | 8.33% |
| 0.40 | image_only | extra_trees | 0.5000 | [0.5000, 0.5000] | 50.00% | 8.33% |
| 0.40 | text_only | logistic | 0.5000 | [0.5000, 0.5000] | 50.00% | 8.33% |
| 0.40 | text_only | extra_trees | 0.5000 | [0.5000, 0.5000] | 50.00% | 8.33% |
| 0.40 | cross_only | logistic | 0.5002 | [0.4999, 0.5007] | 50.15% | 8.25% |
| 0.40 | cross_only | extra_trees | 0.4995 | [0.4963, 0.5030] | 50.00% | 8.08% |
| 0.40 | text_image | logistic | 0.5000 | [0.4990, 0.5008] | 50.13% | 8.33% |
| 0.40 | text_image | extra_trees | 0.4999 | [0.4971, 0.5025] | 50.00% | 8.42% |

## Controls and interpretation

Image-only and parameter-only views are marginal-balance controls: each complete block contains every generating pair, and every image is scored with every candidate. Their chance performance is partly a property of this balanced design. It is not standalone evidence of covertness or secrecy. Cross-only and combined views test the defined content association using the fixed handcrafted feature families.

A separate synthetic control encodes the generating/candidate pair agreement and must be detected with AUC above 0.99. This checks functioning labels, splits, classifiers and scoring. It does not establish sensitivity to every subtle real-world leakage.

Matching visible session/epoch/header fields identifies the intended channel association. The header control verifies that association; protocol IDs and headers are excluded from content-classifier features. This study does not claim full-view unlinkability against an observer who can use those intentional association fields.

These results supply source-consistent revised content-linkage evidence for the relevant parts of Reviewer 5 comments 1–2. Near-chance results, if observed, describe only these finite classifiers and conditional views; a confidence interval containing chance does not prove absence of linkage or estimate epsilon_unl. Above-chance observations must be retained with their uncertainty. Modern steganalysis and published multichannel/generative-hybrid comparisons remain separate unfinished work.

## Pixel-change deviations and approved audit change

All 10,794 recovered images were checked against their original covers. 1 recovered epochs contained 1 pixels whose absolute change exceeded one level. The exact saved images were retained in content-linkage analysis under explicit author approval; no image was modified, re-embedded, retried or replaced. The full deviation records are in pixel_deviations.json and pixel_deviations.csv. These observations do not support an unqualified claim that this implementation always changes pixels by at most one level.

| Epoch | Cover | Partition | Payload | Maximum absolute change | Pixels exceeding one level |
|---:|---|---|---:|---:|---:|
| 1901 | 7718 | train | 0.40 | 2 | 1 |

## Operational interruption and preserved provenance

Generation stopped after 2,997 attempts because a native-library warning preceded the worker JSON response. The original harness misclassified that already-detected sender preflight failure. A versioned output-parser repair resumed only previously unattempted cases. The immutable original log preserves its initial status; the derived complete analysis view documents the corrected classification. No failed trial was retried, no cover or parameter was replaced, and all frozen protocol, sampling, feature, model-selection and test rules were retained. Recovery source hashes and exact diffs are recorded in recovery_v1/provenance.json.

At 9,376 attempts, epoch 1901 had already been authenticated and recovered exactly when a post-recovery assertion found a single pixel changing from 0 to 2. The run stopped before model training or held-out scoring. The author explicitly approved completing the remaining 1,424 cases with unchanged embedding code and recording pixel deviations separately. recovery_v2/provenance.json records this audit-criterion change and preserves both original raw manifests. Epoch 1901 was reconstructed from its already-saved artifacts, without invoking sender or receiver again; its original failure status and traceback remain. Exactly recovered images with such deviations are included consistently in the fixed complete-block analysis. No parameter, model, feature, sample or validation/test selection rule was changed, and no used selection was rerun.

## Reproducibility and independent checks

The original protocol, image worker and legacy embedder were copied unchanged from the verified pilot. Frozen source hashes, cover/pair/message manifests, source settings, classifiers, seeds and inference rules are recorded in `run_config.json` and `source/README.md`. The immutable `generation_results.jsonl` and `generation_continuation_v1.jsonl`, new `generation_continuation_v2.jsonl`, derived `generation_results_complete_v2.jsonl` and `epochs/` preserve all attempts; `generation_audit.json` records failures and incomplete blocks. `feature_dataset.npz`, `feature_metadata.csv.gz` and `feature_schema.json` bind exactly the classifier inputs. Paths, IDs, keys, headers/nonces, payload labels and generating-pair labels are never input features.

`evaluation/validation_models.csv` and `primary_selection.json` document selection before test scoring. `evaluation/test_metrics.csv`, `primary_metrics.csv`, seed metrics, prediction arrays and model checkpoints preserve all panels. An independent stage reproduced every held-out score from all 72 saved checkpoints and checked AUC/balanced accuracy using sklearn and ranking via a separate calculation. The cover-weighted AUC bootstrap is also checked against sklearn weighted AUC. `verified_summary.json` records the completed checks. No manuscript was edited.
