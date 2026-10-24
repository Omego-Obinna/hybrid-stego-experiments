# Phase 4: bounded revised content-linkage evaluation

This is the author's authorised 5–6-hour linkage work package. It regenerates
the revised CPM-v2/fused-0.75 construction, not the older v1 linkage carriers.
The finite pipeline prepares inputs, generates/reconstructs actual epochs,
audits artifacts, extracts fixed content features, fits classifiers, freezes
validation selection, scores held-out covers, calculates uncertainty, independently
reproduces predictions from checkpoints, and saves the report/figures.

## Fixed design before outcomes

- 300 distinct native BOSSBase covers: 150 train, 50 validation, 100 test,
  selected by a fixed public SHA256 ordering within the prepared clean partitions.
  Every derivative of a cover stays in one partition. Byte-identical covers were
  removed by the earlier full-data audit. Prior pilot and fusion-panel covers are
  excluded. The manifest, cover hashes and selection label are frozen.
- Twelve existing public parameter pairs are copied byte for byte from the
  historical linkage bank and hash-checked. Each object is the complete selected
  text stream, not one line. This retains the old content-probe task while changing
  the executable construction. Results are conditional on these twelve pairs;
  generalisation to unseen parameter banks is not claimed.
- Three application payloads, 0.10/0.20/0.40 bpp: 10,800 attempted epochs.
  Each cover/payload shares one fresh random plaintext across its twelve variants;
  every variant has a unique authenticated epoch/context, nonces and STC seed.
  One fresh local research fixture master/session is used; no production secret.
- Original protocol, image worker and legacy embedder are copied unchanged from
  the verified pilot. All sender settings, including lambda=0.75, remain fixed.
  Three CPU cover jobs run concurrently with one thread per sender/receiver.
  The original independent receiver receives frames, key/session and its state;
  it gets no source cover, sender mask, ground truth, pair ID or attack label.
- Every attempted epoch, preflight failure, recovered plaintext comparison,
  image/transcript hash, fidelity value, byte count and wall time is retained.
  A recognised native-STC preflight failure emits no frames and is not retried or
  replaced. Unexpected operational errors or wrong accepted plaintext stop the
  pipeline. A standalone timeout prevents an indefinitely hung stage.

## Binary and ranking tasks

For each generated image, its true pair is the positive candidate and all eleven
other public pairs are negative candidates. The image, cover and payload stay
fixed across candidates. One-positive/eleven-negative raw tuples are not called
a balanced dataset. Fitting uses balanced class weights; the primary accuracy is
balanced accuracy at threshold 0.5, with chance reference 50%. AUC has chance
reference 0.5. A raw majority-class accuracy of 91.67% is not used as evidence.

All twelve candidates are also ranked for each image. Top-1 chance is 1/12,
top-3 is 3/12, top-5 is 5/12, mean rank is 6.5 and chance MRR is H_12/12.
Scores equal within an absolute 1e-12 tolerance receive fractional tie credit,
avoiding an arbitrary pair-name ordering effect.

The primary cohort contains complete twelve-pair blocks for each cover/payload.
If a native trial fails, all otherwise valid images in that block are excluded
from primary content comparisons and explicitly counted separately. All-attempt
recovery still counts every trial. Linkage inference is therefore conditional on
complete valid blocks; it does not quantify failure/availability leakage.
At least 120/40/80 complete train/validation/test cover blocks per payload are
required before inference. Fewer stops the pipeline for review, without silently
changing the sample, partition or success criterion.

## Features, models and validation selection

The historical handcrafted feature functions are reused unchanged: image
distribution/residual summaries, text summaries/32-dimensional character-ngram
hashes, and fixed text/image signature interactions. Four predefined views are
image-only, text-only, cross-only, and combined text/image/cross. Feature lists
are frozen; only numeric img_, gamma_, cross_ fields are model inputs. Paths,
cover/pair IDs, labels, hashes, headers, epoch/session IDs, nonces, key material,
payload labels and sender-only information never enter the observed-content
classifiers. Fixed feature extraction uses no training/test population fit.

Logistic regression (C=1, standardisation fitted on training data, max_iter=3000)
and ExtraTrees (400 trees, min_samples_leaf=2, sqrt feature subsets) use three
fixed seeds, 13/41/97. Each family/view's score is the mean probability over its
three seeds. Any convergence warning stops for investigation. If validation AUC
is below 0.5 the ensemble score is reversed, with orientation fixed before test.
For each payload, combined-view validation AUC selects logistic versus ExtraTrees;
an exact tie favours logistic. primary_selection.json is written before test
scoring. All 24 predefined held-out panels, all seed results and the three chosen
primary panels are reported. No model, seed, threshold or feature is selected on
test results. No recent deep detector or universal adversary is represented here.

## Controls

- Image-only and text-only panels must have chance AUC and balanced accuracy.
  This follows in part from the complete matched design: all images/candidate
  pairs have exactly balanced conditional label contributions. These are
  diagnostic controls, not independent proof of secrecy.
- A separate synthetic feature set explicitly encodes true/candidate agreement.
  A held-out classifier must detect that engineered leakage with AUC >0.99.
  It checks the functioning learning/scoring pipeline, not sensitivity to every
  possible real-world linkage. These features are never mixed with CPM features.
- Intentional public-header association is checked separately. True C1/C2/C3
  metadata match; other variants have distinct authenticated epochs/headers.
  Content classifiers omit this intended association. This experiment does not
  claim full-view unlinkability when public headers identify the channels.

## Uncertainty and interpretation

3,000 fixed-seed cluster-bootstrap draws resample the 100 attempted test covers,
jointly across all their images, candidates and payloads. A payload uses available
complete blocks in each draw. Models and the twelve-pair bank remain fixed.
Diagnostic intervals are pointwise 95%. The three validation-selected primary
AUC intervals use Bonferroni adjustment (98.3333% each, familywise 95%). This
does not correct every exploratory metric or quantify training/bank variability.

The weighted AUC implementation is mathematically equivalent to weighted
positive/negative score comparisons, including ties. It is tested against sklearn
on synthetic grouped examples and spot-checked on actual bootstrap draws.
The independent reporting stage reproduces every held-out score from all 72
saved checkpoints and recalculates AUC, balanced accuracy and fractional ranks.

Report observed scores and intervals, whether favourable or unfavourable.
Near-chance observations support only the tested finite content objective.
A confidence interval containing chance does not prove equality, establish
epsilon_unl or verify a compositional-security assumption. Practical metadata
linkage, scheduler leakage, untested learned features and other sources remain
outside the experiment. It provides revised evidence for the content-linkage
parts of Reviewer 5 comments 1–2; modern steganalysis and matched published
multichannel/generative baselines remain separate unfinished work.

## Operational artifacts

run_config.json, cover_manifest.csv, parameter_bank.json and case_manifest.json
freeze inputs/source hashes and the design. generation_results.jsonl and epochs/
retain all attempts, frames, metrics, logs and outputs. generation_audit.json
tracks every failure/incomplete block. The feature NPZ/metadata/schema bind model
inputs. evaluation/ retains validation selection, checkpoints, all held-out
predictions, 24 diagnostic panels, primary panels, seeds, controls and bootstrap
counts. verified_summary.json, RESULTS.md, delivery_manifest.json and PNG/SVG
figures are produced automatically after independent checks. The global read-only
dashboard reads aggregate status only. No secret/key/plaintext is served by it.

The pipeline is finite and persistent; no automatic reruns, outcome-driven
sample expansion or hidden fallback are allowed. No manuscript or earlier
experimental output is overwritten. Browser access was previously declined;
the author opens/refreshes the existing localhost dashboard manually.
