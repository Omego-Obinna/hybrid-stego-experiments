# CP-UNL extension: full public view with a finite chosen-query history

Frozen before new embedding or classifier outcomes, 17 October 2026.

This experiment instantiates the manuscript's real-world (h=0) CP-UNL
comparison for a declared finite source and adversary family. It includes
complete challenge frames, correctly generated native HMAC tags, public
decoder context and four earlier embedding queries. It does not turn finite
classifier observations into a proof for every PPT adversary, estimate an H2
hybrid premise, or test network scheduling/anonymity. Those are distinct claims.
The prior content-only study is completed evidence and remains untouched.

## Sources and separation

Select 200 previously unused, byte-distinct BOSSBase covers: 100 training,
40 validation, 60 test, within the existing clean source partitions. Exclude
all covers from the pilot, fusion panel and previous 300-cover linkage run.
Selection uses SHA256 of `cpunl-fullview-20261017:partition:cover_id`.
No outcome-based replacements. A separate old training cover is used only for
development checks and excluded from the recorded study.

Use twelve pairs derived by taking the first 65,536 bytes of each object in
the existing twelve-pair bank. The prefixes are copied to new files; no old
object is edited. Equal lengths deliberately define one public length stratum,
so independent-world carriers have the same conditional marginal given lengths.
The sampler is uniform on this fixed, message-independent finite bank. The
study does not establish naturalness or generalisation to unseen parameter banks.

Three application payloads: 0.10, 0.20, 0.40 bpp. Each cover/payload block has
a fresh independent 256-bit research fixture master key and 128-bit session.
Keys never enter observer features or the dashboard. Separate covers/keys remain
in separate partitions. Source protocol, image worker and fused-0.75 embedder
are copied unchanged from the verified previous run and hash-bound.

## Four queries and matched-context counterfactual challenges

For a block, epochs 0--3 are history and epoch 4 is the challenge.
History parameter pairs are independent uniform bank draws. The first query
message is all zero. Each subsequent query message is the truncated SHAKE256
expansion of the public previous request/response transcript with a fixed domain
separator. Thus the policy chooses known messages adaptively from prior public
responses; it never uses a key, source cover, mask or hidden label.
History responses are the complete three native frames (or an explicit failure
symbol). The challenge message is a publicly known all-zero bit string.

Generate twelve legitimate native challenge outputs, one for each bank pair,
under exactly the SAME challenge session/epoch/header/nonces/length/codec, key,
source cover and message. They are mutually exclusive counterfactual branches,
not twelve live transmissions reusing a nonce. Each branch starts from a copy
of the same four-query receiver state; history and every legitimate challenge
are separately decoded and compared byte for byte. This avoids relabelling
old carriers whose embedding seed/cost context differs from the new context.
There are 200 x 3 x (4 + 12) = 9,600 planned native embedding attempts.

For each complete block, let gamma[a] and s[b] denote its twelve parameters
and twelve native challenge carriers. All objects/lengths/context are public.

* Linked world: a uniform on 0..11, observed content (gamma[a], s[a]).
* Independent-content world: a,b independently uniform on 0..11, observed
  content (gamma[a], s[b]). Include ALL 144 combinations, including the twelve
  a=b cases. Forbidding the diagonal would not be independent sampling.
* Generate a native complete-transcript HMAC for EVERY view with its actual
  canonical fields. Linked tags match native outputs. Independent tags are
  recomputed by the laboratory challenger; this is a distributional comparison,
  not an attacker's forgery. Do not decode independent-content views as if they
  were real encodings of the challenge message. No decryption oracle is offered.
* Both worlds expose the identical four-query history and challenge metadata.
  All challenge nonces are fresh relative to that history. Public headers may
  associate frames; hiding that intended association is not the CP-UNL task.

Save reconstructible complete public views via content-addressed/reference
records and native frames, with canonical transcript hashes and all three wire
frame hashes. Export can reconstruct any view's exact bytes. Never give the
classifier verification outcomes, private keys, sender seeds/costs/masks,
generating-pair indices, filenames, labels, or original covers. Known adversarial
query messages and all transmitted public bytes ARE allowed observations.

## Failures and image deviations

Every native trial is preserved. Recognised preflight failures emit no frames,
are never retried and never replaced. Unexpected operational/receiver errors or
wrong accepted plaintext stop the pipeline. Pixel changes beyond one level are
retained and explicitly recorded, following the author's approved audit policy.
Primary distributional inference requires all four history responses and twelve
challenge carriers in a block. Incomplete blocks are excluded symmetrically from
BOTH worlds, with every otherwise-valid image and failure counted separately.
This is conditional complete-block inference, not an availability-leakage bound.
Minimum complete blocks at each payload: 80 train / 30 validation / 45 test.
Fewer stops for review rather than resampling.

## Fixed public feature maps and models

Cache the unchanged historical 284 content features on the new fixed-length
bank/carriers. Add fixed byte-distribution and adjacent-byte sketches that read
every byte of all three content objects. Include raw public session, epoch,
header, nonce and tag bytes; lengths, codec/version and channel consistency;
public digest/content comparisons; and four ordered prior request/response
representations plus fixed challenge/history difference features. The full
observer records retain every raw byte; the classifiers use these explicitly
bounded feature maps, not an arbitrary universal byte-reading adversary.

Five predefined views: metadata_only (no tag/content), content_only,
authentication_only, full_challenge, full_history. Primary is full_history.
Two fixed families: standardised balanced logistic regression (C=1,
max_iter=3000) and balanced ExtraTrees (256 trees, min_samples_leaf=4,
max_features=sqrt). Seeds 13/41/97; ensemble probability means. Three payloads
x five views x two families x three seeds = 90 fitted models / 30 panels.
All outcomes, seed scores and views are retained. No feature/model is adjusted
using validation or test outcomes. An unresolved convergence warning stops.

There are 12 linked and 144 independent rows per block. Fit with balanced
class weights to represent equal world priors. Report AUC, balanced accuracy at
threshold 0.5, TPR/FPR and the fixed-rule signed and absolute acceptance gap;
never report majority-class accuracy as success. Identical diagonal views
deliberately occur in both labels and stay within one cover partition.
Ensemble orientation is fixed using validation AUC only. Choose the primary
family using full_history validation AUC (exact tie favours logistic), write
selection and its hash BEFORE any test scoring, then evaluate once.

## Controls, uncertainty and verification

Metadata-only AUC must equal chance by the exact matched design; this is a
balance check, not independent evidence of security. Check per-block parameter
and carrier marginals and the 1/12 diagonal overlap exactly. Check every
challenge/history HMAC, header/nonces and canonical frame serialization.

A separately labelled synthetic known-leak control copies a public parameter
digest prefix into image pixels and authenticates the resulting validly framed
counterfactual objects. It uses the SAME observable feature extraction and fixed
classifiers, with distinct synthetic training/test fixtures. Independent draws
retain their diagonal overlap; expected discrimination need not be perfect.
Require control AUC >0.90. This checks sensitivity to the deliberately implanted
public coupling, not every possible real leak. It is never mixed into real data.

Use 3,000 fixed-seed bootstrap draws resampling all 60 attempted test cover IDs
jointly across payloads and world combinations. Complete blocks at a payload
contribute when available; key histories and all counterfactuals stay clustered.
Primary AUCs use Bonferroni-adjusted 98.3333% individual intervals (joint target
95% over three payloads). Other intervals are pointwise 95%, conditional on
fixed models, finite bank and query policy. They do not quantify all key/source
or adversary-family uncertainty and are not epsilon_unl estimates.

Independent verification reloads all 90 checkpoints, reproduces every saved
test probability, recomputes sklearn AUC/balanced accuracy and spot-checks
weighted bootstrap calculations. Recheck raw source/data hashes, separation,
world marginals, complete-view tags, diagonal equality, failures, deviations and
selection chronology. Export report, tables and PNG/SVG plots, inspect the local
figure, then mark completed. Keep unfavourable results. Do not edit manuscript
or old experiments. The existing local dashboard gets a distinct phase. The
user opens it manually; previously declined browser access is not bypassed.
