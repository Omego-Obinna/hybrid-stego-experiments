# Definition-to-experiment alignment

The authoritative definition is CP-UNL at manuscript sn-article.tex:1683.
This is an h=0, finite-source, four-prior-query instantiation. It adds the full
public authentication/context/history view to the completed content-only study.

| Definition requirement | Implementation | Audit |
|---|---|---|
| Matched public metadata Z | Same native epoch-4 context within each cover/payload; both parameter objects are 65,536-byte prefixes; all PGM and wire lengths fixed | Exact native context and metadata-vector equality for all 156 views per block |
| Consistent linked content | Twelve original sender challenge outputs; one parameter pair with its own native carrier | Separate native receiver recovery and native tag equality |
| Independent content with the same conditional marginals | Full 12 x 12 product including its twelve diagonal entries, under the same context and history | Every combination once; both conditional marginals exactly twelve times linked counts |
| Visible tags generated consistently | Original HMAC authentication function applied to each actual canonical transcript | Native verifier checks every view; identical diagonal observations have identical tags/hashes |
| Specified prior history | Four actual native chosen-message request/response pairs; later query messages are public functions of earlier responses | Reconstruct query policy and verify all history tags, chronology and challenge freshness |
| Full public view | All three challenge frames and every historical request/response byte reconstructible from retained native artifacts and per-view records | All canonical, wire and complete-observer hashes verified; raw export tested |
| Secret material absent from adversary input | Feature vectors built only from public objects/context/tags/known queries | Source-frozen explicit column groups; keys/masks/seeds/covers/labels/index fields excluded |
| Uniform hidden world and probability-gap interpretation | Twelve positive versus 144 negative rows, class-balanced fitting, equal-prior balanced accuracy | AUC plus explicit fixed-rule TPR/FPR and signed/absolute gap; no majority accuracy claim |
| Permitted PPT wardens | Two fixed finite classifier families, five public feature views, three seeds | All 90 checkpoints/30 panels retained; no claim of covering every PPT algorithm |

The native branches sharing epoch-4 context are mutually exclusive
counterfactuals. An observer receives ONE challenge, not the entire branch bank.
No live nonce-reuse deployment is simulated. A retagged independent-content
world is a laboratory distribution, not a forgery attack or a claimed encoding
of the known challenge message; there is no decryption oracle.

The source is deliberately uniform over one fixed twelve-pair, equal-length
bank. This controls conditional marginals without asserting sampler naturalness,
unseen-bank generalisation, or security of arbitrary source distributions.
The four-query policy is adaptive in its chosen messages but is one fixed policy,
not an exhaustive test of adaptive histories. Raw bytes are retained in full;
classifiers consume explicitly declared fixed feature maps of those bytes.

Incomplete native blocks are removed symmetrically from both worlds. They remain
in all-attempt failure accounting. This makes the statistical evidence conditional
on complete valid histories/challenge banks; availability leakage is not bounded.
Image deviations are retained and reported under the approved audit policy.

This experiment does not estimate a supremum over all adversaries, certify H2
uniform-mask replacement, measure a cryptographic epsilon_unl, hide intentional
header association, or evaluate network scheduling. Those distinctions preserve
the manuscript's actual conditional claim rather than introducing an anonymity
claim that its definition excludes.
