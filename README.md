# Cover-Parameterised Multichannel Hybrid Steganography

Code and saved evidence accompanying **Cover-Parameterised Multichannel Hybrid
Steganography: Compositional Security, Detectability, and Robustness**.
Publication package: 24 October 2026.

This repository covers the manuscript's component experiments and the CPM
protocol, fusion, linkage, active-receiver, native-baseline and matched HSMNet
studies. It also identifies the two bounded detectability pilots and the
incomplete, paused matched SRM branch. See the
[experiment index](docs/EXPERIMENT_INDEX.md) and
[publication map](docs/PUBLICATION_MAP.md) before interpreting a result.

## Start with the saved evidence

Python 3.10 or newer suffices for the archive/integrity check:

```sh
python tools/verify_release.py
```

To reproduce the matched HSMNet statistics **from saved predictions**, without
training, selecting checkpoints, or running image inference:

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements/analysis.txt
python tools/verify_saved_hsmnet.py
```

That check recomputes all 15 AUCs and pointwise intervals, 12 adjusted paired AUC
contrasts, common-valid counts, saved two-seed ensembles, fidelity statistics
and fresh recovery/timing summaries. It does not reproduce checkpoint forward
passes: checkpoints and images are external artifacts. The original saved
verification receipts record those checks and their scope.

To render the main HSMNet figure from the same saved tables, install
`requirements/plotting.txt` and run `python tools/plot_hsmnet_main.py`.
Outputs go to the ignored `local_runs/figures/` directory; the supplied
publication figure is preserved. The plot uses exact interval endpoints,
including asymmetric intervals that can exclude the point estimate.

## Repository layout

| Directory | Contents |
|---|---|
| `experiments/components/` | Cover streams, encoders, masking, component CNN/SRM, candidate linkage, scheduling, runtime and transport sources |
| `experiments/protocol/` | Epoch protocol, image worker and protocol checks |
| `experiments/studies/<run_id>/` | Frozen source, configurations, results, saved score arrays and relevant operational provenance |
| `results/components/` | Component-study numerical tables and selected numeric logs |
| `results/audit/` | Saved accounting checks and corrected unique transport scenarios |
| `manuscript/` | Editable manuscript, supplement, bibliography and figure inputs |
| `provenance/` | File-by-file original/package hashes, study status and excluded-artifact inventory |
| `requirements/`, `tools/` | Separate analysis dependencies, environment guidance and safe release checks |

## Main evidence boundaries

The HSMNet comparison uses five methods (CPM, S-UNIWARD, the recorded local
MiPOD variance approximation, WOW and HILL), a fixed 100-epoch training budget,
two seeds and validation-only choices. Seven of twelve family-adjusted paired
AUC differences favour CPM; five include zero. All fifteen pointwise AUC
intervals exclude chance. This is finite image-detector evidence, not a
cryptographic upper bound on image-carrier detectability or a universal
security claim. Full-public-view CP-UNL is a separate experiment.

Failures, pixel deviations and unfavourable results remain in the supplied
records. Component CNN configurations differ by payload; the saved configuration
audit, rather than directory names, determines their attribution. The scheduling
emission rate is about one epoch/s per seed. Transport summaries use 25,920
unique scenarios, with the 26,191-row input retained separately.

## Reproducing experiments

See [REPRODUCING.md](docs/REPRODUCING.md),
[ENVIRONMENTS.md](docs/ENVIRONMENTS.md) and
[DATA_AND_ARTIFACTS.md](docs/DATA_AND_ARTIFACTS.md).
Source snapshots retain their original paths and defaults for traceability.
They are not a universal one-command pipeline. Full execution needs separately
supplied datasets, corpus splits, keys, native services and/or model checkpoints.
The release checks never start scientific drivers or resume paused work.

Read [PROVENANCE.md](docs/PROVENANCE.md) for path placeholders, hash semantics,
source versions and the documented HSMNet continuation.

## Manuscript and upload

Upload the contents of `manuscript/` as an Overleaf project, keep `latexmkrc`,
choose `sn-article.tex` and pdfLaTeX, then select `Supplementary.tex` to export
the supplement. A separate ready-to-upload Overleaf ZIP accompanies this release.

[GITHUB_UPLOAD.md](docs/GITHUB_UPLOAD.md) gives upload commands. No GitHub
credentials, datasets or experiment fixture master keys are included.

## Attribution and licensing

See [CITATION.md](CITATION.md), [LICENSING.md](LICENSING.md) and
[THIRD_PARTY.md](docs/THIRD_PARTY.md). Original notices are retained. This
package does not invent a blanket licence for author or third-party code.
