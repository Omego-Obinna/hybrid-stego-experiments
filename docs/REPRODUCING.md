# Reproduction levels and execution

## 1. Check this release

Run `python tools/verify_release.py`. It checks file hashes, required study
records, source syntax and fixed saved-table accounting using Python's standard
library. It reads files only. Run `python tools/verify_saved_hsmnet.py` after
installing `requirements/analysis.txt` for numerical score/interval reproduction.
These commands do not need images, model weights, private keys or a GPU.

## 2. Inspect the recorded experiment

Each study has a byte-preserved `source/` snapshot, descriptive `records/`,
and saved `evaluation/` or `predictions/` when available. Read its design,
configuration, completion/verification receipts and results together. The
file-level manifest records where every copied file came from and whether
publication packaging replaced absolute paths in descriptive records.

## 3. Set up a new, independent execution workspace

Never execute a scientific driver against the packaged evidence or an existing
completed run. Supply your own licensed datasets and resources; make a fresh
directory outside this repository. The experiment entry points are listed in
`EXPERIMENT_INDEX.md`. Most component programs accept command-line paths; frozen
run drivers additionally have recorded absolute paths or environment variables.
For example, the matched image modules use `CPM_IMAGE_RUN` for their run root.
Inspect the selected entry point and update only machine paths in a separate
working copy. Preserve the original snapshot and record an exact diff and new
hashes before execution. Do not simply replace all version strings or weaken
source-integrity assertions. Recompute a working-copy source-hash map only for
documented relocation changes, retaining the original map alongside it.

The placeholders `${ARTIFACT_ROOT}`, `${WORKSPACE_ROOT}` and `${USER_HOME}` in
descriptive records are not executable shell assignments or automatic data
downloads. They stand for the original experiment hierarchy, author working
documents and user home. Resolve them deliberately in a new local configuration.
Input/source hashes in these records authenticate original artifacts; use
`provenance/source_manifest.json` for the hashes of packaged record files.

No universal end-to-end launcher is supplied because the experiments use
different native interfaces and environments. The publication's executed STC
binary hash and some original SLIC launch/package state were not fully saved;
the package cannot turn those omissions into exact-environment reproduction.

## Fixed matched HSMNet protocol

Use 1,200 original cover identities split 700/200/300. The five-method
common-valid blocks and minimum 600/150/240 counts are fixed. Preserve all
18,000 image attempt records, including failures and pixel deviations, plus the
450-attempt known-message timing/recovery panel. There is no new embedding
when analysing the saved study.

The source/configuration fixes native 512×512 grayscale inputs, paired batch 2,
two workers, paired D4 augmentation, Adamax lr 0.001 / eps 1e-8 / weight decay 0,
100 epochs, milestones 60 and 85, gamma 0.1 and seeds 801/802. Selection is
minimum joint validation cross-entropy, with common orientation/threshold chosen
on validation only. The detector cycles an equal-method mixture. Test scoring
occurs only after these choices. A 3,000-draw shared cover bootstrap uses seed
61018; the twelve CPM-minus-baseline AUC contrasts use the saved family
adjustment. Do not retune these choices when interpreting this dataset.

The operational continuation is disclosed in `PROVENANCE.md` and saved files.
It is not a recommended fresh-run training strategy. Do not launch the preserved
host-specific `launch_resume.py` or resume the incomplete SRM branch as a
repository check.

## Other study interfaces

- Protocol/fusion/content/full-view studies require the frozen parameter bank,
  matched covers and native STC dependency, plus a new private fixture key.
- Active-receiver experiments use controlled virtual arrivals and saved image
  fixtures; they are not live network deployment tests.
- Pulsar requires its pinned generator weights and native Python stack.
  Assemblage additionally needs the recorded SageMath environment; its service,
  object size and benign source differ from CPM's.
- Component naturalness uses the recorded corpus splits and generated streams.
  Component image/CNN/SRM cohorts must not be silently replaced by the HSMNet
  cohort. The candidate-ranking encoder is v1, despite a directory's v3.2 name.
- Synthetic transport is a simulator. Use the unique-scenario table for the
  corrected summaries and retain the raw duplicate rows separately.

Run-specific configuration records, rather than an illustrative README command
or a folder name, are the scientific specification.
