# Source and accounting provenance

`provenance/source_manifest.json` maps every copied file to its source identity,
source SHA-256, packaged SHA-256, byte count and transformation. Scientific
executable snapshots are byte-identical. Descriptive records have machine paths
replaced by placeholders; their original digests and embedded original-artifact
hash maps remain in the manifest/records. A packaged configuration's file hash
therefore need not equal the original configuration hash inside a saved receipt.
Use `SHA256SUMS` for the final upload tree.

Four copies of the upstream HSMNet example `train.py` are retained byte for
byte as reference material. That upstream example has an unterminated string
and was not the trainer executed in these studies. The study-specific
`image_hsm_train.py` / pilot trainers and the imported model modules pass the
syntax check. `non_executable_reference_files.json` identifies the unused
examples by exact path and digest; release verification excludes only these
fixed reference files from its executable-source syntax check.

The component public-source snapshot was commit
`e25ef2ea4cd2a53ea4161497adce1587470c45a7` of
<https://github.com/Omego-Obinna/hybrid-multi-stego>. Additional experiment
sources are the recorded local frozen run snapshots. This package does not
assert that the older public commit already contains the completed new studies.

## HSMNet continuation

The first driver stopped after the durable epoch-26 checkpoint, during a partial
epoch 27. The versioned continuation retained completed epochs 1–26 and resumed
epoch 27 from the saved model, Adamax and scheduler. Only uncheckpointed partial
updates were lost. The original checkpoint lacked RNG state. The recorded
fixed-seed reconstruction replayed known dropout random calls on constant
feature-shaped tensors and validation-loader seed draws; it did not repeat
native forward/gradient computation or completed epochs. Later last-checkpoints
save CPU/CUDA RNG states atomically. The source and exact operational details
are preserved under `resume_20261021_v1/`.

The fixed sample, architecture, budget, augmentation, optimiser, schedule,
selection and uncertainty choices stayed unchanged. The report-quality pass
corrected baseline core timing using saved `cost_sec + stc_sec`, retained pixel
deviations and preserved provenance. Its original before/after hashes are
included. Full before-artifact backups remain in the external run archive;
they are not silently represented as included by a hash receipt.

## Corrected component accounting

- The scheduling summary's pooled overlapping time spans inflated an emission
  rate to about three epochs/s. The manuscript reports about one epoch/s within
  each seed. The original summary/source remains available as a provenance
  artifact, not as the corrected publication value.
- The transport input has 26,191 rows and 271 repeated parameter/seed scenarios
  with identical simulated metrics. `results/audit/transport_unique_runs.csv`
  retains 25,920 distinct scenarios. Raw rows stay in
  `results/components/transport/run_summary_raw.csv`.
- The fused-v3.2 runtime source CSV has a 153-column principal partition and a
  different appended 178-column 0.30-bpp partition. The saved raw audit and
  publication summaries retain the correct partition and finite timing counts.
- Candidate ranking uses `ours_qim_embed_SAFE_v1.py`. Exact baseline version
  labels, including local MiPOD approximation, are retained. Filenames containing
  `legacy` or `historical` are immutable source identifiers, not publication prose.
