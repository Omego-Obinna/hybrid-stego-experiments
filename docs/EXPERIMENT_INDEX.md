# Experiment index

Identifiers below are immutable artifact identifiers. Names such as `revised`
inside an identifier do not describe a different unpublished result.

| Experiment | Location / entry point | Status and interpretation |
|---|---|---|
| Naturalness and cover-parameter streams | `experiments/components/cover_generation/` | Corpus splits, generated streams, MAUVE/POS and RoBERTa; numerical summaries in `results/components/cover_parameter_hybrid-stego_results/` |
| Image embedding and fidelity | `experiments/components/embeddings/` | Exact named CPM and four baseline implementations; local MiPOD approximation must keep its label |
| Masking ablations | `experiments/components/masking_ablations/` | Generation, analysis and existing-recovery summary; not a proof of PRF security |
| Component classical steganalysis | `experiments/components/steganalysis/SRM_EC/` | Saved component SRM+EC study; distinct from the paused full matched SRM branch |
| Component CNNs | `experiments/components/steganalysis/CNN/` | Ye-Net and Yedroudj-Net; six actual configurations in `results/components/cnn/configuration_audit.json` |
| Candidate-ranking linkage | `experiments/components/linkability_probes/` | Twelve-pair ranking uses the v1 encoder; do not attribute it to fused-v3.2 |
| Synthetic transport | `experiments/components/transport/` | Virtual loss/jitter/ECC/replay/tamper simulator; corrected unique-scenario table in `results/audit/` |
| Scheduling footprint | `experiments/components/scheduling/` | Three seeds, 10,000 epochs each; per-seed emission denominator |
| Runtime and overhead | `experiments/components/runtime/` | Fused-v3.2 0.1/0.2/0.4-bpp partition; mixed-schema raw CSV excluded from ordinary analysis |
| Protocol regression | `experiments/protocol/` | Received-field reconstruction, authentication/freshness and native image-worker checks |

The following paths are relative to `experiments/studies/`.

| Run identifier | Main source entry | Recorded status |
|---|---|---|
| `cpm_v2_pilot_100` | `experiments/preparation/run_pilot.py` with the frozen protocol sources | Completed, 300 attempts |
| `cpm_v2_fusion_sensitivity_25` | `source/run_sensitivity.py` | Completed, 375 attempts, one exact-recovery failure |
| `cpm_v2_revised_linkage_300` | `source/run_pipeline.py` | Completed, content-only linkage; 72 models and 24 panels |
| `cpm_v2_fullview_cpunl_200` | `source/run_pipeline.py` | Completed, full-public-view experiment; 90 models and 30 panels |
| `cpm_v2_active_window` | `source/run_active.py` | Completed, 110 virtual-arrival traces |
| `cpm_v2_native_generative` | `source/run_native_pipeline.py` | Completed, native Pulsar |
| `cpm_v2_assemblage_local` | `source/run_native_pipeline.py` | Completed, local Assemblage; different native interface |
| `cpm_v2_hsmnet_matched_1200` | `source/run_hsmnet_pipeline.py`; documented operational continuation | Completed and independently checked, 6 models / 15 ensembles |
| `cpm_v2_srm_detectability_pilot_160` | `source/pilot.py` | Completed bounded pilot, exploratory |
| `cpm_v2_hsmnet_detectability_pilot_300` | `source/pilot.py` | Completed bounded pilot, exploratory |
| `cpm_v2_matched_image_1200` | `source/run_image_pipeline.py` | **Paused and incomplete. Do not automatically resume.** Shared image preparation was reused by HSMNet |

Preparation/dependency probes without a completed scientific experiment are not
additional independent studies. Their design/configuration lineage is retained
where referenced by the completed-run provenance. Failed attempts within a study
remain in its accounting rather than being replaced.
