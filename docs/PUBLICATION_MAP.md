# Map from publication to evidence

Use document labels and titles as the stable references if pagination changes.

| Publication area | Primary evidence |
|---|---|
| Section 5.1, setup | Study `records/run_config.json`, designs, `results/audit/recorded_parameters.json` |
| Cover naturalness and component probes | `results/components/cover_parameter_hybrid-stego_results/` and component stream code |
| Section 5.3, fidelity | Component fidelity tables; fusion sensitivity records; HSMNet `paired_fidelity.csv` and `fidelity_contrasts.csv` |
| Section 5.4.1, classical and CNN-based evaluation | SRM+EC tables and six saved CNN configurations; their cohorts/configurations must remain distinct |
| Section 5.4.2, matched HSMNet evaluation | HSMNet `detectability.csv`, `auc_paired_contrasts.csv`, saved predictions and checked-review receipt |
| Content-only linkage | `cpm_v2_revised_linkage_300/evaluation/`; distinguishes wrong-candidate ranking from CP-UNL |
| Full-public-view CP-UNL | `cpm_v2_fullview_cpunl_200/evaluation/`, `world_audit.json` and primary selection |
| Authenticated delivery / active receiver | Protocol checks and `cpm_v2_active_window/records/` |
| Native comparisons | Pulsar/Assemblage records, their own recovery, cost and delivery definitions |
| Section 5.8.4, modern steganalysis discussion | Cross-reference to the matched HSMNet results, limited to image-carrier detection |
| Supplement, scheduling / transport / runtime | Component records plus corrected accounting under `results/audit/` |
| Abstract, evidence summary and conclusion | Preserve favourable and unfavourable findings; no outstanding-HSMNet claim |

HSMNet concerns the empirical image-carrier detectability component,
`epsilon_emb`. AUC, a fixed-rule acceptance gap, total variation and a
cryptographic upper bound are different quantities. The full-public-view
CP-UNL evidence stays in its own section and cannot be supplied by a baseline
that has no such interface. High fidelity and exact recovery do not establish
undetectability. See the manuscript for the conditional security statements.
