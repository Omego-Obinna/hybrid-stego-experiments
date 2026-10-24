# Publication table notes

Method: `Ours-QIM-v3.2-Fused-CF`

Rows in log: `30000`

Payloads detected: `[0.1, 0.2, 0.4]`

Detected base-cost type: `fused_suniward_mipod_spatial`

Detected STC backend: `pystc, failed`

Detected direction mode: `none`

Recommended use:

1. Put `table_ours_qim_v32_configuration.tex` in the supplementary material or method appendix.
2. Put `table_ours_qim_v32_embedding_quality.tex` in the main manuscript results section if space permits.
3. Put `table_ours_qim_v32_content_adaptivity.tex` in the supplementary material if the main manuscript is space-limited.
4. Use `table_ours_qim_v32_runtime.tex` when discussing computational practicality.

LaTeX packages required:

```latex
\usepackage{booktabs}
\usepackage{threeparttable}
\usepackage{adjustbox}
```

Suggested caption wording:

`Ours-QIM-v3.2-Fused-CF combines hybrid QIM payload masking, a S-UNIWARD-dominant fused S-UNIWARD/MiPOD base-cost map, complex-feature-aware region refinement, SLIC-guided region constraints, conservative cost spreading, and STC-based additive-distortion minimization.`
