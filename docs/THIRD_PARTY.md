# Third-party sources

Retain the source-level notices and inspect upstream terms before reuse.

| Source | Exact recorded identity / role |
|---|---|
| HSMNet | `diqi12/HSMNet`, commit `2c9f57be9c24753308931ee7e016c9cb2958813d`; official architecture under study `source/HSMNet/`. No separate licence file was present in the retrieved source snapshot; no blanket licence is granted here |
| Pulsar | `spacelab-ccny/pulsar`; recorded commit starts `52c263976770`. Exact per-file hashes and full identifiers, where recorded, are in native run configurations; MIT notice retained |
| Assemblage | `spacelab-ccny/assemblage`; recorded commit starts `ce16a471ea0a`. Its Pulsar submodule starts `ae39c51440b5`; MIT notices retained separately |
| Generator | `google/ddpm-church-256`; recorded revision starts `5a1118b5a35e`; weights excluded, use upstream terms |
| Component CNN/SRM implementations | Source-level provenance in `experiments/components/steganalysis/`; retained code notices apply; no new blanket redistribution licence is asserted |
| Journal template | `sn-jnl.cls`, bibliography styles and their existing notices in `manuscript/` |
| STC and other installed dependencies | External packages; this archive does not relicense their code or native binaries |

The four spatial-baseline implementations are the exact named local scripts.
The MiPOD script uses a local variance approximation; it must not be described
as an exact canonical reference implementation. Native Pulsar and Assemblage
have their own benign-source, format and recovery interfaces, distinct from CPM.
