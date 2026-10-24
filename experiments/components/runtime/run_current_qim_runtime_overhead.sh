#!/usr/bin/env bash
set -euo pipefail
ROOT="/home/omego/Documents/Hybrid_Stego_Eva"
python "${ROOT}/current_qim_runtime_overhead/build_current_qim_runtime_overhead.py" \
  --runtime-summary-csv "${ROOT}/ours_qim_v3_2_fused075_payloads_FULL/logs/publication_tables/table_ours_qim_v32_runtime.csv" \
  --raw-log-root "${ROOT}/ours_qim_v3_2_fused075_payloads_FULL" \
  --stego "0.10=${ROOT}/ours_qim_v3_2_fused075_payloads_FULL/ours_qim_0.100" \
  --stego "0.20=${ROOT}/ours_qim_v3_2_fused075_payloads_FULL/ours_qim_0.200" \
  --stego "0.40=${ROOT}/ours_qim_v3_2_fused075_payloads_FULL/ours_qim_0.400" \
  --gamma1 "${ROOT}/cover_parameter/hybrid-stego/gamma1_alice_selected_seed42.txt" \
  --gamma2 "${ROOT}/cover_parameter/hybrid-stego/gamma2_alice_selected_seed42.txt" \
  --out-dir "${ROOT}/current_qim_runtime_overhead_outputs" \
  --image-width 512 --image-height 512 --max-files 10000 \
  --heartbeat-cadence 10 \
  --hardware-note "Replace with CPU, RAM, Ubuntu, Python, and storage details used for the existing 10,000-image experiment" \
  --formats pdf png svg --dpi 300 --width 8.0 --height 4.2
