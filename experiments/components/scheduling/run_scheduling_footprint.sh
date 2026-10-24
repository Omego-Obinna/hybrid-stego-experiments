#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/omego/Documents/Hybrid_Stego_Eva"
SCRIPT_ROOT="${ROOT}/scheduling_footprint"
OUT_ROOT="${ROOT}/scheduling_footprint_outputs"
MODE="${1:-smoke}"

case "${MODE}" in
  smoke)
    EPOCHS=1000
    BOOTSTRAP=100
    MMD_PERM=100
    CLF_PERM=50
    MMD_N=1000
    CLF_N=1000
    ;;
  paper)
    EPOCHS=10000
    BOOTSTRAP=1000
    MMD_PERM=1000
    CLF_PERM=200
    MMD_N=5000
    CLF_N=5000
    ;;
  *)
    echo "Usage: $0 {smoke|paper}" >&2
    exit 2
    ;;
esac

mkdir -p "${OUT_ROOT}/traces" "${OUT_ROOT}/analysis"

python "${SCRIPT_ROOT}/01_generate_scheduling_traces.py" \
  --out-dir "${OUT_ROOT}/traces" \
  --epochs "${EPOCHS}" \
  --seeds 42,43,44 \
  --epoch-period-ms 1000 \
  --rho-ms 100 \
  --skew-delta-ms 100 \
  --timeout-ms 200 \
  --window-w 5 \
  --heartbeat-h 10 \
  --cover-phi 0.80 \
  --cover-noise-ms 18 \
  --channel-jitter-ms 5 \
  --unshaped-jitter-ms 0.40 \
  --network-delay-ms 20 \
  --network-jitter-ms 5 \
  --include-heartbeats

python "${SCRIPT_ROOT}/02_analyze_scheduling_footprint.py" \
  --trace-csv "${OUT_ROOT}/traces/scheduling_event_traces.csv" \
  --out-dir "${OUT_ROOT}/analysis" \
  --time-field send_time_ms \
  --rho-ms 100 \
  --skew-delta-ms 100 \
  --bootstrap-repeats "${BOOTSTRAP}" \
  --ks-max-sample 10000 \
  --mmd-max-samples "${MMD_N}" \
  --mmd-permutations "${MMD_PERM}" \
  --classifier-max-per-seed "${CLF_N}" \
  --classifier-permutations "${CLF_PERM}" \
  --seed 20260605 \
  --formats pdf png svg \
  --dpi 300

printf '\n[OK] Scheduling-footprint %s run completed.\n' "${MODE}"
printf 'Outputs: %s\n' "${OUT_ROOT}/analysis"
