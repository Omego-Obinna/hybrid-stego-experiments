#!/usr/bin/env bash
set -euo pipefail

KEY="${STEGO_KEY:-your-shared-secret-key}"

mkdir -p logs/gpu_logs results corpus/external

run_gpu_logged () {
  local gpu_log="$1"
  local main_log="$2"
  shift 2

  nvidia-smi \
    --query-gpu=timestamp,index,name,utilization.gpu,utilization.memory,memory.total,memory.used,temperature.gpu,power.draw \
    --format=csv \
    -l 1 > "$gpu_log" &
  GPU_PID=$!

  /usr/bin/time -v "$@" 2>&1 | tee "$main_log"

  kill "$GPU_PID" 2>/dev/null || true
}

run_external () {
  local run_id="$1"
  local ref_a="$2"
  local ref_b="$3"
  local test_book="$4"
  local label="$5"
  local seed="$6"

  echo "============================================================"
  echo "External Run ${run_id}: Reference=${ref_a}+${ref_b}, Test=${test_book}, Seed=${seed}"
  echo "============================================================"

  local ref_file="corpus/external/ext${run_id}_${label}_reference_full.txt"
  local ref_sample="corpus/external/ext${run_id}_${label}_reference_1000.txt"
  local test_sample="corpus/external/ext${run_id}_${label}_test_1000.txt"

  local gamma1="gamma1_ext${run_id}_${label}_selected_seed${seed}.txt"
  local gamma2="gamma2_ext${run_id}_${label}_selected_seed${seed}.txt"

  cat "corpus/${ref_a}" "corpus/${ref_b}" > "$ref_file"

  python3 sample_lines.py \
    --infile "$ref_file" \
    --outfile "$ref_sample" \
    --n 1000 \
    --seed "$((seed + 100))" \
    2>&1 | tee "logs/ext${run_id}_${label}_sample_reference.log"

  python3 sample_lines.py \
    --infile "corpus/${test_book}" \
    --outfile "$test_sample" \
    --n 1000 \
    --seed "$seed" \
    2>&1 | tee "logs/ext${run_id}_${label}_sample_test.log"

  /usr/bin/time -v python3 keyed_select_gamma.py \
    --corpus "$ref_file" \
    --key "$KEY" \
    --n_sent 1000 \
    --out1 "$gamma1" \
    --out2 "$gamma2" \
    2>&1 | tee "logs/ext${run_id}_${label}_keyed_selection.log"

  run_gpu_logged \
    "logs/gpu_logs/ext${run_id}_${label}_mauve_gamma1_gpu.csv" \
    "logs/ext${run_id}_${label}_mauve_gamma1_vs_test.log" \
    python3 compute_mauve.py \
      --p "$gamma1" \
      --q "$test_sample" \
      --device 0 \
      --num_buckets 50

  run_gpu_logged \
    "logs/gpu_logs/ext${run_id}_${label}_mauve_gamma2_gpu.csv" \
    "logs/ext${run_id}_${label}_mauve_gamma2_vs_test.log" \
    python3 compute_mauve.py \
      --p "$gamma2" \
      --q "$test_sample" \
      --device 0 \
      --num_buckets 50

  run_gpu_logged \
    "logs/gpu_logs/ext${run_id}_${label}_mauve_human_gpu.csv" \
    "logs/ext${run_id}_${label}_mauve_human_vs_test.log" \
    python3 compute_mauve.py \
      --p "$ref_sample" \
      --q "$test_sample" \
      --device 0 \
      --num_buckets 50

  run_gpu_logged \
    "logs/gpu_logs/ext${run_id}_${label}_roberta_gamma1_gpu.csv" \
    "logs/ext${run_id}_${label}_roberta_gamma1_vs_test.log" \
    python3 train_roberta_detector.py \
      --gamma "$gamma1" \
      --benign "$test_sample" \
      --epochs 3 \
      --bsz 8 \
      --lr 2e-5

  run_gpu_logged \
    "logs/gpu_logs/ext${run_id}_${label}_roberta_gamma2_gpu.csv" \
    "logs/ext${run_id}_${label}_roberta_gamma2_vs_test.log" \
    python3 train_roberta_detector.py \
      --gamma "$gamma2" \
      --benign "$test_sample" \
      --epochs 3 \
      --bsz 8 \
      --lr 2e-5
}

# External Run 1: Reference = Alice + Pride, Test = Sherlock
run_external 1 "alice.txt" "pride_prejudice.txt" "sherlock.txt" "alice_pride_to_sherlock" 42

# External Run 2: Reference = Alice + Sherlock, Test = Pride
run_external 2 "alice.txt" "sherlock.txt" "pride_prejudice.txt" "alice_sherlock_to_pride" 42

# External Run 3: Reference = Pride + Sherlock, Test = Alice
run_external 3 "pride_prejudice.txt" "sherlock.txt" "alice.txt" "pride_sherlock_to_alice" 42

echo "External leave-one-book-out validation complete."
