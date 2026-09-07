#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_EXECUTABLE="${PYTHON_EXECUTABLE:-/home/pjpjq/miniconda3/envs/satellite.env/bin/python}"
DEVICE="${DEVICE:-cuda}"
RUN_ID="multiuser_single_seed_150k_20260804"
SUITE_DIR="${PROJECT_ROOT}/results/baseline_compare/multiuser_scaling_${RUN_ID}"
USER_COUNTS=(20 25 30 35 40)

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/leo_switch_matplotlib}"
mkdir -p "${MPLCONFIGDIR}"

if [[ "${DEVICE}" == "cuda" ]]; then
    "${PYTHON_EXECUTABLE}" -c \
        'import torch; assert torch.cuda.is_available(), "CUDA is unavailable; set DEVICE=cpu explicitly to run on CPU"'
fi

cd "${PROJECT_ROOT}"

for num_users in "${USER_COUNTS[@]}"; do
    system_run_dir="${PROJECT_ROOT}/results/full_train_latency_priority_multiuser_u${num_users}_${RUN_ID}"
    output_dir="${SUITE_DIR}/u${num_users}"
    existing_summary="${output_dir}/comparison_summary.json"
    loop_log="${SUITE_DIR}/u${num_users}_seed42_maddpg.log"

    if [[ ! -f "${system_run_dir}/final_model.pt" || ! -f "${existing_summary}" ]]; then
        echo "Missing system checkpoint or existing summary for U${num_users}" >&2
        exit 1
    fi

    echo "=== Training MADDPG: U${num_users}, seed 42, 150000 steps ==="
    "${PYTHON_EXECUTABLE}" scripts/compare_system_baselines.py \
        --run-mode compare_only \
        --system-run-dir "${system_run_dir}" \
        --output-dir "${output_dir}" \
        --episodes 5 \
        --max-steps 512 \
        --total-timesteps 150000 \
        --maddpg-timesteps 150000 \
        --seed 42 \
        --device "${DEVICE}" \
        --num-users "${num_users}" \
        --best-model-metric reward \
        --compare-ranking-metric reward \
        --plot-window 3 \
        --early-stop-patience 0 \
        --reuse-methods-from "${existing_summary}" \
        --baselines maddpg \
        2>&1 | tee -a "${loop_log}"
done

"${PYTHON_EXECUTABLE}" scripts/run_multiuser_scaling_suite.py \
    --aggregate-only \
    --run-id "${RUN_ID}" \
    --user-counts "${USER_COUNTS[@]}" \
    --python-executable "${PYTHON_EXECUTABLE}" \
    --seed 42 \
    --device "${DEVICE}" \
    --total-timesteps 150000 \
    --max-steps 512 \
    --n-steps 1024 \
    --batch-size 512 \
    --learning-rate 0.0001 \
    --n-epochs 4 \
    --eval-interval 25000 \
    --eval-episodes 5 \
    --save-interval 50000 \
    --reward-load-balance-weight 0.05 \
    --compare-episodes 5 \
    --plot-window 3 \
    --early-stop-patience 0

echo "MADDPG multi-user loop completed: ${SUITE_DIR}"
