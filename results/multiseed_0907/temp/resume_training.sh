#!/usr/bin/env bash
set -euo pipefail
cd /home/pjpjq/LEO_switch

# Run inside tmux with the original training environment activated.
export CUDA_VISIBLE_DEVICES=0
export MPLCONFIGDIR=/home/pjpjq/LEO_switch/results/multiseed_0907/temp
extra=()
if [[ ${1:-} == --dry-run ]]; then
  extra=(--dry-run)
elif [[ $# -gt 0 ]]; then
  echo 'Usage: bash resume_training.sh [--dry-run]' >&2
  exit 2
fi

common=(
  --run-id additional_4seeds_150k
  --results-root /home/pjpjq/LEO_switch/results/multiseed_0907
  --pretrained-han-path '/home/pjpjq/LEO_switch/results/full_train_latency_priority_multiuser_u{num_users}_multiuser_single_seed_150k_20260804/best_model.pt'
  --total-timesteps 150000 --max-steps 512 --batch-size 512
  --pdqn-lr 0.001 --learning-rate 0.0001 --n-steps 1024 --n-epochs 4
  --eval-interval 25000 --eval-episodes 5 --compare-episodes 5
  --save-interval 50000 --early-stop-patience 0 --graph-update-interval 1
  --reward-load-balance-weight 0.05
  --best-model-metric reward --compare-ranking-metric reward
  --baselines han_mappo mappo_no_han maddpg pdqn min_distance full_local
  --reuse-learned-checkpoints --device cuda
)

# Resume the interrupted comparison; reuse the three finished HAN/MAPPO models.
python -u scripts/run_multiuser_scaling_suite.py "${common[@]}" \
  --seeds 46 --user-counts 25 "${extra[@]}"

# These twelve combinations had not started when this script was prepared.
python -u scripts/run_multiuser_scaling_suite.py "${common[@]}" \
  --seeds 43 44 45 46 --user-counts 30 35 40 "${extra[@]}"

# Include all seven previously finished combinations in the final aggregation.
if [[ ${#extra[@]} -eq 0 ]]; then
  python -u scripts/run_multiuser_scaling_suite.py "${common[@]}" \
    --seeds 43 44 45 46 --user-counts 20 25 30 35 40 --aggregate-only
fi
