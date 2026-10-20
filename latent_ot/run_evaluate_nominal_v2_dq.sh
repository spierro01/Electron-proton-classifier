#!/usr/bin/env bash
# Evaluate frozen-head PID efficiencies, then train domain discriminators.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
source "$HOME/.venvs/latent-ot/bin/activate"

OUT="results/nominal_v2_dq_evaluation_$(date +%Y%m%d_%H%M%S)"
mkdir -p logs
LOG="logs/$(basename "$OUT").log"
printf 'Evaluation output: %s\nLog file: %s\n' "$OUT" "$LOG"

python -u evaluate_nominal_v2_dq.py \
    --maps results/nominal_v2_dq_20261015_115613 \
    --baseline ../transformer_no_veto_v2/results/task0_selected_v2_dq_20261015_113235 \
    --ckpt ../transformer_no_veto_v2/checkpoints/pid_spectra_v2_no_veto_selected_20261013_181104/best.pt \
    --mc derived_v2_dq/latents_MC.npz \
    --tb derived_v2_dq/latents_TB.npz \
    --mask derived_v2/tb_is_derivation_v2_dq.npy \
    --device cuda --batch-size 256 --threads 8 \
    --stage all --out "$OUT" \
    2>&1 | tee "$LOG"

printf '\nEfficiency and closure evaluations completed: %s\n' "$OUT"
