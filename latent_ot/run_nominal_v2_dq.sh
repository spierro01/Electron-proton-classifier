#!/usr/bin/env bash
# Validate selected archives, run a smoke test, then train three nominal maps.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
source "$HOME/.venvs/latent-ot/bin/activate"

RUN_ROOT="results/nominal_v2_dq_$(date +%Y%m%d_%H%M%S)"
if [[ -e "$RUN_ROOT" ]]; then
    printf 'Output already exists: %s\n' "$RUN_ROOT" >&2
    exit 1
fi
mkdir -p "$RUN_ROOT"
exec > >(tee "$RUN_ROOT/run.log") 2>&1
printf 'Run directory: %s\n' "$RUN_ROOT"

COMMON=(
    --mc derived_v2_dq/latents_MC.npz
    --tb derived_v2_dq/latents_TB.npz
    --mask derived_v2/tb_is_derivation_v2_dq.npy
    --device cuda --hidden 2048 --layers 4 --input-modulation
    --batch-size 1024 --seed 42
)

for cls in e p C; do
    python -u train_conditional_ot_v2_dq.py "${COMMON[@]}" \
        --class-name "$cls" --check-only --out "$RUN_ROOT/$cls"
done

python -u train_conditional_ot_v2_dq.py "${COMMON[@]}" \
    --class-name e --steps 20 --identity-steps 10 --log-every 1 \
    --out "$RUN_ROOT/smoke_e"

for cls in e p C; do
    python -u train_conditional_ot_v2_dq.py "${COMMON[@]}" \
        --class-name "$cls" --steps 10000 --identity-steps 1000 \
        --log-every 100 --out "$RUN_ROOT/$cls"
done

printf '\nAll three nominal training runs completed: %s\n' "$RUN_ROOT"
