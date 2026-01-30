#!/bin/bash
set -euo pipefail

export JAX_DEFAULT_MATMUL_PRECISION=highest

mkdir -p ./logs ./results

python learning/train_jax_ppo.py \
  --env_name=HandManipulateBlock-v0 \
  --num_envs=16 \
  --total_steps=5000000 \
  --output_dir=./results/run1 \
  2>&1 | tee ./logs/ppo_run1.log
