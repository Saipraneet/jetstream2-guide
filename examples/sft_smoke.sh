#!/usr/bin/env bash
# 3-step SFT smoke test: proves the fine-tuning env works before you spend real GPU hours.
# Usage: examples/sft_smoke.sh [instance]   (after: js2 launch mybox a100-10 && js2 setup mybox sft)
# Costs ~5 min of instance time.
set -euo pipefail
inst=("$@")
js2 ft "${inst[@]}" defaults sft --model Qwen/Qwen3-1.7B
js2 ft "${inst[@]}" train sft --model Qwen/Qwen3-1.7B --dataset trl-lib/Capybara --max-steps 3 --output /workspace/experiments/smoke
