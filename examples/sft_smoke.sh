#!/usr/bin/env bash
# 3-step SFT smoke test: proves the fine-tuning env works before you spend real GPU hours.
# Costs ~5 min of instance time. Run after: js2 launch mybox a100-10 && js2 setup sft
set -euo pipefail
js2 ft defaults sft --model Qwen/Qwen3-1.7B
js2 ft train sft --model Qwen/Qwen3-1.7B --dataset trl-lib/Capybara --max-steps 3 --output /workspace/experiments/smoke
