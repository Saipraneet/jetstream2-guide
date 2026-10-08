# Reference

The [README](../README.md) is the path for a first session. This page has the rest.

## All `js2` commands

```bash
js2 launch [name] [flavor] [volume]  # flavor: a100-10 (default) | a100-20 | a100 | h100 | cpu | tiny
                                     # volume: new (default, 100 GB) | new:500 | <existing volume name> | none
js2 list                             # your instances: name, status, flavor, IP, volumes
js2 status [inst]                    # one instance in detail
js2 gpus [inst]                      # GPU name and memory
js2 setup [inst] [venv|sft|docker]   # venv (default): vLLM + torch; sft: + TRL, PEFT, bitsandbytes, Unsloth
js2 ssh [inst] [cmd...]              # log in, or run one command
js2 push-code [inst]                 # source files laptop -> instance, checksummed
js2 pull-results [inst] [subpath]    # everything that is not source, instance -> laptop, checksummed
js2 serve [inst] <model> [vllm args] # vLLM server in tmux + tunnel to http://localhost:8000/v1
js2 tunnel [inst]                    # reopen that tunnel
js2 ft [inst] defaults|bench|train sft|dpo|grpo ...   # fine-tuning (needs `setup sft`)
js2 stop [inst]                      # SHELVE: billing stops; disk, volume, IP kept
js2 start [inst]                     # unshelve and remount /workspace
js2 finish [inst]                    # pull-results, verify, delete the instance (volume survives)
js2 kill [inst]                      # delete the instance without pulling (volume survives)
js2 vol list|create <name> [GB]|attach [inst] <vol>|detach [inst] <vol>|delete <vol>
```

`[inst]` can be omitted when you have exactly one instance. Timings: launch ~2-3 min, `setup` 1-6 min, shelve
30 s, unshelve 1 min. Each command takes a few seconds because every OpenStack call does.

**Config.** `~/.jetstream2/js2.conf` holds account defaults (`KEY_NAME`, `SSH_KEY`, `FLAVOR`, `VOLUME_GB`, see
`js2.conf.example`). A `.js2.conf` file in your project can set `EXP_LOCAL` (the folder to sync, default
`<repo>/experiments` if it exists, else the repo root) and `EXP_REMOTE` (its path on the instance, default
`/workspace/<repo name>`).

**Volumes.** `js2 launch mybox` creates a volume `mybox-vol` mounted at `/workspace`. It outlives the instance:
`js2 launch mybox2 a100 mybox-vol` attaches the same disk to a bigger machine with the environment and model
cache already on it. Team quota is 1000 GB across all volumes. `js2 up` and `js2 down` sync whole trees in
either direction and can overwrite things; prefer `push-code` / `pull-results`.

## The machine

Ubuntu 24.04 with the NVIDIA driver (CUDA 13), Python 3.12, docker, tmux, rsync, and nothing else. The 60 GB
root disk has about 11 GB free, so `js2 setup` puts the Python environment, the `uv` cache and the Hugging Face
cache on the volume. `/workspace/env.sh` activates the environment and sets `HF_HOME=/workspace/hf`; add
`export HF_TOKEN=hf_...` to it for gated models (Llama, Gemma). Install extra packages with
`uv pip install <pkg>` after sourcing it. The GPU slices (`a100-10`, `a100-20`) are vGPU profiles: a normal CUDA
device with 10 or 20 GB, of which about 1.3 GB is reserved by the hypervisor.

## Serving a model (inference, evals, agents)

```bash
js2 serve mybox Qwen/Qwen3-8B --max-model-len 4096     # on a100-20 or bigger
python examples/query_server.py "Explain the KV cache."
```

`js2 serve` starts vLLM in a tmux session on the instance, adds GPU-appropriate memory and batching flags, and
opens an SSH tunnel so the OpenAI-compatible API is at `http://localhost:8000/v1` on your laptop. Any OpenAI
client works with `base_url="http://localhost:8000/v1"` and any API key. Always pass `--max-model-len` sized to
your task (longest prompt + output); the model's own maximum wastes memory on unused context. Send many requests
concurrently (asyncio) or the GPU sits idle; vLLM prints "Maximum concurrency for N tokens: Cx" at startup.
Rough fits: 10 GB slice up to 3B bf16 (8B in 4-bit), 20 GB up to 8B, A100 40 GB up to 14B bf16, H100 up to
32B bf16 or 70B FP8. Run models at their released precision; do not re-quantize.

## Fine-tuning (SFT, DPO, GRPO)

```bash
js2 setup mybox sft                                        # ~6 min once
js2 ft mybox defaults sft --model Qwen/Qwen3-1.7B          # the config it would use for this GPU + model
js2 ft mybox train sft --model Qwen/Qwen3-1.7B --dataset data.jsonl --max-steps 3   # smoke test
js2 ft mybox train sft --model Qwen/Qwen3-1.7B --dataset data.jsonl                 # real run (in tmux)
js2 ft mybox train dpo  --dataset pairs.jsonl              # fields: prompt, chosen, rejected
js2 ft mybox train grpo --dataset prompts.jsonl --reward reward.py:score
```

`js2 ft` picks batch size, accumulation, LoRA settings and vLLM memory from the GPU and model size using
measured throughput (table in [SKILL.md](../.agents/skills/jetstream2-experiments/SKILL.md)); any field can be
overridden with `--set key=value`. Datasets are Hugging Face ids or local `.jsonl` files (chat `messages` or
`text` for SFT). Output lands in `/workspace/experiments/out` (`--output` changes it) and comes back with
`js2 pull-results`. `examples/sft_smoke.sh mybox` is a complete 3-step test. On the 10 GB and 20 GB slices use
models of 4B parameters or less; LoRA on an 8B model needs the 40 GB A100, and GRPO on 8B needs the H100.

## Using a coding agent

`.agents/skills/jetstream2-experiments/SKILL.md` teaches a coding agent (Claude Code, Codex, pi, Cursor and
similar) how to size, launch, run and shut down Jetstream2 instances, with the budget rules built in.

- Open this repo in the agent: `AGENTS.md` and `CLAUDE.md` load the skill automatically.
- To use it from your project repo, copy or symlink `.agents/skills/jetstream2-experiments` into your project's
  `.agents/skills/` (and `.claude/skills/` for Claude Code), or into `~/.agents/skills/`, `~/.claude/skills/`
  or `~/.codex/skills/` to have it everywhere.

Tell the agent to state the flavor and the credit estimate before every launch, and to `js2 stop` when done.
