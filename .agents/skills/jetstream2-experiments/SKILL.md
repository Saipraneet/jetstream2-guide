---
name: jetstream2-experiments
description: Launch, size, and run GPU experiments (vLLM inference, LLM/agentic benchmarks, SFT/DPO/GRPO fine-tuning, model-internals work) on Jetstream2 (NSF ACCESS OpenStack cloud) instances from any project. Use when the user wants to run on Jetstream2, start/stop (shelve) an instance, create/attach/mount a volume, pick an A100 slice vs full A100 vs H100 flavor, sync code or results, serve a model, fine-tune with measured defaults, or set up the env (venv, docker, or the fine-tuning venv) on a bare Jetstream2 image.
---

# Jetstream2 experiments

**Instances burn ACCESS allocation (SUs) while ACTIVE or SHUTOFF; only a SHELVED instance is free.** If your
instructions require confirmation before paid or GPU jobs, ask before `js2 launch` and say which flavor and
roughly how long. `js2 stop` shelves; `js2 kill`/`js2 finish` delete. Never leave an idle GPU instance ACTIVE.

**Course budget (CSCI 567): each team has a hard cap of 10,000 SUs, about 90 GPU-hours. There is no top-up.**
Before launching, estimate SUs = rate x hours (h100 128/h, a100 64/h, a100-20 32/h, a100-10 16/h) and say it.
Prefer the smallest flavor that fits; develop on `a100-10`/`a100-20` and move to a big card only for the real run.
Check spend in Exosphere (Allocations page) before any run over an hour. See the repo README for details.

Three files in this skill's `scripts/`: `js2` (symlinked to `~/bin/js2` by the repo's `install.sh`; call it by
full path if `~/bin` is not on PATH) wraps the `openstack` CLI (`~/bin/openstack`, cloud from `~/.config/openstack/clouds.yaml`);
`js2_remote.sh` is what `js2` runs on the instance (volume mount, env setup); `js2_ft.py` is fine-tuning
(defaults, benchmark, train) and runs on the instance via `js2 ft`.

**The images are barebones.** Featured-Ubuntu24 has the NVIDIA driver (595, CUDA 13.2), Python 3.12, docker,
tmux, rsync and nothing else: no torch, vLLM, uv or `nvcc`. The 60 GB root disk has ~11 GB free, so every
environment, model and image store lives on a volume mounted at `/workspace`. `js2 launch` creates and mounts one.

## Quick start

```bash
js2 launch [name] [flavor] [volume]  # flavor h100 | a100 | a100-20 | a100-10 | cpu | tiny | <name>
                                     # volume new (default, 250 GB) | new:500 | <existing volume> | none
js2 setup [inst] [venv|docker|sft]   # venv: vLLM for serving (default); docker: vllm/vllm-openai; sft: fine-tuning
js2 push-code                        # source files only, verified
js2 gpus                             # what you got - size the run from this
js2 ssh                              # long jobs in tmux, tee'd to a log on /workspace; env: source /workspace/env.sh
js2 serve <model> --max-model-len N  # vLLM in tmux + tunnel to localhost:8000, GPU-aware defaults (table below)
js2 ft defaults|bench|train sft|dpo|grpo ...   # fine-tuning with measured defaults (section below)
js2 stop / js2 start                 # shelve (free, ~30 s) / unshelve + remount (~1 min)
js2 finish                           # pull-results + verify + delete (volume survives)
js2 vol list|create|attach|detach|delete
```

`[inst]` is optional when one instance exists. `js2` alone prints every verb. Measured: launch with a new
volume 1m46s, `setup venv` ~4 min, `setup sft` ~6 min, shelve 33 s, unshelve 68 s.

## Flavors, and what fully uses them

| alias | flavor | GPU | vCPU / RAM | fits | `js2 serve` adds |
|---|---|---|---|---|---|
| `h100` | g5.xl | 1x H100 80 GB SXM | 20 / 235 GB | <=70B FP8/4-bit, <=32B bf16 | util 0.95, seqs 1024, batched 16384 |
| `a100` | g3.xl | 1x A100 **40 GB** | 32 / 117 GB | <=14B bf16, <=32B FP8 | util 0.92, seqs 512, batched 4096 |
| `a100-20` | g3.large | A100 20 GB vGPU slice | 16 / 60 GB | <=8B bf16, <=14B 4-bit | util 0.90, seqs 256, batched 4096 |
| `a100-10` | g3.medium | A100 10 GB vGPU slice | 8 / 30 GB | <=3B bf16, <=8B 4-bit | util 0.90, seqs 128, batched 2048 |
| `cpu` / `tiny` | m3.medium / m3.tiny | none | 8 / 30 GB, 1 / 3 GB | orchestration / lifecycle tests | |

- `js2 serve` adds the three flags only when you did not pass them, and prints the final command. Always set
  `--max-model-len` from the task (longest prompt + output cap, next power of two): never the model maximum,
  since unused context is KV cache taken from concurrency. vLLM logs "Maximum concurrency for L tokens: Cx";
  make the client send >= C requests at once, and check `nvidia-smi` util after warm-up.
- **Multi-GPU:** this allocation exposes single-GPU flavors only. If a multi-H100 flavor appears in
  `openstack flavor list`, pass its name; `serve` then adds `--tensor-parallel-size N` (use `--data-parallel-size`
  if the model fits one card). Otherwise scale out: N `h100` instances, `JS2_LOCAL_PORT=800N js2 serve run-N ...`.
- Slices are vGPU profiles, not MIG: a plain CUDA device with 10/20 GB. Quota: 214 cores (~10 H100 instances),
  25 instances, 10 volumes / 1000 GB.
- Run each model at its released precision (FP8/MXFP4/...): no `--dtype`, never re-quantize; record what loaded.

**Measured, g5.xl, Qwen3-8B bf16, `--max-model-len 4096`, 512 concurrent requests (vLLM 0.31):** 2.2k output
tok/s at 1024-in/128-out (prefill-bound, ~29k prefill tok/s), 8.5k output tok/s at 512/512 (38 ms TPOT).
That is a healthy H100; a run far below it is client-bound or misconfigured.

## Fine-tuning: `js2 ft` (venv_sft: vLLM 0.31 / torch 2.13, TRL 1.13, PEFT 0.21, bitsandbytes, Unsloth 2026.10)

```bash
js2 ft defaults grpo --model Qwen/Qwen3-8B [--gpu-gb 40]     # the config it would use (no GPU needed with --gpu-gb)
js2 ft bench sft --model M --seq-len 2048 [--batch 4,8,16]    # largest batch that fits + tok/s (also dpo, grpo)
js2 ft train sft --dataset D [--full] [--max-steps K] [--set key=value ...]
js2 ft train dpo --dataset D                                   # prompt/chosen/rejected
js2 ft train grpo --dataset D --reward reward.py:fn [--max-completion 512] [--max-prompt 512]
```

`defaults` picks batch sizes from GPU memory and model size using the peaks measured below; `train` uses them
(`--set` overrides any field) and runs in the foreground, so launch it inside `js2 ssh` + tmux for long jobs.
Datasets are HF ids or local `.jsonl` (chat `messages`/`text` for SFT; `prompt`,`chosen`,`rejected` for DPO;
`prompt` for GRPO). Smoke the real config with `--max-steps 3` before a long run.

**Policy: Unsloth for SFT and DPO (2x PEFT's tokens/s at a third of the memory); GRPO on either backend with a
large generation round; DAPO loss by TRL default.** Measured on g5.xl, Qwen3-8B, LoRA r=16, 2026-10-07:

| task / config | throughput | peak GB | default |
|---|---|---|---|
| SFT unsloth LoRA, 2048 ctx | 10.9k tok/s (flat from batch 4) | 19-27 | batch 8, accumulate to 64; 10.4k at 4096 (21 GB), 9.4k at 8192 |
| SFT unsloth QLoRA (NF4) | 10.8k tok/s | 11-19 | same speed; use on slices / to co-host |
| SFT unsloth full FT (`--full`, adamw_8bit) | 7.6k tok/s | 48 | batch 8; 80 GB card, <=8B |
| SFT PEFT LoRA / QLoRA / full | 5.5k / 3.2k / 6.7k tok/s | 53 / 55 / 68 | fallback bar only |
| SFT Qwen3-1.7B full (fp32 fused AdamW) | 22.8k tok/s | 44 | small models: full FT beats LoRA |
| DPO unsloth LoRA, 2048-token pairs, ref = adapter off | 1.8 pairs/s, 7.3k tok/s (flat batch 2-8) | 30 / 44 / 73 | batch 4, accumulate to 32, beta 0.1 |
| DPO PEFT | 1.2 pairs/s | 49 | |
| GRPO, 8 gens, 512-token completions, 128-token prompts: **batch 16 x `steps_per_generation` 8 (128 per vLLM call), engine 0.5** | 5.6 completions/s, 2.8k gen tok/s | 70 | the default round; `defaults` gives 8 x 16 for 512-token prompts |
| GRPO untuned (32 per call, engine 0.35) | 1.1 completions/s | 34 | don't |
| GRPO Unsloth with its compile on | 0.8 completions/s | 66 | `UNSLOTH_COMPILE_DISABLE=1` (set by `js2 ft`) |
| GRPO batch 8 x 32 (256 per call), engine 0.6 | 6.1 completions/s | 76 | +10% for 10 GB; not default |

- **GRPO is generation-bound:** a round of 128 completions takes 16 s, a train step 1.1 s. The generation
  batch (`per_device_train_batch_size x steps_per_generation`) is the lever; 32 per call is latency-bound.
  Run time scales ~linearly with `max_completion_length`, so budget it from the task.
- **Unsloth's GRPO trainer torch.compiles its log-prob kernel and recompiles per completion-length shape**
  (py-spy: minutes inside Inductor, 0% GPU, "hung at step 0"). `js2 ft` sets `UNSLOTH_COMPILE_DISABLE=1`, after
  which it matches TRL's colocate path. TRL+PEFT (`--backend peft`) needs no flag.
- **DAPO is TRL's default and stays the default:** `loss_type="dapo"`, `beta=0.0` (no reference model),
  `epsilon=0.2`, token-level importance sampling, group-scaled rewards. Opt-in extras from the DAPO paper:
  `--set epsilon_high=0.28 --set mask_truncated_completions=True`.
- QLoRA + colocated vLLM is unsupported (TRL 1.13 rejects the bitsandbytes model): on slices run GRPO on a bf16
  model that fits (<=4B on 20 GB, <=1.7B on 10 GB), or `vllm_mode="server"` against `js2 serve` on another
  instance. On a 2-GPU flavor put vLLM on GPU 1 the same way so generation and training overlap.
- Per-flavor starting points (`js2 ft defaults` computes them): H100: as above. A100 40 GB: SFT LoRA batch 8,
  DPO batch 2, GRPO batch 4 x 32 with engine 0.4, ~0.4x the H100 rates. 20 GB slice: QLoRA SFT batch 8, DPO
  batch 1-2, GRPO only <=4B. 10 GB slice: QLoRA SFT batch 2-4 for <=4B (8B at batch 1), no DPO/GRPO on 8B.
- Keep gradient checkpointing on (off: +30% speed, OOM at 4 x 2048 on 8B). Pack sequences (`packing=True`,
  default). Effective batch = per-device x accumulation; pick per-device from memory, the rest by accumulation.
- Record GPU, method, base dtype/quant, r, seq len, batch, accumulation, optimizer and measured tok/s in the
  results metadata; sanity-check the first losses against a known run.

## Environments (all on the volume; `/workspace/env.sh` is never overwritten: add `HF_TOKEN` there)

- **venv** (`/workspace/venv`): uv, vLLM, torch, datasets, accelerate. `env.sh` sets `HF_HOME=/workspace/hf`,
  `HF_HUB_DISABLE_XET=1`, `VLLM_USE_FLASHINFER_SAMPLER=0` (no `nvcc`: flashinfer's JIT sampler crashes vLLM at
  warm-up without it). For hooks / nnsight use this venv (vLLM exposes no activations).
- **sft** (`/workspace/venv_sft`): vLLM installed first (pins torch), then TRL, PEFT, bitsandbytes, Unsloth.
  Extra packages: `JS2_EXTRA_PIP="..." js2 setup sft`, or `uv pip install` after activating (uv in `~/.local/bin`).
- **docker**: the image ships nvidia-container-toolkit; setup wires the runtime, bind-mounts `/var/lib/containerd`
  (docker's image store, shared with Exosphere's web-shell containers) onto `/workspace/containerd`, pulls
  `vllm/vllm-openai` (32 GB unpacked). `js2 serve` uses docker when no venv exists or with `JS2_SERVE_MODE=docker`
  (writes `/workspace/vllm_docker.sh`: 127.0.0.1 only, HF cache and token mounted).
- Everything survives `js2 stop`/`start` and `js2 kill` + `js2 launch <name> h100 <old-volume>` (an existing
  volume is mounted, never reformatted; only blank volumes get mkfs).

## Running jobs and syncing 

- Long jobs: `tmux` + `tee` to a log on `/workspace`; verify the session exists and the log grows. Wait from the
  agent with your harness's background mechanism, never a sleep loop or `pgrep -f "<cmd>"`. To stop a run, kill
  the tmux session AND `pkill -f` the script; orphaned python children keep writing results.
- `js2 serve` binds vLLM to 127.0.0.1 and tunnels it (`js2 tunnel` reopens; `JS2_LOCAL_PORT` changes the port).
  The `exosphere` security group opens only 22/tcp inbound; never expose the server port.
- **Code goes up, results come down, never the reverse.** `push-code` sends only
  `*.py|sh|md|txt|toml|cfg|yaml|yml` and verifies by checksum; `pull-results` pulls everything else and
  auto-reverts a pulled tracked file that matches a commit older than HEAD (stale data file on the volume).
  `finish` pulls, verifies, then deletes: never `kill` an instance whose results are not verified locally.
  `up`/`down` sync whole trees (use `up` only to seed a volume).
- Preflight: smoke the real call path on the instance (its library versions differ from the lab box), assert
  model/dataset ids are fetchable (probe `config.json`), validate first outputs, make parsers fail loudly.

## Configuration

`~/.jetstream2/js2.conf` (account): `OS_CLOUD`, `KEY_NAME` (OpenStack keypair), `SSH_KEY`
(default `~/.ssh/id_ed25519`), `IMAGE` (Featured-Ubuntu24), `VOLUME_GB` (250), `VOLUME` (`new`), `NETWORK`,
`SECGROUPS` (`exosphere`), `FLAVOR` (default flavor; the course config sets `a100-10`). Nearest `.js2.conf` upward from cwd (project): `EXP_LOCAL` (default
`<git root>/experiments` if present, else the git root), `EXP_REMOTE` (default `/workspace/<project dirname>`).

**First-time setup:** follow the repo README (ACCESS account, Exosphere application credential saved to
`~/.config/openstack/clouds.yaml`, SSH keypair uploaded, then `./install.sh`, which installs the client and
writes `~/.jetstream2/js2.conf`). Verify with `js2 list`, `js2 launch test tiny new:10`, `js2 stop`, `js2 start`,
`js2 finish test`, `js2 vol delete test-vol` (costs ~1 SU).

## Gotchas

- **Keypair mismatch = locked out:** an instance launched elsewhere with a keypair whose private key is not here
  cannot be reached by `js2 ssh` (no root password). `js2` can still shelve or delete it.
- **Shelve, don't stop:** `openstack server stop` still bills. Shelved instances hide their volume in `js2 list`
  (`js2 vol list` still shows it); `js2 start` remounts via the `nofail` UUID fstab line.
- Floating IPs are reused across tenants, so `js2` keeps its own `~/.jetstream2/known_hosts`.
- `js2 launch <name>` creates `<name>-vol`; to reattach an old volume pass its name as the third argument.
- `ensurepip` makes only `pip3` in a venv; setup symlinks `pip` so a bare `pip install` cannot hit the system
  python (`env.sh` warns if it would).
- The `openstack` CLI takes 2-4 s per call; a `js2` verb taking 10-20 s is normal. `/software` (ceph, lmod) and
  `/opt` are image tooling; ignore them.
