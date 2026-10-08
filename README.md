# Jetstream2 for the CSCI 567 class project

Jetstream2 is an NSF cloud (part of ACCESS) that gives you a Linux VM with a dedicated NVIDIA A100 or H100.
This repo is everything you need to run your project on it: a one-time setup, a small command-line helper
(`js2`) that launches, stops and syncs instances, and ready-made environments for vLLM inference and
SFT / DPO / GRPO fine-tuning. It also doubles as a skill for coding agents (see [Using a coding agent](#using-a-coding-agent)).

## The budget, read this first

**Each team has a hard cap of 10,000 SUs, which is about 90 GPU-hours. When it is gone, it is gone.**
There is no top-up, so treat the allocation like a lab budget.

| flavor alias | GPU | SU / hour | hours you get for 10,000 SU |
|---|---|---|---|
| `a100-10` | 1/4 A100 (10 GB) | 16 | 625 |
| `a100-20` | 1/2 A100 (20 GB) | 32 | 312 |
| `a100` | full A100 (40 GB) | 64 | 156 |
| `h100` | full H100 (80 GB) | 128 | 78 |
| `cpu` / `tiny` | none | 8 / 1 | n/a (debugging only) |

Rates are from the [Jetstream2 flavor table](https://docs.jetstream-cloud.org/general/vmsizes/).
Rules that keep you within budget:

1. **An instance bills while it is ACTIVE or SHUTOFF. Only a SHELVED instance is free.** `js2 stop` shelves.
   Never log off with a GPU instance running; a forgotten H100 burns 3,000 SUs overnight.
2. **Develop small, run big.** Debug your pipeline on `a100-10` (16 SU/h); move to `a100` or `h100` only for
   the runs you will report. Smoke every run with a few steps (`--max-steps 3`) before the long one.
3. **Estimate before launching.** SUs = rate x hours. Say the number out loud (or have your agent say it).
4. **Check spend weekly** in [Exosphere](https://jetstream2.exosphere.app) under your allocation. The cap is
   enforced by the course, not by the cloud, so nothing stops you at 10,000 except you.
5. One instance at a time per team unless you have a reason. Volumes (disk) are cheap and persist across
   instances, so you lose nothing by deleting an instance at the end of a session.

## One-time setup (about 20 minutes)

1. **ACCESS account and allocation.** Create an account at <https://allocations.access-ci.org> (use your USC
   email), then send your ACCESS username to the course staff. We add you to the course allocation; you get an
   email when that is done.
2. **Exosphere credentials.** Log in to <https://jetstream2.exosphere.app> with your ACCESS account, pick the
   course allocation, then create an *application credential*: Exosphere shows an "OpenStack CLI / clouds.yaml"
   option under your allocation's settings (Horizon works too: *Identity > Application Credentials > Create*,
   then *Download clouds.yaml*). Save the file as `~/.config/openstack/clouds.yaml`. Treat it like a password.
3. **SSH keypair.** If you do not have one, run `ssh-keygen -t ed25519`. Upload the public key
   (`~/.ssh/id_ed25519.pub`) in Exosphere under *SSH public keys* and note the name you gave it.
4. **Install the helper.**
   ```bash
   git clone https://github.com/Saipraneet/jetstream2-guide.git
   cd jetstream2-guide && ./install.sh
   ```
   This creates a small Python venv with the `openstack` CLI, symlinks `openstack` and `js2` into `~/bin`,
   and writes `~/.jetstream2/js2.conf`. Open that file and set `KEY_NAME` to the keypair name from step 3.
   Windows users: do this inside WSL.
5. **Verify with a 1-SU test:**
   ```bash
   js2 list                          # empty table = credentials work
   js2 launch test tiny new:10       # ~2 min: tiny CPU VM + 10 GB volume
   js2 ssh test 'df -h /workspace'   # the volume is mounted
   js2 stop test && js2 start test   # shelve (free) and resume
   js2 finish test                   # delete the instance
   js2 vol delete test-vol           # delete the test volume
   ```

## Daily workflow

```bash
js2 launch mybox a100-10          # instance + 100 GB volume mounted at /workspace (~2 min)
js2 setup mybox                   # venv with vLLM + torch on the volume (~4 min); or `setup mybox sft` for fine-tuning
js2 push-code                     # your project's source files -> /workspace/<project>/ (checksummed)
js2 ssh                           # work on the instance; `source /workspace/env.sh` activates the env
js2 pull-results                  # results (everything that is not source) -> local
js2 stop                          # SHELVE when you step away: billing stops, disk and volume stay
js2 start                         # resume later (~1 min); /workspace is remounted
js2 finish                        # pull-results + verify + delete the instance; the volume survives
```

`js2` with no arguments lists every verb. The instance name can be omitted when you have exactly one.
Run `js2` from inside your project repo: `push-code` and `pull-results` sync `<repo>/experiments/` if it exists,
otherwise the repo root (override with a `.js2.conf` in the repo, see `js2.conf.example`).

**Long jobs:** run them in `tmux` on the instance and `tee` to a log on `/workspace`, so an SSH drop does not kill
them. Then `js2 stop` is **not** what you want while a job runs (shelving suspends everything); shelve only when
the job is done.

**The image is bare.** Ubuntu 24.04 with the NVIDIA driver, Python 3.12, docker, tmux, rsync and nothing else.
The root disk has about 11 GB free, so `js2 setup` puts the environment and the Hugging Face cache on the volume.
Add your own packages with `uv pip install ...` after `source /workspace/env.sh`, or put your Hugging Face token
in `/workspace/env.sh` (`export HF_TOKEN=...`) for gated models.

## Serving a model (inference, evals, agents)

```bash
js2 serve mybox Qwen/Qwen3-8B --max-model-len 4096
python examples/query_server.py "Explain the KV cache."
```

`js2 serve` starts vLLM in tmux on the instance, picks GPU-appropriate batching flags, and opens an SSH tunnel
so the OpenAI-compatible API is at `http://localhost:8000/v1` on your laptop (`js2 tunnel` reopens it). Always
pass `--max-model-len` sized to your task; the model's default maximum wastes memory on unused context.
Rough fits: 10 GB slice up to 3B (8B in 4-bit), 20 GB up to 8B, A100 40 GB up to 14B, H100 up to 32B bf16 or 70B
FP8. Send many requests concurrently or the GPU idles.

## Fine-tuning (SFT, DPO, GRPO)

```bash
js2 setup mybox sft                                        # venv with TRL, PEFT, bitsandbytes, Unsloth (~6 min)
js2 ft defaults sft --model Qwen/Qwen3-1.7B                # the config it would use for this GPU + model
js2 ft train sft --dataset your_data.jsonl --max-steps 3   # smoke test first
js2 ft train sft --dataset your_data.jsonl                 # the real run (inside js2 ssh + tmux)
js2 ft train dpo  --dataset pairs.jsonl                    # fields: prompt, chosen, rejected
js2 ft train grpo --dataset prompts.jsonl --reward reward.py:score
```

`js2 ft` picks batch sizes from GPU memory and model size using measured numbers (see the table in
[SKILL.md](.agents/skills/jetstream2-experiments/SKILL.md)); `--set key=value` overrides any field.
Datasets are Hugging Face ids or local `.jsonl`. `examples/sft_smoke.sh` is a complete 3-step test.
On the 10 GB and 20 GB slices use models of 4B parameters or less; LoRA on an 8B model needs the 40 GB A100.

## Using a coding agent

This repo is also a *skill*: `.agents/skills/jetstream2-experiments/SKILL.md` tells a coding agent (Claude Code,
Codex, pi, Cursor and the like) how to size, launch, run and shut down Jetstream2 instances, with the budget
rules built in. Two ways to use it:

- **Work inside this repo.** Open the repo in your agent; `AGENTS.md` and `CLAUDE.md` load the skill automatically.
- **Use it from your project repo.** Copy or symlink `.agents/skills/jetstream2-experiments` into your project's
  `.agents/skills/` (and `.claude/skills/` for Claude Code), or into `~/.agents/skills/`, `~/.claude/skills/`,
  `~/.codex/skills/` for all projects.

Ask the agent to confirm the flavor and the SU estimate before every launch, and to shelve when done.

## Troubleshooting

- **`js2 launch` hangs at "waiting for SSH"**: the featured image takes 2 to 4 minutes to boot. If it exceeds 15
  minutes, `js2 kill <name>` and retry; if your keypair name is wrong you can never log in (no password).
- **"no OS_CLOUD" or authentication errors**: `~/.config/openstack/clouds.yaml` is missing or the application
  credential expired. Make a new one in Exosphere and rerun `./install.sh`.
- **Flavor not available / quota exceeded**: another team is holding the GPUs, or you already have an instance.
  `js2 list` shows yours; try a different slice size or later.
- **Out of disk on the instance**: everything should be under `/workspace`; `df -h / /workspace` tells you which
  filled up. Delete old checkpoints and the `/workspace/hf` cache entries you no longer need.
- **vLLM crashes at startup with a flashinfer error**: `source /workspace/env.sh` first; it sets
  `VLLM_USE_FLASHINFER_SAMPLER=0` (the image has no `nvcc`).
- **Shelved instance disappeared from `js2 list` with its volume**: shelved instances hide the volume; `js2 vol list`
  still shows it, and `js2 start` remounts it.
- Each `openstack` call takes a few seconds; a `js2` verb taking 10 to 20 seconds is normal.

Questions: post in the course forum, and include the output of `js2 list` and `js2 status <name>`.
