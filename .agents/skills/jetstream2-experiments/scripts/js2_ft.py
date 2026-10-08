#!/usr/bin/env python
"""js2_ft.py - fine-tuning on a Jetstream2 instance: defaults, benchmark, train (SFT / DPO / GRPO), one file.
Runs ON the instance in /workspace/venv_sft (`js2 ft ...` copies it there and runs it).

  js2 ft defaults sft|dpo|grpo [--model M] [--seq-len N] [--max-completion N]   print the config it would use
  js2 ft bench    sft|dpo|grpo [--model M] [--batch 1,2,4,...] [...]           largest batch that fits + tok/s
  js2 ft train    sft|dpo|grpo --dataset D [--model M] [--max-steps K] [...]    train with the defaults
                  (any default can be overridden with --set key=value; grpo needs --reward file.py:fn)

Policy (from the 2026-10-07 H100 benchmarks in SKILL.md):
  SFT/DPO : Unsloth LoRA r=16 all-linear, "unsloth" gradient checkpointing, fused AdamW; NF4 base on A100 slices.
            Full FT = Unsloth full_finetuning + adamw_8bit, 80 GB card, <=8B only (--full).
  GRPO    : TRL defaults (DAPO loss, beta=0 -> no reference model), colocated vLLM at 0.5 of the card,
            128-completion generation rounds (per_device_batch x steps_per_generation); bf16 base only.
            Unsloth's GRPO trainer recompiles per shape -> UNSLOTH_COMPILE_DISABLE=1 is set here; or --backend peft.
  Bench   : synthetic fixed-length token data, 2 warm-up + N timed steps; OOM ends the batch sweep.
"""
import argparse, json, os, sys, time, gc, importlib.util

TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
LORA = dict(r=16, lora_alpha=32, lora_dropout=0, target_modules=TARGETS)


# ----------------------------------------------------------------------------------------------- defaults
def gpu_gb():
    import torch
    return torch.cuda.get_device_properties(0).total_memory / 2**30


def params_b(model):
    """Billions of params from the HF config (no weight download); falls back to a '7B'-style name hint."""
    try:
        from transformers import AutoConfig
        c = AutoConfig.from_pretrained(model)
        h, L, v = c.hidden_size, c.num_hidden_layers, c.vocab_size
        inter = getattr(c, "intermediate_size", 4 * h)
        kvh = getattr(c, "num_key_value_heads", c.num_attention_heads)
        hd = h // c.num_attention_heads
        return (L * (h * (h + 2 * kvh * hd) + h * h + 3 * h * inter) + 2 * v * h) / 1e9
    except Exception:
        import re
        m = re.search(r"(\d+(?:\.\d+)?)[bB]", model)
        return float(m.group(1)) if m else 8.0


def pow2(x, lo, hi):
    x = int(max(lo, min(hi, x)))
    return max(lo, 1 << (x.bit_length() - 1))


def defaults(task, model="Qwen/Qwen3-8B", seq_len=2048, max_completion=512, gb=None, pb=None, backend=None, max_prompt=512):
    gb = gb or gpu_gb(); pb = pb or params_b(model)
    tier = "h100" if gb >= 70 else "a100" if gb >= 35 else "slice20" if gb >= 15 else "slice10"
    load_4bit = tier in ("slice20", "slice10") or (tier == "a100" and pb > 14)
    scale = max(pb / 8, 0.25)
    free = gb * 0.9 - (pb * 0.6 if load_4bit else pb * 2)
    common = dict(bf16=True, logging_steps=10, save_strategy="steps", save_steps=200, report_to=[],
                  lr_scheduler_type="cosine", warmup_ratio=0.03, optim="adamw_torch_fused",
                  gradient_checkpointing="unsloth", dataloader_num_workers=2, output_dir="/workspace/experiments/out")
    env = {"HF_HOME": "/workspace/hf", "HF_HUB_DISABLE_XET": "1", "VLLM_USE_FLASHINFER_SAMPLER": "0"}
    out = dict(task=task, model=model, params_b=round(pb, 2), gpu_gb=round(gb, 1), tier=tier,
               backend=backend or "unsloth", load_in_4bit=load_4bit, lora=LORA, env=env, notes=[])
    if task == "sft":   # measured: unsloth-lora 8B@2048 ~0.35 GB/sample over the base; flat tok/s from batch 4
        bs = pow2(free / (0.35 * seq_len / 2048 * scale) / 4, 1, 32)
        out["config"] = dict(common, max_length=seq_len, packing=True, per_device_train_batch_size=bs,
                             gradient_accumulation_steps=max(1, 64 // bs), num_train_epochs=1,
                             learning_rate=1e-4 if load_4bit else 2e-4)
        if tier == "h100" and pb <= 8.5:
            out["full_ft"] = dict(optim="adamw_8bit", per_device_train_batch_size=8 if pb > 4 else 16, learning_rate=1e-5,
                                  gradient_checkpointing=True)
            out["notes"].append("--full: unsloth full_finetuning, 7.6k tok/s at 48 GB for 8B@2048 (LoRA: 10.9k)")
    elif task == "dpo":  # measured: unsloth DPO 8B@2048 = 30/44/73 GB at batch 2/4/8; flat pairs/s
        bs = pow2(free / (7.0 * seq_len / 2048 * scale), 1, 8)
        out["config"] = dict(common, max_length=seq_len, beta=0.1, per_device_train_batch_size=bs,
                             gradient_accumulation_steps=max(1, 32 // bs), num_train_epochs=1,
                             learning_rate=1e-5 if load_4bit else 5e-6, precompute_ref_log_probs=False)
        out["notes"].append("ref_model=None: the reference is the LoRA policy with the adapter off (no second model)")
    elif task == "grpo":  # measured: 8B, 512-token completions, 128-round: ~14 GB train side at batch 16
        out["load_in_4bit"] = False
        if load_4bit:
            out["notes"].append(f"WARNING: bf16 {pb:.1f}B does not fit on {gb:.0f} GB and QLoRA+colocated vLLM is "
                                "unsupported (TRL 1.13): use a smaller model or vllm_mode='server' on another instance")
        util = 0.5 if tier == "h100" else 0.4
        train_free = gb * 0.92 - gb * util - pb * 2
        bs = pow2(train_free / (0.9 * (max_prompt + max_completion) / 640 * scale), 2, 16)
        spg = max(1, (128 if tier in ("h100", "a100") else 64) // bs)
        out["config"] = dict(common, gradient_checkpointing=True, per_device_train_batch_size=bs,
                             gradient_accumulation_steps=1, steps_per_generation=spg, num_generations=8,
                             max_completion_length=max_completion, temperature=1.0, learning_rate=1e-6,
                             use_vllm=True, vllm_mode="colocate", vllm_gpu_memory_utilization=util,
                             vllm_max_model_length=max_prompt + max_completion, loss_type="dapo", beta=0.0, epsilon=0.2,
                             scale_rewards="group", num_iterations=1, num_train_epochs=1)
        env["UNSLOTH_COMPILE_DISABLE"] = "1"
        out["max_prompt"] = max_prompt
        out["notes"] += [f"generation round = {bs * spg} completions per vLLM call; time ~ linear in max_completion_length",
                         f"prompts are truncated to the last {max_prompt} tokens (--max-prompt)",
                         "DAPO extras (opt-in): --set epsilon_high=0.28 --set mask_truncated_completions=True"]
    else:
        raise ValueError(task)
    return out


def apply_env(d):
    for k, v in d["env"].items():
        os.environ.setdefault(k, v)
    os.environ.setdefault("WANDB_DISABLED", "true")


# ----------------------------------------------------------------------------------------------- models
def load_policy(d, a, max_seq, grpo=False):
    """Returns (model, tokenizer, peft_config). unsloth: adapter already applied; peft: config for the trainer."""
    import torch
    if d["backend"] == "unsloth":
        from unsloth import FastLanguageModel
        kw = dict(max_seq_length=max_seq, dtype=torch.bfloat16, load_in_4bit=d["load_in_4bit"],
                  full_finetuning=getattr(a, "full", False))
        if grpo:
            kw.update(fast_inference=True, gpu_memory_utilization=d["config"]["vllm_gpu_memory_utilization"],
                      max_lora_rank=LORA["r"])
        m, tok = FastLanguageModel.from_pretrained(a.model, **kw)
        if not getattr(a, "full", False):
            m = FastLanguageModel.get_peft_model(m, use_gradient_checkpointing="unsloth", **LORA)
        return m, tok, None
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig
    kw = dict(dtype=torch.bfloat16, attn_implementation="sdpa")
    if d["load_in_4bit"]:
        kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
    m = AutoModelForCausalLM.from_pretrained(a.model, **kw)
    return m, AutoTokenizer.from_pretrained(a.model), LoraConfig(task_type="CAUSAL_LM", **LORA)


def make_trainer(task, d, a, cfg, ds, reward=None):
    import torch
    from transformers import TrainerCallback
    grpo = task == "grpo"
    seq = cfg.get("max_length") or (d.get("max_prompt", 512) + cfg.get("max_completion_length", 512))
    model, tok, peft_cfg = load_policy(d, a, seq, grpo=grpo)
    if d["backend"] == "unsloth" and task == "dpo":
        from unsloth import PatchDPOTrainer; PatchDPOTrainer()
    if d["backend"] == "peft":
        cfg["gradient_checkpointing"] = True; cfg["gradient_checkpointing_kwargs"] = {"use_reentrant": False}
    if task == "sft":
        from trl import SFTConfig, SFTTrainer
        if "text" not in ds.column_names and "messages" in ds.column_names:
            # Unsloth's patched SFTTrainer wants a text field; render chat rows with the model's template
            ds = ds.map(lambda r: {"text": tok.apply_chat_template(r["messages"], tokenize=False)}, remove_columns=ds.column_names)
        cfg["dataset_text_field"] = "text"
        return SFTTrainer(model=model, processing_class=tok, train_dataset=ds, peft_config=peft_cfg, args=SFTConfig(**cfg)), tok
    if task == "dpo":
        from trl import DPOConfig, DPOTrainer
        return DPOTrainer(model=model, ref_model=None, processing_class=tok, train_dataset=ds, peft_config=peft_cfg, args=DPOConfig(**cfg)), tok
    from trl import GRPOConfig, GRPOTrainer
    if d["backend"] == "unsloth":   # unsloth patches GRPOTrainer onto its own engine; these keys are TRL-colocate only
        for k in ("vllm_max_model_length",): cfg.pop(k, None)
    return GRPOTrainer(model=model, reward_funcs=reward, processing_class=tok, train_dataset=ds, peft_config=peft_cfg, args=GRPOConfig(**cfg)), tok


# ----------------------------------------------------------------------------------------------- bench
def bench(task, a):
    import torch
    from datasets import Dataset
    from transformers import AutoTokenizer, TrainerCallback
    d = defaults(task, a.model, a.seq_len, a.max_completion, backend=a.backend, max_prompt=a.max_prompt); apply_env(d)
    tok = AutoTokenizer.from_pretrained(a.model)
    gb = gpu_gb()
    rand = lambda n: tok.decode(torch.randint(1000, 20000, (n,)).tolist())
    batches = [int(b) for b in a.batch.split(",")] if a.batch else [d["config"]["per_device_train_batch_size"]]
    hdr = dict(gpu=torch.cuda.get_device_name(0), gpu_gb=round(gb, 1), model=a.model, task=task, backend=d["backend"],
               load_in_4bit=d["load_in_4bit"], full=getattr(a, "full", False))
    if task == "sft":   # manual loop: forward+backward+step on random ids, no trainer overhead
        hdr["seq_len"] = a.seq_len
        model, _, peft_cfg = load_policy(d, a, a.seq_len)
        if peft_cfg is not None:
            from peft import get_peft_model
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            model.enable_input_require_grads(); model = get_peft_model(model, peft_cfg)
        model.train(); model.config.use_cache = False
        params = [q for q in model.parameters() if q.requires_grad]
        if getattr(a, "full", False):
            import bitsandbytes as bnb; opt = bnb.optim.AdamW8bit(params, lr=1e-5)
        else:
            opt = torch.optim.AdamW(params, lr=1e-4, fused=True)
        hdr["trainable_params_M"] = round(sum(q.numel() for q in params) / 1e6, 1)
        print(json.dumps(hdr), flush=True)
        for bs in batches:
            torch.cuda.reset_peak_memory_stats(); gc.collect(); torch.cuda.empty_cache()
            try:
                def step():
                    x = torch.randint(100, 30000, (bs, a.seq_len), device="cuda")
                    model(input_ids=x, labels=x).loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)
                step(); step(); torch.cuda.synchronize(); t0 = time.time()
                for _ in range(a.steps): step()
                torch.cuda.synchronize(); dt = (time.time() - t0) / a.steps
                print(json.dumps(dict(batch=bs, tok_s=round(bs * a.seq_len / dt), step_s=round(dt, 3),
                                      peak_gb=round(torch.cuda.max_memory_allocated() / 2**30, 1))), flush=True)
            except torch.cuda.OutOfMemoryError:
                print(json.dumps(dict(batch=bs, oom=True)), flush=True); break
            finally:
                opt.zero_grad(set_to_none=True); gc.collect(); torch.cuda.empty_cache()
        return

    class Timer(TrainerCallback):
        def __init__(s): s.t = []
        def on_step_begin(s, *x, **k): s.t0 = time.time()
        def on_step_end(s, *x, **k): s.t.append(time.time() - s.t0)

    for bs in batches:
        cfg = dict(d["config"], per_device_train_batch_size=bs, gradient_accumulation_steps=1, logging_steps=1,
                   save_strategy="no", warmup_steps=0, lr_scheduler_type="constant", dataloader_num_workers=0,
                   output_dir="/tmp/js2_ft_bench")
        spg = cfg.get("steps_per_generation", 1) if task == "grpo" else 1
        if task == "grpo" and a.spg: spg = cfg["steps_per_generation"] = a.spg
        if task == "grpo" and a.vllm_util: cfg["vllm_gpu_memory_utilization"] = a.vllm_util
        n_steps = (a.steps + 2) * spg
        cfg["max_steps"] = n_steps; cfg.pop("num_train_epochs", None)
        if task == "dpo":
            comp = a.seq_len - 256
            ds = Dataset.from_list([dict(prompt=rand(256), chosen=rand(comp), rejected=rand(comp)) for _ in range(bs * n_steps)])
            reward = None
        else:
            ds = Dataset.from_list([dict(prompt=rand(64) + " Continue this text at length.") for _ in range(bs * n_steps * 2)])
            reward = lambda completions, **k: [len(c) / 1000 for c in completions]   # worst case: max-length completions
        timer = Timer()
        torch.cuda.reset_peak_memory_stats()
        try:
            trainer, _ = make_trainer(task, d, a, cfg, ds, reward)
            trainer.add_callback(timer); trainer.train()
        except torch.cuda.OutOfMemoryError:
            print(json.dumps(dict(batch=bs, oom=True)), flush=True); break
        steps = timer.t[2 * spg:] or timer.t
        st = sum(steps) / len(steps)
        r = dict(hdr, batch=bs, step_s=round(st, 2), peak_gb=round(torch.cuda.max_memory_allocated() / 2**30, 1))
        if task == "dpo":
            r.update(max_len=a.seq_len, pairs_per_s=round(bs / st, 2), tok_s=round(bs * 2 * a.seq_len / st))
        else:
            ml = next((h["completions/mean_length"] for h in reversed(trainer.state.log_history) if "completions/mean_length" in h), a.max_completion)
            r.update(steps_per_generation=spg, generation_round=bs * spg, max_completion=a.max_completion, mean_completion_len=round(ml),
                     completions_per_s=round(bs / st, 2), gen_tok_s=round(bs * ml / st), gen_round_s=round(max(steps), 1),
                     train_only_step_s=round(sorted(steps)[len(steps) // 2], 2), vllm_util=cfg["vllm_gpu_memory_utilization"])
        print("RESULT " + json.dumps(r), flush=True)
        del trainer; gc.collect(); torch.cuda.empty_cache()
        if task == "grpo": break   # a colocated vLLM engine cannot be rebuilt in-process; one batch per run


# ----------------------------------------------------------------------------------------------- train
def train(task, a):
    import torch
    from datasets import load_dataset
    d = defaults(task, a.model, a.seq_len, a.max_completion, backend=a.backend, max_prompt=a.max_prompt); apply_env(d)
    cfg = d["config"]
    if getattr(a, "full", False):
        assert d.get("full_ft"), "--full needs an 80 GB card and <=8B"
        cfg.update(d["full_ft"])
    for kv in a.set:
        k, v = kv.split("=", 1)
        try: v = json.loads(v)
        except ValueError: pass
        cfg[k] = v
    cfg["output_dir"] = a.output
    if a.max_steps > 0: cfg["max_steps"] = a.max_steps; cfg.pop("num_train_epochs", None)
    print("config:", json.dumps({k: cfg[k] for k in sorted(cfg) if k not in ("report_to",)}), flush=True)
    for n in d["notes"]: print("note:", n)
    ds = load_dataset("json", data_files=a.dataset)["train"] if a.dataset.endswith((".jsonl", ".json")) else load_dataset(a.dataset, split=a.split)
    reward = None
    if task == "grpo":
        path, fn = a.reward.split(":")
        spec = importlib.util.spec_from_file_location("reward_mod", path); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
        reward = getattr(mod, fn)
    if task == "grpo":   # TRL 1.13 has no max_prompt_length: cut prompts to the last max_prompt tokens ourselves
        from transformers import AutoTokenizer
        t = AutoTokenizer.from_pretrained(a.model); mp = d["max_prompt"]
        def cut(r):
            p_ = r["prompt"]
            if isinstance(p_, str):
                ids = t(p_, add_special_tokens=False)["input_ids"]
                if len(ids) > mp: r["prompt"] = t.decode(ids[-mp:])
            return r
        ds = ds.map(cut)
    trainer, tok = make_trainer(task, d, a, cfg, ds, reward)
    trainer.train()
    trainer.save_model(a.output); tok.save_pretrained(a.output)
    print("saved", a.output, "| peak GB:", round(torch.cuda.max_memory_allocated() / 2**30, 1))


# ----------------------------------------------------------------------------------------------- cli
if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("cmd", choices=["defaults", "bench", "train"])
    p.add_argument("task", choices=["sft", "dpo", "grpo"])
    p.add_argument("--model", default="Qwen/Qwen3-8B")
    p.add_argument("--backend", choices=["unsloth", "peft"], default=None, help="default unsloth")
    p.add_argument("--seq-len", type=int, default=2048, help="sft/dpo: max_length (prompt+completion)")
    p.add_argument("--max-completion", type=int, default=512, help="grpo: max_completion_length")
    p.add_argument("--max-prompt", type=int, default=512, help="grpo: prompts truncated to this many tokens; sizes the vLLM context")
    p.add_argument("--full", action="store_true", help="sft: unsloth full fine-tune instead of LoRA")
    p.add_argument("--gpu-gb", type=float, help="defaults: pretend this GPU size (no GPU needed)")
    p.add_argument("--params-b", type=float, help="defaults: pretend this model size")
    # bench
    p.add_argument("--batch", help="bench: comma list of per-device batch sizes (default: the defaults' batch)")
    p.add_argument("--steps", type=int, default=6, help="bench: timed steps per config")
    p.add_argument("--spg", type=int, help="bench grpo: steps_per_generation override")
    p.add_argument("--vllm-util", type=float, help="bench grpo: vllm_gpu_memory_utilization override")
    # train
    p.add_argument("--dataset", help="train: HF dataset id or local .jsonl")
    p.add_argument("--split", default="train")
    p.add_argument("--reward", help="train grpo: reward.py:function_name (completions, **kwargs) -> list[float]")
    p.add_argument("--max-steps", type=int, default=-1)
    p.add_argument("--output", default="/workspace/experiments/out")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="train: override any config field")
    a = p.parse_args()
    if a.cmd == "defaults":
        d = defaults(a.task, a.model, a.seq_len, a.max_completion, a.gpu_gb, a.params_b, a.backend, a.max_prompt)
        print(f"{a.task} | {a.model} ({d['params_b']}B) on {d['tier']} {d['gpu_gb']} GB | backend={d['backend']} 4bit={d['load_in_4bit']}")
        for k, v in d["config"].items(): print(f"  {k}={v!r}")
        for n in d["notes"]: print(f"  # {n}")
        print("  env: " + " ".join(f"{k}={v}" for k, v in d["env"].items()))
    elif a.cmd == "bench":
        bench(a.task, a)
    else:
        assert a.dataset, "--dataset required"
        assert a.task != "grpo" or a.reward, "grpo needs --reward file.py:fn"
        train(a.task, a)
