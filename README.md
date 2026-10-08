# Jetstream2 for the CSCI 567 class project

Jetstream2 is a free (to you) cloud that gives your team a Linux computer with an NVIDIA A100 or H100 GPU.
This repo gets you from zero to running your code on that GPU with five commands, even if you have never used
a remote server before. Everything here works from a Mac, Linux, or Windows (inside WSL) terminal.

## The one rule: 10,000 credits per team, no refills

Your team's budget is **10,000 SUs (about 90 GPU-hours)**. The GPU computer costs credits for every hour it
exists, *whether or not you are using it*, until you **shelve** it (`js2 stop`). A forgotten GPU burns through
1,500 credits overnight.

| flavor | GPU memory | credits / hour | what it is good for |
|---|---|---|---|
| `a100-10` (default) | 10 GB | 16 | developing and debugging; models up to ~3B parameters |
| `a100-20` | 20 GB | 32 | models up to ~8B |
| `a100` | 40 GB | 64 | fine-tuning 8B models, serving 14B |
| `h100` | 80 GB | 128 | the final big run only |

So: **always run `js2 stop` before you walk away**, start on `a100-10`, and move up only when your code works.

## Setup (once per person, about 20 minutes)

1. **Get an ACCESS account.** Sign up at <https://allocations.access-ci.org> with your USC email. Send your
   ACCESS username to the course staff; we add you to the course allocation and email you when it is done.
2. **Download your cloud credentials.** Log in to <https://jetstream2.exosphere.app>, choose the course
   allocation, and create an *application credential* (Exosphere shows a "Download clouds.yaml" option for this).
   Save the file as `~/.config/openstack/clouds.yaml`. Treat it like a password: never commit it to git.
3. **Make an SSH key** if you do not have one: `ssh-keygen -t ed25519` (press Enter at every prompt). Then in
   Exosphere go to *SSH public keys*, upload the contents of `~/.ssh/id_ed25519.pub`, and give it a name.
4. **Install the helper:**
   ```bash
   git clone https://github.com/Saipraneet/jetstream2-guide.git
   cd jetstream2-guide && ./install.sh
   ```
   It prints your keypair names. Open `~/.jetstream2/js2.conf` in any editor and set `KEY_NAME` to the one you
   made in step 3, keeping the quotes: `KEY_NAME="my laptop key"`.
5. **Check it works:** `js2 list` should print an empty table (or your team's instance), not an error.

## Your first session (costs about 2 credits)

```bash
js2 launch mybox          # creates the GPU computer "mybox" with a 100 GB disk. Takes ~3 minutes.
js2 setup mybox           # installs Python, PyTorch, vLLM and Hugging Face tools on it. ~5 minutes.
js2 ssh mybox             # log in. You are now on the GPU computer.
```

Once logged in, run these on the GPU computer:

```bash
source /workspace/env.sh  # activate the Python environment (do this every time you log in)
nvidia-smi                # see your GPU
python -c "import torch; print(torch.cuda.is_available())"   # should print True
exit                      # back to your laptop
```

Then, **before you close your laptop:**

```bash
js2 stop mybox            # shelve: credits stop. Your disk, files and environment are kept.
```

## Every session after that

```bash
js2 start mybox           # ~1 minute. Everything is as you left it.
js2 push-code mybox       # copy your project's code from your laptop to /workspace/<project> on the GPU computer
js2 ssh mybox             # log in, source /workspace/env.sh, run your code
js2 pull-results mybox    # copy results (json, csv, logs, checkpoints) back to your laptop
js2 stop mybox            # shelve!
```

Run `js2 push-code` and `js2 pull-results` from inside your project folder. They sync the `experiments/` folder
if your project has one, otherwise the whole project. Code (.py, .sh, .md, .txt, .yaml) only ever goes up and
results only ever come down, so you cannot overwrite your work by accident. Save results as `.json`, `.csv` or
`.log`, not `.txt`.

Everything you put under `/workspace` on the GPU computer survives `stop`/`start`. Anything else on the machine
does not fit (the system disk is nearly full), so always work under `/workspace`.

## Running something that takes hours

If you close your laptop or lose Wi-Fi, a normal SSH session dies and takes your job with it. Use `tmux`:

```bash
js2 ssh mybox
tmux new -s train                         # opens a session that keeps running after you disconnect
source /workspace/env.sh
cd /workspace/myproject
python train.py 2>&1 | tee train.log      # output goes to the screen and to train.log
# press Ctrl-b then d to detach; close your laptop; go to sleep
```

Later: `js2 ssh mybox`, then `tmux attach -t train` to see it, or `tail -f /workspace/myproject/train.log`.
Only run `js2 stop` once the job is finished: shelving freezes the machine.

## Sharing one computer within a team

One `mybox` per team is plenty. Each teammate does the Setup steps with their own ACCESS account and SSH key, and
the person who runs `js2 launch` adds the others' public keys to the machine (`js2 ssh mybox`, then append each
key to `~/.ssh/authorized_keys`). Only one person should run `js2 launch`; after that everybody can
`js2 start`/`stop`/`ssh` it. Agree in the team chat who has the machine so nobody shelves a running job.

## When you are done with the project

```bash
js2 pull-results mybox    # make sure everything you need is on your laptop
js2 finish mybox          # deletes the computer (it re-checks the results first); the disk survives
js2 vol delete mybox-vol  # deletes the disk too, once you are sure
```

## Help, something is wrong

- **`js2: command not found`**: run `~/bin/js2` instead, or add `export PATH="$HOME/bin:$PATH"` to your
  `~/.bashrc` or `~/.zshrc` and open a new terminal.
- **"authentication" or "no OS_CLOUD" error**: `~/.config/openstack/clouds.yaml` is missing or expired. Repeat
  Setup step 2 and rerun `./install.sh`.
- **`js2 launch` says "waiting for SSH" for 15 minutes**: your `KEY_NAME` is wrong. Run `js2 kill mybox`, fix
  `~/.jetstream2/js2.conf`, and launch again. If it says "host key changed", run the command it prints.
- **"flavor not available"**: all GPUs of that size are busy. Try a different size, or try later.
- **CUDA out of memory**: your model is too big for the slice. Use a smaller model, smaller batch, or `js2 kill`
  and relaunch bigger: `js2 launch mybox a100 mybox-vol` reuses your existing disk on a 40 GB card.
- **"No space left on device"**: you are writing outside `/workspace`, or the disk is full of old checkpoints.
- **Anything else**: post in the course forum with the output of `js2 list` and the full error message.

## More

- [docs/reference.md](docs/reference.md): every `js2` command, serving a model with vLLM as an OpenAI-style API,
  one-command SFT / DPO / GRPO fine-tuning with tuned defaults, volumes, and using a coding agent (Claude Code,
  Codex, Cursor) that already knows all of this.
- [Jetstream2 documentation](https://docs.jetstream-cloud.org/) and the [Exosphere dashboard](https://jetstream2.exosphere.app)
  where you can see your team's credit usage.
