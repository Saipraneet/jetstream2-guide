#!/usr/bin/env bash
# js2_remote.sh - everything js2 runs ON the instance (copied to /tmp by js2; not called by hand).
#   sudo js2_remote.sh mount <volume_id> <mountpoint>   attach-side of `js2 launch`/`js2 vol attach`: find the
#        Cinder volume by serial, mkfs ext4 ONLY if blank, mount, UUID fstab line (nofail), chown to the SSH user
#   js2_remote.sh setup venv|docker|sft [mountpoint]     `js2 setup`: build the env on the volume. Idempotent.
#        venv   : uv venv + vLLM at $MOUNT/venv (serving, offline inference)
#        docker : nvidia runtime + bind-mount the containerd image store onto the volume + pull vllm/vllm-openai
#        sft    : uv venv at $MOUNT/venv_sft with vLLM (pins torch), TRL, PEFT, bitsandbytes, unsloth (SFT/DPO/GRPO)
# Jetstream2 featured images are barebones (Ubuntu 24.04, NVIDIA driver 595 / CUDA 13.2, python3, docker, tmux,
# rsync; no uv/torch/vLLM/nvcc) with a 60 GB root disk that has ~11 GB free: everything lands on the volume.
set -euo pipefail
case "${1:-}" in
  mount)
vid="${2:?volume id}"; mnt="${3:-/workspace}"
serial="${vid:0:20}"
dev=""
for _ in $(seq 1 30); do
  dev=$(lsblk -dno NAME,SERIAL | awk -v s="$serial" 'index($2,s)==1 {print "/dev/"$1}' | head -1)
  [ -n "$dev" ] && break
  udevadm settle 2>/dev/null || true; sleep 2
done
[ -n "$dev" ] || { echo "no block device with serial $serial (lsblk -o NAME,SIZE,SERIAL):" >&2; lsblk -o NAME,SIZE,SERIAL >&2; exit 1; }
fstype=$(blkid -o value -s TYPE "$dev" || true)
if [ -z "$fstype" ]; then
  echo "formatting $dev as ext4 (blank volume)"
  mkfs.ext4 -q -L workspace "$dev"
elif [ "$fstype" != ext4 ]; then
  echo "WARNING: $dev has filesystem $fstype; mounting as-is" >&2
fi
uuid=$(blkid -o value -s UUID "$dev")
mkdir -p "$mnt"
if mountpoint -q "$mnt"; then
  cur=$(findmnt -no SOURCE "$mnt")
  [ "$cur" = "$dev" ] || { echo "$mnt already has $cur mounted" >&2; exit 1; }
else
  mount "$dev" "$mnt"
fi
sed -i "\#[[:space:]]$mnt[[:space:]]#d" /etc/fstab
echo "UUID=$uuid $mnt ext4 defaults,nofail,noatime,x-systemd.device-timeout=10s 0 2" >> /etc/fstab
systemctl daemon-reload 2>/dev/null || true
chown "${SUDO_USER:-ubuntu}:${SUDO_USER:-ubuntu}" "$mnt"
echo "MOUNTED $dev at $mnt ($(df -h --output=size,avail "$mnt" | tail -1 | xargs))"
;;
  setup)
mode="${2:-venv}"; MOUNT="${3:-/workspace}"
mountpoint -q "$MOUNT" || echo "WARNING: $MOUNT is not a mounted volume - env goes on the 60 GB root disk" >&2
mkdir -p "$MOUNT/hf" "$MOUNT/experiments"
[ -f "$MOUNT/env.sh" ] || cat > "$MOUNT/env.sh" <<EOT
export HF_HOME=$MOUNT/hf
export HF_HUB_DISABLE_XET=1
# no nvcc on the image: flashinfer's JIT sampler fails at engine warmup (vLLM 0.31, 2026-10); use torch sampler
export VLLM_USE_FLASHINFER_SAMPLER=0
[ -x $MOUNT/venv/bin/activate ] || [ -f $MOUNT/venv/bin/activate ] && source $MOUNT/venv/bin/activate
case "\$(command -v pip 2>/dev/null)" in $MOUNT/venv/*|"") ;; *)
  echo "WARNING: pip is \$(command -v pip), NOT the venv's - use 'python -m pip'" >&2 ;;
esac
EOT
# Jetstream2 images mount a read-only /software (ceph) with lmod; harmless. The driver supports CUDA 13.x
# (595.x as of 2026-10), so PyPI vLLM's CUDA-13 wheels load; no cu129 workaround needed.
drv=$(nvidia-smi 2>/dev/null | sed -n 's/.*CUDA Version: *\([0-9]*\)\.\([0-9]*\).*/\1/p' | head -1)
echo "driver CUDA major: ${drv:-none}"

case "$mode" in
  venv)
    if [ -x "$MOUNT/venv/bin/vllm" ]; then
      source "$MOUNT/venv/bin/activate"
      python -c 'import torch, vllm; print(f"SETUP_OK (existing venv) vllm={vllm.__version__} torch={torch.__version__} gpu={torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}")'
      exit 0
    fi
    command -v uv >/dev/null || { curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null; export PATH="$HOME/.local/bin:$PATH"; }
    export UV_CACHE_DIR="$MOUNT/.uv-cache"   # keep the multi-GB wheel cache off the root disk
    uv venv "$MOUNT/venv" --python 3.12 -q
    source "$MOUNT/venv/bin/activate"
    python -m ensurepip --upgrade >/dev/null
    # ensurepip on Ubuntu 24.04 creates only pip3; without a `pip` the system /usr/bin/pip shadows it
    [ -e "$MOUNT/venv/bin/pip" ] || ln -s pip3 "$MOUNT/venv/bin/pip"
    if [ -n "$drv" ] && [ "$drv" -lt 13 ]; then
      echo "driver supports CUDA $drv.x only - installing vLLM 0.25.1 from the cu129 wheel index"
      uv pip install -q vllm==0.25.1 --index https://wheels.vllm.ai/0.25.1/cu129 --torch-backend=cu128
      uv pip uninstall -q torchaudio torchcodec 2>/dev/null || true
    else
      uv pip install -q vllm --torch-backend=auto
    fi
    uv pip install -q datasets ninja accelerate
    [ -n "${JS2_EXTRA_PIP:-}" ] && uv pip install -q $JS2_EXTRA_PIP
    python - <<'EOT'
import torch, vllm
print(f"SETUP_OK vllm={vllm.__version__} torch={torch.__version__} cuda={torch.cuda.is_available()} gpu={torch.cuda.get_device_name(0)}")
EOT
    ;;
  docker)
    # The image already ships nvidia-container-toolkit (1.20, 2026-10) and its apt source; only wire the runtime.
    if ! command -v nvidia-ctk >/dev/null; then
      sudo apt-get update -qq && sudo apt-get install -y -qq nvidia-container-toolkit >/dev/null
    fi
    if ! docker info 2>/dev/null | grep -q 'Runtimes:.*nvidia'; then
      sudo nvidia-ctk runtime configure --runtime=docker >/dev/null; sudo systemctl restart docker
    fi
    # Docker here uses the containerd image store (/var/lib/containerd), which also holds Exosphere's web-shell
    # containers, so neither a docker data-root move nor a wipe works. Relocate the whole store onto the volume
    # with a bind mount (fstab, ordered after the volume) so images never fill the 60 GB root disk.
    if ! findmnt -n /var/lib/containerd >/dev/null; then
      echo "moving /var/lib/containerd to $MOUNT/containerd (bind mount)"
      sudo systemctl stop docker docker.socket containerd
      sudo mkdir -p "$MOUNT/containerd"
      sudo rsync -aHAX --remove-source-files /var/lib/containerd/ "$MOUNT/containerd/" && sudo find /var/lib/containerd -depth -type d -empty -delete
      sudo mkdir -p /var/lib/containerd
      grep -q ' /var/lib/containerd ' /etc/fstab || echo "$MOUNT/containerd /var/lib/containerd none bind,nofail,x-systemd.requires-mounts-for=$MOUNT 0 0" | sudo tee -a /etc/fstab >/dev/null
      sudo systemctl daemon-reload; sudo mount /var/lib/containerd
      sudo systemctl start containerd docker
    fi
    # undo an earlier data-root move if one was tried (it does not apply to the containerd store)
    if [ -f /etc/docker/daemon.json ] && grep -q '"data-root"' /etc/docker/daemon.json; then
      sudo python3 - <<'EOT'
import json; p='/etc/docker/daemon.json'; d=json.load(open(p)); d.pop('data-root',None); json.dump(d,open(p,'w'),indent=2)
EOT
      sudo systemctl restart docker
    fi
    sudo usermod -aG docker "$USER" 2>/dev/null || true
    sudo docker image prune -f >/dev/null 2>&1 || true   # drop partial layers from an interrupted pull
    sudo docker pull -q vllm/vllm-openai:latest
    sudo docker run --rm --gpus all --entrypoint python3 vllm/vllm-openai:latest -c 'import vllm,torch; print("SETUP_OK docker vllm="+vllm.__version__+" torch="+torch.__version__)'
    sudo docker run --rm --gpus all --entrypoint nvidia-smi vllm/vllm-openai:latest --query-gpu=name --format=csv,noheader | sed 's/^/SETUP_OK docker gpu=/'
    echo "root disk: $(df -h --output=avail / | tail -1 | xargs) free; containerd store on $(findmnt -no SOURCE /var/lib/containerd)"
    ;;
  sft)
    export PATH="$HOME/.local/bin:$PATH" UV_CACHE_DIR="$MOUNT/.uv-cache"
    command -v uv >/dev/null || { curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null; export PATH="$HOME/.local/bin:$PATH"; }
    if [ -x "$MOUNT/venv_sft/bin/python" ] && "$MOUNT/venv_sft/bin/python" -c 'import trl, peft, vllm' 2>/dev/null; then
      "$MOUNT/venv_sft/bin/python" -c 'import torch,trl,peft,unsloth; print(f"SETUP_OK (existing venv_sft) torch={torch.__version__} trl={trl.__version__} peft={peft.__version__} unsloth={unsloth.__version__}")' 2>/dev/null | tail -1
      exit 0
    fi
    uv venv "$MOUNT/venv_sft" --python 3.12 -q
    source "$MOUNT/venv_sft/bin/activate"
    python -m ensurepip --upgrade >/dev/null; [ -e "$MOUNT/venv_sft/bin/pip" ] || ln -s pip3 "$MOUNT/venv_sft/bin/pip"
    # vLLM first: it pins torch (2.13+cu132 for 0.31); everything else installs against that
    uv pip install -q vllm --torch-backend=auto
    uv pip install -q "trl>=1.0" peft accelerate datasets bitsandbytes
    uv pip install -q unsloth || echo "WARNING: unsloth install failed (TRL/PEFT path still works)" >&2
    [ -n "${JS2_EXTRA_PIP:-}" ] && uv pip install -q $JS2_EXTRA_PIP
    python -c 'import torch,trl,peft,vllm,bitsandbytes as b
try:
    import unsloth; u=unsloth.__version__
except Exception as e: u="FAILED:"+type(e).__name__
print(f"SETUP_OK sft torch={torch.__version__} vllm={vllm.__version__} trl={trl.__version__} peft={peft.__version__} bnb={b.__version__} unsloth={u} gpu={torch.cuda.get_device_name(0)}")' 2>/dev/null | tail -1
    ;;
  *) echo "setup mode must be venv, docker or sft" >&2; exit 1 ;;
esac
;;
  *) echo "usage: js2_remote.sh mount <vid> <mnt> | setup venv|docker|sft [mnt]" >&2; exit 1 ;;
esac
