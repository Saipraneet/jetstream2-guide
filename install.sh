#!/usr/bin/env bash
# install.sh - one-time setup of the js2 helper on your laptop / workstation (Linux or macOS).
# Prereqs (see README): ~/.config/openstack/clouds.yaml from Exosphere, an SSH keypair uploaded to Jetstream2.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SKILL="$REPO/.agents/skills/jetstream2-experiments"
VENV="$HOME/.venvs/openstack-client"

for t in python3 ssh rsync; do command -v "$t" >/dev/null || { echo "missing: $t (install it first)" >&2; exit 1; }; done

echo "1/4 openstack CLI -> $VENV"
[ -x "$VENV/bin/openstack" ] || { python3 -m venv "$VENV"; "$VENV/bin/pip" install -q --upgrade pip; "$VENV/bin/pip" install -q python-openstackclient pyyaml; }

echo "2/4 symlinks in ~/bin"
mkdir -p "$HOME/bin"
ln -sfn "$VENV/bin/openstack" "$HOME/bin/openstack"
ln -sfn "$SKILL/scripts/js2" "$HOME/bin/js2"
case ":$PATH:" in *":$HOME/bin:"*) ;; *) echo "   NOTE: add 'export PATH=\"\$HOME/bin:\$PATH\"' to your shell rc (or call ~/bin/js2)";; esac

echo "3/4 config"
mkdir -p "$HOME/.jetstream2"
if [ ! -f "$HOME/.config/openstack/clouds.yaml" ]; then
  echo "   MISSING ~/.config/openstack/clouds.yaml - download it from Exosphere (README step 2), then rerun." >&2; exit 1
fi
cloud=$(python3 -c 'import yaml,os; print(next(iter(yaml.safe_load(open(os.path.expanduser("~/.config/openstack/clouds.yaml")))["clouds"])))')
if [ ! -f "$HOME/.jetstream2/js2.conf" ]; then
  sed "s/^OS_CLOUD=.*/OS_CLOUD=$cloud/" "$REPO/js2.conf.example" > "$HOME/.jetstream2/js2.conf"
  echo "   wrote ~/.jetstream2/js2.conf (cloud: $cloud) - now set KEY_NAME in it"
else
  echo "   ~/.jetstream2/js2.conf exists, left as is"
fi

echo "4/4 checking access"
export OS_CLOUD="$cloud"
"$HOME/bin/openstack" keypair list -f value -c Name | sed 's/^/   keypair: /' || { echo "   openstack call failed - check clouds.yaml" >&2; exit 1; }
echo "done. Set KEY_NAME in ~/.jetstream2/js2.conf to one of the keypairs above, then: js2 list"
