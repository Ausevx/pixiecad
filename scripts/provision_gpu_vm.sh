#!/usr/bin/env bash
# Provision a GCP GPU VM able to run PixieCAD's dense (S2b) stage.
#
# Everything here was verified end-to-end on 2026-08-02 against project
# pixie-504305: probe, rsync round-trip, docker GPU passthrough, and a
# CUDA-linked COLMAP. The non-obvious parts are commented — they cost an hour
# to discover.
#
#   ./scripts/provision_gpu_vm.sh [instance-name] [zone] [gpu]
#     gpu: t4 (default, plenty for COLMAP dense) | l4 (24GB, needed for TRELLIS.2)
#
# Tear down when done:  gcloud compute instances delete <name> --zone=<zone>

set -euo pipefail

NAME="${1:-pixiecad-gpu}"
ZONE="${2:-asia-south1-a}"
GPU="${3:-t4}"
PROJECT="${PIXIECAD_GCP_PROJECT:-$(gcloud config get-value project 2>/dev/null)}"

case "$GPU" in
  t4) ACCEL="nvidia-tesla-t4"; MACHINE="n1-standard-4" ;;
  l4) ACCEL="nvidia-l4";       MACHINE="g2-standard-4" ;;
  *)  echo "unknown gpu: $GPU (use t4 or l4)" >&2; exit 2 ;;
esac

# The image family matters: 'common-cu124-*' does NOT exist. List current ones:
#   gcloud compute images list --project=deeplearning-platform-release --filter="family~cu1"
IMAGE_FAMILY="common-cu129-ubuntu-2204-nvidia-580"

STARTUP=$(mktemp)
cat > "$STARTUP" <<'EOF'
#!/bin/bash
# Guardrail: hard shutdown after 45 min so a forgotten VM cannot drain credits.
# (--max-run-duration below is the belt to this suspenders.)
(sleep 2700 && shutdown -h now) &
EOF

echo "creating $NAME ($MACHINE + $ACCEL, SPOT) in $ZONE ..."
# SPOT + DELETE-on-preempt: our stage cache makes a killed job a no-op re-run,
# so the 60-90% discount is free money. Expect stockouts — try other zones.
gcloud compute instances create "$NAME" \
  --project="$PROJECT" --zone="$ZONE" \
  --machine-type="$MACHINE" --accelerator="type=$ACCEL,count=1" \
  --provisioning-model=SPOT --instance-termination-action=DELETE \
  --max-run-duration=3600s \
  --image-family="$IMAGE_FAMILY" --image-project=deeplearning-platform-release \
  --boot-disk-size=100GB --boot-disk-type=pd-balanced \
  --metadata-from-file=startup-script="$STARTUP" \
  --metadata=install-nvidia-driver=True
rm -f "$STARTUP"

echo "installing docker + CUDA container runtime ..."
# The Deep Learning VM image does NOT ship docker, despite having the NVIDIA
# container *base* libraries at 1.17.8. Installing the repo's newest
# nvidia-container-toolkit (1.19.x) fails on unmet deps against those pinned
# base packages, so the toolkit version must be pinned to match.
gcloud compute ssh "$NAME" --project="$PROJECT" --zone="$ZONE" --quiet --command='
set -e
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg
echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu jammy stable" \
  | sudo tee /etc/apt/sources.list.d/docker.list >/dev/null
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /etc/apt/keyrings/nvidia-container-toolkit.gpg
curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed "s#deb https://#deb [signed-by=/etc/apt/keyrings/nvidia-container-toolkit.gpg] https://#g" \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list >/dev/null
sudo apt-get update -qq
BASE_VER=$(dpkg-query -W -f="\${Version}" nvidia-container-toolkit-base 2>/dev/null || echo "")
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y \
  docker-ce docker-ce-cli containerd.io \
  ${BASE_VER:+nvidia-container-toolkit=$BASE_VER}
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
sudo usermod -aG docker $USER
sudo docker pull colmap/colmap:latest
'

echo "registering ssh alias ..."
# Writes a Host block to ~/.ssh/config so plain `ssh <alias>` works, which is
# what SSHExecutor (and `pixiecad probe --host`) require.
gcloud compute config-ssh --project="$PROJECT" >/dev/null

ALIAS="$NAME.$ZONE.$PROJECT"
echo
echo "ready:  pixiecad probe --host $ALIAS"
echo "delete: gcloud compute instances delete $NAME --zone=$ZONE --quiet"
