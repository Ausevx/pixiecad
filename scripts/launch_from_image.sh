#!/usr/bin/env bash
# Launch a worker VM from a baked machine image. Ready in ~2 minutes.
#
#   ./scripts/launch_from_image.sh <instance> <zone> <machine-image> [project]
#
# The counterpart to bake_image.sh. Everything the slow path builds -- docker,
# the CUDA container runtime, the Hunyuan shape and paint images, SAM and its
# checkpoint, and the downloaded model weights -- is already on the disk, so
# this only has to boot.
#
# Spot by default. Preemption is cheap now: relaunching from the image costs
# ~2 minutes instead of the ~25-minute rebuild that made spot a bad bet before.

set -euo pipefail

NAME="${1:?usage: launch_from_image.sh <instance> <zone> <machine-image> [project]}"
ZONE="${2:?usage: launch_from_image.sh <instance> <zone> <machine-image> [project]}"
IMAGE="${3:?usage: launch_from_image.sh <instance> <zone> <machine-image> [project]}"
PROJECT="${4:-${PIXIECAD_GCP_PROJECT:-$(gcloud config get-value project 2>/dev/null)}}"

# Same guardrail as provision_gpu_vm.sh: a forgotten GPU VM is the expensive
# failure mode, so it hard-stops itself even if nobody tears it down.
STARTUP=$(mktemp)
cat > "$STARTUP" <<'EOF'
#!/bin/bash
(sleep 6600 && shutdown -h now) &
EOF

ONDEMAND_FLAGS=()
if [ "${PIXIECAD_ONDEMAND:-0}" != "1" ]; then
  ONDEMAND_FLAGS=(--provisioning-model=SPOT --instance-termination-action=DELETE)
fi

echo "launching '$NAME' from machine image '$IMAGE' ..."
gcloud compute instances create "$NAME" \
  --project="$PROJECT" \
  --zone="$ZONE" \
  --source-machine-image="$IMAGE" \
  --metadata-from-file=startup-script="$STARTUP" \
  --max-run-duration=7200s \
  --instance-termination-action=DELETE \
  "${ONDEMAND_FLAGS[@]}" \
  --quiet
rm -f "$STARTUP"

echo "waiting for sshd ..."
for _ in $(seq 1 30); do
  if gcloud compute ssh "$NAME" --project="$PROJECT" --zone="$ZONE" --quiet \
       --command=true >/dev/null 2>&1; then
    break
  fi
  sleep 10
done

# Prove the baked contents actually survived the image round-trip, rather than
# discovering a missing image in the middle of a job.
gcloud compute ssh "$NAME" --project="$PROJECT" --zone="$ZONE" --quiet --command='
set -e
nvidia-smi --query-gpu=name --format=csv,noheader
sudo docker images --format "{{.Repository}}:{{.Tag}}" | sort
'

gcloud compute config-ssh --quiet >/dev/null 2>&1 || true
echo
echo "ready:  $NAME.$ZONE.$PROJECT"
echo "delete: gcloud compute instances delete $NAME --zone=$ZONE --quiet"
