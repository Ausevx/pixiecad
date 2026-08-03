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
# Whether the weights came along in the image decides whether the first job
# pays ~10 GB of download at the GPU rate. Report it rather than find out.
echo "-- cached weights on this host --"
du -sh "$HOME/.cache/huggingface" "$HOME/.cache/hy3dgen" 2>/dev/null || echo "  none cached"
'

# Self-healing patch for images baked before a dependency was discovered.
#
# The paint stack remeshes before painting, and trimesh 4.x delegates that to
# fast_simplification rather than implementing it. A machine image baked before
# that was known texturs nothing: the stage dies with ModuleNotFoundError after
# the pipeline has already loaded, and the job completes untextured with only a
# warning. Rather than requiring everyone to remember to rebake, the launch
# checks and repairs in place -- a no-op on a current image, ~20 s on an old
# one, and it means a stale image degrades to "slightly slower launch" instead
# of "textures silently do not work".
#
# Delete this block once no pre-fix machine images remain.
echo "checking paint image dependencies ..."
gcloud compute ssh "$NAME" --project="$PROJECT" --zone="$ZONE" --quiet --command='
set -e
IMG=pixiecad-hunyuan-paint:latest
if ! sudo docker image inspect "$IMG" >/dev/null 2>&1; then
  echo "  no paint image on this host; skipping"
  exit 0
fi
if sudo docker run --rm "$IMG" python -c "import fast_simplification" 2>/dev/null; then
  echo "  fast_simplification present"
  exit 0
fi
echo "  patching $IMG with fast_simplification ..."
sudo docker rm -f pixiecad-paintfix >/dev/null 2>&1 || true
sudo docker run --name pixiecad-paintfix "$IMG" \
  pip install --no-cache-dir --quiet fast_simplification
sudo docker commit pixiecad-paintfix "$IMG" >/dev/null
sudo docker rm pixiecad-paintfix >/dev/null
sudo docker run --rm "$IMG" python -c "import fast_simplification; print(\"  patched OK\")"
' || echo "WARNING: paint dependency check failed; texturing may not work"

gcloud compute config-ssh --quiet >/dev/null 2>&1 || true
echo
echo "ready:  $NAME.$ZONE.$PROJECT"
echo "delete: gcloud compute instances delete $NAME --zone=$ZONE --quiet"
