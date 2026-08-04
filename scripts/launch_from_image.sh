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
#
# Two hours suits a Hunyuan job, which is minutes of work on a warm image. It
# does NOT suit a cold TRELLIS run: ~9 GB image pull plus ~19 GB of weights
# before any compute starts. maxRunDuration cannot be raised on a running
# instance, and the action is DELETE, so an underestimate here does not stall
# the job -- it destroys the VM mid-download and throws the whole 28 GB away.
# Hence configurable, with the old value as the default.
MAX_RUN_SECONDS="${PIXIECAD_MAX_RUN_SECONDS:-7200}"
# Shut down from inside a little early, so the graceful path wins the race
# against the hard delete.
INNER_SLEEP=$(( MAX_RUN_SECONDS - 600 ))

STARTUP=$(mktemp)
cat > "$STARTUP" <<EOF
#!/bin/bash
(sleep ${INNER_SLEEP} && shutdown -h now) &
EOF

ONDEMAND_FLAGS=()
if [ "${PIXIECAD_ONDEMAND:-0}" != "1" ]; then
  ONDEMAND_FLAGS=(--provisioning-model=SPOT --instance-termination-action=DELETE)
fi

echo "launching '$NAME' from machine image '$IMAGE' ..."
echo "  hard stop after ${MAX_RUN_SECONDS}s (override with PIXIECAD_MAX_RUN_SECONDS)"
gcloud compute instances create "$NAME" \
  --project="$PROJECT" \
  --zone="$ZONE" \
  --source-machine-image="$IMAGE" \
  --metadata-from-file=startup-script="$STARTUP" \
  --max-run-duration="${MAX_RUN_SECONDS}s" \
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
# Self-healing dependency check, one entry per module an older baked image
# might be missing. Each of these was found the expensive way -- after a job
# had already spent GPU minutes generating a mesh -- so they are checked at
# launch, when a fix costs seconds:
#
#   fast_simplification  paint image; trimesh 5 delegates decimation to it
#   rtree                shape image; trimesh builds its triangle bounds tree
#                        with it inside ProximityQuery.on_surface, which the
#                        normal-map bake calls
#
# Both are baked into images built from scratch by setup_hunyuan_vm.sh. This
# block exists for images baked before that, which cannot be un-baked.
# Every container runs as root, so anything docker creates inside a mounted
# cache is root-owned -- and a machine image bakes that ownership in. The
# TRELLIS path then fails writing its HF token to ~/.cache/huggingface as the
# SSH user, with "Permission denied" and no hint that ownership is the cause.
# Cheap to correct at launch, so it is corrected at launch.
echo "fixing cache ownership ..."
gcloud compute ssh "$NAME" --project="$PROJECT" --zone="$ZONE" --quiet --command='
for d in ~/.cache/huggingface ~/.cache/hy3dgen ~/.u2net; do
  [ -e "$d" ] || continue
  sudo chown -R "$USER":"$USER" "$d" 2>/dev/null || true
done
echo "  caches owned by $USER"
' || echo "WARNING: could not fix cache ownership; the TRELLIS backend may fail"

echo "checking worker image dependencies ..."
gcloud compute ssh "$NAME" --project="$PROJECT" --zone="$ZONE" --quiet --command='
check_dep() {
  IMG="$1"; MOD="$2"
  if ! sudo docker image inspect "$IMG" >/dev/null 2>&1; then
    echo "  $IMG not on this host; skipping $MOD"
    return 0
  fi
  if sudo docker run --rm "$IMG" python -c "import $MOD" 2>/dev/null; then
    echo "  $MOD present in $IMG"
    return 0
  fi
  echo "  patching $IMG with $MOD ..."
  sudo docker rm -f pixiecad-depfix >/dev/null 2>&1 || true
  sudo docker run --name pixiecad-depfix "$IMG" \
    pip install --no-cache-dir --quiet "$MOD"
  sudo docker commit pixiecad-depfix "$IMG" >/dev/null
  sudo docker rm pixiecad-depfix >/dev/null
  sudo docker run --rm "$IMG" python -c "import $MOD" \
    && echo "  $MOD patched OK"
}
check_dep pixiecad-hunyuan-paint:latest fast_simplification
check_dep pixiecad-hunyuan:latest rtree

# Not a missing module but the same shape of problem: an image baked before a
# fix, which cannot be un-baked. trimesh 5.0.0 changed
# simplify_quadric_decimation'"'"'s first positional argument from a face count to
# a fraction, so Hunyuan'"'"'s paint code raises "target_reduction must be between
# 0 and 1" -- and only after the GPU has loaded the whole paint pipeline.
IMG=pixiecad-hunyuan-paint:latest
if sudo docker image inspect "$IMG" >/dev/null 2>&1; then
  if sudo docker run --rm "$IMG" grep -q "simplify_quadric_decimation(face_count=" \
       /opt/hunyuan/hy3dpaint/utils/simplify_mesh_utils.py 2>/dev/null; then
    echo "  simplify_quadric_decimation already patched in $IMG"
  else
    echo "  patching $IMG simplify_quadric_decimation call ..."
    sudo docker rm -f pixiecad-simplifyfix >/dev/null 2>&1 || true
    sudo docker run --name pixiecad-simplifyfix "$IMG" \
      sed -i "s/courent.simplify_quadric_decimation(target_count)/courent.simplify_quadric_decimation(face_count=target_count)/" \
      /opt/hunyuan/hy3dpaint/utils/simplify_mesh_utils.py
    sudo docker commit pixiecad-simplifyfix "$IMG" >/dev/null
    sudo docker rm pixiecad-simplifyfix >/dev/null
    echo "  simplify_quadric_decimation patched OK"
  fi
fi
' || echo "WARNING: dependency check failed; texturing or baking may not work"

# --project matters: without it this configures whatever the gcloud default
# happens to be, which silently leaves THIS project's alias pointing at a dead
# instance. A recreated VM reuses its name and often its IP, so the stale entry
# looks correct right up until ssh rejects the new host key.
gcloud compute config-ssh --project="$PROJECT" --quiet >/dev/null 2>&1 || true

# Prove the alias works before calling the host ready. Everything downstream
# reaches this VM through plain ssh, and a host-key mismatch surfaces there as
# "backend not available" -- a long way from the actual cause.
ALIAS="$NAME.$ZONE.$PROJECT"
if ! ssh -o BatchMode=yes -o ConnectTimeout=10 "$ALIAS" true 2>/tmp/pixiecad-ssh-check; then
  echo
  echo "WARNING: the VM is up but 'ssh $ALIAS' fails:"
  sed 's/^/  /' /tmp/pixiecad-ssh-check | tail -3
  echo "  Jobs will fail with 'backend not available' until this is fixed."
  echo "  Usually: gcloud compute config-ssh --project=$PROJECT"
fi
rm -f /tmp/pixiecad-ssh-check

echo
echo "ready:  $ALIAS"
echo "delete: gcloud compute instances delete $NAME --zone=$ZONE --quiet"
