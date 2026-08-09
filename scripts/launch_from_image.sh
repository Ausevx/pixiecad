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

# L4 capacity runs out per-zone, and when it does the request fails outright:
#
#   ZONE_RESOURCE_POOL_EXHAUSTED_WITH_DETAILS ... reason: stockout
#
# That is not quota and not a bad request -- the zone simply has no L4 left,
# and asia-southeast1-b (the zone every earlier run used) hit it. Since the
# error is indistinguishable from a hard failure to anyone reading the log,
# sweep instead: try the requested zone first, then its siblings, and say
# which one won. Paying ~4x for on-demand to dodge a stockout is the wrong
# trade when a neighbouring zone usually has stock.
CANDIDATE_ZONES=("$ZONE")
for z in "${PIXIECAD_ZONE_SWEEP:-asia-southeast1-a asia-southeast1-b asia-southeast1-c asia-south1-a asia-south1-b asia-south1-c}"; do
  for zz in $z; do
    [ "$zz" = "$ZONE" ] || CANDIDATE_ZONES+=("$zz")
  done
done

echo "launching '$NAME' from machine image '$IMAGE' ..."
echo "  hard stop after ${MAX_RUN_SECONDS}s (override with PIXIECAD_MAX_RUN_SECONDS)"

LAUNCHED=""
for z in "${CANDIDATE_ZONES[@]}"; do
  echo "  trying $z ..."
  if OUT=$(gcloud compute instances create "$NAME" \
      --project="$PROJECT" \
      --zone="$z" \
      --source-machine-image="$IMAGE" \
      --metadata-from-file=startup-script="$STARTUP" \
      --max-run-duration="${MAX_RUN_SECONDS}s" \
      --instance-termination-action=DELETE \
      "${ONDEMAND_FLAGS[@]}" \
      --quiet 2>&1); then
    ZONE="$z"
    LAUNCHED=yes
    echo "  launched in $ZONE"
    break
  fi
  if echo "$OUT" | grep -q "ZONE_RESOURCE_POOL_EXHAUSTED\|stockout"; then
    echo "    no L4 capacity in $z, trying the next zone"
    continue
  fi
  # Anything else -- quota, permissions, a bad image name -- is not going to
  # be fixed by moving zones, so fail loudly rather than sweeping through
  # every zone printing the same error six times.
  echo "$OUT" >&2
  rm -f "$STARTUP"
  exit 1
done
rm -f "$STARTUP"

if [ -z "$LAUNCHED" ]; then
  echo "no L4 capacity in any candidate zone: ${CANDIDATE_ZONES[*]}" >&2
  echo "retry later, or set PIXIECAD_ONDEMAND=1 to pay list price for guaranteed stock" >&2
  exit 1
fi

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

# Photogrammetry, for the >=16-photo regime. The dense stage shells out to
# this exact tag (geometry/dense.py: DOCKER_COLMAP_IMAGE), and sparse SfM runs
# locally via pycolmap, so only the dense half needs to be on the host.
# Missing it means an orbit capture dies at S2b after the user has already
# shot 40 photos.
if ! sudo docker image inspect colmap/colmap:latest >/dev/null 2>&1; then
  echo "  pulling colmap/colmap:latest (photogrammetry dense stage) ..."
  sudo docker pull colmap/colmap:latest >/dev/null 2>&1 \
    && echo "  colmap pulled OK" || echo "  WARNING: colmap pull failed; S2b dense will not run"
else
  echo "  colmap/colmap:latest present"
fi
# ffmpeg turns a 30-second orbit video into the 20-40 frames the photogrammetry
# regime needs, which is a far easier capture than 40 deliberate stills.
command -v ffmpeg >/dev/null 2>&1 || {
  echo "  installing ffmpeg ..."
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq ffmpeg >/dev/null 2>&1 \
    && echo "  ffmpeg installed" || echo "  WARNING: ffmpeg install failed"
}

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

  # The paint stage remeshes before texturing, to a target_count HARDCODED at
  # 40,000, and we then publish that mesh as the model. So a job that asked for
  # 300,000 faces, built them and exported them, came back with 39,997 the
  # moment texturing was on -- and the only clue was a face count.
  #
  # PixieCAD passes -e HUNYUAN_SIMPLIFY_TARGET, but the image had no code to
  # read it: the variable was set on every run and did nothing at all for weeks.
  # Teach the image to honour it, defaulting to the upstream 40,000.
  if sudo docker run --rm "$IMG" grep -q "HUNYUAN_SIMPLIFY_TARGET" \
       /opt/hunyuan/hy3dpaint/utils/simplify_mesh_utils.py 2>/dev/null; then
    echo "  HUNYUAN_SIMPLIFY_TARGET already honoured in $IMG"
  else
    echo "  patching $IMG to honour HUNYUAN_SIMPLIFY_TARGET ..."
    sudo docker rm -f pixiecad-targetfix >/dev/null 2>&1 || true
    sudo docker run --name pixiecad-targetfix "$IMG" python -c "
import pathlib
p = pathlib.Path(\"/opt/hunyuan/hy3dpaint/utils/simplify_mesh_utils.py\")
s = p.read_text()
s = s.replace(\"import trimesh\", \"import os\nimport trimesh\", 1)
s = s.replace(
    \"def mesh_simplify_trimesh(inputpath, outputpath, target_count=40000):\",
    \"def mesh_simplify_trimesh(inputpath, outputpath, target_count=None):\n\"
    \"    if target_count is None:\n\"
    \"        target_count = int(os.environ.get(\x27HUNYUAN_SIMPLIFY_TARGET\x27, 40000))\",
)
p.write_text(s)
"
    sudo docker commit pixiecad-targetfix "$IMG" >/dev/null
    sudo docker rm pixiecad-targetfix >/dev/null
    sudo docker run --rm "$IMG" grep -q "HUNYUAN_SIMPLIFY_TARGET" \
      /opt/hunyuan/hy3dpaint/utils/simplify_mesh_utils.py \
      && echo "  HUNYUAN_SIMPLIFY_TARGET patched OK"
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
