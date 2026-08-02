#!/usr/bin/env bash
# Capture a fully-built worker VM as a reusable GCP machine image.
#
#   ./scripts/bake_image.sh <instance> <zone> [image-name] [project]
#
# Why: a fresh VM currently installs docker, builds a 24 GB Hunyuan image from
# Tencent's source, compiles two CUDA extensions, and downloads ~10 GB of model
# weights -- about 25 minutes of identical work every single time, billed at the
# GPU rate even though the GPU is idle for nearly all of it. Measured on this
# project: a cold run is ~35 minutes, of which ~7 is actual compute.
#
# Baking that once turns every later launch into a ~2 minute boot.
#
# Cost shape (indicative GCP rates, verify against your billing):
#   machine image storage   ~$0.05/GB-month on the compressed size; our ~60 GB
#                           disk stores for roughly $2-3/month
#   a stopped VM instead    pays full persistent-disk price -- a 200 GB
#                           pd-balanced disk is ~$20/month, ten times more, for
#                           the same benefit
# So an image is the cheaper way to keep a machine "ready" than keeping a
# stopped VM around.
#
# It also changes the spot calculus. Preemption deletes the boot disk, which
# today means losing the whole build; from an image, a preemption costs a
# 2-minute relaunch, so spot (~4x cheaper than on-demand) becomes the sensible
# default rather than a gamble.

set -euo pipefail

NAME="${1:?usage: bake_image.sh <instance> <zone> [image-name] [project]}"
ZONE="${2:?usage: bake_image.sh <instance> <zone> [image-name] [project]}"
IMAGE="${3:-pixiecad-worker-$(date +%Y%m%d)}"
PROJECT="${4:-${PIXIECAD_GCP_PROJECT:-$(gcloud config get-value project 2>/dev/null)}}"

echo "baking '$NAME' ($ZONE) into machine image '$IMAGE' ..."

# Stop first: imaging a running instance can capture half-written files, and
# the docker graph is exactly the kind of thing that corrupts quietly.
STATUS=$(gcloud compute instances describe "$NAME" --zone="$ZONE" --project="$PROJECT" \
  --format="value(status)" 2>/dev/null || echo "")
if [ -z "$STATUS" ]; then
  echo "no such instance: $NAME in $ZONE" >&2
  exit 2
fi
if [ "$STATUS" = "RUNNING" ]; then
  echo "stopping instance for a consistent image ..."
  gcloud compute instances stop "$NAME" --zone="$ZONE" --project="$PROJECT" --quiet
fi

gcloud compute machine-images create "$IMAGE" \
  --source-instance="$NAME" \
  --source-instance-zone="$ZONE" \
  --project="$PROJECT" \
  --description="PixieCAD worker: docker + hunyuan shape/paint + SAM, prebuilt"

echo
echo "baked: $IMAGE"
echo "launch:  ./scripts/launch_from_image.sh <new-name> $ZONE $IMAGE"
echo "delete:  gcloud compute machine-images delete $IMAGE --quiet"
echo "the source instance is now STOPPED and still costs disk; delete it with:"
echo "         gcloud compute instances delete $NAME --zone=$ZONE --quiet"
