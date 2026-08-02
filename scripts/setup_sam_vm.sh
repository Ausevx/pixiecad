#!/usr/bin/env bash
# Build the SAM segmentation image on an already-provisioned GPU VM.
#
#   ./scripts/setup_sam_vm.sh <instance-name> <zone> [project] [model-type]
#
# Runs happily on the same VM as the Hunyuan worker -- SAM needs a few GB of
# VRAM, so generation and segmentation share one box and one cold start.
#
# The checkpoint is downloaded to the host (~/.cache/sam) and bind-mounted in,
# not baked into the image: 2.4 GB in a layer would be re-pulled on every
# rebuild of the thin Python layer above it.

set -euo pipefail

NAME="${1:?usage: setup_sam_vm.sh <instance> <zone> [project] [model-type]}"
ZONE="${2:?usage: setup_sam_vm.sh <instance> <zone> [project] [model-type]}"
PROJECT="${3:-${PIXIECAD_GCP_PROJECT:-$(gcloud config get-value project 2>/dev/null)}}"
MODEL_TYPE="${4:-vit_h}"

case "$MODEL_TYPE" in
  vit_h) CKPT_URL="https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth" ;;
  vit_l) CKPT_URL="https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth" ;;
  vit_b) CKPT_URL="https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth" ;;
  *) echo "unknown model type: $MODEL_TYPE (vit_h|vit_l|vit_b)" >&2; exit 2 ;;
esac

REMOTE_SCRIPT=$(cat <<REMOTE
set -e
mkdir -p \$HOME/.cache/sam
CKPT=\$HOME/.cache/sam/\$(basename "$CKPT_URL")
if [ ! -s "\$CKPT" ]; then
  echo "downloading $MODEL_TYPE checkpoint ..."
  curl -fL --retry 3 -o "\$CKPT" "$CKPT_URL"
fi

mkdir -p \$HOME/sam-build
cat > \$HOME/sam-build/Dockerfile <<'DOCKEREOF'
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime
ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1
# git is needed at build time: segment-anything is not on PyPI, so pip has to
# clone it. The runtime base image does not ship git, unlike the devel one.
RUN apt-get update && apt-get install -y --no-install-recommends \
      git libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*
# runtime base, not devel: SAM ships no custom CUDA kernels, so there is
# nothing to compile and the devel image would triple the pull for nothing.
RUN pip install --no-cache-dir \
      "numpy<2" opencv-python-headless pycocotools \
      git+https://github.com/facebookresearch/segment-anything.git
DOCKEREOF

sudo docker build -t pixiecad-sam:latest \$HOME/sam-build

# Fail here rather than mid-job: an import or CUDA problem found now costs
# seconds, found during a run it costs the render upload and a VM-hour.
sudo docker run --rm --gpus all -v \$HOME/.cache/sam:/opt/sam pixiecad-sam:latest python -c "
import torch, os
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
print('torch', torch.__version__, 'cuda', torch.cuda.is_available())
ckpt = '/opt/sam/\$(basename "$CKPT_URL")'
assert os.path.exists(ckpt), ckpt
sam = sam_model_registry['$MODEL_TYPE'](checkpoint=ckpt).to('cuda')
SamAutomaticMaskGenerator(sam)
print('SAM $MODEL_TYPE load OK')
"
echo BUILD_OK
REMOTE
)

echo "building SAM image on $NAME ($ZONE) ..."
gcloud compute ssh "$NAME" --project="$PROJECT" --zone="$ZONE" --quiet \
  --command="$REMOTE_SCRIPT"

echo
echo "ready:  checkpoint at ~/.cache/sam/$(basename "$CKPT_URL")"
