#!/usr/bin/env bash
# Build the Hunyuan3D worker image on an already-provisioned GPU VM.
#
#   ./scripts/setup_hunyuan_vm.sh <instance-name> <zone> [project]
#
# Why we build our own instead of pulling a published image: every third-party
# Hunyuan image on Docker Hub was inspected first, and the most popular one
# opens a public Cloudflare tunnel to a vendor domain and uploads results to
# that vendor's S3 bucket. Building from Tencent's own Dockerfile keeps the
# workload on your VM and your data on your disk.
#
# Provision the VM first (L4 is enough for shape generation):
#   ./scripts/provision_gpu_vm.sh pixiecad-hy asia-southeast1-a l4 trellis
# (the 'trellis' workload profile just sizes the machine and disk; the image it
# pre-pulls is unused here and costs only disk.)

set -euo pipefail

NAME="${1:?usage: setup_hunyuan_vm.sh <instance> <zone> [project]}"
ZONE="${2:?usage: setup_hunyuan_vm.sh <instance> <zone> [project]}"
PROJECT="${3:-${PIXIECAD_GCP_PROJECT:-$(gcloud config get-value project 2>/dev/null)}}"

REPO="https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1.git"

echo "building Hunyuan3D image on $NAME ($ZONE) ..."
gcloud compute ssh "$NAME" --project="$PROJECT" --zone="$ZONE" --quiet --command='
set -e
sudo mkdir -p /opt/hunyuan
sudo chown -R $USER /opt/hunyuan
if [ ! -d /opt/hunyuan/.git ]; then
  git clone --depth 1 '"$REPO"' /opt/hunyuan
fi
cd /opt/hunyuan

# Build with a wide arch list so the image is not pinned to one GPU
# generation the way the TRELLIS image was (that one shipped sm_89 only and
# would not run on a T4, A100 or H100 at any price).
cat > /opt/hunyuan/Dockerfile.pixiecad <<DOCKER
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel
ENV DEBIAN_FRONTEND=noninteractive
ENV TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9;9.0"
ENV PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends \
      git build-essential libgl1 libglib2.0-0 libeigen3-dev ninja-build \
 && rm -rf /var/lib/apt/lists/*
COPY . /opt/hunyuan
WORKDIR /opt/hunyuan
# Drop the China mirrors from requirements: they are slow or unreachable from
# most GCP regions and PyPI has every one of these packages.
RUN sed -i "/extra-index-url/d" requirements.txt \
 && pip install --no-cache-dir -r requirements.txt
ENV PYTHONPATH=/opt/hunyuan/hy3dshape:/opt/hunyuan
DOCKER

sudo docker build -f /opt/hunyuan/Dockerfile.pixiecad -t pixiecad-hunyuan:latest /opt/hunyuan
sudo docker run --rm --gpus all pixiecad-hunyuan:latest python -c "
import torch
print(\"torch\", torch.__version__, \"cuda\", torch.cuda.is_available(), torch.cuda.get_device_name(0))
import sys; sys.path.insert(0, \"/opt/hunyuan/hy3dshape\")
from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline
print(\"hy3dshape import OK\")
"
echo BUILD_OK
'

ALIAS="$NAME.$ZONE.$PROJECT"
echo
echo "ready:  pixiecad run images/ --host $ALIAS --backend hunyuan-remote"
