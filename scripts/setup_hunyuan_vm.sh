#!/usr/bin/env bash
# Build the Hunyuan3D worker image on an already-provisioned GPU VM.
#
#   ./scripts/setup_hunyuan_vm.sh <instance-name> <zone> [project]
#
# Why we build our own instead of pulling a published image: every third-party
# Hunyuan image on Docker Hub was inspected first, and the most popular one
# opens a public Cloudflare tunnel to a vendor domain and uploads results to
# that vendor's S3 bucket. Building from Tencent's own source keeps the
# workload on your VM and your data on your disk.
#
# Provision the VM first:
#   ./scripts/provision_gpu_vm.sh pixiecad-hy asia-southeast1-b l4 hunyuan

set -euo pipefail

NAME="${1:?usage: setup_hunyuan_vm.sh <instance> <zone> [project]}"
ZONE="${2:?usage: setup_hunyuan_vm.sh <instance> <zone> [project]}"
PROJECT="${3:-${PIXIECAD_GCP_PROJECT:-$(gcloud config get-value project 2>/dev/null)}}"

REPO="https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1.git"

REMOTE_SCRIPT=$(cat <<'REMOTE'
set -e
sudo mkdir -p /opt/hunyuan
sudo chown -R $USER /opt/hunyuan
if [ ! -d /opt/hunyuan/.git ]; then
  git clone --depth 1 __REPO__ /opt/hunyuan
fi
cd /opt/hunyuan

cat > /opt/hunyuan/Dockerfile.pixiecad <<'DOCKEREOF'
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel
ENV DEBIAN_FRONTEND=noninteractive
# Wide arch list on purpose: the TRELLIS image we tried first shipped sm_89
# cubins only, which silently excluded every GPU except the L4.
ENV TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9;9.0"
ENV PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends \
      git build-essential libgl1 libglib2.0-0 libeigen3-dev ninja-build \
 && rm -rf /var/lib/apt/lists/*
COPY . /opt/hunyuan
WORKDIR /opt/hunyuan
# Shape generation only, so the texture and UI stack is deliberately excluded:
#   bpy          Blender, for texture baking -- and has no py3.11 build at all
#   realesrgan   texture upscaling
#   basicsr      pulled in by realesrgan; imports a torchvision module that has
#                since been removed, so it breaks on any current torchvision
#   gradio/fastapi/uvicorn/pythreejs   the demo web UI
#   deepspeed    training only
# Pins are relaxed as well: upstream pins numpy 1.24.4 alongside scipy 1.14.1,
# which cannot both be satisfied here. numpy stays below 2 because pymeshlab
# still expects the old ABI.
RUN pip install --no-cache-dir \
      "numpy<2" scipy pandas \
      transformers==4.46.0 diffusers==0.30.0 accelerate \
      huggingface_hub safetensors einops omegaconf pyyaml configargparse \
      trimesh pygltflib xatlas pymeshlab \
      opencv-python-headless imageio scikit-image \
      onnxruntime rembg torchdiffeq timm tqdm psutil \
      pytorch-lightning
ENV PYTHONPATH=/opt/hunyuan/hy3dshape:/opt/hunyuan
DOCKEREOF

sudo docker build -f /opt/hunyuan/Dockerfile.pixiecad -t pixiecad-hunyuan:latest /opt/hunyuan

# Prove the GPU and the shape pipeline both work before any job is submitted:
# an import error found here costs seconds, but found mid-build it costs a
# provisioned VM and a model download.
sudo docker run --rm --gpus all pixiecad-hunyuan:latest python -c "
import torch, sys
print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0))
sys.path.insert(0, '/opt/hunyuan/hy3dshape')
from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline
print('hy3dshape import OK')
"
echo BUILD_OK
REMOTE
)

echo "building Hunyuan3D image on $NAME ($ZONE) ..."
gcloud compute ssh "$NAME" --project="$PROJECT" --zone="$ZONE" --quiet \
  --command="${REMOTE_SCRIPT//__REPO__/$REPO}"

ALIAS="$NAME.$ZONE.$PROJECT"
echo
echo "ready:  pixiecad run images/ --host $ALIAS --backend hunyuan-remote"
