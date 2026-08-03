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
# The multi-view checkpoint only exists for the 2.x line and is loaded by
# hy3dgen, a different package from 2.1's hy3dshape. Both repos ship so one
# image can serve single-view (better geometry) and multi-view (uses your
# actual photos) without a rebuild.
sudo mkdir -p /opt/hunyuan2
sudo chown -R $USER /opt/hunyuan2
if [ ! -d /opt/hunyuan2/.git ]; then
  git clone --depth 1 https://github.com/Tencent-Hunyuan/Hunyuan3D-2.git /opt/hunyuan2
fi
cp -r /opt/hunyuan2 /opt/hunyuan/_hunyuan2_src
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
#
# trimesh is pinned because leaving it open already cost a run. Unpinned, it
# drifted to 5.0.0, where simplify_quadric_decimation stopped being a built-in
# and became a delegating import to fast_simplification -- so an image that was
# correct when built started failing to texture, silently, after the GPU had
# already loaded the paint pipeline. 5.0.0 is the version actually verified in
# pixiecad-worker-v2; move the pin deliberately, not by accident.
RUN pip install --no-cache-dir \
      "numpy<2" scipy pandas \
      transformers==4.46.0 diffusers==0.30.0 accelerate \
      huggingface_hub safetensors einops omegaconf pyyaml configargparse \
      "trimesh==5.0.0" pygltflib xatlas pymeshlab \
      opencv-python-headless imageio scikit-image \
      onnxruntime rembg torchdiffeq timm tqdm psutil \
      pytorch-lightning
RUN cp -r /opt/hunyuan/_hunyuan2_src /opt/hunyuan2
ENV PYTHONPATH=/opt/hunyuan/hy3dshape:/opt/hunyuan:/opt/hunyuan2
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
sys.path.insert(0, '/opt/hunyuan2')
from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline as MV
print('hy3dgen (multi-view) import OK')
"

# Pull the weights NOW, into the host cache, so a bake captures them.
#
# The smoke test above only imports; it never calls from_pretrained, and it
# runs --rm. So nothing was ever downloaded at build time and the first real
# job on every fresh VM paid for ~10 GB of HuggingFace traffic at the GPU
# rate. Worse, whether a machine image contained the weights depended on
# whether someone happened to run a job before baking it -- which is not a
# property you want decided by accident.
#
# The mounts match the ones hunyuan_remote.py uses at job time, so this warms
# exactly the directories a job will later read. Both live on the boot disk,
# which is what a machine image captures.
echo "prefetching model weights into the host cache (~10 GB, once) ..."
mkdir -p "$HOME/.cache/huggingface" "$HOME/.cache/hy3dgen"
sudo docker run --rm --gpus all \
  -v "$HOME/.cache/huggingface":/root/.cache/huggingface \
  -v "$HOME/.cache/hy3dgen":/root/.cache/hy3dgen \
  pixiecad-hunyuan:latest python -c "
import sys
sys.path.insert(0, '/opt/hunyuan/hy3dshape')
from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline
Hunyuan3DDiTFlowMatchingPipeline.from_pretrained('tencent/Hunyuan3D-2.1')
print('shape weights cached')
"
sudo chown -R "$USER":"$USER" "$HOME/.cache/huggingface" "$HOME/.cache/hy3dgen" 2>/dev/null || true
du -sh "$HOME/.cache/huggingface" "$HOME/.cache/hy3dgen" 2>/dev/null || true

echo BUILD_OK
REMOTE
)

echo "building Hunyuan3D image on $NAME ($ZONE) ..."
gcloud compute ssh "$NAME" --project="$PROJECT" --zone="$ZONE" --quiet \
  --command="${REMOTE_SCRIPT//__REPO__/$REPO}"

ALIAS="$NAME.$ZONE.$PROJECT"
echo
echo "ready:  pixiecad run images/ --host $ALIAS --backend hunyuan-remote"
