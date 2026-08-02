#!/usr/bin/env bash
# Add Hunyuan3D-2.1's texture (paint) stack on top of the shape image.
#
#   ./scripts/setup_hunyuan_texture.sh <instance-name> <zone> [project]
#
# Run scripts/setup_hunyuan_vm.sh first; this builds FROM that image.
#
# Kept as a separate layer on purpose. The shape pipeline is pure Python and
# builds in a couple of minutes; the paint pipeline needs two compiled
# extensions and a 64 MB upscaler checkpoint, and it is the part that breaks.
# Splitting them means a paint failure never costs a working shape image.
#
# Three things the paint stack needs that shape does not:
#   custom_rasterizer       CUDA extension, compiled against the local torch
#   DifferentiableRenderer  pybind11 C++ module for texture inpainting
#   RealESRGAN_x4plus.pth   upscaler, downloaded to the host cache

set -euo pipefail

NAME="${1:?usage: setup_hunyuan_texture.sh <instance> <zone> [project]}"
ZONE="${2:?usage: setup_hunyuan_texture.sh <instance> <zone> [project]}"
PROJECT="${3:-${PIXIECAD_GCP_PROJECT:-$(gcloud config get-value project 2>/dev/null)}}"

ESRGAN_URL="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"

REMOTE_SCRIPT=$(cat <<REMOTE
set -e
cd /opt/hunyuan

# The checkpoint lives in the build context so the image is self-contained at
# run time; it is only 64 MB, unlike the multi-GB diffusion weights which stay
# on a bind mount.
mkdir -p /opt/hunyuan/hy3dpaint/ckpt
if [ ! -s /opt/hunyuan/hy3dpaint/ckpt/RealESRGAN_x4plus.pth ]; then
  curl -fL --retry 3 -o /opt/hunyuan/hy3dpaint/ckpt/RealESRGAN_x4plus.pth "$ESRGAN_URL"
fi

cat > /opt/hunyuan/Dockerfile.paint <<'DOCKEREOF'
FROM pixiecad-hunyuan:latest
ENV TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9;9.0"
# realesrgan pulls basicsr, which imports torchvision.transforms.functional_tensor
# -- removed from torchvision years ago, so the import dies on any current
# version. Installing without deps and patching the import is the accepted fix;
# the module only ever used rgb_to_grayscale from it.
RUN pip install --no-cache-dir realesrgan==0.3.0 --no-deps \
 && pip install --no-cache-dir "basicsr==1.4.2" --no-deps \
 && pip install --no-cache-dir addict yapf future lmdb tb-nightly \
 && python - <<'PY'
import pathlib, basicsr
p = pathlib.Path(basicsr.__file__).parent / "data" / "degradations.py"
s = p.read_text()
p.write_text(s.replace(
    "from torchvision.transforms.functional_tensor import rgb_to_grayscale",
    "from torchvision.transforms.functional import rgb_to_grayscale"))
print("patched", p)
PY

COPY hy3dpaint /opt/hunyuan/hy3dpaint

# Compiled against the torch already in the image, so this must happen here
# rather than being copied in from anywhere.
RUN cd /opt/hunyuan/hy3dpaint/custom_rasterizer && pip install --no-cache-dir -e .
RUN cd /opt/hunyuan/hy3dpaint/DifferentiableRenderer && bash compile_mesh_painter.sh

ENV PYTHONPATH=/opt/hunyuan/hy3dshape:/opt/hunyuan/hy3dpaint:/opt/hunyuan:/opt/hunyuan2
DOCKEREOF

sudo docker build -f /opt/hunyuan/Dockerfile.paint -t pixiecad-hunyuan-paint:latest /opt/hunyuan

# Prove both compiled extensions import before any job is submitted.
sudo docker run --rm --gpus all pixiecad-hunyuan-paint:latest python -c "
import custom_rasterizer
print('custom_rasterizer OK')
import sys; sys.path.insert(0, '/opt/hunyuan/hy3dpaint')
from DifferentiableRenderer.MeshRender import MeshRender
print('DifferentiableRenderer OK')
from textureGenPipeline import Hunyuan3DPaintPipeline, Hunyuan3DPaintConfig
print('paint pipeline import OK')
"
echo PAINT_BUILD_OK
REMOTE
)

echo "building Hunyuan3D paint image on $NAME ($ZONE) ..."
gcloud compute ssh "$NAME" --project="$PROJECT" --zone="$ZONE" --quiet \
  --command="$REMOTE_SCRIPT"
