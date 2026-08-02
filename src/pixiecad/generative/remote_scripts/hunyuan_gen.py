"""Runs inside the Hunyuan3D container on the GPU host. Not imported locally.

Staged next to the input images by ``hunyuan_remote`` and executed there, so
this file can change without rebuilding the container image.

Shape only, deliberately. The paint pipeline pulls in a custom CUDA rasterizer
and a RealESRGAN checkpoint, and PixieCAD already owns UV unwrap and normal
baking downstream — geometry is the part only the GPU can provide.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, "/opt/hunyuan/hy3dshape")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="tencent/Hunyuan3D-2.1")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--octree-resolution", type=int, default=256)
    ap.add_argument("--guidance-scale", type=float, default=5.0)
    args = ap.parse_args()

    import torch
    from PIL import Image

    from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline
    from hy3dshape.rembg import BackgroundRemover

    image = Image.open(args.image)
    # The pipeline conditions on a cut-out subject; feeding it an opaque
    # background makes it model the backdrop as geometry.
    if image.mode != "RGBA" or image.getextrema()[3][0] == 255:
        image = BackgroundRemover()(image.convert("RGB"))

    pipe = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(args.model)

    kwargs = {
        "image": image,
        "num_inference_steps": args.steps,
        "octree_resolution": args.octree_resolution,
        "guidance_scale": args.guidance_scale,
    }
    if args.seed is not None:
        kwargs["generator"] = torch.manual_seed(args.seed)

    mesh = pipe(**kwargs)[0]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(str(out))

    # Report proportions so the caller can reject a degenerate result without
    # loading the mesh itself.
    meta = {"faces": int(len(mesh.faces)), "vertices": int(len(mesh.vertices))}
    try:
        ext = mesh.extents
        meta["extents"] = [float(v) for v in ext]
    except Exception:
        pass
    Path(out.parent / "gen_meta.json").write_text(json.dumps(meta))
    print(f"wrote {out} faces={meta['faces']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
