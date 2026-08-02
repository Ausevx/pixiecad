"""Runs inside the Hunyuan3D container on the GPU host. Not imported locally.

Staged next to the input images by ``hunyuan_remote`` and executed there, so
this file can change without rebuilding the container image.

Handles both models:

- single view  -> Hunyuan3D-2.1 via ``hy3dshape``   (newer, better geometry)
- multi view   -> Hunyuan3D-2mv via ``hy3dgen``     (uses up to 4 real views)

They live in different packages because the multi-view checkpoint was only
ever released for the 2.x line, so both repos are on the image.

Shape only, deliberately. The paint pipeline pulls in a custom CUDA rasterizer
and a RealESRGAN checkpoint, and PixieCAD already owns UV unwrap and normal
baking downstream — geometry is the part only the GPU can provide.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Both repos on the path: 2.1 for single view, 2 for the multi-view checkpoint.
sys.path.insert(0, "/opt/hunyuan/hy3dshape")
sys.path.insert(0, "/opt/hunyuan2")


def _load(path: str):
    from PIL import Image

    from hy3dshape.rembg import BackgroundRemover

    image = Image.open(path)
    # The pipeline conditions on a cut-out subject; an opaque background gets
    # modelled as geometry.
    if image.mode != "RGBA" or image.getextrema()[3][0] == 255:
        image = BackgroundRemover()(image.convert("RGB"))
    return image


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", help="single conditioning view")
    ap.add_argument("--views", help="JSON mapping view tag -> filename")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="tencent/Hunyuan3D-2.1")
    ap.add_argument("--subfolder", default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--octree-resolution", type=int, default=256)
    ap.add_argument("--guidance-scale", type=float, default=5.0)
    args = ap.parse_args()

    import torch

    kwargs: dict = {
        "num_inference_steps": args.steps,
        "octree_resolution": args.octree_resolution,
        "guidance_scale": args.guidance_scale,
    }
    if args.seed is not None:
        kwargs["generator"] = torch.manual_seed(args.seed)

    if args.views:
        # Multi-view: hy3dgen's preprocessor keys on these exact tags and
        # sorts them into its own view order, so tags matter, order does not.
        from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline

        mapping = json.loads(Path(args.views).read_text())
        images = {tag: _load(str(Path("images") / name)) for tag, name in mapping.items()}
        pipe = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
            args.model,
            subfolder=args.subfolder or "hunyuan3d-dit-v2-mv",
            use_safetensors=True,
            device="cuda",
        )
        kwargs["image"] = images
        used = sorted(images)
    else:
        from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline

        pipe = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(args.model)
        kwargs["image"] = _load(args.image)
        used = [Path(args.image).name]

    mesh = pipe(**kwargs)[0]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(str(out))

    # Report proportions so the caller can reject a degenerate result without
    # loading the mesh itself.
    meta = {
        "faces": int(len(mesh.faces)),
        "vertices": int(len(mesh.vertices)),
        "views_used": used,
    }
    try:
        meta["extents"] = [float(v) for v in mesh.extents]
    except Exception:
        pass
    (out.parent / "gen_meta.json").write_text(json.dumps(meta))
    print(f"wrote {out} faces={meta['faces']} views={used}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
