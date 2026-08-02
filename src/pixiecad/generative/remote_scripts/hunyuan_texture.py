"""Runs inside the Hunyuan3D paint container on the GPU host.

Takes an untextured mesh plus the conditioning photo and writes a textured
GLB. Kept separate from ``hunyuan_gen.py`` because texturing is a distinct
stage: it can be re-run on an existing mesh without regenerating geometry,
which matters when the shape is already good and only the texture needs
another attempt.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, "/opt/hunyuan/hy3dshape")
sys.path.insert(0, "/opt/hunyuan/hy3dpaint")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", required=True)
    # More views means better coverage of concave regions but more VRAM and
    # time; 6 is upstream's default and fits an L4 comfortably.
    ap.add_argument("--max-views", type=int, default=6)
    ap.add_argument("--resolution", type=int, default=512)
    args = ap.parse_args()

    from textureGenPipeline import Hunyuan3DPaintConfig, Hunyuan3DPaintPipeline

    conf = Hunyuan3DPaintConfig(args.max_views, args.resolution)
    pipeline = Hunyuan3DPaintPipeline(conf)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    result = pipeline(
        mesh_path=args.mesh,
        image_path=args.image,
        output_mesh_path=str(out),
    )

    produced = Path(result) if result else out
    if produced != out and produced.exists():
        produced.replace(out)

    meta = {"textured": out.exists(), "max_views": args.max_views}
    try:
        import trimesh

        loaded = trimesh.load(str(out), force="mesh", process=False)
        meta["faces"] = int(len(loaded.faces))
        visual = getattr(loaded, "visual", None)
        image = getattr(getattr(visual, "material", None), "image", None)
        meta["texture_size"] = list(image.size) if image is not None else None
    except Exception as exc:  # pragma: no cover - diagnostic only
        meta["inspect_error"] = str(exc)

    (out.parent / "texture_meta.json").write_text(json.dumps(meta))
    print(f"wrote {out} {meta}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
