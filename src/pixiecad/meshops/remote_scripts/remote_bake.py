"""Runner executed inside the Hunyuan container to bake a normal map.

Kept deliberately thin. The baking algorithm itself is ``bake.py``, shipped
alongside this file and imported unchanged, so the remote result is produced by
exactly the same code as a local bake rather than by a second implementation
that could drift.

Meshes arrive as .npz rather than .glb on purpose: a GLB round-trip has to
preserve UVs through TextureVisuals, and if it ever silently failed to, the
bake would produce a plausible-looking but wrong map. Raw arrays cannot lose
them.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import trimesh


def _load(path: Path, *, with_uv: bool) -> trimesh.Trimesh:
    data = np.load(path)
    mesh = trimesh.Trimesh(
        vertices=data["vertices"].astype(np.float64),
        faces=data["faces"].astype(np.int64),
        process=False,
    )
    if with_uv:
        mesh.visual = trimesh.visual.TextureVisuals(uv=data["uv"].astype(np.float64))
    return mesh


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dense", required=True)
    ap.add_argument("--low", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--resolution", type=int, default=1024)
    ap.add_argument("--padding", type=int, default=4)
    args = ap.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from bake import bake_object_space_normals, save_normal_map

    dense = _load(Path(args.dense), with_uv=False)
    low = _load(Path(args.low), with_uv=True)
    print(f"dense {len(dense.faces)} faces, low {len(low.faces)} faces", flush=True)

    img = bake_object_space_normals(
        dense, low, resolution=args.resolution, padding_px=args.padding
    )
    save_normal_map(img, Path(args.out))
    print(f"wrote {args.out} ({img.shape[1]}x{img.shape[0]})", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
