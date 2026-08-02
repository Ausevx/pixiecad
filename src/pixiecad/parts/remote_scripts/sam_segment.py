"""Runs inside the container on the GPU host. Not imported locally.

Reads ``views/*.png``, runs SAM's automatic mask generator on each, and writes
one ``labels/<stem>.npy`` per view holding an int32 label map (-1 background).

The one non-obvious step is flattening. SAM returns a *list of overlapping*
masks -- a wheel, its rim, and the whole corner of the car can all be returned
at once, nested inside each other. The fusion step downstream treats two masks
in the same view as evidence that they are different parts, which is only
sound if each view is a partition. So masks are painted largest-first and
allowed to overwrite: the smallest mask covering a pixel wins, nesting is
resolved toward the most specific part, and every pixel ends up in exactly one
mask.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--views", default="views")
    ap.add_argument("--out", default="labels")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--model-type", default="vit_h")
    ap.add_argument("--points-per-side", type=int, default=32)
    # Below this fraction of the frame a mask is a shading artefact rather
    # than a part; SAM happily returns dozens of them on a creased surface.
    ap.add_argument("--min-area-frac", type=float, default=0.002)
    args = ap.parse_args()

    import cv2
    import torch
    from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

    sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
    sam.to("cuda" if torch.cuda.is_available() else "cpu")
    generator = SamAutomaticMaskGenerator(
        sam,
        points_per_side=args.points_per_side,
        # Renders are clean synthetic images with no texture noise, so these
        # sit above the library defaults: a mask we accept here should be one
        # SAM is confident about, and false parts cost more than missed ones.
        pred_iou_thresh=0.88,
        stability_score_thresh=0.92,
    )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {}

    for path in sorted(Path(args.views).glob("*.png")):
        image = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]
        min_area = args.min_area_frac * h * w

        masks = [m for m in generator.generate(image) if m["area"] >= min_area]
        # Largest first, so smaller masks paint over them; see module docstring.
        masks.sort(key=lambda m: m["area"], reverse=True)

        labels = np.full((h, w), -1, dtype=np.int32)
        for i, m in enumerate(masks):
            labels[m["segmentation"]] = i

        np.save(out_dir / f"{path.stem}.npy", labels)
        summary[path.stem] = len(masks)
        print(f"{path.stem}: {len(masks)} masks", flush=True)

    (out_dir / "summary.json").write_text(json.dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
