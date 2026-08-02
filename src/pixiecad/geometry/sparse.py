"""Stage 2a: Sparse Structure-from-Motion via pycolmap."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from ..workspace import Workspace, fingerprint_file


class SparseFailure(Exception):
    """Raised when pycolmap sparse reconstruction fails to produce a model."""


@dataclass
class SparseResult:
    n_images_in: int
    n_registered: int
    n_points3d: int
    mean_reproj_error: float
    model_dir: str
    cached: bool = False


def run_sparse(
    images_dir: Path,
    workspace: Workspace,
    *,
    matcher: str = "exhaustive",
) -> SparseResult:
    if matcher not in ("exhaustive", "sequential"):
        raise ValueError(f"Unknown matcher: {matcher}")

    images_dir = Path(images_dir)
    images = sorted(
        p
        for p in images_dir.iterdir()
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if len(images) < 3:
        raise ValueError(
            f"At least 3 images required for sparse SfM, found {len(images)} in {images_dir}"
        )

    fingerprints = [fingerprint_file(p) for p in images]
    params = {"matcher": matcher}
    run = workspace.begin_stage("s2-sparse", params, fingerprints)

    result_path = run.dir / "result.json"
    if run.cached:
        data = json.loads(result_path.read_text())
        data["cached"] = True
        return SparseResult(**data)

    import pycolmap  # Lazy import inside run_sparse

    database_path = run.dir / "database.db"
    sparse_dir = run.dir / "sparse"
    sparse_dir.mkdir(parents=True, exist_ok=True)

    pycolmap.extract_features(database_path, images_dir)

    if matcher == "exhaustive":
        pycolmap.match_exhaustive(database_path)
    elif matcher == "sequential":
        pycolmap.match_sequential(database_path)
    else:
        raise ValueError(f"Unknown matcher: {matcher}")

    maps = pycolmap.incremental_mapping(database_path, images_dir, sparse_dir)

    if not maps:
        raise SparseFailure(
            f"Sparse reconstruction failed: pycolmap produced no valid reconstructions "
            f"from {len(images)} images in {images_dir}. Check photo overlap and quality."
        )

    rec = max(maps.values(), key=lambda r: r.num_reg_images())

    n_registered = rec.num_reg_images()
    n_points3d = rec.num_points3D()
    try:
        mean_reproj_error = float(rec.compute_mean_reprojection_error())
    except (AttributeError, TypeError):
        mean_reproj_error = 0.0

    result = SparseResult(
        n_images_in=len(images),
        n_registered=n_registered,
        n_points3d=n_points3d,
        mean_reproj_error=mean_reproj_error,
        model_dir=str(sparse_dir),
        cached=False,
    )

    result_path.write_text(json.dumps(asdict(result), indent=2) + "\n")
    workspace.finish_stage(run, asdict(result))

    return result
