"""Run SAM over rendered views on a remote GPU host.

Follows the same shape as the Hunyuan backend -- rsync in, one-shot container
run, rsync out -- so a single provisioned VM can generate the mesh and then
segment it without changing hosts.

The split of responsibility is deliberate: rendering and fusion are cheap,
deterministic and stay local, and only the model inference goes remote. That
keeps the expensive, hard-to-debug part of the pipeline as a pure
image-in/labels-out step, and means re-tuning the fusion threshold costs
nothing because the masks are already on disk.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from ..executors.base import Executor

from .render import Render

DEFAULT_SAM_IMAGE = "pixiecad-sam:latest"
DEFAULT_CHECKPOINT = "/opt/sam/sam_vit_h_4b8939.pth"
DEFAULT_MODEL_TYPE = "vit_h"

_SCRIPT = Path(__file__).parent / "remote_scripts" / "sam_segment.py"


class SegmentationError(RuntimeError):
    """Raised when the remote segmenter fails or returns nothing usable."""


def build_sam_script(
    *,
    image: str = DEFAULT_SAM_IMAGE,
    checkpoint: str = DEFAULT_CHECKPOINT,
    model_type: str = DEFAULT_MODEL_TYPE,
    points_per_side: int = 32,
) -> str:
    """Shell script that turns ``views/`` into ``labels/`` on the GPU host."""
    # Writes to "out": SSHExecutor rsyncs exactly that directory back, so a
    # job that puts results anywhere else looks like a silent no-op.
    return (
        "set -e\n"
        "mkdir -p out\n"
        "docker run --rm --gpus all "
        '-v "$PWD":/work '
        # Checkpoint lives on the host, not in the image: a 2.4 GB weight file
        # baked into a layer is re-pulled on every image change.
        '-v "$HOME/.cache/sam":/opt/sam '
        "-w /work "
        f"{image} python /work/sam_segment.py "
        f"--views views --out out --checkpoint {checkpoint} "
        f"--model-type {model_type} --points-per-side {int(points_per_side)}\n"
    )


class RemoteSAMSegmenter:
    """Segments a whole batch of renders in one remote call.

    Not a ``Segmenter2D``: that protocol is one image at a time, and paying an
    SSH round trip and a container start per view would dominate the runtime.
    ``segment_renders`` returns label maps for every view at once, and
    ``PrecomputedSegmenter`` adapts the result back to the per-image protocol
    that ``segment_semantic`` expects.
    """

    def __init__(
        self,
        executor: Executor,
        *,
        image: str = DEFAULT_SAM_IMAGE,
        checkpoint: str = DEFAULT_CHECKPOINT,
        model_type: str = DEFAULT_MODEL_TYPE,
        points_per_side: int = 32,
        timeout_s: int = 1800,
    ) -> None:
        self.executor = executor
        self.image = image
        self.checkpoint = checkpoint
        self.model_type = model_type
        self.points_per_side = points_per_side
        self.timeout_s = timeout_s

    def segment_renders(
        self, renders: list[Render], *, work_dir: Path | None = None
    ) -> dict[str, np.ndarray]:
        """Return ``{view_name: label_map}`` for every render."""
        from ..executors.base import Job
        from PIL import Image

        if not renders:
            return {}

        tmp_ctx = tempfile.TemporaryDirectory() if work_dir is None else None
        base = Path(work_dir) if work_dir is not None else Path(tmp_ctx.name)  # type: ignore[union-attr]
        try:
            views = base / "views"
            views.mkdir(parents=True, exist_ok=True)
            for render in renders:
                Image.fromarray(render.rgb).save(views / f"{render.camera.name}.png")

            script = base / "sam_segment.py"
            shutil.copyfile(_SCRIPT, script)
            out_dir = base / "labels"

            job = Job(
                command=[
                    "sh",
                    "-c",
                    build_sam_script(
                        image=self.image,
                        checkpoint=self.checkpoint,
                        model_type=self.model_type,
                        points_per_side=self.points_per_side,
                    ),
                ],
                inputs=[views, script],
                output_dir=out_dir,
                remote_subdir=f"sam-{abs(hash(tuple(r.camera.name for r in renders))) % 10**8:08d}",
                timeout_s=self.timeout_s,
            )
            result = self.executor.run(job)
            if not result.ok:
                raise SegmentationError(
                    f"SAM failed on the GPU host: {(result.stderr_tail or '')[-1500:]}"
                )

            labels = {}
            for render in renders:
                path = out_dir / f"{render.camera.name}.npy"
                if not path.exists():
                    raise SegmentationError(
                        f"SAM returned no labels for view '{render.camera.name}'"
                    )
                labels[render.camera.name] = np.load(path)
            return labels
        finally:
            if tmp_ctx is not None:
                tmp_ctx.cleanup()


class PrecomputedSegmenter:
    """Adapts already-computed label maps to the ``Segmenter2D`` protocol.

    ``segment_semantic`` calls a segmenter per render, but the remote path
    produces every view in one batch. Keyed on the RGB array's identity rather
    than the view name because the protocol only passes the image.
    """

    def __init__(self, renders: list[Render], labels: dict[str, np.ndarray]) -> None:
        self._by_id = {id(r.rgb): labels[r.camera.name] for r in renders if r.camera.name in labels}

    def segment(self, rgb: np.ndarray) -> np.ndarray:
        try:
            return self._by_id[id(rgb)]
        except KeyError:  # pragma: no cover - guards a caller mistake
            raise SegmentationError(
                "PrecomputedSegmenter was given a render it has no labels for; "
                "pass the same Render objects that produced the labels"
            ) from None


def make_segmenter(executor: Executor, **kwargs: Any) -> RemoteSAMSegmenter:
    """Factory; needs a live Executor so it is not constructed at import."""
    return RemoteSAMSegmenter(executor, **kwargs)
