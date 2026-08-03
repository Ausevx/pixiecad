"""Tests for baking the normal map on the GPU host.

No network and no GPU: a fake executor stands in for the host, unpacking the
same .npz files a real one would receive and running the real bake on them.
That is the point -- it exercises the actual serialisation round trip, which is
where a remote path silently produces a wrong-but-plausible map if UVs are lost
on the way out.

Meshes are kept small deliberately. The dev machine is a 16 GB MacBook Air and
the whole reason this module exists is that a full-size bake costs 3.4 GB.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import trimesh

from pixiecad.executors.base import JobResult
from pixiecad.meshops import unwrap_uv
from pixiecad.meshops.bake_remote import (
    RemoteBakeError,
    _save_mesh,
    bake_object_space_normals_remote,
    build_bake_script,
)


def _meshes():
    dense = trimesh.creation.icosphere(subdivisions=3)
    low = unwrap_uv(trimesh.creation.icosphere(subdivisions=2)).mesh
    return dense, low


class _FakeHost:
    """Runs the real bake on the payload, as the container would."""

    def __init__(self) -> None:
        self.jobs: list = []

    def run(self, job):
        self.jobs.append(job)
        from PIL import Image

        from pixiecad.meshops.bake import bake_object_space_normals

        by_name = {p.name: p for p in job.inputs}
        dense = self._load(by_name["dense.npz"])
        low = self._load(by_name["low.npz"], uv=True)

        # Resolution is carried only in the shell script, exactly as it is for
        # the real host; parsing it back proves it actually crossed over.
        res = int(job.command[-1].split("--resolution ")[1].split()[0])
        img = bake_object_space_normals(dense, low, resolution=res)

        job.output_dir.mkdir(parents=True, exist_ok=True)
        Image.fromarray(img, mode="RGB").save(job.output_dir / "normal.png")
        return JobResult(0, "ok", "", 1.0)

    @staticmethod
    def _load(path: Path, *, uv: bool = False):
        d = np.load(path)
        m = trimesh.Trimesh(
            vertices=d["vertices"].astype(np.float64),
            faces=d["faces"].astype(np.int64),
            process=False,
        )
        if uv:
            m.visual = trimesh.visual.TextureVisuals(uv=d["uv"].astype(np.float64))
        return m


class _BrokenHost:
    def __init__(self, *, ok: bool = False, empty: bool = False) -> None:
        self.ok, self.empty = ok, empty

    def run(self, job):
        if self.empty:
            job.output_dir.mkdir(parents=True, exist_ok=True)
        return JobResult(0 if self.ok else 1, "", "boom: cuda not found", 1.0)


def test_round_trip_matches_a_local_bake():
    """The remote path must produce the same map as baking locally.

    Same algorithm, same inputs -- if this ever diverges, the serialisation
    lost something rather than the maths changing.
    """
    from pixiecad.meshops.bake import bake_object_space_normals

    dense, low = _meshes()
    remote = bake_object_space_normals_remote(_FakeHost(), dense, low, resolution=64)
    local = bake_object_space_normals(dense, low, resolution=64)
    assert remote.shape == local.shape == (64, 64, 3)
    assert np.array_equal(remote, local)


def test_uvs_survive_the_transfer():
    """A bake with no UVs is uniform background; a real one is not.

    Cheap guard against the failure this module is most exposed to: UVs
    silently dropped in transit still produce a valid-looking PNG.
    """
    dense, low = _meshes()
    img = bake_object_space_normals_remote(_FakeHost(), dense, low, resolution=64)
    assert len(np.unique(img.reshape(-1, 3), axis=0)) > 1


def test_missing_uvs_are_refused_before_any_upload():
    dense, _ = _meshes()
    host = _FakeHost()
    with pytest.raises(RemoteBakeError, match="no UVs"):
        bake_object_space_normals_remote(host, dense, dense, resolution=64)
    assert host.jobs == [], "must not ship a payload it already knows is unusable"


def test_failure_does_not_silently_fall_back_to_local():
    """Choosing the host is a statement about THIS machine's memory.

    Quietly baking locally on failure would spend the 3.4 GB the user chose
    the host to avoid, so it has to raise.
    """
    dense, low = _meshes()
    with pytest.raises(RemoteBakeError, match="cuda not found"):
        bake_object_space_normals_remote(_BrokenHost(), dense, low, resolution=64)


def test_success_without_an_image_is_an_error():
    dense, low = _meshes()
    with pytest.raises(RemoteBakeError, match="no image"):
        bake_object_space_normals_remote(
            _BrokenHost(ok=True, empty=True), dense, low, resolution=64
        )


def test_bake_script_requests_no_gpu():
    """The bake does no CUDA work. Asking for --gpus would queue this job
    behind whatever actually needs the device, for nothing."""
    script = build_bake_script(resolution=1024)
    assert "--gpus" not in script
    assert "--resolution 1024" in script


def test_payload_is_compressed_float32(tmp_path):
    """Vertices ship as float32: half the upload, and still far finer than the
    texel grid the bake samples onto."""
    dense, _ = _meshes()
    out = tmp_path / "d.npz"
    _save_mesh(out, dense)
    d = np.load(out)
    assert d["vertices"].dtype == np.float32
    assert d["faces"].dtype == np.int32
