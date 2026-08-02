"""Tests for generative 3D backend base interface, registry, and caching."""

from pathlib import Path

import pytest
import trimesh

from pixiecad.generative import (
    BackendUnavailable,
    FakeBackend,
    GenerateRequest,
    GenerateResult,
    GenerativeBackend,
    GenerativeError,
    available_backends,
    get_backend,
    register_backend,
    run_generate,
)
from pixiecad.generative.base import _REGISTRY
from pixiecad.spec import ObjectSpec
from pixiecad.workspace import Workspace


@pytest.fixture(autouse=True)
def preserve_registry():
    """Fixture to ensure tests don't permanently alter the global registry."""
    snapshot = dict(_REGISTRY)
    yield
    _REGISTRY.clear()
    _REGISTRY.update(snapshot)


def test_generate_request_validation(tmp_path: Path):
    img1 = tmp_path / "img1.png"
    img1.write_text("dummy1")
    img2 = tmp_path / "img2.png"
    img2.write_text("dummy2")
    img3 = tmp_path / "img3.png"
    img3.write_text("dummy3")
    img4 = tmp_path / "img4.png"
    img4.write_text("dummy4")
    img5 = tmp_path / "img5.png"
    img5.write_text("dummy5")
    nonexistent = tmp_path / "missing.png"

    # 0 images raises ValueError
    with pytest.raises(ValueError, match="1 to 4 images"):
        GenerateRequest(images=[])

    # 5 images raises ValueError
    with pytest.raises(ValueError, match="1 to 4 images"):
        GenerateRequest(images=[img1, img2, img3, img4, img5])

    # Missing file raises ValueError
    with pytest.raises(ValueError, match="does not exist"):
        GenerateRequest(images=[nonexistent])

    # 1 to 4 valid images succeed
    req1 = GenerateRequest(images=[img1])
    assert len(req1.images) == 1

    req4 = GenerateRequest(images=[img1, img2, img3, img4])
    assert len(req4.images) == 4


def test_registry_register_and_available():
    class DummyBackend:
        name = "dummy"

        def available(self) -> bool:
            return True

        def generate(self, req: GenerateRequest, out_dir: Path) -> GenerateResult:
            return GenerateResult(
                mesh_path=str(out_dir / "mesh.glb"),
                backend=self.name,
                n_faces=10,
                seconds=0.1,
            )

    def broken_factory():
        raise RuntimeError("Backend initialization failed")

    register_backend(DummyBackend, "dummy")
    register_backend(broken_factory, "broken")

    avail = available_backends()
    assert "dummy" in avail
    assert "broken" not in avail


def test_get_backend_missing_raises():
    with pytest.raises(BackendUnavailable) as exc_info:
        get_backend("nope")
    err_msg = str(exc_info.value)
    assert "nope" in err_msg
    assert "fake" in err_msg


def test_get_backend_none_never_picks_fake():
    """Auto-selection must skip the test double; 'fake' is explicit-only."""
    with pytest.raises(BackendUnavailable, match="No generative backend is usable"):
        get_backend(None)
    # Explicitly named still works.
    assert get_backend("fake").name == "fake"


def test_get_backend_none_prefers_self_hosted(monkeypatch):
    """With both a priority backend and another registered, priority wins."""
    from pixiecad.generative import base as gb

    class Dummy:
        def __init__(self, name):
            self.name = name

        def available(self):
            return True

        def generate(self, req, out_dir):
            raise NotImplementedError

    monkeypatch.setitem(gb._REGISTRY, "trellis-remote", lambda: Dummy("trellis-remote"))
    monkeypatch.setitem(gb._REGISTRY, "other", lambda: Dummy("other"))
    assert get_backend(None).name == "trellis-remote"


def test_fake_backend_generate(tmp_path: Path):
    backend = FakeBackend()
    img = tmp_path / "img.png"
    img.write_text("fake_image")
    req = GenerateRequest(images=[img])

    out_dir = tmp_path / "out"
    res = backend.generate(req, out_dir)

    mesh_file = Path(res.mesh_path)
    assert mesh_file.exists()
    assert res.backend == "fake"
    assert res.n_faces is not None and res.n_faces > 0

    loaded = trimesh.load(str(mesh_file))
    if isinstance(loaded, trimesh.Scene):
        loaded = loaded.to_mesh()
    assert len(loaded.faces) > 0


def test_run_generate_caching(tmp_path: Path):
    ws_dir = tmp_path / "workspace"
    ws = Workspace.create(ws_dir, ObjectSpec())

    img = tmp_path / "img.png"
    img.write_text("fake_image_data")
    req = GenerateRequest(images=[img], seed=42)

    call_count = 0

    class CountingFakeBackend:
        name = "counting_fake"

        def available(self) -> bool:
            return True

        def generate(self, req: GenerateRequest, out_dir: Path) -> GenerateResult:
            nonlocal call_count
            call_count += 1
            fb = FakeBackend()
            res = fb.generate(req, out_dir)
            res.backend = self.name
            return res

    register_backend(CountingFakeBackend, "counting_fake")

    res1 = run_generate(req, ws, backend="counting_fake")
    assert call_count == 1
    assert res1.cached is False
    assert Path(res1.mesh_path).exists()

    res2 = run_generate(req, ws, backend="counting_fake")
    assert call_count == 1
    assert res2.cached is True
    assert res2.mesh_path == res1.mesh_path


def test_run_generate_missing_mesh_raises(tmp_path: Path):
    ws_dir = tmp_path / "workspace"
    ws = Workspace.create(ws_dir, ObjectSpec())

    img = tmp_path / "img.png"
    img.write_text("fake_image_data")
    req = GenerateRequest(images=[img])

    class NonexistentFileBackend:
        name = "nonexistent"

        def available(self) -> bool:
            return True

        def generate(self, req: GenerateRequest, out_dir: Path) -> GenerateResult:
            return GenerateResult(
                mesh_path=str(out_dir / "does_not_exist.glb"),
                backend=self.name,
                n_faces=10,
                seconds=0.1,
            )

    register_backend(NonexistentFileBackend, "nonexistent")

    with pytest.raises(GenerativeError, match="non-existent mesh file"):
        run_generate(req, ws, backend="nonexistent")
