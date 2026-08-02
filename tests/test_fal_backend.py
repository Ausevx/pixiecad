"""Tests for fal.ai TRELLIS generative backend."""

import os
from pathlib import Path
import pytest
import trimesh

from pixiecad.generative.fal_backend import (
    BackendUnavailable,
    FalBackend,
    GenerateRequest,
    GenerativeError,
)


def test_available(monkeypatch):
    monkeypatch.delenv("FAL_KEY", raising=False)
    backend_no_key = FalBackend()
    assert not backend_no_key.available()

    monkeypatch.setenv("FAL_KEY", "fake_key_123")
    backend_with_env = FalBackend()
    assert backend_with_env.available()

    backend_explicit = FalBackend(api_key="explicit_key")
    assert backend_explicit.available()


def test_generate_no_key_raises_backend_unavailable(monkeypatch, tmp_path):
    monkeypatch.delenv("FAL_KEY", raising=False)
    backend = FalBackend()
    img_file = tmp_path / "test.png"
    img_file.write_bytes(b"fake image data")
    req = GenerateRequest(images=[img_file])

    with pytest.raises(BackendUnavailable):
        backend.generate(req, tmp_path)


def test_happy_path(monkeypatch, tmp_path):
    img_file = tmp_path / "test.png"
    img_file.write_bytes(b"fake png content")
    req = GenerateRequest(images=[img_file])

    mesh = trimesh.creation.icosphere()
    glb_bytes = mesh.export(file_type="glb")

    http_calls = []

    def fake_http(self, method, url, body=None, raw=False):
        http_calls.append({"method": method, "url": url, "body": body, "raw": raw})
        if method == "POST" and "fal-ai/trellis-2" in url:
            return {
                "request_id": "req-12345",
                "status_url": "https://queue.fal.run/status/req-12345",
                "response_url": "https://queue.fal.run/response/req-12345",
            }
        elif method == "GET" and url == "https://queue.fal.run/status/req-12345":
            poll_count = sum(1 for c in http_calls if c["url"] == url)
            if poll_count == 1:
                return {"status": "IN_PROGRESS"}
            return {"status": "COMPLETED"}
        elif method == "GET" and url == "https://queue.fal.run/response/req-12345":
            return {
                "model_mesh": {"url": "https://fal.media/files/mesh.glb"}
            }
        elif method == "GET" and url == "https://fal.media/files/mesh.glb":
            assert raw is True
            return glb_bytes
        raise ValueError(f"Unexpected HTTP call: {method} {url}")

    monkeypatch.setattr(FalBackend, "_http", fake_http)

    backend = FalBackend(api_key="test_key", poll_interval_s=0.01)
    res = backend.generate(req, tmp_path)

    expected_file = tmp_path / "fal_trellis.glb"
    assert expected_file.exists()
    assert res.mesh_path == str(expected_file)
    assert res.backend == "fal"
    assert res.n_faces is not None and res.n_faces > 0
    assert res.metadata.get("request_id") == "req-12345"

    status_calls = [
        c for c in http_calls if c["url"] == "https://queue.fal.run/status/req-12345"
    ]
    assert len(status_calls) == 2


def test_timeout(monkeypatch, tmp_path):
    img_file = tmp_path / "test.png"
    img_file.write_bytes(b"fake image")
    req = GenerateRequest(images=[img_file])

    def fake_http(self, method, url, body=None, raw=False):
        if method == "POST":
            return {
                "request_id": "req-timeout",
                "status_url": "https://queue.fal.run/status/req-timeout",
            }
        elif method == "GET":
            return {"status": "IN_PROGRESS"}

    monkeypatch.setattr(FalBackend, "_http", fake_http)

    backend = FalBackend(api_key="test_key", poll_interval_s=0.05, timeout_s=0.1)
    with pytest.raises(GenerativeError, match="timed out"):
        backend.generate(req, tmp_path)


def test_non_2xx_on_submit(monkeypatch, tmp_path):
    img_file = tmp_path / "test.png"
    img_file.write_bytes(b"fake image")
    req = GenerateRequest(images=[img_file])

    def fake_http(self, method, url, body=None, raw=False):
        raise GenerativeError("HTTP 401: Unauthorized")

    monkeypatch.setattr(FalBackend, "_http", fake_http)

    backend = FalBackend(api_key="bad_key")
    with pytest.raises(GenerativeError) as exc_info:
        backend.generate(req, tmp_path)
    assert "401" in str(exc_info.value)


def test_mesh_url_fallback_shape(monkeypatch, tmp_path):
    img_file = tmp_path / "test.png"
    img_file.write_bytes(b"fake image")
    req = GenerateRequest(images=[img_file])

    mesh = trimesh.creation.icosphere()
    glb_bytes = mesh.export(file_type="glb")

    def fake_http(self, method, url, body=None, raw=False):
        if method == "POST":
            return {
                "request_id": "req-fallback",
                "status_url": "https://queue.fal.run/status/req-fallback",
                "response_url": "https://queue.fal.run/response/req-fallback",
            }
        elif method == "GET" and "status" in url:
            return {"status": "COMPLETED"}
        elif method == "GET" and "response" in url:
            return {
                "mesh": {"url": "https://fal.media/files/fallback_mesh.glb"}
            }
        elif method == "GET" and "fallback_mesh.glb" in url:
            return glb_bytes

    monkeypatch.setattr(FalBackend, "_http", fake_http)

    backend = FalBackend(api_key="test_key", poll_interval_s=0.01)
    res = backend.generate(req, tmp_path)
    assert res.backend == "fal"
    assert (tmp_path / "fal_trellis.glb").exists()


def test_payload_check(monkeypatch, tmp_path):
    img1 = tmp_path / "view1.png"
    img2 = tmp_path / "view2.jpg"
    img1.write_bytes(b"png data")
    img2.write_bytes(b"jpg data")

    req = GenerateRequest(
        images=[img1, img2],
        seed=42,
        texture=False,
        options={"texture": True, "custom_opt": 123},
    )

    submitted_payload = None

    def fake_http(self, method, url, body=None, raw=False):
        nonlocal submitted_payload
        if method == "POST":
            submitted_payload = body
            return {
                "request_id": "req-payload",
                "status_url": "https://queue.fal.run/status/req-payload",
                "response_url": "https://queue.fal.run/response/req-payload",
            }
        elif method == "GET" and "status" in url:
            return {"status": "COMPLETED"}
        elif method == "GET" and "response" in url:
            return {"model_mesh": {"url": "https://fal.media/files/mesh.glb"}}
        elif method == "GET" and "mesh.glb" in url:
            mesh = trimesh.creation.icosphere()
            return mesh.export(file_type="glb")

    monkeypatch.setattr(FalBackend, "_http", fake_http)

    backend = FalBackend(api_key="test_key", poll_interval_s=0.01)
    backend.generate(req, tmp_path)

    assert submitted_payload is not None
    assert submitted_payload["image_url"].startswith("data:image/png;base64,")
    assert len(submitted_payload["image_urls"]) == 2
    assert submitted_payload["image_urls"][0].startswith("data:image/png;base64,")
    assert submitted_payload["image_urls"][1].startswith("data:image/jpeg;base64,")
    assert submitted_payload["seed"] == 42
    assert submitted_payload["texture"] is True
    assert submitted_payload["custom_opt"] == 123
