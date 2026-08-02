"""S0.5 triage and naming — all offline; providers are stubbed."""

import json

import pytest
import trimesh

from pixiecad.parts import split_parts
from pixiecad.vision import ProviderUnavailable, name_parts, triage_photos
from pixiecad.vision.base import (
    AnthropicProvider,
    GeminiProvider,
    VisionError,
    VisionReply,
    get_provider,
)
from conftest import checkerboard, save_jpg


@pytest.fixture(autouse=True)
def _no_real_keys(monkeypatch):
    for var in ("ANTHROPIC_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY"):
        monkeypatch.delenv(var, raising=False)


class StubProvider:
    name = "stub"

    def __init__(self, payload):
        self.payload = payload
        self.calls = 0

    def available(self):
        return True

    def ask(self, prompt, images=None, *, max_tokens=1024):
        self.calls += 1
        self.prompt = prompt
        self.images = images or []
        return VisionReply(text=json.dumps(self.payload), provider="stub", model="stub-1")


def _photos(tmp_path, n=4, vary=True):
    d = tmp_path / "p"
    d.mkdir(exist_ok=True)
    out = []
    for i in range(n):
        img = checkerboard(200, 10 + (i * 7 if vary else 0))
        p = d / f"{i:02d}.jpg"
        save_jpg(p, img)
        out.append(p)
    return out


# --- providers -------------------------------------------------------------

def test_no_provider_configured_raises():
    with pytest.raises(ProviderUnavailable, match="no vision provider"):
        get_provider()


def test_named_unknown_provider_raises():
    with pytest.raises(ProviderUnavailable, match="unknown provider"):
        get_provider("mystery")


def test_provider_available_follows_env(monkeypatch):
    assert AnthropicProvider().available() is False
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    assert AnthropicProvider().available() is True
    monkeypatch.setenv("GEMINI_API_KEY", "g-test")
    assert GeminiProvider().available() is True


def test_ask_without_key_raises():
    with pytest.raises(ProviderUnavailable):
        AnthropicProvider().ask("hi")


def test_reply_parses_fenced_json():
    r = VisionReply(text='```json\n{"a": 1}\n```', provider="x", model="y")
    assert r.as_json() == {"a": 1}


def test_reply_rejects_non_json():
    with pytest.raises(VisionError, match="valid JSON"):
        VisionReply(text="not json at all", provider="x", model="y").as_json()


def test_api_key_never_leaks_into_errors(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-super-secret-value")
    p = AnthropicProvider()

    def boom(body):
        raise VisionError("anthropic HTTP 401: unauthorized")

    monkeypatch.setattr(p, "_http", boom)
    with pytest.raises(VisionError) as ei:
        p.ask("hello")
    assert "sk-super-secret-value" not in str(ei.value)


# --- triage ----------------------------------------------------------------

def test_triage_without_provider_still_reports(tmp_path):
    report = triage_photos(_photos(tmp_path, 4))
    assert report.n_photos == 4
    assert report.provider is None
    assert report.distinctness > 0
    assert any("below the ~16" in a for a in report.advice)
    assert "4 photos" in report.summary()


def test_triage_flags_near_identical_photos(tmp_path):
    report = triage_photos(_photos(tmp_path, 4, vary=False))
    assert report.distinctness < 0.04
    assert any("parallax" in a for a in report.advice)


def test_triage_uses_provider_labels(tmp_path, monkeypatch):
    stub = StubProvider(
        {
            "object": "formula 1 car",
            "photos": [
                {"index": 0, "viewpoint": "front", "issues": []},
                {"index": 1, "viewpoint": "left", "issues": ["subject partially cut off"]},
            ],
            "missing_viewpoints": ["top"],
            "advice": ["shoot the right side"],
        }
    )
    monkeypatch.setattr("pixiecad.vision.triage.get_provider", lambda name=None: stub)

    report = triage_photos(_photos(tmp_path, 2))
    assert report.object_guess == "formula 1 car"
    assert report.provider == "stub"
    assert report.photos[0].viewpoint == "front"
    assert report.photos[1].issues == ["subject partially cut off"]
    assert report.missing == ["top"]
    assert "shoot the right side" in report.advice
    assert stub.calls == 1
    assert len(stub.images) == 2


def test_triage_survives_provider_failure(tmp_path, monkeypatch):
    def broken(name=None):
        raise VisionError("provider exploded")

    monkeypatch.setattr("pixiecad.vision.triage.get_provider", broken)
    report = triage_photos(_photos(tmp_path, 3))
    assert report.provider is None
    assert any("skipped" in a for a in report.advice)


def test_triage_rejects_empty():
    with pytest.raises(ValueError, match="no images"):
        triage_photos([])


def test_triage_caps_images_sent(tmp_path, monkeypatch):
    stub = StubProvider({"object": "x", "photos": [], "missing_viewpoints": [], "advice": []})
    monkeypatch.setattr("pixiecad.vision.triage.get_provider", lambda name=None: stub)
    triage_photos(_photos(tmp_path, 20), max_images=5)
    assert len(stub.images) == 5


# --- naming ----------------------------------------------------------------

def _car():
    body = trimesh.creation.box(extents=[4.0, 1.8, 1.0])
    parts = [body]
    for x in (-1.4, 1.4):
        for y in (-0.9, 0.9):
            w = trimesh.creation.cylinder(radius=0.4, height=0.3)
            w.apply_translation([x, y, -0.6])
            parts.append(w)
    return trimesh.util.concatenate(parts)


def test_names_fall_back_to_geometry_without_provider():
    whole = _car()
    parts = name_parts(split_parts(whole, method="connected"), whole)
    assert all(p.is_named for p in parts)
    assert parts[0].name == "body"
    assert all(p.metadata["named_by"] == "geometry" for p in parts)


def test_provider_names_are_used(monkeypatch):
    whole = _car()
    parts = split_parts(whole, method="connected")
    stub = StubProvider(
        {"names": [{"index": p.index, "name": f"component {p.index}"} for p in parts]}
    )
    monkeypatch.setattr("pixiecad.vision.triage.get_provider", lambda name=None: stub)

    named = name_parts(parts, whole, object_hint="a car")
    assert named[0].name == "component 0"
    assert all(p.metadata["named_by"] == "stub" for p in named)
    assert "a car" in stub.prompt


def test_partial_provider_names_fill_from_geometry(monkeypatch):
    whole = _car()
    parts = split_parts(whole, method="connected")
    stub = StubProvider({"names": [{"index": 0, "name": "chassis"}]})
    monkeypatch.setattr("pixiecad.vision.triage.get_provider", lambda name=None: stub)

    named = name_parts(parts, whole)
    assert named[0].name == "chassis"
    assert named[0].metadata["named_by"] == "stub"
    assert all(p.is_named for p in named)
    assert named[1].metadata["named_by"] == "geometry"


def test_naming_survives_bad_provider_json(monkeypatch):
    whole = _car()
    parts = split_parts(whole, method="connected")

    class Junk(StubProvider):
        def ask(self, prompt, images=None, *, max_tokens=1024):
            return VisionReply(text="I'm afraid I can't do that", provider="junk", model="j")

    monkeypatch.setattr("pixiecad.vision.triage.get_provider", lambda name=None: Junk({}))
    named = name_parts(parts, whole)
    assert all(p.is_named for p in named)
    assert all(p.metadata["named_by"] == "geometry" for p in named)
