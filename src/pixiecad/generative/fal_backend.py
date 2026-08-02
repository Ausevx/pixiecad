"""fal.ai TRELLIS generative backend for PixieCAD."""

import base64
from dataclasses import dataclass, field
import json
import mimetypes
import os
from pathlib import Path
import time
from typing import Any
import urllib.error
import urllib.request

# Base interface resolution
try:
    from .base import (
        BackendUnavailable,
        GenerateRequest,
        GenerateResult,
        GenerativeError,
        register_backend,
    )
except ImportError:
    class GenerativeError(Exception):
        """Base exception for generative model errors."""
        pass

    class BackendUnavailable(GenerativeError):
        """Raised when a backend is not configured or unavailable."""
        pass

    @dataclass
    class GenerateRequest:
        images: list[Path]
        seed: int | None = None
        texture: bool = True
        options: dict = field(default_factory=dict)

    @dataclass
    class GenerateResult:
        mesh_path: str
        backend: str
        n_faces: int | None
        seconds: float
        cached: bool = False
        metadata: dict = field(default_factory=dict)

    register_backend = None


def _get_base_exports():
    """Dynamically attempt to import from .base if available, falling back to local definitions."""
    try:
        from .base import (
            BackendUnavailable as BU,
            GenerateRequest as GReq,
            GenerateResult as GRes,
            GenerativeError as GE,
            register_backend as RB,
        )
        return BU, GReq, GRes, GE, RB
    except ImportError:
        return (
            BackendUnavailable,
            GenerateRequest,
            GenerateResult,
            GenerativeError,
            register_backend,
        )


FAL_ENDPOINT = "https://queue.fal.run"
DEFAULT_MODEL_ID = "fal-ai/trellis-2"


def _path_to_data_uri(path: Path) -> str:
    path = Path(path)
    mime, _ = mimetypes.guess_type(path)
    if not mime:
        ext = path.suffix.lower()
        if ext in (".jpg", ".jpeg"):
            mime = "image/jpeg"
        elif ext == ".png":
            mime = "image/png"
        elif ext == ".webp":
            mime = "image/webp"
        else:
            mime = "application/octet-stream"
    with open(path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def _extract_mesh_url(data: dict) -> str | None:
    if not isinstance(data, dict):
        return None

    # 1. model_mesh.url
    model_mesh = data.get("model_mesh")
    if isinstance(model_mesh, dict) and isinstance(model_mesh.get("url"), str):
        return model_mesh["url"]

    # 2. mesh.url
    mesh = data.get("mesh")
    if isinstance(mesh, dict) and isinstance(mesh.get("url"), str):
        return mesh["url"]

    # 3. top-level dict with url ending in .glb or top-level string ending in .glb
    for key, val in data.items():
        if isinstance(val, dict):
            url = val.get("url")
            if isinstance(url, str) and url.endswith(".glb"):
                return url
        elif isinstance(val, str) and val.endswith(".glb"):
            return val

    return None


def _count_faces(mesh_path: Path) -> int | None:
    try:
        import trimesh
        loaded = trimesh.load(str(mesh_path), file_type="glb")
        if isinstance(loaded, trimesh.Scene):
            faces = sum(
                geom.faces.shape[0]
                for geom in loaded.geometry.values()
                if hasattr(geom, "faces") and getattr(geom, "faces", None) is not None
            )
            return int(faces)
        elif hasattr(loaded, "faces") and loaded.faces is not None:
            return int(loaded.faces.shape[0])
    except Exception:
        pass
    return None


class FalBackend:
    name: str = "fal"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model_id: str = DEFAULT_MODEL_ID,
        poll_interval_s: float = 2.0,
        timeout_s: float = 600.0,
    ):
        self._api_key_override = api_key
        self.model_id = model_id
        self.poll_interval_s = poll_interval_s
        self.timeout_s = timeout_s
        self.name = "fal"

    @property
    def api_key(self) -> str | None:
        if self._api_key_override is not None:
            return self._api_key_override
        return os.environ.get("FAL_KEY")

    def available(self) -> bool:
        return bool(self.api_key)

    def _http(
        self,
        method: str,
        url: str,
        *,
        body: dict | None = None,
        raw: bool = False,
    ) -> Any:
        headers = {}
        key = self.api_key
        if key:
            headers["Authorization"] = f"Key {key}"

        if body is not None:
            headers["Content-Type"] = "application/json"
            data = json.dumps(body).encode("utf-8")
        else:
            data = None

        req = urllib.request.Request(
            url, data=data, headers=headers, method=method.upper()
        )

        _, _, _, GenerativeError, _ = _get_base_exports()

        try:
            with urllib.request.urlopen(req) as resp:
                resp_body = resp.read()
                if raw:
                    return resp_body
                if not resp_body:
                    return {}
                return json.loads(resp_body.decode("utf-8"))
        except urllib.error.HTTPError as e:
            err_body = ""
            try:
                err_body = e.read().decode("utf-8", errors="replace")
            except Exception:
                pass
            snippet = err_body[:200] if err_body else ""
            raise GenerativeError(f"HTTP {e.code}: {snippet}") from e
        except urllib.error.URLError as e:
            raise GenerativeError(f"Network error: {e.reason}") from e
        except Exception as e:
            raise GenerativeError(f"HTTP request failed: {e}") from e

    def generate(self, req: Any, out_dir: Any) -> Any:
        BackendUnavailable, _, GenerateResult, GenerativeError, _ = _get_base_exports()

        if not self.available():
            raise BackendUnavailable("FAL_KEY is not set or API key is missing")

        images = getattr(req, "images", None)
        if not images:
            raise GenerativeError("GenerateRequest must contain at least one image")

        payload: dict[str, Any] = {}
        payload["image_url"] = _path_to_data_uri(images[0])
        if len(images) > 1:
            payload["image_urls"] = [_path_to_data_uri(img) for img in images]

        if getattr(req, "seed", None) is not None:
            payload["seed"] = req.seed

        payload["texture"] = getattr(req, "texture", True)

        if getattr(req, "options", None):
            payload.update(req.options)

        submit_url = f"{FAL_ENDPOINT}/{self.model_id}"
        submit_res = self._http("POST", submit_url, body=payload)

        request_id = submit_res.get("request_id")
        status_url = submit_res.get("status_url")
        response_url = submit_res.get("response_url")

        if not status_url:
            raise GenerativeError("fal submit response missing status_url")

        start_time = time.monotonic()
        while True:
            status_res = self._http("GET", status_url)
            status = status_res.get("status")

            if status_res.get("response_url"):
                response_url = status_res.get("response_url")

            if status == "COMPLETED":
                break
            if status in ("FAILED", "ERROR", "CANCELLED"):
                err_msg = status_res.get("error") or status
                raise GenerativeError(
                    f"fal generation failed with status {status}: {err_msg}"
                )

            if time.monotonic() - start_time >= self.timeout_s:
                raise GenerativeError(
                    f"fal generation timed out after {self.timeout_s}s"
                )

            time.sleep(self.poll_interval_s)

        if not response_url:
            raise GenerativeError("fal completion missing response_url")

        result_json = self._http("GET", response_url)
        mesh_url = _extract_mesh_url(result_json)
        if not mesh_url:
            raise GenerativeError("Could not locate mesh URL in fal result")

        mesh_bytes = self._http("GET", mesh_url, raw=True)
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "fal_trellis.glb"
        out_path.write_bytes(mesh_bytes)

        n_faces = _count_faces(out_path)
        elapsed = time.monotonic() - start_time

        return GenerateResult(
            mesh_path=str(out_path),
            backend="fal",
            n_faces=n_faces,
            seconds=elapsed,
            cached=False,
            metadata={"request_id": request_id, "model_id": self.model_id},
        )


# Register backend under "fal" at import time if base is present
try:
    from .base import register_backend
    if register_backend is not None:
        register_backend(lambda: FalBackend(), "fal")
except Exception:
    pass
