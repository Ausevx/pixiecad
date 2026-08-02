"""Provider-agnostic vision/LLM plumbing.

Everything language-shaped in PixieCAD is optional. The pipeline must produce
the same geometry whether or not an API key exists; providers only add names
and prose. So every entry point here degrades to a useful local answer rather
than raising when no provider is configured.
"""

from __future__ import annotations

import base64
import json
import mimetypes
import os
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_MODEL = "claude-sonnet-4-6"
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models"
GEMINI_MODEL = "gemini-3-flash"


class VisionError(Exception):
    """The provider was configured but the call failed."""


class ProviderUnavailable(VisionError):
    """No provider configured — callers should fall back to local-only output."""


@dataclass
class VisionReply:
    text: str
    provider: str
    model: str
    raw: dict[str, Any] = field(default_factory=dict)

    def as_json(self) -> Any:
        """Parse the reply as JSON, tolerating ```json fences."""
        text = self.text.strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            text = text[4:] if text.lower().startswith("json") else text
        try:
            return json.loads(text.strip())
        except json.JSONDecodeError as e:
            raise VisionError(f"provider did not return valid JSON: {e}") from e


class VisionProvider(Protocol):
    name: str

    def available(self) -> bool: ...

    def ask(
        self, prompt: str, images: list[Path] | None = None, *, max_tokens: int = 1024
    ) -> VisionReply: ...


def _encode(path: Path) -> tuple[str, str]:
    mime = mimetypes.guess_type(str(path))[0] or "image/jpeg"
    return mime, base64.b64encode(Path(path).read_bytes()).decode()


class AnthropicProvider:
    name = "anthropic"

    def __init__(self, api_key: str | None = None, model: str = ANTHROPIC_MODEL):
        self._key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.model = model

    def available(self) -> bool:
        return bool(self._key)

    def _http(self, body: dict) -> dict:
        req = urllib.request.Request(
            ANTHROPIC_URL,
            data=json.dumps(body).encode(),
            headers={
                "content-type": "application/json",
                "x-api-key": self._key or "",
                "anthropic-version": "2023-06-01",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            # Never echo the key back in an error.
            raise VisionError(f"anthropic HTTP {e.code}: {e.read()[:300].decode(errors='replace')}") from None
        except Exception as e:
            raise VisionError(f"anthropic request failed: {type(e).__name__}") from None

    def ask(self, prompt, images=None, *, max_tokens=1024) -> VisionReply:
        if not self.available():
            raise ProviderUnavailable("ANTHROPIC_API_KEY is not set")
        content: list[dict] = []
        for img in images or []:
            mime, data = _encode(img)
            content.append(
                {"type": "image", "source": {"type": "base64", "media_type": mime, "data": data}}
            )
        content.append({"type": "text", "text": prompt})
        data = self._http(
            {
                "model": self.model,
                "max_tokens": max_tokens,
                "messages": [{"role": "user", "content": content}],
            }
        )
        text = "".join(b.get("text", "") for b in data.get("content", []))
        return VisionReply(text=text, provider=self.name, model=self.model, raw=data)


class GeminiProvider:
    name = "gemini"

    def __init__(self, api_key: str | None = None, model: str = GEMINI_MODEL):
        self._key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        self.model = model

    def available(self) -> bool:
        return bool(self._key)

    def _http(self, body: dict) -> dict:
        url = f"{GEMINI_URL}/{self.model}:generateContent?key={self._key}"
        req = urllib.request.Request(
            url, data=json.dumps(body).encode(),
            headers={"content-type": "application/json"}, method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            raise VisionError(f"gemini HTTP {e.code}") from None
        except Exception as e:
            raise VisionError(f"gemini request failed: {type(e).__name__}") from None

    def ask(self, prompt, images=None, *, max_tokens=1024) -> VisionReply:
        if not self.available():
            raise ProviderUnavailable("GEMINI_API_KEY is not set")
        parts: list[dict] = []
        for img in images or []:
            mime, data = _encode(img)
            parts.append({"inline_data": {"mime_type": mime, "data": data}})
        parts.append({"text": prompt})
        data = self._http(
            {
                "contents": [{"parts": parts}],
                "generationConfig": {"maxOutputTokens": max_tokens},
            }
        )
        try:
            text = "".join(
                p.get("text", "")
                for p in data["candidates"][0]["content"]["parts"]
            )
        except (KeyError, IndexError):
            raise VisionError("gemini returned no candidates") from None
        return VisionReply(text=text, provider=self.name, model=self.model, raw=data)


def get_provider(name: str | None = None) -> VisionProvider:
    """First configured provider, or the named one. Raises if none is usable."""
    providers: dict[str, VisionProvider] = {
        "anthropic": AnthropicProvider(),
        "gemini": GeminiProvider(),
    }
    if name:
        p = providers.get(name)
        if p is None:
            raise ProviderUnavailable(f"unknown provider {name!r}; have {sorted(providers)}")
        if not p.available():
            raise ProviderUnavailable(f"{name} is not configured (missing API key)")
        return p
    for p in providers.values():
        if p.available():
            return p
    raise ProviderUnavailable(
        "no vision provider configured; set ANTHROPIC_API_KEY or GEMINI_API_KEY"
    )


def available_providers() -> list[str]:
    return [n for n, p in (("anthropic", AnthropicProvider()), ("gemini", GeminiProvider())) if p.available()]
