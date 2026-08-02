"""Generative 3D model reconstruction interfaces and backends."""

from .base import (
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

# Importing the concrete backends is what registers them. fal self-registers
# (it needs only an env var); the remote one cannot, since it requires an
# Executor — construct it with make_trellis_backend(executor).
from .fal_backend import FalBackend  # noqa: E402  (import registers "fal")
from .hunyuan_remote import RemoteHunyuanBackend, make_backend as make_hunyuan_backend  # noqa: E402
from .trellis_remote import RemoteTrellisBackend, make_backend as make_trellis_backend  # noqa: E402

__all__ = [
    "FalBackend",
    "RemoteTrellisBackend",
    "make_trellis_backend",
    "make_hunyuan_backend",
    "RemoteHunyuanBackend",
    "make_hunyuan_backend",
    "BackendUnavailable",
    "FakeBackend",
    "GenerateRequest",
    "GenerateResult",
    "GenerativeBackend",
    "GenerativeError",
    "available_backends",
    "get_backend",
    "register_backend",
    "run_generate",
]
