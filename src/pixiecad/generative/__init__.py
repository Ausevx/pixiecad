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

# Neither remote backend can self-register: both need an Executor, so they are
# constructed explicitly with make_hunyuan_backend(executor) /
# make_trellis_backend(executor).
from .hunyuan_remote import RemoteHunyuanBackend, make_backend as make_hunyuan_backend  # noqa: E402
from .trellis_remote import RemoteTrellisBackend, make_backend as make_trellis_backend  # noqa: E402

__all__ = [
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
