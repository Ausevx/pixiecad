"""Generative 3D model reconstruction interfaces and backends."""

from collections.abc import Mapping

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
from .hunyuan_remote import (  # noqa: E402
    PASSTHROUGH_OPTIONS as _hunyuan_options,
    RemoteHunyuanBackend,
    make_backend as make_hunyuan_backend,
)
from .trellis_remote import (  # noqa: E402
    PASSTHROUGH_OPTIONS as _trellis_options,
    RemoteTrellisBackend,
    make_backend as make_trellis_backend,
)

#: What each backend actually reads out of ``GenerateRequest.options``.
#:
#: Each backend filters the options dict against its own allow-list before
#: building a remote command, which is the right thing -- a typo must not
#: silently change generation settings. The cost is that a REAL setting the
#: user chose is dropped just as quietly.
#:
#: That cost was paid: the dashboard's "generation detail" control sets
#: ``octree_resolution``, which only Hunyuan reads. Every TRELLIS build made
#: from the dashboard therefore ran at the worker's own default however the
#: control was set, and nothing anywhere said so -- five consecutive runs asked
#: for 512 and none of them took effect.
#:
#: Mirrored from each module rather than restated, so a backend that learns a
#: new knob cannot drift out of sync with what we claim it understands.
BACKEND_OPTIONS: dict[str, frozenset[str]] = {
    "hunyuan-remote": frozenset(_hunyuan_options),
    "trellis-remote": frozenset(_trellis_options),
}


def unsupported_options(backend: str, options: Mapping[str, object]) -> list[str]:
    """Which of ``options`` this backend will ignore.

    An unknown backend returns nothing: silence beats warning about a manifest
    we do not have. FakeBackend and any future backend land here until someone
    adds them above.
    """
    known = BACKEND_OPTIONS.get(backend)
    if known is None:
        return []
    return sorted(k for k in options if k not in known)


__all__ = [
    "BACKEND_OPTIONS",
    "unsupported_options",
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
