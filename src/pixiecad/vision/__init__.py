"""S0.5: optional language layer — photo triage and part naming.

Nothing here is required for geometry. With no API key configured, triage still
reports structural coverage and parts still get geometric names.
"""

from .base import (
    ProviderUnavailable,
    VisionError,
    VisionProvider,
    VisionReply,
    available_providers,
    get_provider,
)
from .triage import PhotoNote, TriageReport, name_parts, triage_photos

__all__ = [
    "VisionProvider",
    "VisionReply",
    "VisionError",
    "ProviderUnavailable",
    "get_provider",
    "available_providers",
    "triage_photos",
    "name_parts",
    "TriageReport",
    "PhotoNote",
]
