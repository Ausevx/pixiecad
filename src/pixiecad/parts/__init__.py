"""S5: split a mesh into separately exportable, separately budgeted parts."""

from .export import ExportedPart, export_parts, slugify
from .segment import Part, split_parts

__all__ = ["Part", "split_parts", "ExportedPart", "export_parts", "slugify"]
