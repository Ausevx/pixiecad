"""ObjectSpec: the user's contract with the pipeline.

Declares real-world dimensions (for metric scaling), the polygon budget
(exact triangle count enforced at the retopo stage), and per-part wishes.
"""

from __future__ import annotations

import json
import re
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field, field_validator, model_validator


class Unit(str, Enum):
    MM = "mm"
    CM = "cm"
    M = "m"
    IN = "in"
    FT = "ft"

    @property
    def to_meters(self) -> float:
        return {"mm": 0.001, "cm": 0.01, "m": 1.0, "in": 0.0254, "ft": 0.3048}[self.value]


_DIM_RE = re.compile(r"^\s*([0-9]*\.?[0-9]+)\s*(mm|cm|m|in|ft)\s*$", re.IGNORECASE)


def parse_length(text: str) -> float:
    """Parse a human dimension string like '4.5m' or '45 cm' into meters."""
    m = _DIM_RE.match(text)
    if not m:
        raise ValueError(
            f"Cannot parse dimension {text!r}; expected e.g. '4.5m', '45cm', '18in'"
        )
    value, unit = float(m.group(1)), Unit(m.group(2).lower())
    if value <= 0:
        raise ValueError(f"Dimension must be positive, got {text!r}")
    return value * unit.to_meters


class Dimensions(BaseModel):
    """Known real-world extents of the object's axis-aligned bounding box.

    Any subset may be given; one axis is enough to recover metric scale.
    Values are stored in meters.
    """

    length_m: float | None = Field(default=None, gt=0, description="Longest horizontal extent")
    width_m: float | None = Field(default=None, gt=0)
    height_m: float | None = Field(default=None, gt=0)

    @classmethod
    def parse(
        cls,
        length: str | None = None,
        width: str | None = None,
        height: str | None = None,
    ) -> "Dimensions":
        return cls(
            length_m=parse_length(length) if length else None,
            width_m=parse_length(width) if width else None,
            height_m=parse_length(height) if height else None,
        )

    @property
    def any_known(self) -> bool:
        return any(v is not None for v in (self.length_m, self.width_m, self.height_m))


class PartSpec(BaseModel):
    """A part the user wants exported separately, with its own budget."""

    name: str
    target_faces: int | None = Field(default=None, gt=0)
    export: bool = True

    @field_validator("name")
    @classmethod
    def _clean_name(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("part name must be non-empty")
        return v


class ScaleSource(str, Enum):
    NONE = "none"  # arbitrary scale, whatever reconstruction produced
    DIMENSIONS = "dimensions"  # user-declared bounding-box extents
    ARUCO = "aruco"  # detected marker of known physical size


class ObjectSpec(BaseModel):
    """Everything the user asserts about the object and the desired output."""

    name: str = "object"
    dimensions: Dimensions = Field(default_factory=Dimensions)
    target_faces: int = Field(default=20_000, gt=0, description="Total triangle budget")
    quad_remesh: bool = False
    parts: list[PartSpec] = Field(default_factory=list)
    split_parts: bool = False
    allow_generative_completion: bool = False
    aruco_marker_size_m: float | None = Field(
        default=None, gt=0, description="Physical side length of ArUco marker if used"
    )

    @model_validator(mode="after")
    def _parts_imply_split(self) -> "ObjectSpec":
        if self.parts:
            self.split_parts = True
        return self

    @property
    def scale_source(self) -> ScaleSource:
        if self.aruco_marker_size_m is not None:
            return ScaleSource.ARUCO
        if self.dimensions.any_known:
            return ScaleSource.DIMENSIONS
        return ScaleSource.NONE

    # -- persistence ---------------------------------------------------------

    def save(self, path: Path) -> None:
        path.write_text(self.model_dump_json(indent=2) + "\n")

    @classmethod
    def load(cls, path: Path) -> "ObjectSpec":
        return cls.model_validate(json.loads(path.read_text()))
