from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass
class GPUInfo:
    name: str
    vram_mb: int
    compute_cap: float


@dataclass
class Capabilities:
    hostname: str
    gpu: GPUInfo | None
    reachable: bool
    # Why the probe came back empty. A bare False sent one real job looking
    # for a wrong backend NAME when the actual cause was a stale SSH host key
    # -- the diagnosis existed in ssh's stderr and was thrown away.
    detail: str = ""

    def cuda_ok(self, min_vram_mb: int = 0) -> bool:
        return self.reachable and self.gpu is not None and self.gpu.vram_mb >= min_vram_mb

    def why_not(self, min_vram_mb: int = 0) -> str:
        """Plain-language reason ``cuda_ok`` is False, or "" when it is True."""
        if not self.reachable:
            return f"cannot reach '{self.hostname}'" + (f": {self.detail}" if self.detail else "")
        if self.gpu is None:
            return (
                f"'{self.hostname}' is reachable but reports no GPU"
                + (f": {self.detail}" if self.detail else "")
            )
        if self.gpu.vram_mb < min_vram_mb:
            return (
                f"'{self.hostname}' has {self.gpu.name} with {self.gpu.vram_mb} MB VRAM; "
                f"{min_vram_mb} MB is needed"
            )
        return ""


@dataclass
class Job:
    command: list[str]
    inputs: list[Path]
    output_dir: Path
    remote_subdir: str
    timeout_s: int = 3600


@dataclass
class JobResult:
    returncode: int
    stdout_tail: str
    stderr_tail: str
    duration_s: float

    @property
    def ok(self) -> bool:
        return self.returncode == 0


class Executor(Protocol):
    def probe(self) -> Capabilities:
        ...

    def run(self, job: Job) -> JobResult:
        ...


def parse_nvidia_smi(text: str) -> GPUInfo | None:
    """Parse one CSV line of nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader,nounits.

    Example: 'NVIDIA GeForce GTX 1060, 6144, 6.1'
    Tolerates whitespace, ' MiB' suffixes, and returns None on empty/garbage input.
    """
    if not text or not isinstance(text, str):
        return None

    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    if not lines:
        return None

    parts = [p.strip() for p in lines[0].split(",")]
    if len(parts) != 3:
        return None

    name = parts[0]
    if not name:
        return None

    vram_raw = (
        parts[1]
        .replace("MiB", "")
        .replace("MIB", "")
        .replace("mb", "")
        .replace("MB", "")
        .strip()
    )
    try:
        vram_mb = int(vram_raw)
        compute_cap = float(parts[2].strip())
        return GPUInfo(name=name, vram_mb=vram_mb, compute_cap=compute_cap)
    except (ValueError, TypeError):
        return None
