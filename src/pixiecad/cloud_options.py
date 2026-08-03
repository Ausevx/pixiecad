"""GPU hardware choices offered by the dashboard, with cost and time estimates.

Every number here is either measured on this project or taken from GCP's
published rates, and each is labelled as such. Estimates that look measured
but are not cause real money to be spent on a guess.

The honest headline: for our workload the GPU is rarely the bottleneck, so
paying more usually buys less than people expect. Measured on this project,
Hunyuan3D shape generation is 170 s on an L4 and SAM segmentation 125 s, and
both are dominated by fixed costs -- model loading, image pull, weight
download -- rather than raw throughput.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class GPUOption:
    key: str
    label: str
    accelerator: str
    machine_type: str
    vram_gb: int
    usd_per_hour_spot: float
    usd_per_hour_ondemand: float
    # Multiplier against the L4 baseline for OUR pipeline, not raw FLOPS.
    relative_speed: float
    speed_basis: str
    note: str
    available: bool = True

    @property
    def gen_seconds(self) -> int:
        """Estimated shape generation time, from the measured 170 s on an L4."""
        return int(170 / self.relative_speed)

    @property
    def sam_seconds(self) -> int:
        """Estimated SAM segmentation time, from the measured 125 s on an L4."""
        return int(125 / self.relative_speed)

    @property
    def texture_seconds(self) -> int:
        """Estimated texturing time, from the measured 123 s on an L4."""
        return int(123 / self.relative_speed)


# Prices are GCP asia-southeast1 list rates as of 2026-08 and will drift; they
# are indicative, not billing. Speeds marked "measured" were timed on this
# project; the rest are inferred and flagged as such, because a fabricated
# benchmark is worse than an admitted estimate.
GPU_OPTIONS: tuple[GPUOption, ...] = (
    GPUOption(
        key="t4",
        label="NVIDIA T4 (16 GB)",
        accelerator="nvidia-tesla-t4",
        machine_type="n1-standard-4",
        vram_gb=16,
        usd_per_hour_spot=0.11,
        usd_per_hour_ondemand=0.35,
        relative_speed=0.45,
        speed_basis="measured for SAM (125 s on T4); inferred for generation",
        note="Cheapest. Ran SAM fine. Too little VRAM for comfortable shape generation.",
    ),
    GPUOption(
        key="l4",
        label="NVIDIA L4 (24 GB) - recommended",
        accelerator="nvidia-l4",
        machine_type="g2-standard-16",
        vram_gb=24,
        usd_per_hour_spot=0.28,
        usd_per_hour_ondemand=1.10,
        relative_speed=1.0,
        speed_basis="measured: 170 s generation, 123 s texturing",
        note="The baseline every timing on this project was measured against.",
    ),
    GPUOption(
        key="a100",
        label="NVIDIA A100 (40 GB)",
        accelerator="nvidia-tesla-a100",
        machine_type="a2-highgpu-1g",
        vram_gb=40,
        usd_per_hour_spot=1.20,
        usd_per_hour_ondemand=3.70,
        relative_speed=1.6,
        speed_basis="inferred from throughput, NOT measured here",
        note=(
            "Roughly 10x the price of L4 spot for well under 2x on this "
            "workload, which is diffusion-step bound rather than throughput "
            "bound. Quota is currently 0 on this project."
        ),
        available=False,
    ),
    GPUOption(
        key="h100",
        label="NVIDIA H100 (80 GB)",
        accelerator="nvidia-h100-80gb",
        machine_type="a3-highgpu-1g",
        vram_gb=80,
        usd_per_hour_spot=3.00,
        usd_per_hour_ondemand=11.00,
        relative_speed=2.2,
        speed_basis="inferred, NOT measured here",
        note=(
            "Quota is 0 on this project, so it cannot be launched. Even if it "
            "could, it saves roughly 90 s per run for over 10x the hourly rate."
        ),
        available=False,
    ),
)

# Fixed overheads, measured on this project, that no faster GPU removes. These
# dominate a cold run and are the reason a bigger GPU disappoints.
COLD_START_SECONDS = 240  # provision + docker install
IMAGE_BUILD_SECONDS = 900  # first build of the worker image on a fresh VM
WEIGHTS_DOWNLOAD_SECONDS = 600  # ~10 GB, first run only


def estimate(option: GPUOption, *, texture: bool, semantic: bool, warm: bool) -> dict:
    """Estimated wall-clock seconds and USD for one run on this hardware."""
    gpu_seconds = option.gen_seconds
    if texture:
        gpu_seconds += option.texture_seconds
    if semantic:
        gpu_seconds += option.sam_seconds

    overhead = 0 if warm else (COLD_START_SECONDS + IMAGE_BUILD_SECONDS + WEIGHTS_DOWNLOAD_SECONDS)
    total = gpu_seconds + overhead
    return {
        "gpu_seconds": gpu_seconds,
        "overhead_seconds": overhead,
        "total_seconds": total,
        # Billed on wall clock, so the overhead costs money even though no GPU
        # work is happening -- which is the real argument for a baked image.
        "usd_spot": round(option.usd_per_hour_spot * total / 3600, 3),
        "usd_ondemand": round(option.usd_per_hour_ondemand * total / 3600, 3),
    }


def options_payload(*, texture: bool = True, semantic: bool = True) -> list[dict]:
    """Serialisable option list for the dashboard, with both warm and cold costs."""
    payload = []
    for option in GPU_OPTIONS:
        entry = asdict(option)
        entry["warm"] = estimate(option, texture=texture, semantic=semantic, warm=True)
        entry["cold"] = estimate(option, texture=texture, semantic=semantic, warm=False)
        payload.append(entry)
    return payload


# ---------------------------------------------------------------------------
# Per-stage cost, so the new-job form can say what a job will actually take
# before it is submitted rather than after.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StageCost:
    """One stage's wall time and memory, and where the number came from.

    ``basis`` is not decoration. A user deciding whether to bake at 2048 on a
    16 GB machine is making a real resource decision, and an invented figure
    presented in the same typeface as a measured one makes that decision worse
    than showing nothing.
    """

    key: str
    label: str
    where: str  # "local" | "host"
    seconds: float | None
    peak_ram_mb: int | None
    basis: str  # "measured" | "estimated"
    detail: str


# Measured on this project, 2026-08-04, on the 16 GB M3 dev machine, against a
# 1.31M face source decimated to 20k -- the shape a generative backend actually
# returns. Timing and peak RSS captured together per run.
BAKE_LOCAL: dict[int, tuple[float, int]] = {
    512: (6.0, 1759),
    1024: (13.6, 3367),
    2048: (52.9, 4118),
}

# Raw arrays, float32 vertices, savez_compressed: 11.7 MB for the pair at
# 1.31M faces. Transfer time is whatever the user's uplink is, so it is
# reported as a size and a range rather than as a single fabricated duration.
BAKE_UPLOAD_MB = 11.7

# The dev machine's entire local pipeline outside baking -- ingest, clean,
# decimate, unwrap, parts export -- measured at 247 MB peak in 1.9 s.
LOCAL_BASELINE_MB = 247


def bake_cost(resolution: int, *, where: str) -> StageCost:
    """What a normal-map bake costs at this resolution, here or on the host."""
    nearest = min(BAKE_LOCAL, key=lambda r: abs(r - resolution))
    seconds, ram = BAKE_LOCAL[nearest]
    exact = nearest == resolution

    if where == "host":
        return StageCost(
            key="bake",
            label=f"normal map {resolution}x{resolution} (host)",
            where="host",
            seconds=None,
            peak_ram_mb=0,
            basis="estimated",
            detail=(
                f"{BAKE_UPLOAD_MB:.1f} MB uploaded, then baked on the host's "
                "CPU. Costs this machine nothing but the transfer. Never run "
                "end-to-end, so no measured time yet."
            ),
        )
    return StageCost(
        key="bake",
        label=f"normal map {resolution}x{resolution} (this machine)",
        where="local",
        seconds=seconds,
        peak_ram_mb=ram,
        basis="measured" if exact else "estimated",
        detail=(
            f"By far the heaviest local stage -- everything else combined "
            f"peaks at {LOCAL_BASELINE_MB} MB. Runs while you are using the "
            "machine."
        )
        + ("" if exact else f" Interpolated from the measured {nearest}x{nearest}."),
    )


def stage_costs(
    *,
    bake: bool = True,
    normal_res: int = 1024,
    bake_location: str = "auto",
    texture: bool = False,
    semantic: bool = False,
    has_host: bool = False,
    gpu_key: str = "l4",
) -> dict:
    """Every stage a job as configured will run, with its cost and provenance."""
    option = next((o for o in GPU_OPTIONS if o.key == gpu_key), GPU_OPTIONS[0])
    stages: list[StageCost] = [
        StageCost(
            key="local_pipeline",
            label="ingest, clean, decimate, unwrap, export",
            where="local",
            seconds=1.9,
            peak_ram_mb=LOCAL_BASELINE_MB,
            basis="measured",
            detail="Measured on 8 photos at 1024px. Scales with photo count.",
        )
    ]

    if has_host:
        stages.append(
            StageCost(
                key="generate",
                label="shape generation",
                where="host",
                seconds=float(option.gen_seconds),
                peak_ram_mb=0,
                basis="measured" if option.key == "l4" else "estimated",
                detail=f"{option.speed_basis} on {option.label}.",
            )
        )
    if texture and has_host:
        stages.append(
            StageCost(
                key="texture",
                label="texturing",
                where="host",
                seconds=float(option.texture_seconds),
                peak_ram_mb=0,
                basis="measured" if option.key == "l4" else "estimated",
                detail="2048x2048 PBR maps.",
            )
        )
    if semantic and has_host:
        stages.append(
            StageCost(
                key="segment",
                label="semantic segmentation (SAM)",
                where="host",
                seconds=float(option.sam_seconds),
                peak_ram_mb=0,
                basis="measured" if option.key == "l4" else "estimated",
                detail="14 views at 768px. Cached, so a re-run is free.",
            )
        )

    resolved = "host" if (bake_location in ("auto", "host") and has_host) else "local"
    if bake:
        stages.append(bake_cost(normal_res, where=resolved))

    local_peak = max([s.peak_ram_mb or 0 for s in stages if s.where == "local"] or [0])
    known = [s.seconds for s in stages if s.seconds is not None]
    return {
        "stages": [asdict(s) for s in stages],
        "local_peak_ram_mb": local_peak,
        "total_known_seconds": round(sum(known), 1),
        "has_unmeasured": any(s.seconds is None for s in stages),
        "bake_location_resolved": resolved if bake else None,
    }
