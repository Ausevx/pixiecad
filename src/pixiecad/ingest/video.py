"""Video ingestion module for photogrammetry.

Extracts evenly spaced frames from a video clip, rejecting those degraded by
motion blur or poor exposure, and keeps the sharpest available frame for each
interval to maximize photogrammetry success.
"""

import json
import math

import shutil
import subprocess
import tempfile
from pathlib import Path

import cv2

from pixiecad.ingest.quality import assess_quality

#: Candidate frames extracted per output frame. Enough to find a sharp one in
#: each time slot without decoding the whole clip.
OVERSAMPLE = 6
#: A malformed container can make ffmpeg block forever.
FFMPEG_TIMEOUT_S = 300
#: Reading container metadata is near-instant; this only guards a hung probe.
FFPROBE_TIMEOUT_S = 30


def _probe_frame_count(video: Path) -> int | None:
    """Total frames in the video, or None if ffprobe cannot say.

    Counting exactly (`-count_frames`) decodes the entire clip, which is the
    cost this module exists to avoid, so this reads the container's own
    metadata instead and falls back to duration x frame rate.
    """
    if not shutil.which("ffprobe"):
        return None
    try:
        proc = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=nb_frames,duration,avg_frame_rate",
             "-of", "json", str(video)],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=FFPROBE_TIMEOUT_S,
        )
        streams = json.loads(proc.stdout).get("streams") or []
    except (subprocess.SubprocessError, OSError, json.JSONDecodeError):
        return None
    if not streams:
        return None
    stream = streams[0]

    nb_frames = stream.get("nb_frames")
    if nb_frames not in (None, "N/A"):
        try:
            if int(nb_frames) > 0:
                return int(nb_frames)
        except (TypeError, ValueError):
            pass

    # Matroska and fragmented MP4 leave nb_frames unset. Duration x rate is
    # close enough: the stride only has to land within a frame or two.
    try:
        num, _, den = (stream.get("avg_frame_rate") or "").partition("/")
        fps = float(num) / float(den)
        n = int(float(stream["duration"]) * fps)
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        return None
    return n if n > 0 else None


class InadequateVideoError(Exception):
    """Raised when a video yields too few sharp, usable frames for photogrammetry."""
    
    def __init__(self, usable: int, frames: list[Path], message: str):
        super().__init__(message)
        self.usable = usable
        self.frames = frames


def extract_frames(video: Path, out_dir: Path, *, target: int = 32) -> list[Path]:
    """Extract frames from a video, preserving even temporal coverage and sharpness.

    Why this signature?
    `video` and `out_dir` are standard paths. `target` sets the desired number
    of frames, defaulting to 32 (a solid baseline for photogrammetry). Returns
    a list of paths to the successfully extracted and accepted frames so the
    caller can hand them straight to the reconstruction pipeline.

    Over-extracts frames, scores them, and selects the sharpest valid frame
    from each temporal bucket to guarantee both coverage and quality.
    """
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg not found in PATH; required for video ingestion")

    out_dir.mkdir(parents=True, exist_ok=True)
    # Clear stale frames from an earlier run. Output names are positional, so a
    # shorter run leaves the tail of a longer one behind, and any downstream
    # stage that globs this directory would ingest frames of a different
    # object without noticing.
    for old in out_dir.glob("frame_*.jpg"):
        old.unlink()

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        # Over-extract by a fixed multiple of the target, NOT every frame.
        # Extracting everything wrote 1800 JPEGs for a 30s 60fps clip to keep
        # 32, and scored all of them. A few candidates per output slot is
        # enough to find a sharp one, and it decouples how densely we sample
        # from how many frames the caller asked for -- which was the bug:
        # `target` controlled both, so the same video reported a different
        # number of "usable" frames depending on how many were requested.
        #
        # Sample by STRIDE across the whole clip, never by an output cap.
        # `thumbnail=3` plus `-frames:v 192` looked equivalent and was not:
        # thumbnail emits one frame per batch of 3, and -frames:v stops ffmpeg
        # once 192 are written, so decoding ended at input frame ~576. A 30s
        # 60fps orbit was read over its first 9.6s only -- about 120 degrees of
        # arc -- and the buckets below then spread themselves evenly over that
        # truncated prefix while reporting full coverage. Reconstruction from a
        # third of an orbit fails, with nothing pointing back to ingest.
        n_frames = _probe_frame_count(video)
        # Probe failure keeps every frame: decoding too much is slow, whereas a
        # guessed stride silently crops the orbit again.
        stride = max(1, n_frames // (target * OVERSAMPLE)) if n_frames else 1
        cmd = [
            "ffmpeg",
            "-nostdin",
            "-i", str(video),
            # The comma is escaped because ffmpeg splits filters on it.
            "-vf", f"select=not(mod(n\\,{stride}))",
            "-vsync", "vfr",
            "-q:v", "2",
            str(tmp_path / "frame_%05d.jpg")
        ]

        try:
            proc = subprocess.run(
                cmd,
                check=True,
                stdout=subprocess.DEVNULL,
                # Keep stderr: discarding it left every failure mode -- a
                # corrupt container, an unsupported codec -- reading as the
                # same generic message with nothing to act on.
                stderr=subprocess.PIPE,
                timeout=FFMPEG_TIMEOUT_S,
            )
        except subprocess.TimeoutExpired as e:
            raise RuntimeError(
                f"ffmpeg timed out after {FFMPEG_TIMEOUT_S}s reading {video}"
            ) from e
        except subprocess.CalledProcessError as e:
            tail = (e.stderr or b"").decode("utf-8", "replace").strip().splitlines()
            raise RuntimeError(
                f"ffmpeg could not read {video}: "
                + (tail[-1] if tail else "no diagnostic")
            ) from e
        del proc

        all_frames = sorted(tmp_path.glob("*.jpg"))
        n_candidates = len(all_frames)
        if not all_frames:
            raise InadequateVideoError(0, [], "Video yielded no frames")

        # Group into `target` roughly equal buckets to preserve temporal distribution.
        # This prevents picking 32 sharp frames from a single 2-second static shot.
        buckets = []
        n = len(all_frames)
        for i in range(target):
            start = math.floor(i * n / target)
            end = math.floor((i + 1) * n / target)
            if start < end:
                buckets.append(all_frames[start:end])

        usable_frames = []
        for i, bucket in enumerate(buckets):
            best_frame = None
            best_score = -1.0

            for frame_path in bucket:
                img = cv2.imread(str(frame_path))
                if img is None:
                    continue

                res = assess_quality(img)
                # Keep the sharpest frame that passes all quality checks.
                if res.ok and res.blur_score > best_score:
                    best_score = res.blur_score
                    best_frame = frame_path

            if best_frame is not None:
                dst = out_dir / f"frame_{i:04d}.jpg"
                shutil.copy2(best_frame, dst)
                usable_frames.append(dst)

    # Only a genuine shortage is an error. The caller decides what it needs:
    # detect_regime already applies the >= 16 photogrammetry threshold, and a
    # caller asking for 12 has said 12 is enough. Hardcoding 16 here made
    # target=10, 12 or 15 fail unconditionally on flawless footage.
    if not usable_frames:
        raise InadequateVideoError(
            0, [],
            # Candidates examined, NOT surviving buckets. Reporting buckets is
            # capped at `target`, so a flawless 60-frame clip asked for 15
            # frames replied "only 15 usable frames survived quality checks" --
            # naming a limit of its own sampling grid as a fault in the footage.
            f"No frames in {video} passed the quality checks "
            f"({n_candidates} candidates examined): too blurry, too dark, or "
            "below the minimum resolution."
        )
    return usable_frames
