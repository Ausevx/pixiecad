"""Tests for video ingestion."""

from pathlib import Path
from pixiecad.ingest.video import extract_frames
import numpy as np
import pytest
import shutil
import subprocess

@pytest.fixture(scope="session")
def ffmpeg_available():
    return shutil.which("ffmpeg") is not None


@pytest.fixture
def sharp_video(tmp_path) -> Path:
    """A perfectly sharp 1-second synthetic video."""
    video_path = tmp_path / "sharp.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-f", "lavfi",
            "-i", "testsrc=duration=1:size=640x640:rate=30",
            "-c:v", "libx264",
            "-y",
            str(video_path),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return video_path


@pytest.fixture
def blurry_video(tmp_path) -> Path:
    """A 1.5-second video: first 0.5s blurry, next 1s sharp."""
    video_path = tmp_path / "blurry.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-f", "lavfi",
            "-i", "testsrc=duration=1.5:size=640x640:rate=30",
            "-vf", "boxblur=10:1:enable='between(t,0,0.5)'",
            "-c:v", "libx264",
            "-y",
            str(video_path),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return video_path


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg not installed")
def test_missing_ffmpeg(sharp_video: Path, tmp_path: Path, monkeypatch):
    """A missing ffmpeg produces a clear, actionable error."""
    monkeypatch.setattr(shutil, "which", lambda cmd: None)

    out_dir = tmp_path / "out"
    with pytest.raises(RuntimeError, match="ffmpeg not found in PATH"):
        extract_frames(sharp_video, out_dir)


# ---------------------------------------------------------------------------
# The tests above were shown to pass against a selector that ignored time
# entirely: output frames are renamed positionally, so counting them and
# checking they sort proves nothing about WHERE in the clip they came from.
# These use a clip whose brightness ramps linearly with time, which makes each
# frame's source position readable from its pixels.
# ---------------------------------------------------------------------------

_HAVE_FFMPEG = shutil.which("ffmpeg") is not None


def _ramp_video(path: Path, n: int = 90, size: int = 720) -> Path:
    """A clip whose brightness rises linearly, so time is recoverable."""
    import cv2 as _cv2

    frames = path.parent / "ramp_src"
    frames.mkdir(exist_ok=True)
    rng = np.random.default_rng(0)
    for i in range(n):
        base = int(20 + (200 * i) / (n - 1))
        # Texture on top, or assess_quality rejects every frame as blurred:
        # a flat field has no Laplacian variance at all.
        img = np.clip(
            base + rng.integers(-12, 12, (size, size, 3)), 0, 255
        ).astype(np.uint8)
        _cv2.imwrite(str(frames / f"{i:04d}.png"), img)
    subprocess.run(
        ["ffmpeg", "-y", "-framerate", "30", "-i", str(frames / "%04d.png"),
         "-pix_fmt", "yuv420p", "-crf", "12", str(path)],
        capture_output=True, check=True,
    )
    return path


def _mean_levels(frames: list[Path]) -> list[float]:
    import cv2 as _cv2

    return [float(_cv2.imread(str(f)).mean()) for f in frames]


@pytest.mark.skipif(not _HAVE_FFMPEG, reason="ffmpeg not installed")
def test_frames_span_the_whole_clip(tmp_path):
    """The criterion that matters, and the one the original tests could not
    see: selection must cover the timeline, not the sharpest cluster.

    An orbit captured only over the first two seconds reconstructs nothing --
    the cameras never get round the object.
    """
    video = _ramp_video(tmp_path / "ramp.mp4")
    frames = extract_frames(video, tmp_path / "out", target=16)
    levels = _mean_levels(frames)
    # The clip ramps ~20 -> ~220. Frames drawn from the whole timeline span
    # most of that; frames from any single moment span almost none.
    assert max(levels) - min(levels) > 120, (
        f"selected frames only span brightness {min(levels):.0f}-{max(levels):.0f}, "
        "so they came from a narrow slice of the clip"
    )


def _long_ramp_video(path: Path) -> Path:
    """A 30s 60fps ramp: 1800 frames, far past any plausible output cap.

    Built by ffmpeg rather than frame by frame like `_ramp_video`, because
    1800 PNGs at the 640px minimum resolution is several hundred MB of scratch
    space. testsrc2 supplies the texture that `assess_quality` needs to score
    a frame as sharp; `eq` supplies the brightness ramp that makes each
    frame's position in the clip readable from its pixels.
    """
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi",
         "-i", "testsrc2=size=640x640:rate=60:duration=30",
         "-vf", "eq=brightness=-0.45+(t/30)*0.9:eval=frame",
         "-c:v", "libx264", "-preset", "ultrafast", "-crf", "30",
         "-pix_fmt", "yuv420p", str(path)],
        capture_output=True, check=True,
    )
    return path


@pytest.mark.slow
@pytest.mark.skipif(not _HAVE_FFMPEG, reason="ffmpeg not installed")
def test_a_long_clip_is_sampled_all_the_way_to_its_end(tmp_path):
    """Sampling must reach the end of the clip, however long the clip is.

    Every other fixture here is 90 frames, which is why they all passed while
    `thumbnail=3` with `-frames:v 192` was capping decode at ~576 input
    frames. On a 30s 60fps orbit that cap ended extraction at 9.58s -- about
    120 degrees of arc -- and the buckets then spread themselves evenly over
    that prefix and reported full coverage. This clip is long enough for the
    cap to bite.
    """
    video = _long_ramp_video(tmp_path / "long.mp4")
    levels = _mean_levels(extract_frames(video, tmp_path / "out", target=32))

    # The ramp runs ~70 -> ~180. Truncated at a third of the clip, the last
    # frame measured ~101; sampled across the whole clip it measures ~180.
    assert levels[-1] > 150, (
        f"the last selected frame has brightness {levels[-1]:.0f}, so it came "
        "from partway through the clip, not the end of it"
    )


@pytest.mark.skipif(not _HAVE_FFMPEG, reason="ffmpeg not installed")
def test_frames_are_in_temporal_order(tmp_path):
    """Monotonic brightness means monotonic time, so this also catches a
    selector that shuffles the timeline."""
    video = _ramp_video(tmp_path / "ramp.mp4")
    levels = _mean_levels(extract_frames(video, tmp_path / "out", target=12))
    assert levels == sorted(levels), "frames are not in capture order"


@pytest.mark.skipif(not _HAVE_FFMPEG, reason="ffmpeg not installed")
def test_a_small_target_is_honoured(tmp_path):
    """A caller asking for 12 has said 12 is enough. A hardcoded 16 made
    target=10, 12 and 15 fail unconditionally on flawless footage, blaming the
    video for a limit of its own sampling grid."""
    video = _ramp_video(tmp_path / "ramp.mp4")
    for target in (8, 12, 15):
        got = extract_frames(video, tmp_path / f"out{target}", target=target)
        assert len(got) == target, f"asked for {target}, got {len(got)}"


@pytest.mark.skipif(not _HAVE_FFMPEG, reason="ffmpeg not installed")
def test_a_rerun_does_not_leave_the_previous_run_behind(tmp_path):
    """Output names are positional, so a shorter run would leave the tail of a
    longer one -- and anything globbing the directory would silently ingest
    frames of a different object."""
    video = _ramp_video(tmp_path / "ramp.mp4")
    out = tmp_path / "out"
    extract_frames(video, out, target=20)
    extract_frames(video, out, target=8)
    assert len(list(out.glob("frame_*.jpg"))) == 8
