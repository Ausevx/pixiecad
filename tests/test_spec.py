import pytest

from pixiecad.spec import Dimensions, ObjectSpec, PartSpec, ScaleSource, parse_length


@pytest.mark.parametrize(
    "text,meters",
    [("4.5m", 4.5), ("45cm", 0.45), ("450 mm", 0.45), ("18in", 0.4572), ("2ft", 0.6096)],
)
def test_parse_length(text, meters):
    assert parse_length(text) == pytest.approx(meters)


@pytest.mark.parametrize("bad", ["", "4.5", "m", "-3m", "0cm", "4,5m"])
def test_parse_length_rejects(bad):
    with pytest.raises(ValueError):
        parse_length(bad)


def test_scale_source_priority():
    assert ObjectSpec().scale_source is ScaleSource.NONE
    assert (
        ObjectSpec(dimensions=Dimensions.parse(length="4.5m")).scale_source
        is ScaleSource.DIMENSIONS
    )
    assert (
        ObjectSpec(
            dimensions=Dimensions.parse(length="4.5m"), aruco_marker_size_m=0.1
        ).scale_source
        is ScaleSource.ARUCO
    )


def test_parts_imply_split():
    spec = ObjectSpec(parts=[PartSpec(name="wheels", target_faces=800)])
    assert spec.split_parts


def test_spec_roundtrip(tmp_path):
    spec = ObjectSpec(
        name="car",
        dimensions=Dimensions.parse(length="4.5m", height="1.4m"),
        target_faces=15000,
        parts=[PartSpec(name="chassis"), PartSpec(name="wheels", target_faces=800)],
    )
    p = tmp_path / "spec.json"
    spec.save(p)
    assert ObjectSpec.load(p) == spec
