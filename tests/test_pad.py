from spyral.core.pad_map import PadMap
from spyral.core.config import DEFAULT_MAP
from spyral import PadParameters


def test_default_pads():
    params = PadParameters(
        pad_geometry_path=DEFAULT_MAP,
        pad_time_path=DEFAULT_MAP,
        pad_scale_path=DEFAULT_MAP,
    )

    default_pmap = PadMap(params)

    assert default_pmap.is_valid

    pad0 = default_pmap.get_pad_data(0)
    assert pad0 is not None
    assert pad0.x == -269.95294
    assert pad0.y == 4.2506561
    assert pad0.gain == 1.0
    assert pad0.time_offset == 0
    assert pad0.scale == 1.0
