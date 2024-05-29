from spyral.core.pad_map import PadMap
from spyral.core.config import INVALID_PATH
from spyral import PadParameters


def test_default_pads():
    params = PadParameters(
        is_default=True,
        is_default_legacy=False,
        pad_geometry_path=INVALID_PATH,
        pad_time_path=INVALID_PATH,
        pad_electronics_path=INVALID_PATH,
        pad_scale_path=INVALID_PATH,
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

    reverse0 = default_pmap.get_pad_from_hardware(pad0.hardware)
    assert reverse0 is not None
    assert reverse0 == 0


def test_default_legacy_pads():
    params = PadParameters(
        is_default=False,
        is_default_legacy=True,
        pad_geometry_path=INVALID_PATH,
        pad_time_path=INVALID_PATH,
        pad_electronics_path=INVALID_PATH,
        pad_scale_path=INVALID_PATH,
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

    reverse0 = default_pmap.get_pad_from_hardware(pad0.hardware)
    assert reverse0 is not None
    assert reverse0 == 0
