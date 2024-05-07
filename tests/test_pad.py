from spyral.core.pad_map import PadMap
from spyral.core.config import INVALID_PATH
from spyral import PadParameters


def test_default_pads():
    params = PadParameters(
        is_default=True,
        is_default_legacy=False,
        pad_geometry_path=INVALID_PATH,
        pad_gain_path=INVALID_PATH,
        pad_time_path=INVALID_PATH,
        pad_electronics_path=INVALID_PATH,
        pad_scale_path=INVALID_PATH,
    )

    default_pmap = PadMap(params)

    assert default_pmap.is_valid


def test_default_legacy_pads():
    params = PadParameters(
        is_default=False,
        is_default_legacy=True,
        pad_geometry_path=INVALID_PATH,
        pad_gain_path=INVALID_PATH,
        pad_time_path=INVALID_PATH,
        pad_electronics_path=INVALID_PATH,
        pad_scale_path=INVALID_PATH,
    )

    default_pmap = PadMap(params)

    assert default_pmap.is_valid
