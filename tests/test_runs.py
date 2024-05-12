from spyral.core.run_stacks import (
    form_run_string,
    collect_runs,
    create_run_stacks,
    get_size_path,
)

from pathlib import Path


def test_form_run_string():
    expected = "run_0001"
    created = form_run_string(1)
    assert expected == created


def test_size_path():
    test_path = Path("./__init__.py")
    expected_size = 0
    reported_size = get_size_path(test_path)
    assert reported_size == expected_size


def test_collect():
    fake_trace_path = Path(__file__).parent
    collected = collect_runs(fake_trace_path, 0, 1)
    assert len(collected) == 2
    assert collected[0] != 0
    assert collected[1] != 0


def test_tall_create_run_stack():
    n_stacks = 1
    fake_trace_path = Path(__file__).parent
    tall_stack = create_run_stacks(fake_trace_path, 0, 1, n_stacks)

    assert len(tall_stack) == 1
    assert tall_stack[0][0] == 1
    assert tall_stack[0][1] == 0


def test_wide_create_run_stack():
    n_stacks = 2
    fake_trace_path = Path(__file__).parent
    wide_stack = create_run_stacks(fake_trace_path, 0, 1, n_stacks)

    assert len(wide_stack) == 2
    assert wide_stack[0][0] == 1
    assert wide_stack[1][0] == 0
