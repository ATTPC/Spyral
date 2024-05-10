from pathlib import Path


def form_run_string(run_number: int) -> str:
    """Make the run_* string

    Parameters
    ----------
    run_number: int
        The run number

    Returns
    -------
    str
        The run string
    """
    return f"run_{run_number:04d}"


def get_size_path(path: Path) -> int:
    """
    Get the size of a path item.

    Parameters
    ----------
    path: pathlib.Path
        the path item to be inspected

    Returns
    -------
    int
        the size of the item at the given path, or 0 if no item exists.
    """
    if not path.exists():
        return 0
    else:
        return path.stat().st_size


def collect_runs(trace_path: Path, run_min: int, run_max: int) -> dict[int, int]:
    """Make dict of runs with the size of the raw data file

    Using the Workspace and the Config run_min and run_max, get a dict of run numbers to be processed
    and sort the list based on the size of the raw trace files. The dict key is the run number, and the
    associated data is the size of the raw trace file. Runs that do not exist or have no data are ommitted
    from the list.

    Parameters
    ----------
    trace_path: pathlib.Path
        the path to AT-TPC trace data
    run_min: int
        the first run, inclusive
    run_max: int
        the last run, inclusive

    Returns
    -------
    dict[int, int]
        a dictionary where the keys are run numbers and the values are the size of the associated raw trace files. The
        dict is sorted descending on the size of the raw trace files.
    """
    run_dict = {
        run: get_size_path(trace_path / f"{form_run_string(run)}.h5")
        for run in range(run_min, run_max + 1)
        if get_size_path(trace_path / f"{form_run_string(run)}.h5") != 0
    }
    run_dict = dict(sorted(run_dict.items(), key=lambda item: item[1], reverse=True))
    return run_dict


def create_run_stacks(
    trace_path: Path, run_min: int, run_max: int, n_stacks: int
) -> list[list[int]]:
    """Create a set of runs to be processed for each stack in n_stacks.

    Each stack is intended to be handed off to a single processor. As such,
    the goal is to balance the load of work across all of the stacks. collect_runs
    is used to retrieve the list of runs sorted on their sizes. The list of stacks
    is then snaked, depositing one run in each stack in each iteration.
    This seems to provide a fairly balanced load without too much effort.

    Parameters
    ----------
    trace_path: pathlib.Path
        The path to AT-TPC traces
    run_min: int
        The minimum run number, inclusive
    run_max: int
        The maximum run number, inclusive
    n_stacks: int
        the number of stacks, should be equal to number of processors

    Returns
    -------
    list[list[int]]
        The stacks. Each stack is a list of ints, where each value is a run number for that stack to process.
    """

    # create an empty list for each stack
    stacks = [[] for _ in range(0, n_stacks)]
    total_load = 0
    load_per_stack = [0 for _ in range(0, n_stacks)]
    sorted_runs = collect_runs(trace_path, run_min, run_max)
    if len(sorted_runs) == 0:
        return stacks

    # Snake through the stacks, putting the next run in the next stack
    # This should put the most equal data load across the stacks
    stack_index = 0
    reverse = False
    for run, size in sorted_runs.items():
        if stack_index != -1 and stack_index != n_stacks:
            stacks[stack_index].append(run)
            load_per_stack[stack_index] += size
        elif stack_index == -1:
            stack_index += 1
            stacks[stack_index].append(run)
            load_per_stack[stack_index] += size
            reverse = False
        elif stack_index == n_stacks:
            stack_index -= 1
            stacks[stack_index].append(run)
            load_per_stack[stack_index] += size
            reverse = True
        total_load += size

        if reverse:
            stack_index -= 1
        else:
            stack_index += 1

    # Remove any unused processors
    stacks = [s for s in stacks if len(s) != 0]

    print("Approximate data load per process:")
    for idx, load in enumerate(load_per_stack):
        print(f"Process {idx}: {float(load/total_load) * 100:.2f}%")

    return stacks
