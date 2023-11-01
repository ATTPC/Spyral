from ..core.config import Config
from ..core.workspace import Workspace
from pathlib import Path

def get_size_path(path: Path) -> int:
    if not path.exists():
        return 0
    else:
        return path.stat().st_size
    
def collect_runs(ws: Workspace, run_min: int, run_max: int) -> dict[int, int]:
    run_dict = {run: get_size_path(ws.get_trace_file_path(run)) for run in range(run_min, run_max+1) if get_size_path(ws.get_trace_file_path(run)) != 0}
    run_dict = dict(sorted(run_dict.items(), key=lambda item: item[1], reverse=True))
    return run_dict

def create_run_stacks(config: Config, n_stacks: int) -> list[list[int]]:
    #create an empty list for each stack
    stacks = [[] for _ in range(0, n_stacks)]
    total_load = 0
    load_per_stack = [0 for _ in range(0, n_stacks)]
    ws = Workspace(config.workspace)
    sorted_runs = collect_runs(ws, config.run.run_min, config.run.run_max)
    if len(sorted_runs) == 0:
        return stacks

    #Snake through the stacks, putting the next run in the next stack
    #This should put the most equal data load across the stacks
    stack_index = 0
    reverse = False
    for (run, size) in sorted_runs.items():
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

    print('Approximate data load per process:')
    for idx, load in enumerate(load_per_stack):
        print(f'Process {idx}: {float(load/total_load) * 100:.2f}%')

    return stacks