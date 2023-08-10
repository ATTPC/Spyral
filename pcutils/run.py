from .core.config import Config
from .core.workspace import Workspace
from .phase_1 import phase_1

def run_pcutils(config: Config):

    ws = Workspace(config.workspace)
    pad_map = ws.get_pad_map()
    for idx in range(config.run.run_min, config.run.run_max + 1, 1):
        if config.run.do_phase1:
            phase_1(ws.get_trace_file_path(idx), ws.get_point_cloud_file_path(idx), pad_map, config.trace, config.cross, config.detector)
        if config.run.do_phase2:
            continue
        if config.run.do_phase3:
            continue
        if config.run.do_phase4:
            continue
        if config.run.do_phase5:
            continue