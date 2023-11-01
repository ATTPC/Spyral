from .config import WorkspaceParameters
from .pad_map import PadMap
from .nuclear_data import NuclearDataMap, NucleusData
from pathlib import Path

def form_run_string(run_number: int) -> str:
        return f'run_{run_number:04d}'

class Workspace:
    '''
    The Workspace class represents the disk location to which data can be written/retrieved.
    The workspace can create directories as needed for writting data. Note, that the workspace cannot
    access directories with restricted permissions.
    '''
    def __init__(self, params: WorkspaceParameters):
        self.trace_data_path = Path(params.trace_data_path)
        self.workspace_path = Path(params.workspace_path)

        if not self.workspace_path.exists():
            self.workspace_path.mkdir()

        if not self.workspace_path.is_dir() or not self.trace_data_path.is_dir():
            print(self.workspace_path)
            print(self.trace_data_path)
            raise Exception('Workspace encountered an error! Trace data path and workspace path should point to directories not files!')
        
        if not self.trace_data_path.exists():
            raise Exception('Workspace encountered an error! The trace data path must exist!')
            
        self.point_cloud_path = self.workspace_path / 'clouds'
        if not self.point_cloud_path.exists():
            self.point_cloud_path.mkdir()

        self.cluster_path = self.workspace_path / 'clusters'
        if not self.cluster_path.exists():
            self.cluster_path.mkdir()

        self.estimate_path = self.workspace_path / 'estimates'
        if not self.estimate_path.exists():
            self.estimate_path.mkdir()

        self.physics_path = self.workspace_path / 'physics'
        if not self.physics_path.exists():
            self.physics_path.mkdir()

        self.gate_path = self.workspace_path / 'gates'
        if not self.gate_path.exists():
            self.gate_path.mkdir()

        self.track_path = self.workspace_path / 'tracks'
        if not self.track_path.exists():
            self.track_path.mkdir()

        self.correction_path = self.workspace_path / 'correction'
        if not self.correction_path.exists():
            self.correction_path.mkdir()

        self.pad_geometry_path = Path(params.pad_geometry_path)
        if not self.pad_geometry_path.exists() or not self.pad_geometry_path.is_file():
            raise Exception('Workspace encountered an error! Pad geometry path does not exist!')
        
        self.pad_gain_path = Path(params.pad_gain_path)
        if not self.pad_gain_path.exists() or not self.pad_gain_path.is_file():
            raise Exception('Workspace encountered an error! Pad gain path does not exist!')
        
        self.pad_time_path = Path(params.pad_time_path)
        if not self.pad_time_path.exists() or not self.pad_time_path.is_file():
            raise Exception('Workspace encountered an error! Pad gain path does not exist!')
        
        self.pad_electronics_path = Path(params.pad_electronics_path)
        if not self.pad_electronics_path.exists() or not self.pad_electronics_path.is_file():
            raise Exception('Workspace encountered an error! Pad gain path does not exist!')
        
        self.nuclear_data_path = Path(params.nuclear_data_path)
        if not self.nuclear_data_path.exists() or not self.nuclear_data_path.is_file():
            raise Exception('Workspace encountered an error! Nuclear data path does not exist!')
        
        self.pad_map = PadMap(self.pad_geometry_path, self.pad_gain_path, self.pad_time_path, self.pad_electronics_path)
        self.nuclear_map = NuclearDataMap(self.nuclear_data_path)

    def get_trace_file_path(self, run_number: int) -> Path:
        runstr = form_run_string(run_number)
        return self.trace_data_path / f'{runstr}.h5'
    
    def get_point_cloud_file_path(self, run_number: int) -> Path:
        runstr = form_run_string(run_number)
        return self.point_cloud_path / f'{runstr}.h5'
    
    def get_cluster_file_path(self, run_number: int) -> Path:
        runstr = form_run_string(run_number)
        return self.cluster_path / f'{runstr}.h5'
    
    def get_estimate_file_path_parquet(self, run_number: int) -> Path:
        runstr = form_run_string(run_number)
        return self.estimate_path / f'{runstr}.parquet'
    
    def get_estimate_file_path_csv(self, run_number: int) -> Path:
        runstr = form_run_string(run_number)
        return self.estimate_path / f'{runstr}.csv'
    
    def get_physics_file_path_parquet(self, run_number: int, particle: NucleusData) -> Path:
        runstr = form_run_string(run_number)
        return self.physics_path / f'{runstr}_{particle.isotopic_symbol}.parquet'
    
    def get_physics_file_path_csv(self, run_number: int, particle: NucleusData) -> Path:
        runstr = form_run_string(run_number)
        return self.physics_path / f'{runstr}_{particle.isotopic_symbol}.csv'
    
    def get_gate_file_path(self, gate_name: str) -> Path:
        return self.gate_path / gate_name
    
    def get_pad_map(self) -> PadMap:
        return self.pad_map
    
    def get_nuclear_map(self) -> NuclearDataMap:
        return self.nuclear_map
    
    def get_track_file_path(self, name: str) -> Path:
        return self.track_path / name
    
    def get_correction_file_path(self, name: str) -> Path:
        return self.correction_path / name
        