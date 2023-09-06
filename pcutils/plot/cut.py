from matplotlib.path import Path
from polars import Series
import numpy as np
from typing import Optional
import json

class CutHandler:
    '''
    Handler to recieve vertices from a matplotlib selector (i.e. PolygonSelector).
    Typically will be used interactively, most likely via cmd line interpreter. The onselect
    method should be passed to the selector object at construction. CutHandler can also be used in analysis
    applications to store cuts.

    Example:

    ```[python]
    from evbutils import CutHandler, Cut2D, write_cut_json
    from matplotlib.widgets import PolygonSelector
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1,1)
    handler = CutHandler()
    selector = PolygonSelector(ax, handler.onselect)

    #plot some data here...

    plt.show()
    
    #wait for user to draw a cut and close the window

    mycut = handler.cuts['cut_0']
    mycut.name = 'mycut'
    write_cut_json(mycut, 'mycut.json')
    ```
    '''
    def __init__(self):
        self.cuts: dict[str, Cut2D] = {}

    def onselect(self, vertices: list[tuple[float, float]]):
        cut_default_name = f'cut_{len(self.cuts)}'
        self.cuts[cut_default_name] = Cut2D(cut_default_name, vertices)

class Cut2D:
    '''
    Implementation of 2D cuts as used in many types of graphical analyses with matplotlib
    Path objects. Takes in a name (to identify the cut) and a list of points. The Path
    takes the verticies, and can then be used to check if a point(s) is inside of the polygon using the 
    is_*_inside functions. Can be serialized to json format. Can also retreive Nx2 ndarray of vertices
    for plotting after the fact.
    '''
    def __init__(self, name: str, vertices: list[tuple[float, float]]):
        self.path: Path = Path(vertices, closed=True)
        self.name = name
        
    def is_point_inside(self, x: float, y: float) -> bool:
        return self.path.contains_point((x,  y))

    def is_arr_inside(self, points: list[tuple[float, float]]) -> list[bool]:
        return self.path.contains_points(points)

    def is_cols_inside(self, columns: Series) -> Series:
        data = np.transpose([columns.struct.field(name).to_list() for name in columns.struct.fields])
        return Series(values=self.path.contains_points(data))

    def get_vertices(self) -> np.ndarray:
        return self.path.vertices

    def to_json_str(self) -> str:
        return json.dumps(self, default=lambda obj: {'name': obj.name, 'vertices': obj.path.vertices.tolist()}, indent=4)

def write_cut_json(cut: Cut2D, filepath: str) -> bool:
    json_str = cut.to_json_str()
    try:
        with open(filepath, 'a') as output:
            output.write(json_str)
            return True
    except OSError as error:
        print(f'An error occurred writing cut {cut.name} to file {filepath}: {error}')
        return False

def load_cut_json(filepath: str) -> Optional[Cut2D]:
    try:
        with open(filepath, 'a') as input:
            buffer = input.read()
            cut_dict = json.loads(buffer)
            if not 'name' in cut_dict or not 'vertices' in cut_dict:
                print(f'Data in file {filepath} is not the right format for Cut2D, could not load')
                return None
            return Cut2D(cut_dict['name'], cut_dict['vertices'])
    except OSError as error:
        print(f'An error occurred reading trying to read a cut from file {filepath}: {error}')
        return None
