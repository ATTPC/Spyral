# Extending the Configuration

When modifying Spyral, it is sometimes necessary to expose new configuration parameters. In this section we'll break down how this is done.

## config.py

Pretty much everything you need is found in `spyral/core/config.py`. Each of these parameter groups is represented by an appropriate `dataclass` in `config.py`. For example, the WorkspaceParameters dataclass

```python
@dataclass
class WorkspaceParameters:
    '''
    Parameters describing paths to various resources used across the application
    '''
    trace_data_path: str = ''
    workspace_path: str = ''
    pad_geometry_path: str = ''
    pad_gain_path: str = ''
    pad_time_path: str = ''
    pad_electronics_path: str = ''
```

should look very similar to the Workspace section of our JSON

```json
"Workspace":
{
    "trace_data_path": "/path/to/some/traces/",
    "workspace_path": "/path/to/some/workspace/",
    "pad_geometry_path": "/path/to/some/geometry.csv",
    "pad_gain_path": "/path/to/some/gain.csv",
    "pad_time_path": "/path/to/some/time.csv",
    "pad_electronics_path": "/path/to/some/electronics.csv"
},
```

If you're not familiar with dataclasses, you can read up on them [here](https://docs.python.org/3/library/dataclasses.html). Essentially, this wrapper allows you to make simple data holding structures. We then store all of these parameter sets in a parent Config class. So the only trick is translating the JSON to Python, and thankfully the python standard library already does most of this for us. In the function `json_load_config_hook` we take the dictionary of parsed JSON data and place it into our structures. Check out the following snippet:

```python
def json_load_config_hook(json_data: dict[Any, Any]) -> Config:
    config = Config()
    ws_params = json_data['Workspace']
    config.workspace.trace_data_path = ws_params['trace_data_path']
    config.workspace.workspace_path = ws_params['workspace_path']
    config.workspace.pad_geometry_path = ws_params['pad_geometry_path']
    config.workspace.pad_gain_path = ws_params['pad_gain_path']
    config.workspace.pad_time_path = ws_params['pad_time_path']
    config.workspace.pad_electronics_path = ws_params['pad_electronics_path']
```

So extending the configuration is as simple as:

1. Make a new set of configuration parameters in the JSON
2. Add a new dataclass to config.py
3. Add the new dataclass to the Config parent class
4. Update the json_load_config_hook function
