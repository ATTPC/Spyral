# PointCloud-utils

Code for creating point clouds from HDF5 run files. (Recommended to run on the N103 machine in the NSCL computer
system for best performance)

## Installation

### Download

To download the repository use `git clone --recursive https://github.com/turinath/PointCloud-utils.git`

### Anaconda

On the same directory as this package, run:

```[bash]
conda env create -f environment.yml
```

This creates an Anaconda environment with the name "pcutil-env" with all of the necessary libraries and versions.

### Pip

Or if pip is prefered create a virtual environment using

```[bash]
python -m venv </some/path/to/your/new/environment>
```

Activate the environment using `source </some/path/to/your/new/environment/>/bin/activate`, then install all required dependencies using

```[bash]
pip install -r requirements.txt
```

All dependencies for PointCloud-utils will then be installed to your virtual environment

## Requirements

Anaconda >= 4.10.1

Python >= 3.10

## Usage

### Configuration

User configuration parameters are passed through JSON files. Currently hardcoded to accept a file named config.json at the top level of the repository, but this will change to accept configuration files from the command line.

Configurations contain many parameters. These can be seen in the config.json example given with the repo. These parameters are grouped by the use case:

- Workspace parameters: These are file paths to either raw data, the workspace, or various AT-TPC pad data files.
- Run parameters: Run numbers over which the data should be processed, as well as indications of which types of analysis to run
- Detector parameters: detector conditions and configuration
- Gas parameters: energy loss parameters
- Trace parameters: parameters which are used in the peak identification and baseline removal analysis
- Cross-talk parameters: parameters used in cross-talk removal, after peaks have been identified
- Clustering parameters: point cloud clustering parameters
- More to come...

### Running

To use PointCloud-utils, run the main.py script located at the top level of the repository (i.e. `python3 main.py`) with the virtual environment activated.
