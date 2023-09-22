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

User configuration parameters are passed through JSON files. Configuration files are passed at runtime to the script.

Configurations contain many parameters. These can be seen in the config.json example given with the repo. These parameters are grouped by the use case:

- Workspace parameters: These are file paths to either raw data, the workspace, or various AT-TPC pad data files.
- Run parameters: Run numbers over which the data should be processed, as well as indications of which types of analysis to run
- Detector parameters: detector conditions and configuration
- Trace parameters: parameters which are used in the peak identification and baseline removal analysis
- Cross-talk parameters: parameters used in cross-talk removal, after peaks have been identified
- Clustering parameters: point cloud clustering parameters
- Estimation parameters: used to generate estimates of physical observables
- Solver parameters: used to control the physics solver (minimzation or kalman filter)

### Running

To use PointCloud-utils, run the main.py script located at the top level of the repository (i.e. `python3 main.py`) with the virtual environment activated. Example:

```[bash]
python main.py my_config.json
```

### Plotting

PointCloud-utils also bundles some helpful plotting tools for creating dataset histograms. The default numpy/scipy/matplotlib histogramming solution is not terribly useful for larger datasets. The tools included in pcutils/plot can help generate histograms of large datasets as well as generate gates for use with various analyses. The plotter.py file contains an example of how to generate a particle ID gate and then apply it to a dataset.
