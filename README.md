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

User configuration parameters are passed through the params.txt file located at the top level of the repository. Current user parameters are:

- PATH: The path to the HDF5 data file containing all trace data
- ntuple_PATH: The path to which output physics data will be written (CSV format regardless of specified extension)
- parent_PATH: The path to which temporary files will be written during the execution of the analysis. Temporary files will be removed upon completion.

All parameters must be specified to execute the analysis.

### Running

To use PointCloud-utils, run the main.py script located at the top level of the repository (i.e. `python3 main.py`) with the virtual environment activated.
