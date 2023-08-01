# PointCloud-utils

Code for creating point clouds from HDF5 run files. (Recommended to run on the N103 machine in the NSCL computer
system for best performance)

## Installation

On the same directory as this package, run:

```[bash]
conda env create -f environment.yml
```

This creates an Anaconda environment with the name "python3" with all of the necessary libraries and versions.

## Requirements

Anaconda >= 4.10.1

Python >= 3.7

## Usage

1. Activate the Anaconda environment with: `conda activate python3`

2. Use a text editor to change the PATH variable in line 151 of PCMP.py to the full path of the HDF5 file that you 
want to construct point clouds for.

3. Use a text editor to set the integrated charge thresholds in config.txt

4. Run: `python Phase1.py`

5. Once that is finished running, run: `python Phase2.py`
