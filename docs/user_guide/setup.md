# Installation and Setup

If you're already comfortable using Python and environments the short story is that you can install Spyral through pip using

```bash
pip install attpc_spyral
```

However if you're new to using Python, below are some instructions on how to setup your system and install Spyral.

## Python and Pip

Spyral requires a Python of version > 3.10 and < 3.13. Most operating systems do not have one of these installed, and you will need to install it yourself. The recommended (minimum) version is 3.11.

### Linux

To install on Linux (Debian flavor) use

```bash
sudo apt install python3.11
```

You will also to install the appropriate pip using

```bash
sudo apt install python3-pip
```

### MacOS

Python can be easily installed using [homebrew](https://brew.sh/). To install Python 3.11 use

```bash
brew install python@3.11
```

This will also install the appropriate pip.

### Windows

Windows is mostly a mess for this. You can download from the official [Python](https://www.python.org/downloads/windows/) if you want and manual install. However, I recommend using Visual Studio Installer (see [here](https://learn.microsoft.com/en-us/visualstudio/install/install-visual-studio?view=vs-2022)) to install everything. It will make sure your paths are set and not a huge mess. However, Visual Studio is quite heavy, so your mileage may vary.

### Python @ FRIB

When working on the FRIB systems (both Windows and Linux), in many cases you will not have control over which versions of python are available to you. Instead, the recommended solution is often to use the pre-installed Anaconda distributions on the systems. Below is the Spyral approved method for setting up an environment:

- First setup a new conda environment for Spyral. To do this run: `conda create --name spyral-env python=3.11`. The Python flag `python=3.11` tells conda to use Python version 3.11 for this environment. You can set the version number to any of the supported python versions for Spyral.
- Once that environment is setup, it can be activated at anytime using `conda activate spyral-env`. To deactivate (return to `base`) use `conda activate`. To completely deactivate any conda virtual environment use `conda deactivate`

This method is tested during development and should work in general even on systems not at FRIB where Anaconda is installed and you have limited control over what Python is avaliable to you. However, our general recommendation remains, where possible, to install and manage your Python distributions yourself to avoid overlarge installs and possible dependency clashing (or at the very least use miniconda instead of a full Anaconda).

### Using the right Python

In general when a newer version of Python is installed it is aliased as `python3`. Sometimes if multiple new versions are installed you may have to get more specific and use something like `python3.11`. You can always check by running the interpreter to see which version you are actively using.

## Installation

Before installing it is recommended to create a virtual environment. This can be done using the following command structure:

```bash
python -m venv .venv
```

Note that on some operating systems you may need to use `python3` instead of `python`. Once your virtual environment is created, activate it by running

```bash
source .venv/bin/activate
```

The above example is for Linux/MacOS. Windows users will need to use the slightly different commands. More details on virtualenvs can be found [here](https://docs.python.org/3/library/venv.html). Once the environment is active, use pip to install Spyral.

```bash
pip install attpc_spyral
```

Note the package is named `attpc_spyral` not `spyral`.

**Note**: if you installed Python through Anaconda (i.e. the methods described by the [Python @ FRIB](#python-frib) section) you *do not* need to create a virtual environment as descirbed in this section. You already created a conda virtual environment instead. Simply skip right to using `pip install attpc_spyral` after activating the conda environment.

### Why use virtual environments?

This allows us to keep all of the Spyral dependencies isolated from any other projects you have! Otherwise versions of libraries might clash and result in all kinds of nasty side effects.

### Do I need anything else?

The only other thing you'll need is some data to analyze!

### What are the Spyral dependencies?

These are the packages Spyral needs to run. Here we've listed the big dependencies

- [spyral-utils](https://github.com/gwm17/spyral-utils/) - This includes
    - [shapely](https://shapely.readthedocs.io/en/stable/manual.html)
    - [scipy](https://scipy.org/)
    - [numpy](https://numpy.org/)
    - [polars](https://pola.rs)
    - [pycatima](https://github.com/hrosiak/pycatima)
- [scikit-learn](https://scikit-learn.org/stable/)
- [contourpy](https://contourpy.readthedocs.io/en/v1.2.0/)
- [h5py](https://www.h5py.org/)
- [lmfit](https://lmfit.github.io/lmfit-py/)
- [matplotlib](https://matplotlib.org/)
- [numba](https://numba.readthedocs.io/en/stable/)
- [rocket-fft](https://pypi.org/project/rocket-fft/)
- [tqdm](https://github.com/tqdm/tqdm)

Without these amazing packages, Spyral wouldn't be anywhere near as useful. Thank you to all of these amazing projects!
