# Getting Started with Spyral

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

- First setup a new conda environment for Spyral. To do this run: `conda create --name spyral-env python=3.11`. The Python flag `python=3.11` tells conda to use Python version 3.11 for this environment. You can set the version number to any of the supported versions for Spyral.
- Once that environment is setup, it can be activated at anytime using `conda activate spyral-env`. To deactivate (return to `base`) use `conda activate`. To completely deactivate any conda virtual environment use `conda deactivate`
- Once the environment is activated (you should see `(spyral-env)` on your terminal tag) you can install the dependencies using `pip install -r requirements.txt` from the Spyral repository.

Note that you can't use `conda install` for the requirements. Several of our dependencies are only available via PyPI and not conda or conda-forge. This also can refer to the bundled versions of some libraries like scipy and numpy for Anaconda. We only support using the versions pinned in the requirements.txt file.

This method is tested during development and should work in general even on systems not at FRIB where Anaconda is installed and you have limited control over what Python is avaliable to you. However, our general recommendation remains, where possible, to install and manage your Python distributions yourself to avoid overlarge installs and possible dependency clashing (or at the very least use miniconda instead of a full Anaconda).

### Using the right Python

In general when a newer version of Python is installed it is aliased as `python3`. Sometimes if multiple new versions are installed you may have to get more specific and use something like `python3.11`. You can always check by running the interpreter to see which version you are actively using.

## Installation

To download Spyral it is recommend to use `git`

```bash
git clone https://github.com/attpc/Spyral.git
```

This will download the Spyral source code and install it to a directory named `Spyral`. To install all of the necessary dependencies it is recommended to create a virtual environment and use pip. To create a new virtual environment navigate to the Spyral directory and run the following command:

```bash
python -m venv .venv
```

Note that on some operating systems you may need to use `python3` instead of `python`. Once your virtual environment is created, activate it by running

```bash
source .venv/bin/activate
```

The above example is for Linux/MacOS. Windows users will need to use the slightly different commands. More details on virtualenvs can be found [here](https://docs.python.org/3/library/venv.html). Once the virtualenv is active, use pip to install the dependencies.

```bash
pip install -r requirements.txt
```

**Note**: if you installed Python through Anaconda (i.e. the methods described by the [Python @ FRIB](#python--frib) section) you *do not* need to create a virtual environment as descirbed in this section. You already created a conda virtual environment instead. Simply skip right to using `pip install -r requirements.txt` after downloading Spyral and setting up the conda environment.

To make sure everything went ok you can then run

```bash
python main.py --help
```

from the top level of the repository. This should display the Spyral help message if everything went well.

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
- [click](https://click.palletsprojects.com/en/8.1.x/)
- [contourpy](https://contourpy.readthedocs.io/en/v1.2.0/)
- [h5py](https://www.h5py.org/)
- [ipywidgets](https://ipywidgets.readthedocs.io/en/stable/)
- [ipympl](https://matplotlib.org/ipympl/)
- [jupyterlab](https://jupyter.org/)
- [lmfit](https://lmfit.github.io/lmfit-py/)
- [matplotlib](https://matplotlib.org/)
- [nbformat](https://nbformat.readthedocs.io/en/latest/)
- [numba](https://numba.readthedocs.io/en/stable/)
- [plotly](https://plotly.com/)
- [rocket-fft](https://pypi.org/project/rocket-fft/)
- [tqdm](https://github.com/tqdm/tqdm)

Without these amazing packages, Spyral wouldn't be anywhere near as useful. Thank you to all of these amazing projects!
