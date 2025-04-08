## Getting Started

##  Getting Started
###  1. Clone the repository

````zsh
git clone git@code.usgs.gov:talongi/surf.git
cd surf
````

###  2. Install python and dependencies
**Recommended: Using Conda or Mamba package manager**

SURF is supported for Python 3.10+. [install python - miniconda](https://docs.anaconda.com/miniconda/) or [using mamba](https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html) on your computer.

To avoid package conflicts, create a new Conda environment:

```zsh
$ conda create -n surf-env python=3.10 -c conda-forge numpy pandas scipy pyvista pyproj scikit-learn networkx
$ conda activate surf-env
```

For visualization support, install:

```zsh
$ conda install -c conda-forge jupyter trame rasterio trame-vtk trame-vuetify
````

**Alternatively: Using pip**

If you prefer pip, install dependencies from the provided requirements.txt file:

```zsh
pip install -r requirements.txt
# Optional
pip install jupyter rasterio trame trame-vtk trame-vuetify
```

###  3. Set up your python path
To access `SURF` functions globally, add the package path to your `PYTHONPATH`

```zsh
export PYTHONPATH=PYTHONPATH:$HOME/<where ever SURF is cloned to> 
```
This line could also be added to your bashrc or zshrc file to run at startup.
Something like,

```zsh
echo "export PYTHONPATH=PYTHONPATH:$HOME/surf/" >> ~/.zshrc # if surf is cloned to your home directory
```

Alternatively, modify scripts to include the package path where needed as shown in `./examples/*`


