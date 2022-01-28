# Mechanistic Cell Cycle Simulation

This code implements different mechanistic models of the regulation of replication initiation in E. coli: The auto-regulated initiator titration (AIT) model, two versions of a protein activation model (LD and LDDR model) and a combination of both models. 
The simulation updates relevant parameters of the cell (like the volume, number of chromosomes, number of initiators) for incremental time steps. Whenever the cell divides, one of the two cells is discarded. The simulation therefore produces a single cell line for a given total time T_{max}. 

## How to obtain the project locally
First, the code should be downloaded from the repository to the folder of your choice on your computer. 
The file structure of this project looks like this

```
project
└─── cellcycle
│   │   mainClass.py
│   │   CellCycleCombined.py
│   │   ... (more python files)
└─── notebooks
│   │   Fig1A.ipynb
│   │   ... (more jupyter notebooks)
|   README.md
│   setup.py    
│   input_params.json 
│   LICENSE   
```
The simulations are written in python 3 and require that a python version of ``>=3.6`` is installed on the computer. This tutorial is written for the operating system Linux.
## Optional: Make a virtual environment
 It is recommended to first create a virtual environment in which all required packages will be installed and from which the code can be run. Open a terminal and create a new virtual environment by typing in the terminal
 ```console
 virtualenv ~/.virtualenvs/virtual_env_cellcycle
 ```
 This will create a virtual environment in the folder ``.virtualenvs`` wich is located in the home folder. This virtual environment can be activated via
 ```console
source ~/.virtualenvs/virtual_env_cellcycle
```
In the next step you can then make a package from the code in this virtual environment and whenever you want to use the code or jupyter notebooks you can do this from this virtual environment.

## Making a package from the project
Now a package should be generated from the project in order to use the code and the jupyter notebooks. Open a terminal in the same directory as the file setup.py. Then activate your virtual environment as described in the previous section.
This line of code makes a package of the code that can then be used to run the code:

```console
python3 setup.py sdist bdist_wheel
```
Now, you just need to install the package via
```console
pip install -e .
```
The ``-e`` extension ensures that whenever you work on the code, the package uses the new version of the code and not the once installed version of it. The time to run these two commands should only be a few seconds.

## How to run the simulations
Now all relevant packages for running the code should be installed. In order to run simulations a few variables need to be specified by the user. In the file ``input_params.json`` the following variables need to be specified:

- ``ROOT_PATH`` should be set to the file path where the output of the simulations should be stored. 
- ``FILE_NAME`` will be the name of the folder in which the simulation will be stored.
- ``CODE_PATH`` should be set to the file path where the file ``setup.py`` is located. It will be used to store the version of the git and the ``id`` of the commit that was used to generate the code.
- ``INPUT_FILE_PATH`` specifies the path containing the input parameter set that will be used in the simulations to generate new data
- ``MAKE_NEW_PARAMETER_SET`` should be set to either ``true`` or ``false``. If this variable is set to ``true`` then the ParameterSet class is used to generate a new parameter set and this parameter set is then used to run the simulations. If the variable is set to ``false`` then the parameter set located at ``INPUT_FILE_PATH`` is used when running the simulations.

The simulations can either be started from the terminal or in a jupyter notebook. 

### Run simulations from terminal
From the terminal, the simulations can be run in the virtual environment that contains the installed package via the following command:
```console
cellcycle_run /path_to_input_params/input_params.json
```
where the argument specifies the file path to the input parameters that you have specified.

### Run simulations from jupyter notebook
Alternatively, you can run the simulations from a jupyter notebook. In order to use the packages also in the jupyter notebook we need to install ``jupyter`` in the virtual environment via

```console
pip install jupyter
```

Then we can start jupyter notebook from the terminal via 

```console
jupyter notebook
```
and use the ``cellcycle`` package that we have installed in this virtual environment. The jupyter notebook called ``LaunchNewSimu.ipynb`` can be used to run a simulation via a jupyter notebook. 

## What happens in the simulation

First, the parameter set is loaded (either from input location or via the class ParameterSet depending on the value of the variable ``MAKE_NEW_PARAMETER_SET``). 
Then the simulation checks, whether at the location where the output should be stored (``ROOT_PATH``) there is a folder with the name ``FILE_NAME``. 
If there is already a folder with the same name, the program asks whether the simulation should stop (type any character for yes and ``Enter`` for no). 
If you decide to run the simulations anyways, the folder will be overwritten with new data. The parameter set is stored as ``parameter_set.csv`` in the output folder. Then a loop over the number 'n_series' of simulations is started. For each series, we first make a new directory containing the folder in which the data of the series will be saved. Then we create a dictionary of the parameters used in this series and store it in the series folder as a ``parameters.json`` file. Then the parameter dictionary is used to run a cell cycle using the class 'CellCycleCombined'. After running the cell cycle, the obtained time traces and arrays with division and initiation volumes are stored in hdf5 files. We plot a few relevant variables as a function of time in order to get a first impression of the results. 

If you want to run a simulation with new parameters set the ``MAKE_NEW_PARAMETER_SET`` to ``True`` and modify the parameters as described in the comments in the ParameterSet class.

## Data analysis via jupyter notebook
The data analysis is done via jupyter notebooks that open the hdf5 files and create data frames that are used to analyse the simulations. The jupyter notebooks can either be used to re-analyse existing data or to analyse a newly created dataset via the simulation as explained above. 

### Download data from Zenodo and analyse it via jupyter notebook
The data for all figures of the paper and SI can be downloaded on Zenodo. For each (sub)figure in the paper we have made a separate folder containing the data and a separate jupyter notebook for analysing this data. To run a jupyter notebook on the existing data, you first need to specify the file path to the folder named ``Data`` that contains the entire dataset of the article. You can do this by setting the parameter named ``DATA_FOLDER_PATH`` in the ``input_params.json`` file to the filepath leading to the folder ``Data`` in which the downloaded data is located. 

Then you can simply run all jupyter notebooks, because by default they will analyse the data that is located at the ``DATA_FOLDER_PATH``. 

### Analyse newly generate data via jupyter notebooks
You can also analyse newly generated data using the jupyter notebooks. In this case the variable ``file_path`` in the jupyter notebook needs to be changed and set to the location where you stored your newly generated data. Be careful to put the same number of simulations with the same parameters at the file location in order to be able to analyse new simulations with the existing jupyter notebooks.