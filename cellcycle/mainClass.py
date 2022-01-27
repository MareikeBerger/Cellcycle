from .CellCycleCombined import CellCycleCombined
from . import PlottingTools as plottingTools
from .ParameterSet import ParameterSet
from . import DataAnalysis as dataAnalysis
from . import DataStorage as dataStorage
from . import FilePathTools as filePathTools

import numpy as np
import pandas as pd
import json
import sys


def start_simulation(file_path, confirm):
    """Checks whether a new directory was made. If yes, then simulation will start. If not, then ask user whether he wants to overwrite existing simulation and 
    start a new one. If user types simply enter, then a new simulation will start. If user types any character and then enter, the simulation will stop.

    Args:
        file_path (string): path to where new directory should be made

    Returns:
        boolean: if True -> any existing folders will be overwritten and a new simulation starts
        if False-> stop simulation
    """
    # If this parameter is set to True, simulation is stopped [default: False]
    start_simulation = True
    new_directory_was_made = filePathTools.make_directory(file_path)
    if not new_directory_was_made:
        print('simulation exists already')
        # if no external confirm parameter was given then ask via terminal
        if confirm == None:
            stop_simulation = input('Should simulation start? (Enter=yes, any character=no)') or False      
            start_simulation = not stop_simulation
        else:
            start_simulation = confirm # if confirm is given then set start simulation to this value
    return start_simulation


def data_frame_from_existing_parameter_set(input_file_path):
    return pd.read_csv(input_file_path, sep=';')

def data_frame_from_new_parameter_set(code_path):
    # make an instance of parameterSet class containing all parameters
    parameter_set = ParameterSet(code_path)
    return parameter_set.parameter_set_df

def run_simulations(data_frame_params, confirm, file_name, file_path, parameter_path):
    # Based on given root_path, file_name and code_path makes paths for storing data
    series_names = filePathTools.make_series_names(file_name, data_frame_params["id"])
    series_paths = filePathTools.make_series_paths(file_path, series_names)
    dataset_paths = filePathTools.make_dataset_paths(series_paths)

    # Check whether there is an output directory already and that the additional confirm is true
    if start_simulation(file_path, confirm):
        print('start with simulation')
        # print parameter set to file
        filePathTools.print_parameter_set_to_file(file_path, parameter_path, data_frame_params)

        # Loop over all series and run a cell cycle for each series
        for i_series in range(data_frame_params["n_series"][0]): 
            # make a dictionary from the parameters set with index i_series
            parameter_dict = data_frame_params.iloc[i_series]  

            # create the directory for the simulation i_series and save parameters as json
            filePathTools.make_directory(series_paths[i_series])
            filePathTools.print_dict_to_json(parameter_dict, series_paths[i_series])

            # make new instance of cell cycle class and run it
            myCellCycle = CellCycleCombined(parameter_dict)
            myCellCycle.run_cell_cycle()

            # make data frames out of data arrays and store data in hdf5 file for each series
            dataStorage.saveDataFrameToHdf5(dataset_paths[i_series], myCellCycle.makeDataFrameOfCellCycle(), 'dataset_time_traces')
            dataStorage.saveDataFrameToHdf5(dataset_paths[i_series], myCellCycle.makeDataFrameOfInitEvents(), 'dataset_init_events')
            v_b_v_d_dataframe = myCellCycle.makeDataFrameOfDivisionEvents()
            dataStorage.saveDataFrameToHdf5(dataset_paths[i_series], v_b_v_d_dataframe, 'dataset_div_events')
            # plot a few variables as a function of the simulation time
            plot_time_traces_of_simulation(parameter_dict, series_paths[i_series], myCellCycle, v_b_v_d_dataframe)

    else:
        print('No new simulation was started')

def plot_time_traces_of_simulation(parameter_dict, series_path, myCellCycle, v_b_v_d_dataframe):
    if parameter_dict.version_of_model == 'titration':
        dataAnalysis.plot_time_trace_number_initiators(series_path, myCellCycle, parameter_dict)

    elif parameter_dict.version_of_model == 'switch_titration':
        dataAnalysis.plot_time_trace_switch_titration_combined(series_path, myCellCycle, parameter_dict)
        dataAnalysis.plot_time_trace_number_initiators(series_path, myCellCycle, parameter_dict)
    
    elif parameter_dict.version_of_model == 'switch':
        dataAnalysis.plot_time_trace_number_initiators(series_path, myCellCycle, parameter_dict)
        dataAnalysis.plot_time_trace_active_initiator_concentration(series_path, myCellCycle, parameter_dict)
        
    elif parameter_dict.version_of_model == 'switch_critical_frac':
        dataAnalysis.plot_time_trace_number_initiators(series_path, myCellCycle, parameter_dict)
        dataAnalysis.plot_time_trace_initiator_fraction(series_path, myCellCycle, parameter_dict)
    plottingTools.plot_two_arrays(series_path, myCellCycle.t_init, myCellCycle.v_init_per_ori, r'$t$', r'$v^\ast$', 'init_volume_over_time')

def extract_variables_from_input_params_json(path_to_json):
    with open(path_to_json) as json_file:
        data = json.load(json_file)
    return data

def run():
    file_path_input_params_json = sys.argv[1]
    main(file_path_input_params_json)

def main(file_path_input_params_json, confirm=None):
    input_param_dict = extract_variables_from_input_params_json(file_path_input_params_json)
    # Based on given root_path and file_name makes paths for storing data
    file_path = filePathTools.make_file_path(input_param_dict["ROOT_PATH"], input_param_dict["FILE_NAME"])
    parameter_path = filePathTools.make_parameter_path(file_path)
    if input_param_dict["MAKE_NEW_PARAMETER_SET"]:
        data_frame_params = data_frame_from_new_parameter_set(input_param_dict["CODE_PATH"])
    else:
        data_frame_params = data_frame_from_existing_parameter_set(input_param_dict["INPUT_FILE_PATH"])
    run_simulations(data_frame_params, confirm, input_param_dict["FILE_NAME"], file_path, parameter_path)