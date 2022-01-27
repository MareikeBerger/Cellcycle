import os
import numpy as np

def make_file_path(root_path, file_name):
    return os.path.join(root_path, file_name) # path where the output will be stored

def make_parameter_path(file_path):
    return os.path.join(file_path, 'parameter_set.csv') # path where the entire serie will be stored

def make_dataset_paths(series_path):
    return np.array([os.path.join(item, 'dataset.h5') for item in series_path]) # path to the data_set

def make_series_names(file_name, id):
    return np.array([file_name + '_' + str(item) for item in id]) # the name of each serie
    
def make_series_paths(file_path, series_names):
    return np.array([os.path.join(file_path, item) for item in series_names]) # path to the individual series

def make_directory(filepath):
    """ Tries to make a new directory at filepath. If this was succesfull, returns True. If fails to make new directory, returns false

    Args:
        filepath (string): path to the file where simulation output should be stored

    Returns:
        boolean: True if new directory was made, False if it failed to make new directory
    """
    try:
        os.mkdir(filepath)
    except OSError:
        print("Creation of the directory %s failed" % filepath)
        return False
    else:
        print("Successfully created the directory %s " % filepath)
        return True

def print_parameter_set_to_file(file_path, parameter_path, data_frame):
    """ Transforms the class ParameterSet into a data frame with removed entries and saves it as .csv to the
    parameter path, if the path exists. """
    if os.path.isdir(file_path):
        data_frame.to_csv(parameter_path, sep=';')
        print("Successfully saved parameter_set to the directory %s " % file_path)
    else:
        print("Directory for saving parameter_set df %s is not a directory" % file_path)
        exit()


def print_dict_to_json(dict, series_path):
    """ Saves the input dictionary as a .json file to the series path of the same dictionary. First checks whether the
    series path of the dictionary is a directory.

        Parameters
        ----------
        dict : dictionary
            The dictionary that will be saved to the file

        """
    if os.path.isdir(series_path):
        print("Successfully saved parameter_set to the directory %s " % series_path)
        dict.to_json(os.path.join(series_path, 'parameters.json'))
    else:
        print("Directory for saving parameters df %s failed" % series_path)
        exit()