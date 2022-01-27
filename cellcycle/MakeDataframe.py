import os
from glob import glob
from pprint import pprint
import json
import numpy as np
import pandas as pd
import h5py
from scipy.optimize import fsolve

def generate_file_dict(sub_simu_path):
    simu_path, sub_simu_name = os.path.split(sub_simu_path)
    simu_name = os.path.split(simu_path)[1]
    if os.path.isfile(os.path.join(sub_simu_path, "parameters.json")):
        return {
            "simu_name": simu_name,
            "simu_path": simu_path,
            "sub_simu_name" : sub_simu_name,
            "sub_simu_path" : sub_simu_path,
            "path_parameter_file" : os.path.join(sub_simu_path, "parameters.json"),
            "path_dataset" : os.path.join(sub_simu_path, "dataset.h5")
        }

def generate_parameter_dict(param_filepath):
    with open(param_filepath, 'r') as file_:
        return json.load(file_)


def return_average_conc_from_path(filepath_h5, name):
    dataset = pd.read_hdf(filepath_h5, key='dataset_time_traces')
    return extract_average_conc_from_dataset(dataset, name)


def extract_average_conc_from_dataset(dataset, name):
    volume_time_array = np.array(dataset["volume"])
    average_v = np.mean(volume_time_array[int(volume_time_array.size / 2):-1])
    proteins_time_array = dataset[name]
    average_n = np.mean(proteins_time_array[int(volume_time_array.size / 2):-1])
    average_conc = average_n / average_v
    return average_conc


def return_average_n_ori_from_path(filepath_h5):
    dataset = pd.read_hdf(filepath_h5, key='dataset_time_traces')
    n_ori_time_array = np.array(dataset["n_ori"])
    return np.mean(n_ori_time_array[int(n_ori_time_array.size / 2):-1])

def return_average_volume_from_path(filepath_h5):
    dataset = pd.read_hdf(filepath_h5, key='dataset_time_traces')
    volume_time_array = np.array(dataset["volume"])
    return np.mean(volume_time_array[int(volume_time_array.size / 2):-1])

def return_average_v_init_per_n_ori_from_path(filepath_h5):
    hf = h5py.File(filepath_h5, 'r')
    if 'dataset_init_events' in hf.keys():
        data_frame = pd.read_hdf(filepath_h5, key='dataset_init_events')
        v_init = data_frame.iloc[-1]['v_init']
        n_ori_init = data_frame.iloc[-1]['n_ori_init']
        return v_init / n_ori_init
    else:
        print('v_init could not be returned, return None instead')
        print('not in list')
        return None

def return_average_v_init_from_path(filepath_h5):
    hf = h5py.File(filepath_h5, 'r')
    if 'dataset_init_events' in hf.keys():
        data_frame = pd.read_hdf(filepath_h5, key='dataset_init_events')
        return data_frame.iloc[-1]['v_init']
    else:
        print('v_init could not be returned, return None instead')
        return None


def solveRegulatorConcentrationAnalytically(z,
                                            n_ori,
                                            volume,
                                            basal_rate_regulator,
                                            mich_const_regulator,
                                            hill_coeff_regulator,
                                            rate_growth):
    x = z[0]
    F = basal_rate_regulator * (n_ori / volume) / (
            1 + (x / mich_const_regulator) ** hill_coeff_regulator) - rate_growth * x
    return F

def calculate_average_regulator_conc_th( n_ori,
                                        volume,
                                        basal_rate_regulator,
                                        mich_const_regulator,
                                        hill_coeff_regulator,
                                        rate_growth):
    regulator_conc_approx = (basal_rate_regulator * (n_ori / volume) /
                            (rate_growth * mich_const_regulator)) ** ( 1 / (hill_coeff_regulator + 1)) * mich_const_regulator
    return fsolve(solveRegulatorConcentrationAnalytically, regulator_conc_approx,
                                    args=(n_ori,
                                            volume,
                                            basal_rate_regulator,
                                            mich_const_regulator,
                                            hill_coeff_regulator,
                                            rate_growth))


def calculate_average_initiator_conc_th(regulator_conc,
                                        n_ori,
                                        volume,
                                        basal_rate_initiator,
                                        mich_const_initiator,
                                        hill_coeff_initiator,
                                        rate_growth):
    return basal_rate_initiator * (n_ori / volume) / \
           (rate_growth * (1 + (
                   regulator_conc / mich_const_initiator) ** hill_coeff_initiator))


def add_average_values_to_df(total_df):
    total_df["average_init_conc"] = total_df.apply(lambda row: return_average_conc_from_path(row.path_dataset, "N_init"), axis=1)
    total_df["average_init_conc_normalized"] = total_df.apply(
        lambda row: (row.average_init_conc / row.michaelis_const_initiator), axis=1)
    total_df["average_volume"] = total_df.apply(lambda row: return_average_volume_from_path(row.path_dataset), axis=1)
    total_df["average_n_ori"] = total_df.apply(lambda row: return_average_n_ori_from_path(row.path_dataset), axis=1)
    total_df["v_init_per_n_ori"] = total_df.apply(lambda row:
                                                  (return_average_v_init_per_n_ori_from_path(row.path_dataset)), axis=1)
    total_df["v_init"] = total_df.apply(lambda row:
                                        (return_average_v_init_from_path(row.path_dataset)), axis=1)
    # if lipids were evolved explicitly, then calculate average lipid concentration
    try:
        if total_df["model_lipids_explicitly"][0] == 1:
            total_df["average_lipid_conc"] = total_df.apply(lambda row: return_average_conc_from_path(row.path_dataset, "N_lipids"), axis=1)
            total_df["average_lipid_regulator_conc"] = total_df.apply(lambda row: return_average_conc_from_path(row.path_dataset, "N_regulator_lipids"), axis=1)
    except:
        print('no lipids were modelled explicitly')

    return total_df

def add_theoretical_init_reg_concentrations_to_df(total_df):
    total_df["average_reg_conc_th"] = total_df.apply(lambda row: (
        calculate_average_regulator_conc_th(row.average_n_ori,
                                            row.average_volume,
                                            row.basal_rate_regulator,
                                            row.michaelis_const_regulator,
                                            row.hill_coeff_regulator,
                                            row.rate_growth
                                            )[0]
        ), axis=1)

    total_df["average_init_conc_th"] = total_df.apply(lambda row: (
        calculate_average_initiator_conc_th(row.average_reg_conc_th,
                                            row.average_n_ori,
                                            row.average_volume,
                                            row.basal_rate_initiator,
                                            row.michaelis_const_initiator,
                                            row.hill_coeff_initiator,
                                            row.rate_growth
                                            )
        ), axis=1)

    total_df["average_init_conc_th_normalized"] = total_df.apply(
        lambda row: (row.average_init_conc_th / row.michaelis_const_initiator), axis=1)
    return total_df


def make_dataframe(filepath):
    simu_paths = glob(os.path.join(filepath, "*"))
    pprint(simu_paths)
    file_dicts_list = []
    for simu_path in simu_paths:
        if os.path.isdir(simu_path):
            sub_simu_paths = glob(os.path.join(simu_path, "*"))
            for sub_simu_path in sub_simu_paths:
                if os.path.isdir(sub_simu_path):
                    file_dict = generate_file_dict(sub_simu_path)
                    file_dict.update(generate_parameter_dict(file_dict["path_parameter_file"]))
                    file_dicts_list.append(file_dict)
    return pd.DataFrame(file_dicts_list)