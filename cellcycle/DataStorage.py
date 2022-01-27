import h5py
import pandas as pd

def saveMatrixToHdf5(filepath, matrix, filename, nameDataset):
    with h5py.File(filepath + '/' + filename + ".h5", "w") as file_:
         file_.create_dataset(nameDataset, data = matrix)

def saveDataFrameToHdf5(filepath, DataFrame, nameDataset):
    print('saving data_frame')
    if not DataFrame.empty:
        print('not empty')
        DataFrame.to_hdf(filepath, nameDataset)
    else:
        print('empty')

def openMatrixInHdf5(filepath_series_dataset, nameDataset):
    with h5py.File(filepath_series_dataset, "r") as file_:
        data = file_[nameDataset]
        print(data[:, :])
        return data

def openDataFrameHdf5(filepath, key= None):
    if key:
        return pd.read_hdf(filepath + '/dataset.h5', key=key)
    else:
        return pd.read_hdf(filepath + '/dataset.h5',)


def saveDataFrameToCSVfile(filepath, dataframe):
    dataframe.to_csv(filepath, sep=';')