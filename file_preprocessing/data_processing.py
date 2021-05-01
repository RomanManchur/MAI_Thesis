'''This is help library that makes data preprocessing to make it ready for learning models '''

import pandas as pd
import numpy as np
import sys

from data_visulization import standardPlot
from file_preprocessing.ds_quantization import quantize_ds
from file_preprocessing.ds_normalization import normalize_df
from file_preprocessing.ds_supression import data_supression

def data_processing(name, datatype, measurements, visulalize=False, threshold=1, wavelenght="all", pdfdir='./pdf/'):
    """ Perform data pre-processing (and visualization optional); Returns dictionary with processed data
    :param name <str>
        name of object currently processed
    :param datatype <str>
        data type of object currently processed (V2, CP, UV, etc)
    :param: measurements <dict>
        keys: name of parameter, values: measurements taken for parameter (e.i: "V2": <array>)
    :param: visulalize <boolean>
        defines if data processig is plotted or not (default = False / non-plotted)
    :param threshold: <float>
        defines threshold used for mask filtering and plotting
    :param wavelenght: <string>
        controls if all data is selected or only for specific wavelength
        can either be "low", "medium", "high" or all (default all and corresponds to all data)
    :param pdfdir: <string>
        specifies path to folder where plots are stored

    :return: <dict>
        data after quantization, normalization, compression and interpolation
        keys: data type; values - numpy.array
    """

    # re-arrange the data to PandasDataframe format from input dictionary
    df_loaded_data = pd.DataFrame.from_dict(measurements)

    # get column name that has waves information, used for filtering data
    if "waveV2" in df_loaded_data.columns:
        wave_reference = "waveV2"
    elif "waveCP" in df_loaded_data.columns:
        wave_reference = "waveCP"
    else:
        print("Wrong wave length information in provided dataset, can either be waveV2 or waveCP")
        sys.exit(1)

    if wavelenght == "all":
        measurements_ = df_loaded_data
    elif wavelenght == "low":
        measurements_ = df_loaded_data[df_loaded_data[wave_reference] < 1.6e-6]
    elif wavelenght == "medium":
        measurements_ = df_loaded_data[
            (df_loaded_data[wave_reference] > 1.6e-6) & (df_loaded_data[wave_reference] < 1.7e-6)]
    elif wavelenght == "high":
        measurements_ = df_loaded_data[df_loaded_data[wave_reference] > 1.7e-6]
    else:
        print("Bad filtering parameter applied for wavelength information, expected either low, medium, high or all\n"
              "Got {0}".format(wavelenght))
        sys.exit(1)
    measurements_.reset_index()

    # measurements_ = pd.DataFrame.from_dict(measurements)
    columns = measurements_.columns.to_list()
    x = columns[0]

    # if no data present in filtered dataset exit processing
    if measurements_.empty:
        z_dict = {columns[index]: np.array([]) for index in range(len(columns))}
        return z_dict

    print("Processing file {0}, datatype {1}:.....".format(name, datatype))
    quantized_ds = quantize_ds(measurements_, intervals=100)  # quantize dataset along x-axis
    normalized_ds = normalize_df(quantized_ds, 1,
                                 method="minmax")  # normalize dataset using min-max normalization  - [0..1]"
    z = data_supression(normalized_ds,
                        method="median")  # compress data point in each bucket using mean or median compression
    z.sort_values(by=x, inplace=True)  # sort and replace
    z.interpolate(method="linear", axis=0, direction="forward", inplace=True)

    # Data vizualization after pre-processing
    if visulalize:
        standardPlot.plotData(raw_data=(name, measurements_, threshold),
                              quantized=quantized_ds,
                              normalized=normalized_ds,
                              interpolated=z,
                              dst_folder=pdfdir,
                              plot_type=datatype + "_",
                              scale=wavelenght)
    # postprocessing data
    keys = z.columns.to_list()
    values = z.to_numpy()
    z_dict = {keys[index]: values[:, index] for index in range(len(keys))}
    return z_dict