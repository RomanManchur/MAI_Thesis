import math

import numpy as np
import pandas as pd


def quantize_ds(panda_ds, intervals=100):
    '''
    Function to quantize dataset along x-axis (zero-column)
    :param panda_ds: input data set as pandas dataset object
    :return: quantized representation of input DS along 1st dimension (x-axis)
    '''
    print('Make quantize is called....')
    col_names = panda_ds.columns.to_list()
    x = panda_ds.iloc[:, 0]
    max_value = math.ceil(x.max())
    min_value = math.floor(x.min())
    interval_range = max_value - min_value

    #Pass condition to avoid zero devision if data doesn't span on range and either zero or represented by same x-values
    if interval_range == 0: return panda_ds

    interval_width = interval_range // intervals
    n = len(x)
    quant = np.zeros(n)
    for idx, element in enumerate(x):
        cell = int((element - min_value) // interval_width)
        quantized_element = cell * interval_width + min_value
        quant[idx] = quantized_element

    # print(len(set(quant)))
    #quant_ds = pd.concat([pd.DataFrame(data=quant, columns=[col_names[0]]), panda_ds[col_names[1:]]], axis=1)
    quant_ds = pd.DataFrame(quant,columns=[col_names[0]])
    original_ds_right_part = panda_ds[col_names[1:]]
    original_ds_right_part.reset_index(drop=True,inplace=True)
    quant_ds = quant_ds.join(original_ds_right_part)

    # some bins will have no data, hence need to append those with NaN
    j, missing_x, missing_y = min_value, [], []
    # control = sorted(set(quant))
    while j <= max_value:
        if not len(quant_ds[(quant_ds.iloc[:, 0] >= j) & (quant_ds.iloc[:, 0] < j + interval_width)]):
            missing_x.append(j)
            missing_y.append(np.nan)
        j = interval_width + j
    z = pd.DataFrame({col_names[0]: missing_x, col_names[1]: missing_y})
    quant_ds = quant_ds.append(z, ignore_index=True)
    # print(len(set(quant_ds[col_names[0]])))
    # print(sorted(quant_ds[col_names[0]]))

    # insert well known value at zero
    if col_names[0] == 'base':
        z = dict((a, b) for a, b in zip(col_names, [0, 1]))
        quant_ds = quant_ds.append(z, ignore_index=True)
    elif col_names[0] == 'Bmax':
        z = dict((a, b) for a, b in zip(col_names, [0, 0]))
        quant_ds = quant_ds.append(z, ignore_index=True)

    return quant_ds
