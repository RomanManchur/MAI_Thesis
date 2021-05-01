'''Help library to calculate distance from target sequence to pseudo center sequence'''

import pandas as pd
import numpy as np
from eval_distance import DTWDistance

def make_dtw_distance_0(dataset, feature, window=5, CP=False):
    '''Returns DTW distance between dataset members and pseudo center (zeros sequnce)

    :param: dataset <dictionary>
        keys: celestial object names
        values: Celestial object
    :param feature <tuple>
        [0] - defines wavelength scale
        [1] - defines feature to work on, either V2 or CP
    :param window <int>
        defines window size for DTW calculation
    :return dtw_distance_table <panda.DataFrame>
    '''

    dim = len(dataset)
    dtw_distance_table = np.ones((dim, 1)) * np.inf
    d1, d2 = feature
    for i, ni in enumerate(dataset):
        ts = dataset[ni].post_processing_data[d1][d2]
        if CP:
            dtw_distance_table[i] = DTWDistance(ts, 0.5 * np.ones(ts.shape[0]), w=window)
        else:
            dtw_distance_table[i] = DTWDistance(ts, np.zeros(ts.shape[0]), w=window)
    return pd.DataFrame(dtw_distance_table, index=list(dataset.keys()), columns=["Ref"])