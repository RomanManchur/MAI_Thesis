'''Help library to produce 2D matrix with distance information between each object pairs'''

import  pandas as pd
import numpy as np
from eval_distance import DTWDistance

def make_dtw_distance_table(train_dataset, feature, window=5, test_dataset=None):
    '''Returns DTW distance between each member in dataset stored in pandas table

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

    #This definies condition when computation is done on whole unlabled dataset
    if not test_dataset:
        dim = len(train_dataset)
        dtw_distance_table = np.ones((dim, dim)) * np.inf
        d1, d2 = feature
        for i, ni in enumerate(train_dataset):
            for j, nj in enumerate(train_dataset):
                if ni == nj:  # no distance between two same elements
                    continue
                else:
                    dtw_distance_table[i][j] = DTWDistance(train_dataset[ni].post_processing_data[d1][d2],
                                                           train_dataset[nj].post_processing_data[d1][d2], w=window)

    #This definies condition when matching is done for unknown test sequences with respect to reference labeled dataset
    else:
        dim1, dim2 = len(test_dataset), len(train_dataset)
        dtw_distance_table = np.ones((dim1, dim2)) * np.inf
        d1, d2 = feature
        for i, ni in enumerate(test_dataset):
            for j, nj in enumerate(train_dataset):
                dtw_distance_table[i][j] = DTWDistance(test_dataset[ni].post_processing_data[d1][d2],
                                                       train_dataset[nj].post_processing_data[d1][d2], w=window)
    return pd.DataFrame(dtw_distance_table, index=list(train_dataset.keys()), columns=list(train_dataset.keys()))



