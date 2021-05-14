'''Help library to calculate distance from target sequence to pseudo center sequence'''

import pandas as pd
import numpy as np
from eval_distance import DTWDistance

def euc_pair_distance(object_a, object_b):
    ''' Returns equcledian distance between objects
        :param: object_a and object_b are objects of class celsestial
        returns: <float>
            sqrt( (x_a - x_b)**2 + (y_a - y_b)**2)
    '''
    x1, y1 = object_a.coordinates
    x2, y2 = object_b.coordinates

    return ((x1 - x2)**2 + (y1 - y2)**2)**0.5



def get_coordinates(dataset,window=20):
    ''' Calculates object coordinates in 2D space after bringing sequences to pseudo center '''
    res = []
    indexes = []
    for k, v in dataset.items():
        x_low = DTWDistance(v.post_processing_data['V2_low']['V2'], np.zeros(v.post_processing_data['V2_low']['V2'].shape[0]), window=window)
        x_medium = DTWDistance(v.post_processing_data['V2_medium']['V2'], np.zeros(v.post_processing_data['V2_medium']['V2'].shape[0]), window=window)
        x_high = DTWDistance(v.post_processing_data['V2_high']['V2'], np.zeros(v.post_processing_data['V2_high']['V2'].shape[0]), window=window)

        y_low = DTWDistance(v.post_processing_data['CP_low']['CP'], 0.5 * np.ones(v.post_processing_data['CP_low']['CP'].shape[0]), window=window)
        y_medium = DTWDistance(v.post_processing_data['CP_medium']['CP'], 0.5 * np.ones(v.post_processing_data['CP_medium']['CP'].shape[0]), window=window)
        y_high = DTWDistance(v.post_processing_data['CP_high']['CP'], 0.5 * np.ones(v.post_processing_data['CP_high']['CP'].shape[0]), window=window)

        x = (x_low**2 + x_medium**2 + x_high**2)**0.5
        y = (y_low**2 + y_medium**2 + y_high**2)**0.5

        # dataset[k].coordinates = (x,y)
        res.append({"x":x, "y":y})
        indexes.append(k)
    return pd.DataFrame(res, index=indexes)


# def make_cluster_distances(dataset):
#     '''
#         Returns  distance matrix between pair of celestial objects
#          - diagonal values are zero
#          - lower and upper values are diagonal symetric
#
#         :param dataset: <dictionary>
#             keys: names of celsestial objects
#             values: object class Celestial
#
#         :return <pandas.DataFrame>
#          - indexes : names of object
#          - columns : names of object
#          - values : distances between objects
#     '''
#
#     #Variable to track keys elements of dictionary to avoid symetric computation:
#     linkage_list = list(dataset.keys())
#     n = len(linkage_list)
#
#     #Initialize distance matrix
#     distance_matrix = np.full((n, n), np.inf)
#     # np.fill_diagonal(distance_matrix, 0)
#
#     #Compute distances between pairs and update matrix
#     for i1 in range(n):
#         for i2 in range(0, i1):
#             object_a, object_b = linkage_list[i1], linkage_list[i2]
#             dist = euc_pair_distance(dataset[object_a], dataset[object_b])
#             distance_matrix[i1][i2] = dist
#             distance_matrix[i2][i1] = dist
#
#     return pd.DataFrame(distance_matrix,index=list(dataset.keys()), columns=list(dataset.keys()))


def make_cluster_distances(dataset):
    '''
        Returns  distance matrix between pair of celestial objects
         - diagonal values are zero
         - lower and upper values are diagonal symetric

        :param dataset: pandas.Dataframe
            columns: x = V2composite, y = CP_composite position
            indexes: object name

        :return <pandas.DataFrame>
         - indexes : names of object
         - columns : names of object
         - values : distances between objects
    '''

    #Variable to track keys elements of dictionary to avoid symetric computation:
    linkage_list = list(dataset.index)
    n = len(linkage_list)

    #Initialize distance matrix
    distance_matrix = np.full((n, n), np.inf)
    # np.fill_diagonal(distance_matrix, 0)

    #Compute distances between pairs and update matrix
    for i1 in range(n):
        for i2 in range(0, i1):
            object_a, object_b = linkage_list[i1], linkage_list[i2]
            x1 = dataset.loc[object_a,'x']
            x2 = dataset.loc[object_b, 'x']
            y1 = dataset.loc[object_a, 'y']
            y2 = dataset.loc[object_b, 'y']
            dist = ((x1 - x2)**2 + (y1 - y2)**2)**0.5
            distance_matrix[i1][i2] = dist
            distance_matrix[i2][i1] = dist

    return pd.DataFrame(distance_matrix,index=linkage_list, columns=linkage_list)



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