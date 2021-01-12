import numpy as np

def euclidian_barycenter(TS):
    '''
    Standard Euclidian baricenter computed from Time series set: <m x n> --> rows are samples, columns are averaging attribute
    :param TS: array-like, shape(samples x series)
    :return: numpy.array of shape (series_size x 1)
        Barycenter of provided time series
        =========
        Example: (euclidian_barycenter([[1,2,3],[2,3,4]]) --> [1.5 2.5 3.5]
    '''
    return np.average(TS,axis=0)