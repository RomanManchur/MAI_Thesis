import math

def DTWMatrix(ts1,ts2,window=10):
    '''Help function to compute DTW matrix
    :param ts1: <any iterable object> represents 1st series
    :param ts2: <any iterable object> represents 2nd series
    :param window: <int> max skew window between series to be still considered valid

    '''

    DTWdistances = {}
    window = max(window, abs(len(ts1) - len(ts2)))

    for i in range(-1, len(ts1)):
        for j in range(-1, len(ts2)):
            DTWdistances[(i, j)] = float('inf')
    DTWdistances[(-1, -1)] = 0

    for i in range(len(ts1)):
        for j in range(max(0, i - window), min(len(ts2), i + window)):
            dist = (ts1[i] - ts2[j]) ** 2
            DTWdistances[(i, j)] = dist + min(DTWdistances[(i - 1, j)], DTWdistances[(i, j - 1)],
                                              DTWdistances[(i - 1, j - 1)])

    return DTWdistances


def DTWDistance(ts1, ts2, window=10):
    '''
    Function that calculates dynamic time wrapping distance between two series
    :param ts1: <any iterable object> represents 1st series
    :param ts2: <any iterable object> represents 2nd series
    :param w: <int> max skew window between series to be still considered valid
    :return: <float> DTW distance between series
    '''
    DTWdistances = DTWMatrix(ts1,ts2,window)
    return math.sqrt(DTWdistances[len(ts1) - 1, len(ts2) - 1])



