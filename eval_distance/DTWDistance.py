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



def DTWcompound(query, reference, window):
    ''' Function to compute complex distance between objects on basis of DTW distances per wavelengths of V2 and CP

    :param query:
        Celestial object from test set
    :param reference:
        Celestial object from reference set
    :param window: size of wrapping window
    :return: <float>
        composed distance as Eucledian distance between V2 (component per wavelengths) and CP (component per wavelength)

    '''

    #Calculate DTW distance between query and reference objects for each wavelenght on basis of V2 attribute
    V2_dtw_low = DTWDistance(query.post_processing_data['V2_low']['V2'],reference.post_processing_data['V2_low']['V2'], window)
    V2_dtw_medium = DTWDistance(query.post_processing_data['V2_medium']['V2'], reference.post_processing_data['V2_medium']['V2'], window)
    V2_dtw_high= DTWDistance(query.post_processing_data['V2_high']['V2'], reference.post_processing_data['V2_high']['V2'], window)

    # Calculate DTW distance between query and reference objects for each wavelenght on basis of CP attribute
    CP_dtw_low = DTWDistance(query.post_processing_data['CP_low']['CP'], reference.post_processing_data['CP_low']['CP'], window)
    CP_dtw_medium = DTWDistance(query.post_processing_data['CP_medium']['CP'], reference.post_processing_data['CP_medium']['CP'], window)
    CP_dtw_high = DTWDistance(query.post_processing_data['CP_high']['CP'], reference.post_processing_data['CP_high']['CP'], window)

    #return compose distance: sum of each distance for filtered attribute and get Eucledian distance based on sums
    return ((V2_dtw_low + V2_dtw_medium + V2_dtw_high)**2 + (CP_dtw_low + CP_dtw_medium + CP_dtw_high)**2)**0.5

