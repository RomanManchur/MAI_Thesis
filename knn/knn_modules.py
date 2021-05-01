'''This library contains all fuctions related to knn calculation logic'''

import pandas as pd
import numpy as np
import random
from sklearn.metrics import classification_report
from eval_distance import DTWDistance

def generate_data_sets(file_names, split_ratio=0.8):
    '''Perfroms random split on input dataset to produce training and test set
    :param file_names: <dictionary>
        keys: object types
        values: object names
    :param split_ratio: <float>
        represents how dataset is splitted in train / test data set ratios

    :return: <tuple>
        elements are list of strings with names of celestial objects
    '''

    if len(file_names) > 0:  # file_names is dictionary in form {object_type:<name, name, name>, ...}
        train_ref, test_ref = [], []
        for k, v_ in file_names.items():
            v = list(v_)
            random.shuffle(v)
            # perfrom 80/20 split
            index = int(len(v) * (split_ratio * 100) // 100)
            train_ref.extend(v[:index])
            test_ref.extend(v[index:])
    return (train_ref, test_ref)


def get_compound_dtw(query, reference, window):
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



def get_nn(query, train_ref, data_set_as_dict, w, n_count=1, composed_distance = False, scale = 'V2_all',attribute='V2'):
    """
    Returns dict-like object of closest neighbors and their distance
    :param query <array-like>
        any query sequence that need to be classified
    :param train_ref <array-like>
        key: object_name for which labels are known, values: sequence representing the object
    :param w <int>
        window size that is used in DTW metric calculation
    :param n_count <int>
        defines how many neighbors to hold in response
    :param composed_distance <bool>
        controls if distance is calculated as DTW distance on full sequence or as composed distance per wavelenghts

    :return: pandas dataframe with closest object(s) from train set to the test query and respective distance(s)
    """
    min_dist = np.ones(n_count) * np.inf
    closest_n = [""] * n_count
    result = pd.DataFrame({"Closest Neighbor": closest_n, "Distance": min_dist})
    #compute distance between current object and all other objects in labeled dataset
    for train_name in train_ref:
        if not composed_distance:
            current_distance = DTWDistance(query.post_processing_data[scale][attribute], data_set_as_dict[train_name].post_processing_data[scale][attribute], window=w)
        else:
            current_distance = get_compound_dtw(query, data_set_as_dict[train_name], window=w)

        if (current_distance < result['Distance']).any():
            result.loc[0] = [train_name, current_distance] #keep the highest distance at the top
            result.sort_values(by="Distance", ascending=False, inplace=True, ignore_index=True)
            # finally sort in ascending order as we will need to check for object with min distance while evaluating model
    result.sort_values(by="Distance", ascending=True, inplace=True, ignore_index=True)
    return result



def knn_classification(train_ref, test_ref , data_set_as_dict, split_ratio=0.8, window=5, n_neighbors=3, composed_distance = False, scale = 'V2_all',attribute='V2'):
    """
    Function that implements classification of test_ds samples based on nearest neighbors in train_ds.
    Uses KNN nearest neighbor method and DTW metric
    :param file_names: <dict> in form:
        keys: celestial object type
        values: set of celestial object names
    :param data_set_as_dict: <dict> in form:
        keys: celestial object name
        values: object of class Celestial
    :param split_ratio: <float>
        represents how dataset is splitted in train / test data set ratios
    :param measurements_type: <str>
        defines what measurement type is used to build model, can either take "V2" or "CP"
    :param window: int
        defines window size used in DTW metric calculation
    :param n_neighbors: int
        defines number of neighbors that is kept during evaluation

    :return: None
    """

    print("=====================================================\n")
    print("=================<KNN classification> ===============\n")
    print("=====================================================\n")

    print("Trainset: {0}".format(train_ref))
    print("Testset: {0}".format(test_ref))

    # test model accuracy
    preds, ground_truth = [], []
    for query_object_name in test_ref:
        query_object_data = data_set_as_dict[query_object_name] #get all data associtated with current query object

        # call function to get NN
        n_closest = get_nn(query_object_data, train_ref, data_set_as_dict,  w=window, n_count=n_neighbors, composed_distance = composed_distance, scale=scale, attribute = attribute )
        print('Query object name', query_object_name)
        print('Window size', window)
        print(n_closest)
        first_closest = n_closest.loc[0]['Closest Neighbor']  # get name of the 1st closest neighbor
        predicted_type = data_set_as_dict[first_closest].object_type
        true_type = data_set_as_dict[query_object_name].object_type
        if predicted_type and true_type and predicted_type != true_type:
            print("===================================================")
            print("Classification mismatch for {0}: predicted {1}, actual {2}".format(query_object_name,
                                                                                      predicted_type, true_type))
            print("===================================================")
            print("Other possible neighbors and distances\n", n_closest)

        preds.append(predicted_type)
        ground_truth.append(true_type)

    print("=================<Model Accuracy (query on samples)> ===============\n")
    return classification_report(ground_truth, preds, zero_division=0, output_dict=True)




def get_closest_neighbors(ds1, window, nn, ds2=None):
    ''' Returns closest neighbors between elements of input dataset(s)

    :param ds1: <dict>
        keys: object names
        values: objects of class Celestial
    :param window: <int>
        contols wrapping window size
    :param nn:  <int>
        controls number of nearest neighbors to be found
    :param ds2: <dict> [optional]
        keys: object names
        values: objects of class Celestial
        if passed then K-nn is calculated between all members in ds2 against all members in ds1;
        otherwise K-nn is calculated between all pairs in ds1

    :return: <dict> nearest_neighbors
        keys: celestial object name
        values: <dict>
            keys: closest neighbor(s) name
            values: distance
    '''
    nearest_neighbors = {}

    if not ds2:
        test_ds = ds1
    else:
        test_ds = ds2

    for names_i, values_i in test_ds.items():
        #At each iteration of outer loop create temporary dictionary structure of fixed length
        #that will hold information about closest neighbors and respective distance to them.
        tmp_dict = {x:(None,np.inf) for x in range(nn)}
        for names_j, values_j in ds1.items():
            if names_i != names_j:
                #compute compound distance between object pairs
                current_distance = get_compound_dtw(values_i, values_j, window)
                #check within temporary dictionary structure if current distance is less than stored maximum
                #if yes -> remove maximum element from dictionary and update with current distance
                #otherwise -> continue
                max_distance = 0
                #getting max distance from temp dictionary and associated key
                for k,v in tmp_dict.items():
                    if v[1] > max_distance:
                        max_distance = v[1]
                        max_element = k
                #update after checking all elements in case current distance is lower than maximum stored in tmd_dict
                if current_distance < max_distance:
                    del tmp_dict[max_element]
                    tmp_dict[names_j] = (values_j.object_type,current_distance)
                c=0
        nearest_neighbors[names_i] = tmp_dict

    return nearest_neighbors
