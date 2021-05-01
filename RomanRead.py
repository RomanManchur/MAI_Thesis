import os
import random

import pandas as pd
from sklearn.metrics import classification_report
import ReadOIFITS as oifits
from data_visulization import *
from eval_distance import *
from file_preprocessing import *
from tslearn.clustering import TimeSeriesKMeans
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import re, sys

# Specifying the location of the data file
dir = "/Users/rmanchur/Documents/MAI_Thesis/data/"
# fitsdir = dir + "all_data/all_sets/"
# fitsdir = dir + "all_data/2stars/"
# fitsdir = dir + "all_data/StellarSurface/"
fitsdir = dir + "all_data/Single/"
# fitsdir = dir + "all_data/renamed/"
target_list_path = "data/points_to_check/target_list_small.txt"
# target_list_path = "data/points_to_check/broken.txt"
pdfdir = dir + "pdf/"
csvdir = dir + "csv/"
data_set_as_dict = {}


def print_section(infostring):
    print("====================================")
    print(infostring)
    print("====================================")
    return None


def get_object_name(object_name_):
    """Takes celestial object name as input and removes measurement index if one is present; returns object name
    :param object_name <str>
        supported types are:
            objectName_measurementsNumber'  or 'objectName'
    """
    if re.search(r'([\w_]+)_\d+$', object_name_):
        object_name = re.search(r'([\w_\+]+)_\d+$', object_name_).group(1)
    else:
        object_name = object_name_
    return object_name


def get_type_name(target_object, fname):
    """"Returns tupple of (file_type, file_name) if file type encoded in filename, otherwise returns filename
    :param target_object <str>
        reference to true object name (obtained from target list file)
    :param fname <str>
        file name of FITS file processed;
        Standardized values are:
            'objectType_objectName_measurementsNumber' or
            'objectName_measurementsNumber'  or
            'objectName'
    """
    # This defines condtion when type is encoded in filename
    if not fname.startswith(target_object):
        type_end_index = fname.index("_")
        object_type = fname[:type_end_index]
        return (target_object, object_type)
    else:
        return (target_object,)


# Those parameters control if knn will be run on  files directly
def default_value():
    return set()


knn_on_files = False
file_names = defaultdict(default_value)


def get_nn(query, train_ds, w, n_count=1):
    """
    Returns dict-like object of closest neighbors and their distance
    :param query <array-like>
        any query sequence that need to be classified
    :param train_ds <array-like>
        key: object_name for which labels are known, values: sequence representing the object
    :param w <int>
        window size that is used in DTW metric calculation
    :param n_count <int>
        defines how many neighbors to hold in response
    """
    min_dist = np.ones(n_count) * np.inf
    closest_n = [""] * n_count
    result = pd.DataFrame({"Closest Neighbor": closest_n, "Distance": min_dist})
    for train_name, train_seq in train_ds.items():
        current_distance = DTWDistance(query, train_seq, w)
        if (current_distance < result['Distance']).any():
            result.loc[0] = [train_name, current_distance]
            result.sort_values(by="Distance", ascending=False, inplace=True, ignore_index=True)
            # finally sort in ascending order as we will need to check for object with min distance while evaluating model
    result.sort_values(by="Distance", ascending=True, inplace=True, ignore_index=True)
    return result


def make_dtw_distance_table(dataset, feature, window=5):
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

    dim = len(dataset)
    dtw_distance_table = np.ones((dim, dim)) * np.inf
    d1, d2 = feature
    for i, ni in enumerate(dataset):
        for j, nj in enumerate(dataset):
            if ni == nj:  # no distance between two same elements
                continue
            else:
                dtw_distance_table[i][j] = DTWDistance(dataset[ni].post_processing_data[d1][d2],
                                                       dataset[nj].post_processing_data[d1][d2], w=window)
    return pd.DataFrame(dtw_distance_table, index=list(dataset.keys()), columns=list(dataset.keys()))


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


class Celestial:
    """Generic conteiner to store object information"""

    def __init__(self, name, data, object_type=None, post_processing_data={}):
        """
        Create instance of class
        :param name: <str>
            name of celestial object obtained from filename
        :param data <nested dict>
            keys: represent value type (V2, CP, uv, etc) and values in inner dictionary with
            keys: equal to measurements type and values are actual measurements for the parameter

        :param object_type: <str>
            type of celestial object obtained from filename; default - None
        :param post_processing_data <nested dict>
            keys: represent value type (V2, CP, uv, etc) and values in inner dictionary with
            keys: equal to measurements type and values are actual measurements for the parameter after data processing
        """
        self.name = name
        self.data = data
        self.object_type = object_type
        self.post_processing_data = post_processing_data

    def update_data(self, new_data):
        """"Combines measurements taken for the same object in different days to single dataset
        :param: new_data <nested dict>
            data that needs to be appended to data already associated with object
            keys: represent value type (V2, CP, uv, etc) and values in inner dictionary with
            keys: equal to measurements type and values are actual measurements for the parameter
        :returns None
        """
        for outer_key, inner_dict in new_data.items():
            for inner_key, inner_value in inner_dict.items():
                self.data[outer_key][inner_key] = np.append(self.data[outer_key][inner_key],
                                                            new_data[outer_key][inner_key])
        return None


def data_processing(name, datatype, measurements, visulalize=False, threshold=1, wavelenght="all"):
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


def DBA_model(samples, samples_names, num_clusters, dst_folder, data_type, plot=False):
    """
    Builds DBA clustering model based on input data, prints data points associated with cluster
    and plots data and avg sequence

    :return: <numpy.array> center sequences
    """
    seed = 1234
    dba_model = TimeSeriesKMeans(n_clusters=num_clusters, n_init=2, metric="dtw", max_iter_barycenter=10, verbose=True,
                                 random_state=seed)
    dba_predict = dba_model.fit_predict(samples)
    for cluster_id in range(num_clusters):
        # get indexes of rows associated with current cluster in dataset
        object_incluster_indexes = np.argwhere(dba_predict == cluster_id).ravel()
        print("============Cluster#{0}=============".format(cluster_id + 1))
        for each_object in object_incluster_indexes:
            print("--> associated object: {0}".format(
                samples_names[each_object]))  # print name of celestial object associated with cluster
        # plot clusters and data
        if plot:
            plt.subplot(num_clusters, 1, 1 + cluster_id)
            for xx in samples[dba_predict == cluster_id]:
                plt.plot(xx.ravel(), "k-", alpha=.2)
            plt.plot(dba_model.cluster_centers_[cluster_id].ravel(), "r-")
            plt.text(0.55, 0.85, "Cluster %d" % (cluster_id + 1),
                     transform=plt.gca().transAxes)
            plt.ylim(0.95 * samples.min(), 1.05 * samples.max())
            if cluster_id == 0:
                plt.title("DBA $k$-means")
            # plt.tight_layout()
            plt.subplots_adjust(hspace=0.5)
            f = plt.gcf()
            f.set_size_inches(8, 10)
            if cluster_id == num_clusters:
                plt.savefig(dst_folder + data_type + "_clustering.pdf", dpi=100)
                plt.close()

    return dba_model.cluster_centers_


def make_clusters(data_set_as_dict, wavelength_scale="V2_all", data_type="V2", num_clusters=12, plot=False):
    '''
    Function that builds clusters using sequence avaraging

    :param data_set_as_dict <dict>
        keys: celestial object names; values: object of Celestial type
    :param wavelength_scale <str>
        can either be - V2_low, V2_medium, V2_high or V2_all and similar values for CP parameter
    :param: data_type <str>
        can either be V2 or CP
    :param num_clusters <int>
        defines number of clusters build
    '''
    celestial_object_data = list(data_set_as_dict.values())
    d1, d2 = len(data_set_as_dict), len(celestial_object_data[0].post_processing_data[wavelength_scale][data_type])
    samples = np.zeros((0, d2))  # building data matrix used in clustering decision
    samples_names = []  # building associative list with celestial object names (that is to address fact that list are non-ordered)
    for idx, cel_object in enumerate(celestial_object_data):
        # there might be empty datasets after filtering skip processing for those and continue with next object
        # print(cel_object.name,cel_object.post_processing_data[wavelength_scale][data_type], cel_object.post_processing_data[wavelength_scale][data_type].size)
        if cel_object.post_processing_data[wavelength_scale][data_type].size == 0 or \
                cel_object.post_processing_data[wavelength_scale][data_type].size != d2:
            data_set_as_dict.pop(cel_object.name, None)  # remove objects with no measurements or non-aligned dimensions
            continue
        samples = np.vstack((samples, cel_object.post_processing_data[wavelength_scale][data_type]))
        # samples[idx] = cel_object.post_processing_data[data_type][data_type]#this can't be used as we might have missing records and indexing will be affected with removal opertation
        samples_names.append(cel_object.name)
    cluster_centers = DBA_model(samples, samples_names, num_clusters=num_clusters, dst_folder=pdfdir,
                                data_type=wavelength_scale, plot=plot)

    # calculate distance to cluster center from each object, e.i: certanity rate
    # re-defining dimensions: d1 may be lower than original due to filtered fields, d2 - correspond to avg.sequence size
    d1, d2 = len(samples_names), cluster_centers.shape[0]
    distance_to_centers_ = np.array(np.ones((d1, d2)) * np.inf)
    distance_to_centers = pd.DataFrame(distance_to_centers_, index=samples_names,
                                       columns=[x for x in range(num_clusters)])
    for i, cel_object in enumerate(data_set_as_dict.values()):
        for j, cluster in enumerate(cluster_centers):
            distance_to_centers.iloc[i, j] = DTWDistance(cel_object.post_processing_data[wavelength_scale][data_type],
                                                         cluster)
    distance_to_centers.to_csv(csvdir + wavelength_scale + data_type + '_distances.csv')

    return cluster_centers


def knn_classification(file_names, data_set_as_dict, split_ratio=0.8, wavelength_scale="V2_all",
                       meassurements_type="V2", window=5, n_neighbors=3):
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
    :param meassurements_type: <str>
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
    # control check if file_names dictionary contains data, that will be in case knn on files is set to True
    if len(file_names) > 0:  # file_names is dictionary in form {object_type:<name, name, name>, ...}
        train_ref, test_ref = [], []
        for k, v_ in file_names.items():
            v = list(v_)
            random.shuffle(v)
            # perfrom 80/20 split
            index = int(len(v) * (split_ratio * 100) // 100)
            train_ref.extend(v[:index])
            test_ref.extend(v[index:])

        # make train and test datasets based on V2 measurements
        train_ds, test_ds = {}, {}
        for celestial_object in data_set_as_dict.values():
            if celestial_object.name in train_ref:
                train_ds[celestial_object.name] = celestial_object.post_processing_data[wavelength_scale][
                    meassurements_type]
            else:
                test_ds[celestial_object.name] = celestial_object.post_processing_data[wavelength_scale][
                    meassurements_type]

        print("Trainset: {0}".format(train_ref))
        print("Testset: {0}".format(test_ref))

        # test model accuracy
        preds, ground_truth = [], []
        for query_object_name, query_object_data in test_ds.items():
            n_closest = get_nn(query_object_data, train_ds, w=window, n_count=n_neighbors)
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
        print(classification_report(ground_truth, preds, zero_division=0))


# cleaning up
for pdf in os.listdir(fitsdir):
    if pdf.endswith("pdf"):
        os.remove(fitsdir + "/" + pdf)

#########################################################
#                   Get object list                     #
#########################################################
print_section("Reading object list infomration.....")
with open(target_list_path, 'r') as target_list_fd:
    target_list = target_list_fd.read().splitlines()
# target_list.sort()
random.shuffle(target_list)  # as data set is big and can't be loaded in memory as whole, I will be working on random batches

#########################################################
#   Reading and storing input data object list          #
#########################################################
files_in_data_directory = os.listdir(fitsdir)
batch_id, batch_size = 0, len(target_list)

for object_count, target_object in enumerate(target_list):

    if batch_id == 3: break # this one for debug only

    # Iterate over files in data directory and process files in given batch
    for each_file in files_in_data_directory:
        if target_object in each_file:  # matching if current file is one of the targets to be processed
            # print(target_object, each_file)
            # Reading the oifits file
            data = oifits.read(fitsdir, each_file)

            # A plot to quickly visualise the dataset. CPext sets the vertical limits for the closure phases (CPs) plot:
            # By construction CPs are between -180 and 180 degrees but sometimes the signal is small.
            data.plotV2CP(CPext=15, lines=False,save=True, name=each_file+".pdf")

            # Creating a dictionary with meaningful data (easier to manipulate)
            datadict = data.givedataJK()

            # Extracting the meaningful information from the dictionary
            # extract the squared Visibilities and errors
            V2, V2e = datadict["v2"]

            # extract closure phases and errors
            CP, CPe = datadict["cp"]
            # extract baselines (let"s start with those 1D coordinates instead of u and v coordinates)
            base, Bmax = oifits.Bases(datadict)
            # extract u and v coordinates
            u, u1, u2, u3 = datadict["u"]
            v, v1, v2, v3 = datadict["v"]
            # extract wavelengths for squared visibilities and closure phases
            waveV2, waveCP = datadict["wave"]

            #################################################
            ####### Now you can play with the data!##########
            #################################################
            knn_on_files = True
            print("Processing file {0}:...........".format(each_file))
            object_type_name = get_type_name(target_object, each_file)
            # print("Extracted name [and type optional]:.......... {0}".format(object_type_name))
            # track association between known object types and their names; used later in prediction and accuracy check
            if knn_on_files and len(object_type_name) == 2:
                object_name, object_type = object_type_name
                file_names[object_type].add(object_name)
            else:
                # in case no labels are known or KNN is not applied we still need to get object names
                # to combine data from multiple measurements on same object and clustering
                object_name = object_type_name[0]
                object_type = None

            # Read data and store to dictionary for easy access
            # get V2 and CP measurements
            V2_data = {"base": base, "V2": V2, "V2err": V2e, "waveV2": waveV2}
            CP_data = {"Bmax": Bmax, "CP": CP, "CPerr": CPe, "waveCP": waveCP}

            # if there was some data already processed for current object - update data in dictionary
            if object_name in data_set_as_dict.keys():
                data_set_as_dict[object_name].update_data({"V2": V2_data, "CP": CP_data})
                c = 0
            # if it's first time data for the celestial object is processed, initialize object and add it to dictionary
            else:
                current_object = Celestial(object_name, {"V2": V2_data, "CP": CP_data}, object_type)
                data_set_as_dict[object_name] = current_object
                c = 0

    #Track batch state and reset dataset information if full batch is processed
    if (object_count+1) % batch_size == 0:
        batch_id += 1

        #At this point all data is read from FITS files and stored to 'data_set_as_dict' within currect batch,
        #each key represent object name and associated value is object of class 'Celestial'

        #Run pre-processing on data
        for object_name, celestial_object in data_set_as_dict.items():
            #all wavelengths
            V2_data_processed = data_processing(object_name, "V2", celestial_object.data["V2"], visulalize=True, threshold=1,
                                                wavelenght="all")
            tmp_pd = pd.DataFrame(data=celestial_object.data["V2"], columns=list(celestial_object.data["V2"].keys()))
            print(object_name)
            print(tmp_pd)


            CP_data_processed = data_processing(object_name, "CP", celestial_object.data["CP"], visulalize=True, threshold=180,
                                                wavelenght="all")

            tmp_pd = pd.DataFrame(data=celestial_object.data["CP"], columns=list(celestial_object.data["CP"].keys()))
            print(object_name)
            print(tmp_pd)

            #low wavelengths
            V2_data_processed_low = data_processing(object_name, "V2", celestial_object.data["V2"], visulalize=False, threshold=1,
                                                wavelenght="low")
            CP_data_processed_low = data_processing(object_name, "CP", celestial_object.data["CP"], visulalize=False, threshold=180,
                                                wavelenght="low")
            #medium wavelengths
            V2_data_processed_medium = data_processing(object_name, "V2", celestial_object.data["V2"], visulalize=False, threshold=1,
                                                wavelenght="medium")
            CP_data_processed_medium = data_processing(object_name, "CP", celestial_object.data["CP"], visulalize=False, threshold=180,
                                                wavelenght="medium")
            #high wavelengths
            V2_data_processed_high = data_processing(object_name, "V2", celestial_object.data["V2"], visulalize=False, threshold=1,
                                                wavelenght="high")
            CP_data_processed_high = data_processing(object_name, "CP", celestial_object.data["CP"], visulalize=False, threshold=180,
                                                wavelenght="high")

            celestial_object.post_processing_data = {"V2_all":V2_data_processed, "CP_all": CP_data_processed,
                                                     "V2_low":V2_data_processed_low, "CP_low": CP_data_processed_low,
                                                     "V2_medium":V2_data_processed_medium, "CP_medium": CP_data_processed_medium,
                                                     "V2_high":V2_data_processed_high, "CP_high": CP_data_processed_high}


        # ###############
        # ###V2 model####
        # ###############
        V2_cluster_centers = make_clusters(data_set_as_dict, wavelength_scale="V2_all", data_type="V2", num_clusters=7,plot=True)
        V2_cluster_centers_low = make_clusters(data_set_as_dict, wavelength_scale="V2_low", data_type="V2", num_clusters=7)
        V2_cluster_centers_medium = make_clusters(data_set_as_dict, wavelength_scale="V2_medium", data_type="V2", num_clusters=7)
        V2_cluster_centers_high = make_clusters(data_set_as_dict, wavelength_scale="V2_high", data_type="V2", num_clusters=7)



        plot_avarage_sequence(V2_cluster_centers,V2_cluster_centers_low, V2_cluster_centers_medium, V2_cluster_centers_high,
                              legend=["all", "low", "medium", "high"],
                              plot_type="V2",
                              path=pdfdir + "V2_averages.pdf",
                              num_clusters=7)

        #Reset
        V2_cluster_centers, V2_cluster_centers_low, V2_cluster_centers_medium, V2_cluster_centers_high = None, None, None, None

        # ###############
        # ###CP model####
        # ###############
        CP_cluster_centers = make_clusters(data_set_as_dict, wavelength_scale="CP_all", data_type="CP", num_clusters=7,plot=True)
        CP_cluster_centers_low = make_clusters(data_set_as_dict, wavelength_scale="CP_low", data_type="CP", num_clusters=7)
        CP_cluster_centers_medium = make_clusters(data_set_as_dict, wavelength_scale="CP_medium", data_type="CP", num_clusters=7)
        CP_cluster_centers_high = make_clusters(data_set_as_dict, wavelength_scale="CP_high", data_type="CP", num_clusters=7)



        plot_avarage_sequence(CP_cluster_centers,CP_cluster_centers_low, CP_cluster_centers_medium, CP_cluster_centers_high,
                              legend=["all", "low", "medium", "high"],
                              plot_type="CP",
                              path=pdfdir + "CP_averages.pdf",
                              num_clusters=7)
        #Reset
        CP_cluster_centers,CP_cluster_centers_low, CP_cluster_centers_medium, CP_cluster_centers_high = None, None, None, None

        # ###################
        # ###KNN on files####
        # ###################
        #KNN on V2 values
        knn_classification(file_names,data_set_as_dict)

        #KNN on CP values
        knn_classification(file_names,data_set_as_dict,wavelength_scale="CP_all", meassurements_type="CP")

        #Calculate sum of squared disatances for each wavelenght
        V2_dtw_low = make_dtw_distance_table(data_set_as_dict, ("V2_low", "V2"))
        V2_dtw_medium = make_dtw_distance_table(data_set_as_dict, ("V2_medium", "V2"))
        V2_dtw_high = make_dtw_distance_table(data_set_as_dict, ("V2_high", "V2"))

        V2_dtw_total = V2_dtw_low + V2_dtw_medium.values + V2_dtw_high.values

        # print("{0}\n{1}\n{2}\n{3}\n".format(V2_dtw_low,V2_dtw_medium, V2_dtw_high, V2_dtw_total))
        closest_neighbor = V2_dtw_total.idxmin(axis=1)
        print(closest_neighbor)


        #Calculate sum of squared disatances for each wavelenght based on CP
        print("===============CP====================")

        CP_dtw_low = make_dtw_distance_table(data_set_as_dict, ("CP_low", "CP"))
        CP_dtw_medium = make_dtw_distance_table(data_set_as_dict, ("CP_medium", "CP"))
        CP_dtw_high = make_dtw_distance_table(data_set_as_dict, ("CP_high", "CP"))

        CP_dtw_total = CP_dtw_low + CP_dtw_medium.values + CP_dtw_high.values

        # print("{0}\n{1}\n{2}\n{3}\n".format(CP_dtw_low,CP_dtw_medium, CP_dtw_high, CP_dtw_total))
        closest_neighbor = CP_dtw_total.idxmin(axis=1)
        print(closest_neighbor)

        #check if any pattern is observed if distance is calculated from 0,0
        V2_dtw0_low = make_dtw_distance_0(data_set_as_dict, ("V2_low", "V2"))
        V2_dtw0_medium = make_dtw_distance_0(data_set_as_dict, ("V2_medium", "V2"))
        V2_dtw0_high = make_dtw_distance_0(data_set_as_dict, ("V2_high", "V2"))
        V2_dtw0_total = V2_dtw0_low + V2_dtw0_medium + V2_dtw0_high

        CP_dtw0_low = make_dtw_distance_0(data_set_as_dict, ("CP_low", "CP"),CP=True)
        CP_dtw0_medium = make_dtw_distance_0(data_set_as_dict, ("CP_medium", "CP"),CP=True)
        CP_dtw0_high = make_dtw_distance_0(data_set_as_dict, ("CP_high", "CP"),CP=True)
        CP_dtw0_total = CP_dtw0_low + CP_dtw0_medium + CP_dtw0_high

        plt.subplot(1,1,1)
        plt.scatter(V2_dtw0_total,CP_dtw0_total)
        legend = V2_dtw0_total.index.tolist()
        for i,txt in enumerate(legend):
            plt.annotate(txt,(V2_dtw0_total.iloc[i,0],CP_dtw0_total.iloc[i,0]))
        plt.savefig("./" + str(batch_id) + "_distance.pdf", dpi=100)

        #Reset dataset information and clean memory
        data_set_as_dict = {}


# # moving files
for pdf in os.listdir(fitsdir):
    if pdf.endswith("pdf"):
        os.rename(fitsdir + "/" + pdf, pdfdir + pdf)
