import os
import random

import pandas as pd
from sklearn.metrics import classification_report
import ReadOIFITS as oifits
from data_visulization import standardPlot
from eval_distance import *
from file_preprocessing import *
from Barycenters_avarage import euclidian_barycenter
from tslearn.clustering import TimeSeriesKMeans
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict


# Specifying the location of the data file
dir = "/Users/rmanchur/Documents/MAI_Thesis/data/"
fitsdir = dir + "all_data/all_sets/"
# fitsdir = dir + "all_data/2stars/"
# fitsdir = dir + "all_data/StellarSurface/"
# fitsdir = dir + "all_data/Single/"
pdfdir = dir + "pdf/"
csvdir = dir + "csv/"
data_set = []

#Those parameters control if knn will be run on  files directly
def default_value():
    return []

def get_type_name(fname):
    separator_index = fname.index("_")
    file_type = fname[:separator_index]
    file_name = fname[separator_index+1:]
    return (file_type, fname)

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
    min_dist = np.ones(n_count)*np.inf
    closest_n = [""]*n_count
    result = pd.DataFrame({"Closest Neighbor": closest_n, "Distance":min_dist})
    for train_name, train_seq in train_ds.items():
        current_distance = DTWDistance(query,train_seq,w)
        if (current_distance < result['Distance']).any():
            result.loc[0] = [train_name, current_distance]
            result.sort_values(by="Distance", ascending=False,inplace=True, ignore_index=True)
            # finally sort in ascending order as we will need to check for object with min distance while evaluating model
    result.sort_values(by="Distance", ascending=True, inplace=True, ignore_index=True)
    return result


class Celestial:
    """Generic conteiner to store object information"""
    def __init__(self, name, data,post_processing_data):
        """
        Create instance of class
        :param type: <str>
            description of object type
        :param data <dict>
            keys reprsent value type (V2, CP, uv, etc) and values are actual measurements in numpy.array
        :param post_processing_data <dict>
            keys reprsent value type (V2, CP, uv, etc) and values are actual measurements in numpy.array
        """
        self.name = name
        self.data = data
        self.post_processing_data = post_processing_data


    def getType(self):
        return self.name.split("_")[0]



def data_processing(name, datatype, measurements, visulalize=False, threshold=1):
    """
    :param name <str>
        name of object currently processed
    :param datatype <str>
        data type of object currently processed (V2, CP, UV, etc)
    :param: measurements <dict>
        keys: name of parameter, values: measurements taken for parameter (e.i: "V2": <array>)
    :param: visulalize <boolean>
        defines if data processig is plotted or not (default = False / non-plotted)
    :return: <dict>
    data after quantization, normalization, compression and interpolation
    keys: data type; values - numpy.array
    """

    #re-arrange the data to PandasDataframe format from input dictionary
    measurements_ = pd.DataFrame.from_dict(measurements)
    columns = measurements_.columns.to_list()
    x = columns[0]

    print("Processing file {0}, datatype {1}:.....".format(name,datatype))
    quantized_ds = quantize_ds(measurements_, intervals=50)  # quantize dataset along x-axis
    normalized_ds = normalize_df(quantized_ds, 0,
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
                              plot_type=datatype+"_")
    #postprocessing data
    keys = z.columns.to_list()
    values = z.to_numpy()
    z_dict = {keys[index]:values[:,index] for index in range(len(keys))}
    return z_dict


def DBA_model(samples,samples_names, num_clusters,dst_folder, data_type):
    """
    Builds DBA clustering model based on input data, prints data points associated with cluster and plots data and avg sequence
    :return: <numpy.array> center sequences
    """
    seed = 1234
    dba_model = TimeSeriesKMeans(n_clusters=num_clusters,n_init=2, metric="dtw",max_iter_barycenter=10,verbose=True,random_state=seed)
    dba_predict = dba_model.fit_predict(samples)
    for cluster_id in range(num_clusters):
        plt.subplot(num_clusters, 1, 1 + cluster_id)
        #get indexes of rows associated with current cluster in dataset
        object_incluster_indexes = np.argwhere(dba_predict == cluster_id).ravel()
        print("============Cluster#{0}=============".format(cluster_id))
        for each_object in object_incluster_indexes:
            print("--> associated object: {0}".format(samples_names[each_object]))#print name of celestial object associated with cluster
        #plot clusters and data
        for xx in samples[dba_predict == cluster_id]:
            plt.plot(xx.ravel(), "k-", alpha=.2)
        plt.plot(dba_model.cluster_centers_[cluster_id].ravel(), "r-")
        plt.text(0.55, 0.85,"Cluster %d" % (cluster_id + 1),
                 transform=plt.gca().transAxes)
        if cluster_id == 1:
            plt.title("DBA $k$-means")
    # plt.tight_layout()
    plt.savefig(dst_folder + data_type + "_clustering.pdf", dpi=100)
    plt.close()

    return dba_model.cluster_centers_


def make_clusters(data_set, data_type="V2", num_clusters=12):
    d1, d2 = len(data_set), len(data_set[0].post_processing_data[data_type][data_type])
    samples = np.zeros((d1, d2))
    samples_names = []
    for idx, cel_object in enumerate(data_set):
        samples[idx] = cel_object.post_processing_data[data_type][data_type]
        samples_names.append(cel_object.name)
    cluster_centers = DBA_model(samples, samples_names, num_clusters=num_clusters, dst_folder=pdfdir, data_type=data_type)

    # calculate distance to cluster center from each object, e.i: certanity rate
    d2 = cluster_centers.shape[0]
    distance_to_centers_ = np.array(np.ones((d1, d2)) * np.inf)
    distance_to_centers = pd.DataFrame(distance_to_centers_, index=samples_names,
                                       columns=[x for x in range(num_clusters)])
    for i, cel_object in enumerate(data_set):
        for j, cluster in enumerate(cluster_centers):
            distance_to_centers.iloc[i, j] = DTWDistance(cel_object.post_processing_data[data_type][data_type], cluster)
    distance_to_centers.to_csv(csvdir + data_type + '_distances.csv')

    return cluster_centers


# cleaning up
for pdf in os.listdir(fitsdir):
    if pdf.endswith("pdf"):
        os.remove(fitsdir + "/" + pdf)




# processing data
for each_file in os.listdir(fitsdir):
    # Reading the oifits file
    data = oifits.read(fitsdir, each_file)

    # A plot to quickly visualise the dataset. CPext sets the vertical limits for the closure phases (CPs) plot:
    # By construction CPs are between -180 and 180 degrees but sometimes the signal is small.
    # data.plotV2CP(CPext=15, lines=False,save=True, name=each_file+".pdf")

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
    if knn_on_files:
        object_type, object_name = get_type_name(each_file)
        file_names[object_type].append(object_name.split('.')[0])

    #get object name
    cel_object_name = each_file.split(".")[0]

    #get V2 and do processing for those
    V2_data = {"base": base, "V2": V2, "V2err": V2e, "waveV2": waveV2}
    CP_data = {"Bmax": Bmax, "CP": CP, "CPerr": CPe, "waveCP": waveCP}
    V2_data_processed = data_processing(cel_object_name,"V2",V2_data,visulalize=False,threshold=1)
    CP_data_processed = data_processing(cel_object_name,"CP", CP_data, visulalize=False,threshold=180)
    current_object = Celestial(cel_object_name,V2_data,
                               post_processing_data={"V2":V2_data_processed,
                                                     "CP": CP_data_processed})

    #build full dataset
    data_set.append(current_object)


###############
###V2 model####
###############
V2_cluster_centers = make_clusters(data_set, data_type="V2", num_clusters=12)

###############
###CP model####
###############
CP_cluster_centers = make_clusters(data_set, data_type="CP", num_clusters=12)



####################
####KNN on files####
####################
print("=====================================================\n")
print("=================<KNN classification> ===============\n")
print("=====================================================\n")
#control check if file_names dictionary contains data, that will be in case knn on files is set to True
if len(file_names)>0:
    train_ref, test_ref = [], []
    for k, v in file_names.items():
        random.shuffle(v)
        #perfrom 80/20 split
        index = len(v) * 80 // 100
        train_ref.extend(v[:index])
        test_ref.extend(v[index:])

#make train and test datasets
train_ds, test_ds = {}, {}
for cel_object in data_set:
    if cel_object.name in train_ref:
        train_ds[cel_object.name] = cel_object.post_processing_data["V2"]["V2"]
    else:
        test_ds[cel_object.name] = cel_object.post_processing_data["V2"]["V2"]

#test model accuracy
preds, ground_truth = [], []
for cel_object, query in test_ds.items():
    n_closest = get_nn(query,train_ds,w=5,n_count=3)
    first_closest =n_closest.loc[0]['Closest Neighbor'] #get 1st closest neighbor
    first_closest_type,_ = get_type_name(first_closest)
    true_type,cel_object_name = get_type_name(cel_object)
    if first_closest_type != true_type:
        print("===================================================")
        print("Classification mismatch for {0}: predicted {1}, actual {2}".format(cel_object_name,first_closest_type,true_type))
        print("===================================================")
        print("Other possible neighbors and distances\n", n_closest)

    preds.append(first_closest_type)
    ground_truth.append(true_type)

print("=================<Model Accuracy (query on samples)> ===============\n")
print(classification_report(ground_truth,preds,zero_division=0))



# for i in range(len(samples)-1):
#     for j in range(i,len(samples)):
#         print(DTWDistance(samples[i],samples[j]))
#         print(euclidian_barycenter(samples))


# moving files
for pdf in os.listdir(fitsdir):
    if pdf.endswith("pdf"):
        os.rename(fitsdir + "/" + pdf, pdfdir + pdf)
