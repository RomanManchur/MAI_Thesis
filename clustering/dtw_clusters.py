from tslearn.clustering import TimeSeriesKMeans
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from eval_distance import DTWDistance


def DBA_model(samples, samples_names, num_clusters, path_to_pdf, plot=False):
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
            if cluster_id == num_clusters-1:
                plt.savefig(path_to_pdf, dpi=100)
                plt.close()
    print("Results are saved to: {0}".format(path_to_pdf))
    return dba_model.cluster_centers_

def make_dtw_clusters(measurements ,num_clusters=12, path_to_pdf='', plot=True):
    '''
    Function that builds clusters using sequence avaraging

    :param pandas.Dataframe
        Table with measurements for certain attribute [V2 or CP] filtered on wavelength
    :param num_clusters <int>
        defines number of clusters build
    '''
    csvdir = './data/csv/'
    pdfdir = './data/pdf/'
    data_type = 'V2'

    samples_names = list(measurements.index)
    for target_object in samples_names:
        ts = measurements.loc[target_object]
        if ts.isnull().any():
            measurements.drop(target_object, inplace=True) # drop rows with NaN

    measurements_data = measurements.values # get measurements data
    samples_names = list(measurements.index)  # building associative list with celestial object names
    d1, d2 = measurements.shape
    samples = np.zeros((0, d2))  # building data matrix used in clustering decision
    for idx, ts in enumerate(measurements_data):
        samples = np.vstack((samples, ts))
    cluster_centers = DBA_model(samples, samples_names, num_clusters=num_clusters,path_to_pdf=path_to_pdf, plot=plot)

    # calculate distance to cluster center from each object, e.i: certainty rate
    # re-defining dimensions: d1 may be lower than original due to filtered fields, d2 - correspond to avg.sequence size
    d1, d2 = len(samples_names), cluster_centers.shape[0]
    distance_to_centers_ = np.array(np.ones((d1, d2)) * np.inf)
    distance_to_centers = pd.DataFrame(distance_to_centers_, index=samples_names, columns=[x for x in range(num_clusters)])
    for i, ts in enumerate(measurements_data):
        for j, cluster in enumerate(cluster_centers):
            distance_to_centers.iloc[i, j] = DTWDistance(ts, cluster)
    distance_to_centers.to_csv(csvdir + data_type + '_distances.csv')

    return cluster_centers


# def make_dtw_clusters(data_set_as_dict, wavelength_scale="V2_all", data_type="V2", num_clusters=12, plot=False):
#     '''
#     Function that builds clusters using sequence avaraging
#
#     :param data_set_as_dict <dict>
#         keys: celestial object names; values: object of Celestial type
#     :param wavelength_scale <str>
#         can either be - V2_low, V2_medium, V2_high or V2_all and similar values for CP parameter
#     :param: data_type <str>
#         can either be V2 or CP
#     :param num_clusters <int>
#         defines number of clusters build
#     '''
#     csvdir = './data/csv/'
#     celestial_object_data = list(data_set_as_dict.values())
#     d1, d2 = len(data_set_as_dict), len(celestial_object_data[0].post_processing_data[wavelength_scale][data_type])
#     samples = np.zeros((0, d2))  # building data matrix used in clustering decision
#     samples_names = []  # building associative list with celestial object names (that is to address fact that list are non-ordered)
#     for idx, cel_object in enumerate(celestial_object_data):
#         # there might be empty datasets after filtering skip processing for those and continue with next object
#         # print(cel_object.name,cel_object.post_processing_data[wavelength_scale][data_type], cel_object.post_processing_data[wavelength_scale][data_type].size)
#         if cel_object.post_processing_data[wavelength_scale][data_type].size == 0 or \
#                 cel_object.post_processing_data[wavelength_scale][data_type].size != d2:
#             data_set_as_dict.pop(cel_object.name, None)  # remove objects with no measurements or non-aligned dimensions
#             continue
#         samples = np.vstack((samples, cel_object.post_processing_data[wavelength_scale][data_type]))
#         # samples[idx] = cel_object.post_processing_data[data_type][data_type]#this can't be used as we might have missing records and indexing will be affected with removal opertation
#         samples_names.append(cel_object.name)
#     cluster_centers = DBA_model(samples, samples_names, num_clusters=num_clusters, dst_folder=pdfdir,
#                                 data_type=wavelength_scale, plot=plot)
#
#     # calculate distance to cluster center from each object, e.i: certainty rate
#     # re-defining dimensions: d1 may be lower than original due to filtered fields, d2 - correspond to avg.sequence size
#     d1, d2 = len(samples_names), cluster_centers.shape[0]
#     distance_to_centers_ = np.array(np.ones((d1, d2)) * np.inf)
#     distance_to_centers = pd.DataFrame(distance_to_centers_, index=samples_names,
#                                        columns=[x for x in range(num_clusters)])
#     for i, cel_object in enumerate(data_set_as_dict.values()):
#         for j, cluster in enumerate(cluster_centers):
#             distance_to_centers.iloc[i, j] = DTWDistance(cel_object.post_processing_data[wavelength_scale][data_type],
#                                                          cluster)
#     distance_to_centers.to_csv(csvdir + wavelength_scale + data_type + '_distances.csv')
#
#     return cluster_centers