import os
import random

import pandas as pd
import ReadOIFITS as oifits
from Celestial import Celestial
from data_visulization import *
from eval_distance import *
import matplotlib.pyplot as plt
from collections import defaultdict
from file_preprocessing.data_processing import data_processing
from knn.knn_modules import generate_data_sets, get_closest_neighbors
from clustering.hierarchical import agglomerative_clusters, make_condense_matrix
from clustering.dtw_clusters import make_dtw_clusters
from knn.knn_modules import knn_classification
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
import numpy as np
import sys



args = sys.argv
mode = None
benchmarking, knn, clustering = False, False, False
dir, fitsdir, targets = '', '', ''
max_window_size = None
#Below variables control K-nn calculation
dtw_window_size, nearest_neighbors = None, 3
#Below variable specifies distane matrix
csvfile = None # CSV file to read / write coordinates of dataset in 2D space
#Below variable defines cluster type: k-means or hierarchical
cluster_type = None
#Below defines linkage type in case of hierarchical clusters
linkage_type = None
#Below defines number of clusters  in case of k-means clusters
k_clusters = None
#optional attributes
vizualize_data = False
pdfdir, csvdir, out = None, None, None
saved = False # controls if distance matrix is read from file (False) or saved to specified file(True)

meassurement_type, wavelength = None, None
# Specifying the location of the data file
# dir = "/Users/rmanchur/Documents/MAI_Thesis/data/"
# fitsdir = dir + "all_data/all_sets/"
# fitsdir = dir + "all_data/2stars/"
# fitsdir = dir + "all_data/StellarSurface/"
# fitsdir = dir + "all_data/Single/"
# fitsdir = dir + "all_data/renamed/"
# targets = "data/points_to_check/targetlist.txt"
# targets = "data/points_to_check/target_list_small.txt"
# targets = "data/points_to_check/broken.txt"
# pdfdir = dir + "pdf/"
# csvdir = dir + "csv/"
data_set_as_dict = {}
knn_on_files = True


arg_index = 1
print(args)

while arg_index < len(args):
    if args[arg_index] == '-mode': # expected values: [benchmarking, knn, clustering]
        mode = args[arg_index+1]

    elif args[arg_index] == '-dir': #directory with project files
        dir = args[arg_index+1]

    elif args[arg_index] == '-fitsdir':  # directory with dataset files
        fitsdir = args[arg_index + 1]

    elif args[arg_index] == '-targets':  # path to file with objects to run analysis on
        targets = args[arg_index + 1]

    elif args[arg_index] == '-pdfdir':  #path to directory that will be used to save PDF files
        pdfdir = args[arg_index + 1]

    elif args[arg_index] == '-csvdir':  # path to directory that will be used to save and read CSV files
        csvdir = args[arg_index + 1]

    elif args[arg_index] == '-knn_on_files':  # only used in case of applying KNN with reference set containing labled data
        knn_on_files = args[arg_index + 1]

    elif args[arg_index] == '-max_window_size':  # defines max window DTW wrapping window size when running benchmarks
        max_window_size = args[arg_index + 1]

    elif args[arg_index] == '-dtw_window_size':  # controls max DTW wrapping window in KNN
        dtw_window_size = args[arg_index + 1]

    elif args[arg_index] == '-nearest_neighbors':  # controls max number of nearest neighbors to track
        nearest_neighbors = args[arg_index + 1]

    elif args[arg_index] == '-cluster_type':  # expected types: [hierarchical, k-means, dtw]
        cluster_type = args[arg_index + 1]

    elif args[arg_index] == '-linkage_type':  # defines linkage type in hierarchical clustering
        linkage_type = args[arg_index + 1]

    elif args[arg_index] == '-num_clusters':  # controls number of clusters for K-means and DBA k-means (DTW)
        num_clusters = args[arg_index + 1]

    elif args[arg_index] == '-meassurement_type':  # Defines measurement type used in DBA k-means (DTW) either V2 or CP
        meassurement_type = args[arg_index + 1]

    elif args[arg_index] == '-wavelength':  # Defines wavelength type used in DBA k-means (DTW), supported values [all, low, medium, high]
        wavelength = args[arg_index + 1]

    elif args[arg_index] == '-vizualize_data':  # bolean parameter that defines if to plot data during pre-processing
        vizualize_data = str(args[arg_index + 1])

    elif args[arg_index] == '-out':  # specify location to store results
        out = args[arg_index + 1] # specifies path to store the data

    elif args[arg_index] == '-csv_file':  # path to csv file with pre-processed data to full avoid recalculation on dataset
        csvfile = args[arg_index + 1]

    elif args[arg_index] == '-saved':
        saved = args[arg_index + 1] #if False reads distM from file, else saves to file
        if saved == "False": saved = False
        else: saved = True

    arg_index+=2

# Check for mandatory parameters; Exit if any mandatory is missing
err = False
#In case of full processing we need to specify path to project directory, directory with DB files and target list file
if len(dir) == 0:
    print('Mandatory parameter "-dir" is missing. Please provide path to project directory...')
    err = True
if len(fitsdir) == 0 and not saved:
    print('Mandatory parameter "-fitsdir" is missing. Please provide path to database files...')
    err = True
if len(targets) == 0 and not saved:
    print('Mandatory parameter "-targets" is missing. Please provide path to a file with target data...')
    err = True

#In case processing is run using stored data, then CSV file needs to be provided
if csvfile is None:
    print('Mandatory parameter "-csv_file" is missing. Provide the file to read/write pre-processed data')
    err = True

#Mode is madatory parameter that defines what analysis will be run on dataset
if mode and mode == 'benchmarking':
    benchmarking = True
elif mode and mode == 'knn':
    knn = True
elif mode and mode == 'clustering':
    clustering = True
else:
    print('Got unsupported mode {0}. Expected modes of operation are: benchmarking | knn | clustering'.format(mode))
    err = True

#In case mode is clustering, cluster type needs to be provided.
if mode == 'clustering' and (cluster_type is None or cluster_type not in ['hierarchical', 'k-means', 'dtw']):
    print('Mandatory parameter "-cluster_type" is missing. In case running in clustering mode, then type of cluster needs'
          'to be provided.')
    err = True

#In case mode is DBA k-means, both measurement_type and wavelengths parameters need to be provided
if cluster_type == 'dtw' and (meassurement_type is None or wavelength is None) and not saved:
    print('BDA mode of operation mandatory parameter "-meassurement_type" or -wavelength missing.')
    err = True

if err:
    print("Expected usage:....")
    print("1. Benchmarking mode to detect best window size")
    print("./run -mode benchmarking -max_window_size <size> -dir <project_path>  -fitsdir <path_to_IOFS> -targets <file_with_targets>")
    print("Optional parameters: \n"
          "[-pdfdir] - specifies path to store measurements and pre-processing plots (only relevant if vizualize_data is True)\n"
          "[-vizualize_data] - boolean (default False), defines if data needs to be plotted or not\n")

    print("2. K-nn mode")
    print("./run -mode knn -dtw_window_size <size> -nearest_neighbors <neigh_numbers> -dir <project_path>  -fitsdir <path_to_IOFS> -targets <file_with_targets>")
    print("Optional parameters: \n"
          "[-pdfdir] - specifies path to store measurements and pre-processing plots (only relevant if vizualize_data is True)\n"
          "[-out] - specifies path to file where results are stored (default: nn_small_results.txt)\n"
          "[-vizualize_data] - boolean (default False), defines if data needs to be plotted or not\n")

    print("3. Clustering mode")
    print("3.1 Hierarchical clusters")
    print(
        "./run -mode clustering -cluster_type <hierarchical> -linkage_type <linkage_type> -dtw_window_size <size> -dir <project_path>  -fitsdir <path_to_IOFS> -targets <file_with_targets>")

    print("3.2 K-means clusters")
    print(
        "./run -mode clustering -cluster_type <k-means> -num_clusters <k> -dtw_window_size <size> -dir <project_path>  -fitsdir <path_to_IOFS> -targets <file_with_targets>")
    print("3.3 DTW clusters")
    print(
        "./run -mode clustering -cluster_type <dtw> -num_clusters <k> -dtw_window_size <size> -dir <project_path>  -fitsdir <path_to_IOFS> -targets <file_with_targets>")
    print("Optional parameters: \n"
          "[-pdfdir] - specifies path to store measurements and pre-processing plots (only relevant if vizualize_data is True)\n"
          "[-vizualize_data] - boolean (default False), defines if data needs to be plotted or not\n"
          "[-csv_file] - if passed reads distance matrix from file, instead of recomputing\n"
          "[-saved] - if true, saves distance matrix to specified file; otherwise reads from it (default: False)"
          )
    sys.exit(1)


# Check for optional parameters; Use default values if arguments are not passed in command line
if mode == 'benchmarking' and max_window_size is None:
    max_window_size = 50
if dtw_window_size is None:
    dtw_window_size = 20
if cluster_type == 'hierarchical' and linkage_type is None:
    linkage_type  = 'complete'
if (cluster_type == 'k-means' or cluster_type == 'dtw') and num_clusters is None:
    num_clusters = 7
if pdfdir is None:
    pdfdir = dir + "pdf/"
if csvdir is None:
    csvdir = dir + "csv/"

def print_section(infostring):
    print("====================================")
    print(infostring)
    print("====================================")
    return None

# Those parameters control if knn will be run on  files directly
def default_value():
    return set()
file_names = defaultdict(default_value)

# cleaning up
# for pdf in os.listdir(fitsdir):
#     if pdf.endswith("pdf"):
#         os.remove(fitsdir + "/" + pdf)


#Main program
if not saved:
    ##################################################################
    #                   Get object list from file                    #
    ##################################################################
    print_section("Reading object list infomration.....")
    target_list = {}
    with open(targets, 'r') as target_list_fd:
        for line in target_list_fd:
            line = line.strip()
            elements = line.split(',')
            if len(elements) == 2: #in case file contains information of object name and object type
                target_list[elements[0]] = elements[1]
            elif len(elements) == 1:#in case file contains only information on object name
                target_list[elements[0]] = None
            else:
                continue

    #########################################################
    #   Reading and storing input data object list          #
    #########################################################
    files_in_data_directory = os.listdir(fitsdir)

    # Iterate over files in data directory and process files in given batch
    for object_name, object_type in target_list.items():
        for each_file in files_in_data_directory:
            if object_name in each_file:  # matching if current file is one of the targets to be processed
                # print(target_object, each_file)
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
                ####### Filling data required for K-nn ##########
                #################################################
                print("Processing file {0}:...........".format(each_file))
                print("File is associated with object {0} of type {1}".format(object_name, object_type))
                # Build a dictionary with mapping between object_types as keys and all object names associated with that type
                # E.g:  {'StellarSurface': {'ANTARES', 'ALTAIR', 'R_CRA', 'BETELGEUSE'}, 'StarEZdust': {'AU_MIC', 'BETA_PIC'}, '2stars': {'HD167971', 'HD152314', 'HD54662', 'HIP11231', 'HD152233'}, ...
                # That information is used in KNN for predicting object type based on type of the nearest neighbor
                if knn_on_files and object_type:
                    file_names[object_type].add(object_name)

                ########################################################################################
                ####### Building 'data_set_as_dict' that stores all data associated with files #########
                ########################################################################################
                # Read data and store to dictionary for easy access
                # get V2 and CP measurements
                V2_data = {"base": base, "V2": V2, "V2err": V2e, "waveV2": waveV2}
                CP_data = {"Bmax": Bmax, "CP": CP, "CPerr": CPe, "waveCP": waveCP}

                # if there was some data already processed for current object - update data in dictionary
                if object_name in data_set_as_dict.keys():
                    data_set_as_dict[object_name].update_data({"V2": V2_data, "CP": CP_data})
                # if it's first time data for the celestial object is processed, initialize object and add it to dictionary
                else:
                    current_object = Celestial(object_name, {"V2": V2_data, "CP": CP_data}, object_type)
                    data_set_as_dict[object_name] = current_object

    #Full set is read at this point and associated dictionary elements are created; now we can do data processing

    ################################################################################
    #   Iterate over full dataset (data_set_as_dict) and perform data preprocessing#
    ################################################################################
    for object_name, celestial_object in data_set_as_dict.items():
        #all wavelengths
        V2_data_processed = data_processing(object_name, "V2", celestial_object.data["V2"], visulalize=vizualize_data, threshold=1,
                                            wavelenght="all", pdfdir=pdfdir)
        # tmp_pd = pd.DataFrame(data=celestial_object.data["V2"], columns=list(celestial_object.data["V2"].keys()))
        # print(object_name)
        # print(tmp_pd)


        CP_data_processed = data_processing(object_name, "CP", celestial_object.data["CP"], visulalize=vizualize_data, threshold=180,
                                            wavelenght="all", pdfdir=pdfdir)

        # tmp_pd = pd.DataFrame(data=celestial_object.data["CP"], columns=list(celestial_object.data["CP"].keys()))
        # print(object_name)
        # print(tmp_pd)

        #low wavelengths
        V2_data_processed_low = data_processing(object_name, "V2", celestial_object.data["V2"], visulalize=vizualize_data, threshold=1,
                                            wavelenght="low", pdfdir=pdfdir)
        CP_data_processed_low = data_processing(object_name, "CP", celestial_object.data["CP"], visulalize=vizualize_data, threshold=180,
                                            wavelenght="low", pdfdir=pdfdir)
        #medium wavelengths
        V2_data_processed_medium = data_processing(object_name, "V2", celestial_object.data["V2"], visulalize=vizualize_data, threshold=1,
                                            wavelenght="medium", pdfdir=pdfdir)
        CP_data_processed_medium = data_processing(object_name, "CP", celestial_object.data["CP"], visulalize=vizualize_data, threshold=180,
                                            wavelenght="medium", pdfdir=pdfdir)
        #high wavelengths
        V2_data_processed_high = data_processing(object_name, "V2", celestial_object.data["V2"], visulalize=vizualize_data, threshold=1,
                                            wavelenght="high", pdfdir=pdfdir)
        CP_data_processed_high = data_processing(object_name, "CP", celestial_object.data["CP"], visulalize=vizualize_data, threshold=180,
                                            wavelenght="high", pdfdir=pdfdir)

        celestial_object.post_processing_data = {"V2_all":V2_data_processed, "CP_all": CP_data_processed,
                                                 "V2_low":V2_data_processed_low, "CP_low": CP_data_processed_low,
                                                 "V2_medium":V2_data_processed_medium, "CP_medium": CP_data_processed_medium,
                                                 "V2_high":V2_data_processed_high, "CP_high": CP_data_processed_high}


# Data is preprocessed and models can be applied

###################
###KNN on files####
###################
#RUN below section only while benchmarking to find best window size.
if benchmarking:
    try:
        max_window_size = int(max_window_size)
    except ValueError:
        print("Wrong parameter {0} can not be converted to integer".format(max_window_size))

    train_ref, test_ref = generate_data_sets(file_names, 0.8)
    a_v2, a_cp, a_total = [], [], []
    for window in range(1,max_window_size):
        print("WINDOW SIZE:", window)
        # #KNN on V2 values
        accuracy = knn_classification(train_ref, test_ref ,data_set_as_dict,window=window)['accuracy']
        a_v2.append(accuracy)

        #KNN on CP values
        accuracy = knn_classification(train_ref, test_ref ,data_set_as_dict,scale = 'CP_all',attribute='CP', window=window)['accuracy']
        a_cp.append(accuracy)

        # #KNN on composite metric
        accuracy = knn_classification(train_ref, test_ref ,data_set_as_dict,composed_distance=True, window=window)['accuracy']
        a_total.append(accuracy)

    x = [i for i in range(1,max_window_size)]
    fig, axs = plt.subplots(3,1)
    axs[0].plot(x,a_v2, 'r')
    axs[0].set_title("Effect of window size on accuracy using V2 metric")
    axs[0].set_xlabel('window size')
    axs[0].set_ylabel('Acccuracy')

    axs[1].plot(x,a_cp, 'g')
    axs[1].set_title("Effect of window size on accuracy using CP metric")
    axs[1].set_xlabel('window size')
    axs[1].set_ylabel('Acccuracy')

    axs[2].plot(x,a_total, 'b')
    axs[2].set_title("Effect of window size on accuracy using compound metric")
    axs[2].set_xlabel('window size')
    axs[2].set_ylabel('Acccuracy')

    plt.subplots_adjust(hspace=0.5)
    plt.subplots_adjust(wspace=0.5)
    f = plt.gcf()
    f.set_size_inches(8, 10)
    plt.savefig( "accuracy.pdf", dpi=100)  # save accuracy plots
    plt.close()

    print("Benchmarking completed succesfully!")
    print("Results are stored in accuracy.pdf")
    sys.exit(0)

#Section K-nn on full dataset with fixed wrapping window size
if knn:
    try:
        window = int(dtw_window_size)
    except ValueError:
        print("Wrong parameter {0} can not be converted to integer".format(dtw_window_size))

    try:
        nearest_neighbors = int(nearest_neighbors)
    except ValueError:
        print("Wrong parameter {0} can not be converted to integer".format(nearest_neighbors))
    print(data_set_as_dict)
    get_closest_neighbors(ds1=data_set_as_dict, window=window, nn=nearest_neighbors,out=csvdir+out)
    print("{0}-nn completed succesfully.\n Results are stored to {1}".format(str(nearest_neighbors),csvdir+out))


# Section clusters
if clustering and not cluster_type == 'dtw':
    try:
        window = int(dtw_window_size)
    except ValueError:
        print("Wrong parameter {0} can not be converted to integer".format(dtw_window_size))

    if not saved:
        #Calculate co-orditanates of object in 2D space: x-axis composite distance for V2; y-axis - composite distance for CP
        coordinates_matrix = get_coordinates(dataset=data_set_as_dict, window=window)
        coordinates_matrix.to_csv(csvdir+csvfile)
        print("Coordinates matrix calculated successfully and stored to {0}{1}".format(csvdir, csvfile))
    else:
        #Load coordinates from file
        coordinates_matrix = pd.read_csv(csvdir+csvfile, index_col=0)

    # Calculate clusters
    # Hierarchical clustering with compound distance
    if cluster_type == 'hierarchical':
        # Make distance matrix between pairs
        distance_martix = make_cluster_distances(coordinates_matrix)
        # print(agglomerative_clusters(distance_martix,linkage_type))
        # Use standard packages to make dendogram
        # 1. Create condenced distance matrix
        dist_condenced = make_condense_matrix(distance_martix)
        # 2. Make linkage
        Z = linkage(dist_condenced, linkage_type)
        # 3. Plot
        plt.figure()
        labels = np.asarray(list(distance_martix.columns))
        # dendrogram(Z, orientation='left', labels=labels, truncate_mode='level', p=7)
        dendrogram(Z, orientation='left', labels=labels)
        f = plt.gcf()
        f.set_size_inches(15, 10)
        plt.savefig(pdfdir+'hierarchical.pdf', dpi=100)
        print("Built dendrogram successfully. Results are stored to {0}".format(pdfdir+"hierarchical.pdf"))

        #Uncomment below section fetch targets in cluster groups once max_distance parameter is identified from dendogram
        # max_distance = 4
        # clusters = fcluster(Z, max_distance, criterion='distance')
        # with open(csvdir + 'hierarchical_results.txt', 'w') as hierarchical_results:
        #     for i in range(clusters.min(), clusters.max()+1):
        #         hierarchical_results.write("Cluster [" + str(i)+ "]\n")
        #         hierarchical_results.write(str(list(coordinates_matrix[clusters == i].index)))
        #         hierarchical_results.write("\n")
        sys.exit(0)

#K-means with compound distance
    elif cluster_type == 'k-means':
        # Use K-means
        try:
            num_clusters = int(num_clusters)
            if num_clusters <=0:
                print("Number of clusters has to be positive integer, got {0} instead".format(num_clusters))
                sys.exit(1)
        except ValueError:
            print("Number of clusters has to be positive integer, got {0} instead".format(num_clusters))
            sys.exit(1)

        kmeans = KMeans(n_clusters= num_clusters)
        # Get data required for K-means clustering
        # names, coordinates = [], []
        # for k, v in data_set_as_dict.items():
        #     names.append(k)
        #     coordinates.append([v.coordinates[0], v.coordinates[1]])
        # clustering_data = pd.DataFrame(np.asarray(coordinates), columns=['x', 'y'], index=names)

        # Show original distribution of points in 2D space
        plt.figure()
        plt.scatter(coordinates_matrix.loc[:,'x'],coordinates_matrix.loc[:,'y'])
        # for id, name in enumerate(coordinates_matrix.index):
        #     plt.annotate(name, (coordinates_matrix.iloc[id,0], coordinates_matrix.iloc[id,1]))
        plt.xlabel("DTW(V2)")
        plt.ylabel("DTW(CP)")
        f = plt.gcf()
        f.set_size_inches(15, 10)
        plt.savefig(pdfdir+'data2D.pdf', dpi=100)

        # Apply clustering
        labels = kmeans.fit_predict(coordinates_matrix)

        # Get unique labels
        uniq_lables = np.unique(labels)

        with open(csvdir + 'kmeans_results.txt', 'w') as kmeans_results:
            # Plot clusters
            centroids = kmeans.cluster_centers_
            for i in uniq_lables:
                plt.scatter(coordinates_matrix.loc[labels == i,'x'], coordinates_matrix.loc[labels == i,'y'], label=i)
                # Uncomment below section to write clustering results to ASCII file
                # kmeans_results.write("Cluster [" + str(i)+ "]\n")
                # kmeans_results.write(str(list(coordinates_matrix[labels == i].index)))
                # kmeans_results.write("\n")
            plt.scatter(centroids[:,0] , centroids[:,1] , s = 80, color = 'k', marker = 11)

            # Uncomment below section to plot labeled points on the same plot with unlabled
            coordinates_matrix_ = pd.read_csv(csvdir + 'coordinates_matrix.csv', index_col=0)
            plt.scatter(coordinates_matrix_.loc[:, 'x'], coordinates_matrix_.loc[:, 'y'], color = 'k', marker='*')
            for id_, name_ in enumerate(coordinates_matrix_.index):
                plt.annotate(name_, (coordinates_matrix_.iloc[id_,0], coordinates_matrix_.iloc[id_,1]))

            plt.legend()
            f = plt.gcf()
            f.set_size_inches(15, 10)
            plt.savefig(pdfdir+'kmeans.pdf', dpi=100)
            print("K-means with k={0} completed successfully. Results are saved to {1}".format(num_clusters,pdfdir+'kmeans.pdf'))
        sys.exit(0)


# DBA k-means clustering section
if clustering and cluster_type == 'dtw':
    ################ #############
    ###DTW V2 clustering model####
    ################ #############
    print("Buildng BDA k-means model for attribute {0} and wavelength {1}".format(meassurement_type, wavelength))
    if not saved:
        measurements = []
        for celestial_object in data_set_as_dict.values():
            ts = celestial_object.post_processing_data[meassurement_type+"_"+wavelength][meassurement_type]
            measurements.append(ts)
        measurements = pd.DataFrame(measurements, index=list(data_set_as_dict.keys()))
        measurements.to_csv(csvdir+csvfile)
    else:
        # Load coordinates from file
        measurements = pd.read_csv(csvdir + csvfile, index_col=0)
    #Run DBA k-means
    try:
        num_clusters = int(num_clusters)
        if num_clusters <= 0:
            print("Number of clusters has to be positive integer, got {0} instead".format(num_clusters))
            sys.exit(1)
    except ValueError:
        print("Number of clusters has to be positive integer, got {0} instead".format(num_clusters))
        sys.exit(1)
    V2_cluster_centers_ = make_dtw_clusters(measurements, num_clusters=num_clusters, path_to_pdf=pdfdir + out + '.pdf')

    print("DBA k-means with k={0}, completed succesfully".format(num_clusters))