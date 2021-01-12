import os
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

# Specifying the location of the data file
dir = '/Users/rmanchur/Documents/MAI_Project/data/'
# fitsdir = dir + 'all_data/all_sets/'
# fitsdir = dir + 'all_data/2stars/'
# fitsdir = dir + 'all_data/Star+exozodiacaldust/'
fitsdir = dir + 'all_data/Single/'
pdfdir = dir + 'pdf/'
csvdir = dir + 'csv/'


class Celestial:
    '''Generic conteiner to store object information'''

    def __init__(self, name, z):
        '''
        Create instance of class
        :param type: <str> description of object type
        :param V2: <series> measurements of V2
        '''
        self.name = name
        self.V2 = z.iloc[:, 1]

    def getType(self):
        return self.name.split('_')[0]



def knn(train, test, w):
    preds, ground_truth = [], []
    for ind, i in enumerate(test):
        min_dist = float('inf')
        closest_n, closest_seq = '', []
        for j in train:
            dist = DTWDistance(i.V2, j.V2, w)
            if dist < min_dist:
                min_dist = dist
                closest_n = j.getType()
        preds.append(closest_n)
        ground_truth.append(i.getType())

    return classification_report(ground_truth, preds)


# cleaning up
for pdf in os.listdir(fitsdir):
    if pdf.endswith('pdf'):
        os.remove(fitsdir + '/' + pdf)

# processing data
sq_visibilities, closure_phase = {}, {}
for each_file in os.listdir(fitsdir):
    # Reading the oifits file
    data = oifits.read(fitsdir, each_file)

    # A plot to quickly visualise the dataset. CPext sets the vertical limits for the closure phases (CPs) plot:
    # By construction CPs are between -180 and 180 degrees but sometimes the signal is small.
    # data.plotV2CP(CPext=15, lines=False,save=True, name=each_file+'.pdf')

    # Creating a dictionary with meaningful data (easier to manipulate)
    datadict = data.givedataJK()

    # Extracting the meaningful information from the dictionary
    # extract the squared Visibilities and errors
    V2, V2e = datadict['v2']

    # extract closure phases and errors
    CP, CPe = datadict['cp']
    # extract baselines (let's start with those 1D coordinates instead of u and v coordinates)
    base, Bmax = oifits.Bases(datadict)
    # extract u and v coordinates
    u, u1, u2, u3 = datadict['u']
    v, v1, v2, v3 = datadict['v']
    # extract wavelengths for squared visibilities and closure phases
    waveV2, waveCP = datadict['wave']

    # Now you can play with the data!
    cel_object = each_file.split('.')[0]
    sq_visibilities[cel_object] = pd.DataFrame({'base': base, 'V2': V2, 'V2err': V2e, 'waveV2': waveV2})
    closure_phase[cel_object] = pd.DataFrame({'Bmax': Bmax,
                                              'CP': CP})





traintest = []
for k, v in sq_visibilities.items():
    print('Processing file:.....', k)
    quantized_ds = quantize_ds(v, intervals=50)  # quantize dataset along x-axis
    normalized_ds = normalize_df(quantized_ds, 0,
                                 method='minmax')  # normalize dataset using min-max normalization  - [0..1]'
    z = data_supression(normalized_ds,
                        method='median')  # compress data point in each bucket using mean or median compression
    z.sort_values(by='base', inplace=True)  # sort and replace
    z.interpolate(method='linear', axis=0, direction='forward', inplace=True)

    # Data vizualization after pre-processing
    standardPlot.plotData(raw_data=(k, v, 1),
                          quantized=quantized_ds,
                          normalized=normalized_ds,
                          interpolated=z,
                          dst_folder=pdfdir,
                          plot_type="V2_")

#     traintest.append(Celestial(k, z))#construct training set
#
#     z.to_csv(csvdir+'quantize_'+k+'.csv')
#     plt.figure()
#     z.plot.scatter(x=z.columns[0],y=z.columns[1],)
#     plt.savefig(pdfdir+'quantize'+ k + '.pdf')

# random.shuffle(traintest)
# train = traintest[0:int(len(traintest)*0.8)]
# test = traintest[int(len(traintest)*0.8):]
#
# train_samples = [i.name for i in train]
# test_samples = [ i.name for i in test]
# #calculate distance between each test example and each train example
# for i in test:
#     for j in train:
#         w = DTWDistance(i.V2, j.V2)
#         print('Distance metric between: ', i.name, j.name, 'is: ', w)
#
# print("Train samples:\t" + str(train_samples) + '\n' + "Test samples:\t" + str(test_samples))
# print(knn(train,test,3))
#
#



DTWDistance([1,2,3,4,5],[6,7,8,9,0],2)
print(euclidian_barycenter([[1,2,3],[2,3,4]]))

seed = 1234
dba_model = TimeSeriesKMeans(n_clusters=2,n_init=2, metric="dtw",max_iter_barycenter=10,verbose=True,random_state=seed)
TS = [[1,1,2,2,1,1],[1,2,2,1,1,1],[2,1,2,1,2,1],[1,2,1,2,1,2]]
TS = np.array(TS)
dba_predict = dba_model.fit_predict(TS)
sz = 2
for yi in range(2):
    plt.subplot(3, 3, 4 + yi)
    for xx in TS[dba_predict == yi]:
        plt.plot(xx.ravel(), "k-", alpha=.2)
    plt.plot(dba_model.cluster_centers_[yi].ravel(), "r-")
    # plt.xlim(0, sz)
    # plt.ylim(-4, 4)
    plt.text(0.55, 0.85,'Cluster %d' % (yi + 1),
             transform=plt.gca().transAxes)
    if yi == 1:
        plt.title("DBA $k$-means")

plt.tight_layout()
plt.show()





# moving files
for pdf in os.listdir(fitsdir):
    if pdf.endswith('pdf'):
        os.rename(fitsdir + '/' + pdf, pdfdir + pdf)
