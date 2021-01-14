import matplotlib.pyplot as plt


def plotData(raw_data, quantized, normalized, interpolated, dst_folder, plot_type):
    '''
    :param <tupple>
            data: <dictionary> keys are filenames and associated value is pandas dataframe with values to plot
            threshold: <int> threshold on errors value used to create color map
    :param quantized: <pandas DF> quantized represenation of input data
    :param normalized: <pandas DF> normalized represenation of input data
    :param interpolated: <pandas DF> interpolation over missing points
    :param dst_folder: <str> path to folder in which figures needs to be stored
    :param plot_type: <str> prefix of resulting figure
    :return: None
    '''
    fig, axs = plt.subplots(2, 2)
    cname, data, threshold = raw_data
    color_map = data.iloc[:, 3]
    cell = 0
    for current_figure in [data, quantized, normalized, interpolated]:
        i = cell // 2
        j = cell % 2
        mask = current_figure.iloc[:, 2] < threshold  # set mask for color map, here errors < defined threshold
        x = current_figure.iloc[:, 0]
        y = current_figure.iloc[:, 1]
        labels = current_figure.columns.tolist()
        xlabel, ylabel = labels[0:2]
        if cell != 3:
            axs[i, j].scatter(x[mask], y[mask], c=color_map, s=0.5, cmap='gist_rainbow_r')
        else:
            color_map = current_figure.iloc[:, 3]
            axs[i, j].scatter(x[mask], y[mask], c=color_map[mask], s=0.5, cmap='gist_rainbow_r')
        axs[i, j].set_xlabel(xlabel)
        axs[i, j].set_ylabel(ylabel)
        cell += 1
    axs[0, 0].set_title("Original data")
    axs[0, 1].set_title("Quantized data")
    axs[1, 0].set_title("Normalized data")
    axs[1, 1].set_title("Supressed+Interpolation")
    plt.subplots_adjust(hspace=0.5)
    plt.subplots_adjust(wspace=0.5)
    f = plt.gcf()
    f.set_size_inches(8, 10)
    plt.savefig(dst_folder + plot_type + cname + ".pdf", dpi=100)  # save data from current block
    plt.close()

# def standardPlot(data, threshold = 1):
#     '''
#
#     :param data: <dictionary> keys are filenames and associated value is pandas dataframe with values to plot
#     :return: None
#     '''
#     size = len(data.keys())
#     keys = list(data.keys())
#     fig, axs = plt.subplots(2, 2)
#     cell = 0
#     if size >=4:
#         block = keys[:4]
#         for k in block:
#             object_data = data.pop(k)#fetch pandas frame associated with given celestial object
#             i = cell//2
#             j = cell%2
#             x = object_data.iloc[:,0]
#             y = object_data.iloc[:,1]
#             mask = object_data.iloc[:,2] < threshold #set mask for color map, here errors < defined threshold
#             color_map = object_data.iloc[:,3]
#             axs[i,j].scatter(x[mask],y[mask],c=color_map, s=0.1,cmap='gist_rainbow_r')
#             axs[i,j].set_xlabel('base')
#             axs[i,j].set_ylabel('V2')
#             cell+=1
#         plt.savefig("_"+"".join(random.choice(string.ascii_lowercase) for i in range(10))+".pdf")#save data from current block
#         standardPlot(data,1)#recursive call
#     else:
#         for k in keys:
#             object_data = data.pop(k)  # fetch pandas frame associated with given celestial object
#             i = cell // 2
#             j = cell % 2
#             x = object_data.iloc[:, 0]
#             y = object_data.iloc[:, 1]
#             mask = object_data.iloc[:, 2] < threshold  # set mask for color map, here errors < defined threshold
#             color_map = object_data.iloc[:, 3]
#             axs[i, j].scatter(x[mask], y[mask], c=color_map, s=0.1,cmap='gist_rainbow_r')
#             axs[i, j].set_xlabel('base')
#             axs[i, j].set_ylabel('V2')
#             cell += 1
#         plt.savefig("_"+"".join(random.choice(string.ascii_lowercase) for i in range(10))+".pdf")# save data basis block
