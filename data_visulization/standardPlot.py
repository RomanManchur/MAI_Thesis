import matplotlib.pyplot as plt


def plotData(raw_data, quantized, normalized, interpolated, dst_folder, plot_type, scale='all'):
    '''
    :param <tupple>
            data: <dictionary> keys are filenames and associated value is pandas dataframe with values to plot
            threshold: <int> threshold on errors value used to create color map
    :param quantized: <pandas DF> quantized represenation of input data
    :param normalized: <pandas DF> normalized represenation of input data
    :param interpolated: <pandas DF> interpolation over missing points
    :param dst_folder: <str> path to folder in which figures needs to be stored
    :param plot_type: <str> prefix of resulting figure
    :param scale: <str> represents data_scale, e.i: all , low , medium or high wavelength
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
        ccc = x[mask]
        ddd = y[mask]
        if cell != 3:
            if len(ccc) == len(ddd) and len(ccc) == len(color_map):
                axs[i, j].scatter(x[mask], y[mask], c=color_map, s=0.5, cmap='gist_rainbow_r')
                if plot_type == "V2_" and cell == 0:
                    axs[0,0].set_ylim([0,1])
                elif plot_type == "CP_" and cell == 0:
                    axs[0,0].set_ylim([-180,180])
            else:
                axs[i, j].scatter(x[mask], y[mask])
                #TODO add logging process for above condition where threshold is not defined well.
        else:
            color_map = current_figure.iloc[:, 3]
            axs[i, j].scatter(x[mask], y[mask], c=color_map[mask], s=0.5, cmap='gist_rainbow_r')
        axs[i, j].set_xlabel(xlabel)
        axs[i, j].set_ylabel(ylabel)
        cell += 1
    axs[0, 0].set_title("Original data")
    axs[0, 1].set_title("Quantized data")
    axs[1, 0].set_title("Normalized data")
    axs[1, 1].set_title("Averaging  and Interpolation")
    plt.subplots_adjust(hspace=0.5)
    plt.subplots_adjust(wspace=0.5)
    f = plt.gcf()
    f.set_size_inches(8, 10)
    plt.savefig(dst_folder + plot_type + cname + "_" + scale + ".pdf", dpi=100)  # save data from current block
    plt.close()

