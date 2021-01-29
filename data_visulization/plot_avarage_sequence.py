from matplotlib import pyplot as plt


def plot_avarage_sequence(*args, **kwargs):
    data = args
    imax,jmax = kwargs["num_clusters"], len(data)
    target_path = kwargs["path"]
    title = kwargs["plot_type"]
    colors = ["k", "r", "b", "g"]
    labels = [["all", "low", "medium", "high"]] * imax
    ymin, ymax = 1000, -1000
    for i in range(imax):
        for j in range(jmax):
            plt.subplot(imax, 1, i + 1)
            plt.plot(data[j][i], colors[j])
            plt.title("Cluster{0}".format(i + 1))
            ymin_ = data[j][i].min()
            ymax_ = data[j][i].max()
            if ymin_ < ymin:
                ymin = ymin_
            if ymax_ > ymax:
                ymax = ymax_
            if j == jmax-1:  # once all sequences for all wavelenghts are shown for current cluster - set axis limit
                plt.ylim(0.90 * ymin, 1.1 * ymax)
                plt.legend(labels[i])
    plt.subplots_adjust(hspace=0.5)
    f = plt.gcf()
    f.set_size_inches(8, 10)
    plt.savefig(target_path)
