from matplotlib import pyplot as plt
from eval_distance import DTWDistance

def plot_avarage_sequence(centers, *args, **kwargs):
    imax = kwargs["num_clusters"]
    target_path = kwargs["path"]
    colors = ["k", "r", "b", "g"]
    labels = [["all", "low", "medium", "high"]] * imax
    #plot center sequences based on all data
    for i in range(imax):
        plt.subplot(imax, 1, i + 1)
        plt.plot(centers[i], colors[0])
        plt.title("Cluster{0}".format(i + 1))
        ymin = centers[i].min()
        ymax = centers[i].max()
        if ymin > 0:
            plt.ylim(0.90 * ymin, 1.1 * ymax)
        else:
            plt.ylim(1.1 * ymin, 1.1 * ymax)

    # plot center sequences based on wavelength [low, medium, high]
    for j in range(3):
        cluster_sequnce_membership = {k:[] for k in range(imax)}
        for i in range(imax):
            distance = float("inf")
            cluster_id = None
            #find closest central sequence based on DTW disatnce
            #by comparing each sequcence [args[j][i]] with centers from all dataset [centers[c]]
            for c in range(imax):
                current_distance = DTWDistance(args[j][i],centers[c],w=5)
                #that's a dirty trick to avoid multiple component sequences on same graph;
                #there might be condition when two or more sequnces are closer to avarage sequnce from other cluster and that will result in collision
                #need to think on more sophisticated method of avoidance......
                if current_distance < distance:
                    cluster_id = c
                    distance = current_distance
            cluster_sequnce_membership[cluster_id].append([i,distance])


        for k,v in cluster_sequnce_membership.items():
            if len(v) == 1:
                plt.subplot(imax, 1, k+1)
                plt.plot(args[j][v[0][0]],colors[j+1])#plot sequence by refernce to sequnce index assigned to the cluster
            #this is when colision happens and some clusters have no sequnce while other have 2 or more... need to think on that.
            else:
                pass

        for i in range(imax):
            for j in range(4):
                plt.subplot(imax, 1, i + 1)
                plt.legend(labels[i])

    print(cluster_sequnce_membership)
    plt.subplots_adjust(hspace=0.5)
    f = plt.gcf()
    f.set_size_inches(8, 10)
    plt.savefig(target_path)



# def plot_avarage_sequence(*args, **kwargs):
#     data = args
#     imax,jmax = kwargs["num_clusters"], len(data)
#     target_path = kwargs["path"]
#     title = kwargs["plot_type"]
#     colors = ["k", "r", "b", "g"]
#     labels = [["all", "low", "medium", "high"]] * imax
#     ymin, ymax = 1000, -1000
#     for i in range(imax):
#         for j in range(jmax):
#             plt.subplot(imax, 1, i + 1)
#             plt.plot(data[j][i], colors[j])
#             plt.title("Cluster{0}".format(i + 1))
#             ymin_ = data[j][i].min()
#             ymax_ = data[j][i].max()
#             if ymin_ < ymin:
#                 ymin = ymin_
#             if ymax_ > ymax:
#                 ymax = ymax_
#             if j == jmax-1:  # once all sequences for all wavelenghts are shown for current cluster - set axis limit
#                 if ymin > 0:
#                     plt.ylim(0.90 * ymin, 1.1 * ymax)
#                 else:
#                     plt.ylim(1.1 * ymin, 1.1 * ymax)
#                 plt.legend(labels[i])
#     plt.subplots_adjust(hspace=0.5)
#     f = plt.gcf()
#     f.set_size_inches(8, 10)
#     plt.savefig(target_path)
