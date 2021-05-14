import numpy as np
from tqdm import tqdm

def fit(dists):
    """Merge sequences.
    :param series: Iterator over series.
    :return: Dictionary with as keys the prototype indicices and as values all the indicides of the series in
        that cluster.
    """

    nb_series = 3
    cluster_idx = dict()
    min_value = np.min(dists)
    min_idxs = np.argwhere(dists == min_value)
    min_idx = min_idxs[0, :]
    deleted = set()
    cnt_merge = 0

    print('Merging patterns')
    pbar = tqdm(total=dists.shape[0])

    # Hierarchical clustering (distance to prototype)
    while min_value <= np.inf:
        cnt_merge += 1
        i1, i2 = int(min_idx[0]), int(min_idx[1])

        print("Merge {} <- {} ({:.3f})".format(i1, i2, min_value))
        if i1 not in cluster_idx:
            cluster_idx[i1] = {i1}
        if i2 in cluster_idx:
            cluster_idx[i1].update(cluster_idx[i2])
            del cluster_idx[i2]
        else:
            cluster_idx[i1].add(i2)
        # if recompute:
        #     for r in range(i1):
        #         if r not in deleted and abs(len(cur_seqs[r]) - len(cur_seqs[i1])) <= max_length_diff:
        #             dists[r, i1] = self.dist(cur_seqs[r], cur_seqs[i1], **dist_opts)
        #     for c in range(i1+1, len(cur_seqs)):
        #         if c not in deleted and abs(len(cur_seqs[i1]) - len(cur_seqs[c])) <= max_length_diff:
        #             dists[i1, c] = self.dist(cur_seqs[i1], cur_seqs[c], **dist_opts)
        for r in range(i2):
            dists[r, i2] = np.inf
        for c in range(i2 + 1, 3):
            dists[i2, c] = np.inf
        deleted.add(i2)
        if len(deleted) == nb_series - 1:
            break
        if pbar:
            pbar.update(1)
        # min_idx = np.unravel_index(np.argmin(dists), dists.shape)
        # min_value = dists[min_idx]
        min_value = np.min(dists)
        # if np.isinf(min_value):
        #     break
        min_idxs = np.argwhere(dists == min_value)
        min_idx = min_idxs[0, :]
        pbar.update(dists.shape[0] - cnt_merge)

    prototypes = []
    for i in range(3):
        if i not in deleted:
            prototypes.append(i)
            if i not in cluster_idx:
                cluster_idx[i] = {i}
    return cluster_idx

dists = np.full((3,3), np.inf)
np.fill_diagonal(dists,0)
print(dists)

