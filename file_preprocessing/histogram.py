def make_hist(panda_ds, col_names=[], intervals=100):
    '''
    Function to build histogram representation of data set
    :param panda_ds: input data set as pandas dataset object
    :param col_names: columns for which histagram is built
    :return: histogram representation of input DS
    '''
    print('Make histogam called....')
    hist_ds = pd.DataFrame()
    if not len(col_names):
        col_names = panda_ds.columns
    for c in col_names:
        max_value = math.ceil(panda_ds[c].max())
        min_value = math.floor(panda_ds[c].min())
        range = max_value - min_value
        interval_width = range / intervals
        hist = np.zeros(intervals)
        for element in panda_ds[c]:
            cell = int(element // interval_width)
            if cell >= intervals:
                hist[-1] += 1
            else:
                hist[cell] += 1
        hist_ds[c] = hist

    return hist_ds
