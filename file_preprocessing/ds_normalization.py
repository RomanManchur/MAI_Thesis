def normalize_df(panda_ds, x, method='mean'):
    '''
    Function to normalize dataset; normalization is against specifid column - x-axis
    :param panda_ds: input data set as pandas dataset object
    :param x: <int> column index along which normalization is needed; corresponds to x-axis on figures
    :param method: <str> either minmax or mean
    :return: normalized DS
    '''
    print('Make normalize is called....')
    if method == 'mean':
        tmp = (panda_ds.iloc[:, x] - panda_ds.iloc[:, x].mean()) / panda_ds.iloc[:, x].std()  # mean normalization
    else:
        tmp = (panda_ds.iloc[:, x] - panda_ds.iloc[:, x].min()) / (
                    panda_ds.iloc[:, x].max() - panda_ds.iloc[:, x].min())  # min-max normalization

    panda_ds.iloc[:, x] = tmp  # set normalized features along x-axis
    return panda_ds
