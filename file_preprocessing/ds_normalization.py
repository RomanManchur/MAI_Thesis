def normalize_df(panda_ds_input, x, method='mean'):
    '''
    Function to normalize dataset; normalization is against specifid column - x-axis
    :param panda_ds_input: input data set as pandas dataset object
    :param x: <int> column index along which normalization is needed; corresponds to x-axis on figures
    :param method: <str> either minmax or mean
    :return: normalized DS
    '''
    print('Make normalize is called....')
    panda_ds = panda_ds_input.copy()#make a copy of input DF to apply normalization on it, otherwise original DF will be modified
    col = panda_ds.columns.tolist()
    if method == 'mean':
        tmp = (panda_ds.iloc[:, x] - panda_ds.iloc[:, x].mean()) / panda_ds.iloc[:, x].std()  # mean normalization
    else:
        if 'V2' in col:
            tmp = (panda_ds.iloc[:, x] - panda_ds.iloc[:, x].min()) / (
                        panda_ds.iloc[:, x].max() - panda_ds.iloc[:, x].min())  # min-max normalization for V2
        else:
            ymin, ymax = -180, 180
            tmp = (panda_ds.iloc[:, x] - ymin) / (
                    ymax - ymin)  # min-max normalization for CP

    panda_ds.iloc[:, x] = tmp  # set normalized features along x-axis
    return panda_ds
