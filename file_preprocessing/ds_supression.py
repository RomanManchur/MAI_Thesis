import pandas as pd


def data_supression(pandas_ds, method='mean'):
    '''
    Function to supress data along y-axis in quantize bucket using mean or median supression method
    :param panda_ds: input data set as pandas dataset object
    :param method: <str> either median or mean
    :return: supressed DS
    '''
    print('Data supression is called....')
    c_names = pandas_ds.columns
    c = c_names[0]
    pandas_ds = pandas_ds.set_index([c])
    idx = set(list(pandas_ds.index.values))
    tmp = []
    for i in idx:
        elements_in_bin = pandas_ds.loc[i]  # pulls all Y-axis elements in given bin along x-axis
        if isinstance(elements_in_bin, pd.DataFrame):
            if method == 'mean':
                z = elements_in_bin.mean()
            else:
                z = elements_in_bin.median()
        else:  # in case bucket contains just one value, don't do any manipulation with data
            z = elements_in_bin
        val = z.to_list()  # converts processed features into list
        val.insert(0, i)  # inserting into list x-axis value that corresponds to given features vector at position 0
        tmp.append(val)  # append feature vector to resulting list
    compressed_ds = pd.DataFrame(tmp, columns=c_names)  # convert result list to dataframe
    return compressed_ds
