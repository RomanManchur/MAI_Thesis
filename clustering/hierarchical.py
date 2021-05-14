import numpy as np
import sys
import pandas as pd


def agglomerative_clusters(M_, distance_type='complete', clusters=[]):
    '''
        Computes aglomerative clusters based on input distance matrix
        :param M: <pandas.DataFrame>
            - indexes: object names
            - columns: object names
            - values: triangular matrix representing distance between pairs
        :param distance_type: <str>
            - defines distance that will be used between clusters

        :param clisters: <list>
            - keeps track of clusters groups

        :return: <list>
            agglomerative clusters as a nested lists
    '''

    # print(M)
    m, n = M_.shape
    M = M_.copy()

    if m != n:
        print(M)
        print(M.shape[0])
        print(M.shape[1])
        print('Expected square matrix as input. Got {0}x{1}'.format((M.shape[0], M.shape[1])))
        sys.exit(1)

    if m == 1:
        return clusters
    else:
        min_col_values = M.min() # return series of minimum elements in M-matrix in column direction
        col_index = min_col_values.idxmin() # index of minimum element in min_col_values, col index of min element of M
        min_value = min_col_values[col_index] # min value along the column; that is min distance between pair of elements
        row_indexes = M.index[M[col_index] == min_value] # all rows where elements is equal to min_value
        row_index = row_indexes.values[0] #get 1st index with minimum

        # print("row_index: ", row_index, "col_index: ", col_index)

        #contantenate group elements
        clustered_elements = str(row_index) + ' : ' + str(col_index)

        #recompute distances to between all elements and new group
        # 1. Initialize new series that will hold distances between all elements and new cluster
        new_col = np.full(m-2, np.inf)
        # print(new_col)

        # 2. iterate over elements of original matrix and for each element except of merged calculate distance to
        # new cluster --> maximize distance for complete linkage clustering, and use other distances for other models
        idx=0
        for current_object in M.columns:
            if current_object != row_index and current_object != col_index:
                if distance_type == 'complete':
                    distance = max(M.loc[col_index,current_object], M.loc[row_index,current_object])
                elif distance_type == 'single':
                    distance = min(M.loc[col_index, current_object], M.loc[row_index, current_object])

                #make new distance vector
                new_col[idx] = distance #update element in new series
                idx+=1 #update index

        #         print(M[current_object])
        #         print("Distance: ", distance)
        # print("Column that will be inserted: ", new_col)

        #keep track of clustered items
        clusters.append({clustered_elements: min_value})


        # 3. drop elements that clustered together
        M.drop(columns = [col_index,row_index], inplace = True)
        M.drop([col_index,row_index], inplace=True)
        # print(M)

        # 4. update distance matrix by inserting new distance vector for merged elements
        M.insert(0, column=clustered_elements, value=new_col) # add new column
        # print(M)
        tmp={}
        for idx, k in enumerate(M.columns):
            if k == clustered_elements:
                tmp[k] = np.inf
            else:
                tmp[k] = new_col[idx-1]
        M = pd.concat([pd.DataFrame(tmp, index=[clustered_elements]), M])
        # print(M)
        return agglomerative_clusters(M, distance_type = distance_type, clusters=clusters)




def make_condense_matrix(M_):
    '''
        Takes symetric diagonal matrix and creates condensed form of it
        :param M_: <pandas.DataFrame>

        :return: np.array
    '''
    result = []
    n,m = M_.shape
    if n != m:
        print("Expected square matrix for condence operation, got {0}x{1} instead".format((m,n)))
        sys.exit(1)
    for i1 in range(0,n):
        for i2 in range(i1+1,m):
            result.append(M_.iloc[i2, i1])
    return np.asarray(result)


#
M_ = np.asarray([
    [np.inf, 9,3,6,11],
    [9, np.inf, 7, 5, 10],
    [3,7, np.inf, 9, 2],
    [6,5,9,np.inf, 8],
    [11,10,2,8,np.inf]
    ])