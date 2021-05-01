from eval_distance import DTWDistance, DTWMatrix
from matplotlib import pyplot as plt

ts1 = [0,0,1,2,3,4,4,4,2,3,2,1,0,0,1,0]
ts2 = [14,14,5,10,10,11,12,13,14,14,14,5,13,12,11,10]

dtw_matrix = DTWMatrix(ts1,ts2,w=5) #calculate distance matrix
for i in range(0,len(ts1)):
    s = ''
    for j in range(0,len(ts2)):
        s = s + str(dtw_matrix[i,j]) + ','
    # print(s[:-1])


#traceback matrix for optimal distances
i = len(ts1)-1
j = len(ts2)-1
start_point = dtw_matrix[(i,j)]
optimal_path = [(i,j)]
while i != 0 and j != 0:
    up = dtw_matrix[(i-1,j)]
    left = dtw_matrix[(i,j-1)]
    diag = dtw_matrix[(i-1,j-1)]

    #when otimal path is to the left of current point in matrix
    if left < up and left < diag:
        j-=1
    # when otimal path is to the up of current point in matrix
    elif up < left and up < diag:
        i-=1
    # when otimal path is on diagonal of matrix
    else:
        i-=1
        j-=1
    optimal_path.append((i,j))

optimal_path.reverse()
# print(optimal_path)

fig , axs = plt.subplots(2)
axs[0].plot(ts1, color='blue', linestyle='dashed', marker='o', markerfacecolor='black', markersize=5)
axs[0].plot(ts2, color='red', linestyle='dashed', marker='o', markerfacecolor='black', markersize=5)
axs[1].plot(ts1, color='blue', linestyle='dashed', marker='o', markerfacecolor='black', markersize=5)
axs[1].plot(ts2, color='red', linestyle='dashed', marker='o', markerfacecolor='black', markersize=5)

#plot eucledian distance
for id in range(len(ts1)):
    x_values = [id, id]
    y_values=[ts2[id], ts1[id]] #that captures vertical distance between two points
    axs[0].plot(x_values, y_values, 'y--')
    axs[0].set_title("Euclidean alignments")


#plot DTW distance
for p in optimal_path:
    idx1, idx2 = p
    y_values = [ts2[idx2], ts1[idx1]]
    x_values = [idx2, idx1]
    axs[1].plot(x_values, y_values, 'g--')
    axs[1].set_title("DTW alignments")

plt.subplots_adjust(hspace=0.5)
plt.subplots_adjust(wspace=0.5)
f = plt.gcf()
f.set_size_inches(8, 10)
plt.savefig("DTW_distance_example.png")
plt.close()
# print(DTWDistance(ts1,ts2))


#draw ts1 and t2
plt.plot(ts1, "b")
plt.savefig("ts1.png")
plt.close()
plt.plot(ts2, 'r')
plt.savefig("ts2.png")
plt.close()