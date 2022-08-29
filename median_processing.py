# %%
from os import abort
import numpy as np
import json
import pywt
import matplotlib.pyplot as plt


# load the training results
def threshold_selection(data, nonzero_num):
    """
    data: Matrix
    nonzero_num: the number of nonzero elements to be preserved
    GOAL: return the threshold value that keeps the specified nonzero number of data
    """
    if np.sum(data!=0) < nonzero_num:
        print("Threshold is not necessary.")
        return 
    data = abs(data.flatten())
    sorted_data = sorted(data, reverse=True)
    return sorted_data[int(nonzero_num)]
def findWeight(matrix):
    p = matrix.shape[0]
    weight = 0.1
    weightMatrix = matrix + weight * np.identity(p)
    while np.all(np.linalg.eigvals(weightMatrix)>=0) == False:
        weight += 0.1
        weightMatrix = matrix + weight * np.identity(p)
    return weight

rf = open ("ref.txt", mode= "w")

if __name__ == '__main__':

    print("=======generating fastmdmc data========")

    p = 100 # feature num
    # 导入训练数据
    with open("graph_030001.json".format(p), "r") as file:
        jsonfile = json.load(file)
    # 处理json文件中的share部分
    predicted_common_part = jsonfile['share']
    predicted_common_part = np.array(predicted_common_part)
    predicted_common_part = predicted_common_part.reshape((p, p))
    # 得到mask_matrix和inv_mask matrix，代表着论文中sgn(W)和mat{1}-sgn(W)
    diagonal_elimination = np.ones(shape=(p, p)) - np.identity(p)
    mask_matrix = np.copy(predicted_common_part)
    mask_matrix = np.where(mask_matrix!=0, 1, 0) * diagonal_elimination
    print("the number of nonzero entries in the common substructure:", np.sum(mask_matrix))
    print("the number of nonzero entries in the common substructure:", np.sum(mask_matrix), file= rf)

    inv_mask_matrix = np.ones(mask_matrix.shape) - mask_matrix
    inv_mask_matrix = inv_mask_matrix * diagonal_elimination
    # load the testing data
    with open("../processed_data/fast_cov.json".format(p), "r") as file:
        jsonfile2 = json.load(file)
    testing_input_cov = np.array(jsonfile2['fast']) * diagonal_elimination
    # num的数量是指threshold_cov中nonzero_entry的数量，可以人工调节
    for num in range(int(np.sum(mask_matrix)), int(p * p), 100):
    # for num in range(int(np.sum(mask_matrix)), int(np.sum(mask_matrix))+10, 1):
        nonzero_edges = num
        print("===========generating new data===========")
        print("nonzero edges:", nonzero_edges)
        # set sparsity degree
        keep_edges_num = nonzero_edges - np.sum(mask_matrix)
        threshold = threshold_selection(testing_input_cov, keep_edges_num)
        threshold_cov = pywt.threshold(testing_input_cov, threshold, mode='soft')
        # print(threshold_cov)
        print(np.sum(threshold_cov!=0))
        print("threshold:", threshold)
        print(np.sum(threshold_cov!=0), file= rf)
        print("threshold:", threshold, file= rf)
        # last_input_matrix是testing部分的输入
        last_input_matrix = mask_matrix * testing_input_cov + inv_mask_matrix * threshold_cov
        weight = findWeight(last_input_matrix)
        last_input_matrix = last_input_matrix + weight * np.identity(p)
        filename = "./fastmdmc_input/dimension"+str(p)+"nonzero" +str(nonzero_edges) + "fastmdmc_input_cov.json"
        with open(filename, "w") as file:
            file.write(json.dumps(last_input_matrix.tolist()))
            print("======The file has been written!=======")

rf.close()

# %%
