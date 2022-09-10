import imp
from thresholdFunc import hardThreshold,softThreshold
from scipy.sparse.csgraph import connected_components
import scipy.linalg
import numpy.linalg
import numpy as np


def _comps_inv(mat_list):
    comp_inv=[numpy.linalg.inv(mat) for mat in mat_list]
    return comp_inv
    #return numpy.linalg.inv(mat_list)


#TODO:Args for each thresholding function
def improvedEE(cov,thrArgs=0.0,regPar=0.0,func=hardThreshold):
    '''
    FST.
    :param cov: Covariance matrix.
    :param thrArgs: Parameter for thresholding function (\nu).
    :param regPar: Parameter for soft-thresholding (\lambda).
    :param func: The thresholding function we used to obtain the topological graph structure.harThreshold or softThreshold.
    :return: (Omega, var_seq, nonzero_seq)
              Omega: the estimated precision matrix.
              var_seq: Reordered variable sequence.
              nonzero_seq: The non-isolated variable. 
    '''
    var_num=(cov.shape)[0]
    thr_cov=func(cov,thrArgs)
    compNum,bins=connected_components(thr_cov,directed=False)
    compSize=[0 for i in range(compNum)]
    for i in range(len(bins)):
        compSize[bins[i]]+=1
    var_seq=[(i,bins[i]) for i in range(var_num)]
    var_seq=[each[0] for each in sorted(var_seq,key=lambda x:x[1],reverse=False)] # The reordered sequence of vars
    reorder_cov=cov[var_seq]
    reorder_cov=reorder_cov[:,var_seq]
    #Split each connected component
    offset = 0
    comp_cov = []

    nonzero_seq=[]

    for index in range(compNum):
        start = offset
        end = compSize[index] + offset
        offset = end
        each_comp=np.linalg.inv(reorder_cov[start:end, start:end])
        if not each_comp.shape==(1,1):
            each_comp /= np.max(abs(each_comp))
            each_comp = softThreshold(each_comp, regPar)
            nonzero_seq.extend([ i for i in range(start,end)])
        comp_cov.append(each_comp)
    Omega=scipy.linalg.block_diag(*comp_cov)
    #Omega = func(Omega, regPar)
    return Omega,var_seq,nonzero_seq


def getConnect(cov,coord,thrArgs=0.0,regPar=0.0,func=hardThreshold):

    var_num = (cov.shape)[0]
    thr_cov = func(cov, thrArgs)
    compNum, bins = connected_components(thr_cov, directed=False)
    print('Component number ',compNum)
    component={}
    for index in range(var_num):
        if bins[index] not in component:
            component[bins[index]] = [index]
        else:
            component[bins[index]].append(index)

    useless_blk=[]
    useless_index=[]
    num=0
    for each in component:
        if len(component[each])==1:
            useless_blk.append(each)
            useless_index.append(component[each][0])
        else:
            num+=len(component[each])

    component={each:component[each] for each in component if each not in useless_blk}
    var_seq = []
    for each in component:
        var_seq.extend(component[each])

    coord=[coord[index] for index in var_seq]

    colors = ['b','r' , 'c', 'y', 'k', 'm', 'g']
    offset = 0
    node_color = []
    for each in component:
        temp = colors[offset % 7]
        for index in component[each]:
            node_color.append(temp)
        offset+=1

    return component,num,coord,node_color


if __name__ == '__main__':
    import time
    import networkx as nx
    from timeit import default_timer as timer
    import json
    import matplotlib.pyplot as plt
    dimension = 200
    # with open("./third_variation/input_data/random_"+str(dimension) + "dvars_" + str(int(dimension/2)) + "dsamples_test.json", "r") as file:
    #     jsonfile = json.load(file)

    # test_data = np.array(jsonfile).reshape((50, 100))

    # with open("./third_variation/input_data/random_"+str(dimension) + "dvars_" + str(500) + "dsamples.json", "r") as file:
    #     jsonfile = json.load(file)

    # test_data = np.array(jsonfile)[0].reshape((500, 100))

    ###########test time############
    for sample in np.arange(100, 1001, 100):
        filename = "/Users/jojo/Documents/VscodeProjects/tx/TestTime/time_input_data/dimension{}sample{}data.json".format(dimension, sample)
        with open(filename, 'r') as file:
            X = np.array(json.load(file))

        start=timer()
        cov = np.cov(X.T)
        # cov = np.random.rand(dimension, dimension)
        # for i in range(5, 15):
            # thr = i * 0.1
        thr = 0.3
        Omega, __, __ = improvedEE(cov, thr)
            # plt.imshow(Omega)
            # plt.show()
        # Xt= np.transpose(X)
        # M = np.eye(n) + (w2/w1)*(X@omega@Xt)
        # res = (1/w1)*omega - (w2/w1)*(omega@Xt@np.transpose(M)@X@omega)
        print('Time cost : ',timer()-start)

