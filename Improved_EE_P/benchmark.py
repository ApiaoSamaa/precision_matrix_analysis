import numpy as np
import time
import sys
import json

import os
os.environ["MKL_NUM_THREAD"]="1"
os.environ["NUMEXPR_NUM_THREADS"]="1"
os.environ["OMP_NUM_THREADS"]="1"

from improvedEE import improvedEE
from evaluation import accuracy
from EE import EE

if __name__ == '__main__':
    thr=0.23
    result={}
    prePath = '/home/ubuntu/zjq/RFile/improved_EE/'
    graph = np.loadtxt(prePath + 'precision/balance_10000vars_20blks.csv', delimiter=',')
    cov = np.loadtxt(prePath + 'cov/balance_10000vars_20blks_10000samples.csv', delimiter=',')
    print('Finished reading data.')

    result['improved-EE']={}
    print('Improved EE:')
    start=time.time()
    Omega,var_seq=improvedEE(cov,thr,core_num=int(sys.argv[1]))
    #Omega, var_seq = improvedEE(cov, thr, core_num=1)
    result['improved-EE']['time']=time.time()-start
    reorder_graph=graph[var_seq]
    reorder_graph=reorder_graph[:,var_seq]
    Omega/=np.max(abs(Omega)) #Normalization
    #result['improved-EE']['accuracy']=accuracy(reorder_graph,Omega)
    print('Finishd improved EE')
    '''
    result['EE'] = {}
    print('EE:')
    start = time.time()
    ee_model=EE(cov,thr)
    ee_model /= np.max(abs(ee_model))
    result['EE']['time'] = time.time() - start
    #result['EE']['accuracy'] =accuracy(graph, ee_model)
    print('Finishd EE')
    '''
    print(result)


