import numpy as np

def hardThreshold(mat,thr):
    result=mat.copy()
    dim=(mat.shape)[0]
    for i in range(dim):
        for j in range(i+1,dim):
            if abs(result[i,j])<=thr:
                result[i,j]=result[j,i]=0
    return result

def softThreshold(mat,thr):
    result = mat.copy()
    dim = (mat.shape)[0]
    for i in range(dim):
        for j in range(i + 1, dim):
            tmp=0
            if abs(result[i, j]) > thr:
                tmp=1 if result[i,j]>0 else -1
                tmp*=(abs(result[i,j])-thr)
            result[i, j] = result[j, i] = tmp
    return result
