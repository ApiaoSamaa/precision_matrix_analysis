from numpy.linalg import norm

def _maxNorm(X):
    dim=X.shape[0]
    m=0
    for i in range(dim):
        for j in range(dim):
            if abs(X[i,j])>m:
                m=abs(X[i,j])
    return m


def accuracy(real,est):
    dim=real.shape[0]
    tp=0
    fp=0
    tn = 0
    fn=0
    for i in range(dim):
        for j in range(dim):
            if est[i,j]!=0:
                if real[i,j]!=0:
                    tp+=1
                else:
                    fp+=1
            else:
                if real[i,j]!=0:
                    fn+=1
                else:
                    tn+=1
    tpr=tp/(tp+fn)
    fpr=fp/(fp+tn)
    frobenius=norm(real - est, ord='fro')
    maximumnorm=_maxNorm(real-est)
    return {'TPR':tpr,'FPR':fpr,'FrobeniusNorm':frobenius,'MaxNorm':maximumnorm}

