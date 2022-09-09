from thresholdFunc import hardThreshold,softThreshold
import numpy

def EE(cov,thrArgs=0.0,regPar=0.0,func=hardThreshold):
    thr_cov = func(cov, thrArgs)
    mat=numpy.linalg.inv(thr_cov)
    return softThreshold(mat,regPar)