import numpy as np
from nilearn.image import load_img
from improvedEE import improvedEE
from EE import EE
from nilearn import plotting
import matplotlib.pyplot as plt
from sklearn.covariance import graphical_lasso
import time

dir=''

#Loading images
print('Start reading images')
path=dir+'sub-01_func_sub-01_task-rhymejudgment_bold.nii.gz' # path 表示需要读取的 fMRI 数据
img = load_img(path)
affine=img.affine
img=img.get_fdata()
print('Finished reading images')

#Pick out some layers of 4D images
zLen=1
data = img[:,:,:zLen,:]

# convert 4D data to 2D
print('Start reshaping images')
vol_shape=data.shape[:-1]
n_voxels = np.prod(vol_shape)
voxel_by_time = data.reshape(data.shape[-1],n_voxels)
print(voxel_by_time.shape)
print('Finished reshaping images')
del data

# compute sample covarianve
ano_cov=np.cov(voxel_by_time,rowvar=False)
ano_cov/=np.max(abs(ano_cov))

# 这里为了方便预设了参数 \nu 和 \lambda ; 如果效果不好，考虑 grid search 来选择参数。
# 对于不同的数据，应该哦选择不不同的参数。
regPar=0.01 # \lambda
for parameter in [0.7]: # \nu
    print('=========== Parameter %f ============'%parameter)

    # ==========================================
    #                  FST
    print('Start estimating FST......')
    print(time.asctime( time.localtime(time.time()) ))
    start = time.time()
    estimator = improvedEE(ano_cov, parameter, regPar)
    print("Costs : ", time.time() - start)
    seq = estimator[1]
    model = estimator[0]
    # model/=np.max(abs(model))
    print('Finished estimating')
    # stmmetrising
    print('Start symmetrising')
    varNum = model.shape[0]
    for i in range(varNum):
        for j in range(i + 1, varNum):
            model[i, j] = model[j, i]
    print('Finished symmetrising')
    # Plotting images for FST
    reorder_cov = ano_cov[seq]
    reorder_cov = reorder_cov[:, seq]
    # the sample covariance
    figure = plotting.plot_matrix(reorder_cov, cmap='bwr', vmax=1, vmin=-1)
    plt.savefig(fname=dir + 'FST-covariance(iaps-%d-%d)'%(varNum,int(parameter*100)))
    del reorder_cov
    # the estimated precision matrix
    plotting.plot_matrix(model, cmap='bwr', vmax=1, vmin=-1)
    plt.savefig(fname=dir + 'FST-precision(iaps-%d-%d)'%(varNum,int(parameter*100)))
    del model
    # ==========================================

    # ==========================================
    #                  EE
    print('\n Start estimating EE....')
    print(time.asctime(time.localtime(time.time())))
    start = time.time()
    estimator = EE(ano_cov, parameter, regPar)
    varNum = estimator.shape[0]
    print("Costs : ", time.time() - start)
    print('Finished estimating!')
    # the sample covariance
    figure = plotting.plot_matrix(ano_cov, cmap='bwr', vmax=1, vmin=-1)
    plt.savefig(fname=dir + 'EE-covariance(iaps-%d-%d)' % (varNum, int(parameter * 100)))
    # the estimated precision matrix
    plotting.plot_matrix(estimator, cmap='bwr', vmax=1, vmin=-1)
    plt.savefig(fname=dir + 'EE-precision(iaps-%d-%d)' % (varNum, int(parameter * 100)))
    del estimator
    # ==========================================

    # ==========================================
    #                  GLasso
    # print('\n Start estimating GLASSO...')
    # print(time.asctime(time.localtime(time.time())))
    # start = time.time()
    # estimator = graphical_lasso(ano_cov,regPar, max_iter = int(1e4))
    # varNum = estimator.shape[0]
    # print("Costs : ", time.time() - start)
    # print('Finished estimating!')
    # # the sample covariance
    # figure = plotting.plot_matrix(ano_cov, cmap='bwr', vmax=1, vmin=-1)
    # plt.savefig(fname=dirct + 'GLasso-covariance(iaps-%d-%d)' % (varNum, int(parameter * 100)))
    # # the estimated precision matrix
    # plotting.plot_matrix(estimator, cmap='bwr', vmax=1, vmin=-1)
    # plt.savefig(fname=dir + 'GLasso-precision(iaps-%d-%d)' % (varNum, int(parameter * 100)))
    # del estimator
    # ==========================================




