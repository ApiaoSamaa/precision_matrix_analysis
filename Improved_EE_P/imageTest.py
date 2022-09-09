import numpy as np
from nilearn.image import load_img
from improvedEE import improvedEE
from nilearn import plotting
from nilearn.masking import compute_epi_mask
from nilearn.masking import apply_mask
import matplotlib.pyplot as plt
import time


dirct = ""
# Masking
print('Start reading amd masking images')
#path=dirct+'sub-01_func_sub-01_task-motorphotic_acq-1Daccel1shot_asl.nii.gz'
#path=dirct+'sub-01_ses-imageryTest01_func_sub-01_ses-imageryTest01_task-imagery_run-01_bold.nii.gz'
path='sub-01_func_sub-01_task-probabilisticclassification_run-01_bold.nii.gz'
img = load_img(path)
#mean_haxby = mean_img(img)
mask_img = compute_epi_mask(img)
masked_data=apply_mask(img,mask_img)
print(masked_data.shape)
print('Finished reading and masking images')
del img

# Computing sample covariance
print('Start computing covariance')
covariance=np.cov(masked_data,rowvar=False)
covariance/=np.max(abs(covariance))
del masked_data
print('Finished computing covariance')

# Estimating
print('Start estimating')
start=time.time()
estimator=improvedEE(covariance,0.3,0.2)
print("Costs : ",time.time()-start)
seq=estimator[1]
model=estimator[0]
del estimator
print('Finished estimating')

# Symmetrising
print('Start symmetrising')
varNum=model.shape[0]
for i in range(varNum):
    for j in range(i + 1, varNum):
        model[i, j] = model[j, i]
print('Finished symmetrising')


# Plotting images
plotting.plot_matrix(model,cmap='RdGy')
plt.savefig(fname=dirct+'precision(mask-imagenaryTest-%d)'%varNum)
del model

reorder_cov=covariance[seq]
del covariance
reorder_cov=reorder_cov[:,seq]
figure=plotting.plot_matrix(reorder_cov,cmap='OrRd')
plt.savefig(fname=dirct+'covariance(imagenaryTest-%d)'%varNum)
del reorder_cov