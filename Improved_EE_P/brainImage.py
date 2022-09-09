import numpy as np
from nilearn.image import load_img
from improvedEE import improvedEE,getConnect
from nilearn import plotting
from nibabel.affines import apply_affine
import matplotlib.pyplot as plt
import time

#dirct='/home/ubuntu/zjq/pythonFile/'
dirct=''

#Loading images
print('Start reading images')
#path=dirct+'sub-01_func_sub-01_task-iaps_run-01_bold.nii'
path=dirct+'sub-01_ses-imageryTest01_func_sub-01_ses-imageryTest01_task-imagery_run-01_bold.nii.gz'
img = load_img(path)
affine=img.affine
img=img.get_fdata()
print('Finished reading images')


#Pick out some layers of 4D images
zLen=1
base=0
# Get global space coordinate
coord=[]
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        for k in range(zLen):
            coord.append(apply_affine(affine,[i,j,k]))

#4D to 2D
data = img[:,:,base:base+zLen,:]
print('Start reshaping images')
vol_shape=data.shape[:-1]
n_voxels = np.prod(vol_shape)
voxel_by_time = data.reshape(data.shape[-1],n_voxels)
print(voxel_by_time.shape)
print('Finished reshaping images')
#del data

# Get sample covariance
ano_cov=np.cov(voxel_by_time,rowvar=False)
ano_cov/=np.max(abs(ano_cov))

#plotting.plot_matrix(ano_cov,cmap='bwr')
#plotting.show()
# Estimating
regPar=0.2
parameter=0.535

print('Start estimating FST')
component,num,coord,node_color=getConnect(ano_cov,coord, parameter, regPar)
print('Component number after extraction ',len(component.keys()))

mat=np.eye(num)

plotting.plot_connectome(mat,coord,node_color=node_color,node_size=1.5,
                         display_mode='y',title='Coronal')
plotting.show()

'''
coord=[coord[i] for i in nonzero_seq]

model = model[nonzero_seq]
model = model[:, nonzero_seq]

del estimator
#del ano_cov
print('Finished estimating')

print('Start symmetrising')
varNum = model.shape[0]
for i in range(varNum):
    for j in range(i + 1, varNum):
         model[i, j] = model[j, i]
print('Finished symmetrising')


# Plotting images
plotting.plot_connectome(model, coord, title='Connectome',display_mode='z',
                         node_size=0.25,edge_vmax=1,edge_vmin=-1,edge_cmap='bwr',
                         edge_threshold=0.8)
plotting.show()
'''

