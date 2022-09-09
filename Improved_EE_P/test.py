import os
os.environ["MKL_NUM_THREAD"]="32"
os.environ["NUMEXPR_NUM_THREADS"]="32"
os.environ["OMP_NUM_THREADS"]="32"

import numpy as np
import time

def generate(n):
	temp = np.eye(n)
	for i in range(n):
		for j in range(i + 1, n):
			val = 0.7 ** abs(i - j)
			temp[i, j] = val
			temp[j, i] = val
	return temp


for dim in [10000,20000,30000,40000]:
	print('=========== %d Variables ==========='%dim)
	print('Start generating martrix')
	mat=generate(dim)
	print('Fnished generating martrix')

	print('Start generating martrix')
	start=time.time()
	np.linalg.inv(mat)
	print('Time cost : ',time.time()-start)
	print('Finished generating martrix')
	del mat
