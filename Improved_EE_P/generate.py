import numpy as np
import scipy.linalg
from scipy.sparse import dia_matrix
from tempfile import mkdtemp
import os.path as path
import matplotlib.pyplot as plt
from random import randint
from random import seed
import json

def random(p, blk, seed_num):
    #Calculate each block
    blk_size = int(p / blk)
    temps = []
    for index in range(blk):
        temp = np.eye(blk_size)
        for i in range(blk_size):
            for j in range(i + 1, blk_size):
                seed(i * j + seed_num)
                val = randint(6,10) *0.1
                seed(2 * i * j + seed_num)
                val *= (1 if randint(-4, 1)<0 else 0)
                # print(val)
                # val = val * (1 if randint(-1,1)<0 else -1)
                temp[i, j] = val
                temp[j, i] = val
        # temp = np.linalg.inv(temp)
        temps.append(temp)
    
    # Combine them into the final matrix
    filename = path.join(mkdtemp(), 'newfile.dat')
    result = np.memmap(filename, dtype='float32', mode='w+', shape=(p, p))
    # for i in range(p):
    #     for j in range(p):
    #         seed(i * j + seed_num)
    #         result[i][j] = randint(0,1) * 0.1 * (1 if randint(-1,1)<0 else -1)
    for index in range(blk):
        low=index*blk_size
        for i in range(blk_size):
            for j in range(i, blk_size):
                result[low + i, low + j] = temps[index][i,j]
                result[low + j, low + i] = temps[index][j,i]
        if index%10==0:
            print('Finished {} blocks'.format(index))
    # for i in range(p-blk_size):
    #     seed(i + seed_num)
    #     result[i, blk_size + i] = randint(0, 5) * 0.1 * (1 if randint(-1,1)<0 else -1)
    #     result[blk_size + i, i] = result[i, blk_size + i]
    return result

def generateBlk(n):
    result = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            val = 0.7 ** abs(i - j)
            result[i, j] = val
            result[j, i] = val
    return result

def circle(p, blk):
    blk_size = int(p / blk)
    blk_list = []
    for index in range(blk):
        each_blk = np.eye(blk_size)
        for i in range(blk_size):
            j = (i + 1) % blk_size
            each_blk[i, j] = 0.1
            each_blk[j, i] = 0.1
        blk_list.append(each_blk)
    return scipy.linalg.block_diag(*blk_list)

def grid(p, blk, row, col):
    blk_list = []
    each_blk = np.eye(row * col)
    for index in range(blk):
        for i in range(row - 1):
            for j in range(col):
                index = i * col + j
                each_blk[index, index + 1] = 0.05
                each_blk[index + 1, index] = 0.05
                each_blk[index, index + col] = 0.05
                each_blk[index + col, index] = 0.05
        blk_list.append(each_blk)
    return scipy.linalg.block_diag(*blk_list)

def imBalance(a, b, c, d):
    return scipy.linalg.block_diag(*[random(a), random(b), random(c), random(d)])

def weakGraph(p, blk=4):
    # Total 4 blocks and each 2 has connections
    size = int(p / blk)
    firstBlk = scipy.linalg.block_diag(random(size,blk), random(size,blk))
    secondBlk = scipy.linalg.block_diag(random(size,blk), random(size,blk))
    for i in range(10):
        firstBlk[i, size + i] = 0.5
        firstBlk[size + i, i] = 0.5
        secondBlk[i, size + i] = 0.5
        secondBlk[size + i, i] = 0.5
    return scipy.linalg.block_diag(firstBlk, secondBlk)


def addNoise(n, p):
    noise = np.random.randn(n, p)
    noise = noise - np.mean(noise)
    return noise
    


print('Start generating matrix')

blk=4
p = 200
T = 5
n = 500
test_n = 100
adjs = []
datasets = []
for t in range(T):
    # if p<3000:
    #     n=2*p
    # else:
    #     n=p
    mat = random(p, blk, t)
    plt.imshow(mat)
    plt.show()
    adjs.append(mat.tolist())
    
    if t == T-1:
        break
    print('Start generating data')
    np.random.seed(1028)
    data = np.random.multivariate_normal(mean=np.zeros((len(mat),)), cov=np.linalg.inv(mat), size=n, check_valid='ignore',
                                         tol=0.1)
    # plt.imshow(np.cov(data.T))
    # plt.show()
    datasets.append(data.tolist())
    print('Finished generating data  ', p)
    # print('Finished generating matrix   ', p)
    # np.savetxt(str.format('./random_%dvars_%dblks.csv' % (p, blk)),
    #            mat, delimiter=',')
np.random.seed(1028)
test_data = np.random.multivariate_normal(mean=np.zeros((len(mat),)), cov=np.linalg.inv(mat), size=test_n, check_valid='ignore',
                                         tol=0.1)
# plt.imshow(np.cov(test_data.T))
# plt.show()                                    
with open('./third_variation/input_data/new_random_{}dvars_{}dblks.json'.format(p, blk),"w+") as file:
    file.write(json.dumps(adjs[:-1])) 

print("labels of training data: ", np.array(adjs[:-1]).shape)

with open('./third_variation/input_data/new_random_{}dvars_{}dsamples.json'.format(p, n),"w+") as file:
    file.write(json.dumps(datasets)) 

print("shape of training data: ", np.array(datasets).shape)

print('Finished writing training data  ', p)
print("=========================================")

with open('./third_variation/input_data/new_random_{}dvars_{}dblks_test.json'.format(p, blk),"w+") as file:
    file.write(json.dumps(adjs[-1])) 
plt.imshow(adjs[-1])
plt.show()

print("labels of testing data: ", np.array(adjs[-1]).shape)
with open('./third_variation/input_data/new_random_{}dvars_{}dsamples_test.json'.format(p, test_n),"w+") as file:
    file.write(json.dumps(test_data.tolist())) 

print("labels of testing data: ", np.array(test_data).shape)


######################

# blk=20
# for p in [500,1000,2000,3000,5000,10000,15000]:
#     n=2*p
#     print('Finished generating matrix  ',p)
#     mat=circle(p,20)

#     np.savetxt(str.format('data/precision/circle_%dvars_20blks.csv'%p), mat, delimiter=',')
#     print('Finished writing matrix  ',p)
#     data = np.random.multivariate_normal(mean=np.zeros((1, p))[0], cov=np.linalg.inv(mat), size=n, check_valid='ignore',tol=0.1)
#     print('Finished generating data  ',p)
#     np.savetxt(str.format('data/X/circle_%dvars_20blks_%dsamples.csv'%(p,n)),
#                data, delimiter=',')
#     print('Finished writing data  ',p)
#     print("=========================================")


# blk=20
# for pair in [(5,5),(5,10),(10,10),(10,15),(10,25),(10,50),(15,50)]:
#     row,col=pair
#     p=row*col*20
#     n=2*p
#     mat = grid(p, 20,row,col)
#     np.savetxt(str.format('data/precision/grid_%dvars_20blks.csv' % p), mat, delimiter=',')
#     print('Finished writing matrix  ', p)
#     data = np.random.multivariate_normal(mean=np.zeros((1, p))[0], cov=np.linalg.inv(mat), size=n, check_valid='ignore',tol=0.1)
#     print('Finished generating data  ', p)
#     np.savetxt(str.format('data/X/grid_%dvars_20blks_%dsamples.csv' % (p, n)), data, delimiter=',')
#     print('Finished writing data  ', p)
#     print("=========================================")

'''
prePath = '/home/ubuntu/zjq/RFile/improved_EE/'
data = np.loadtxt(prePath + 'data/balance_10000vars_20blks_10000samples.csv', delimiter=',')
print('Finished reading')

rowNum, colNum = data.shape
mean = data.mean(axis=0)
data = data - mean
cov = np.dot(data.T, data)
cov /= np.max(abs(cov))
print('Finished normalizing')
np.savetxt(str.format('/home/ubuntu/zjq/RFile/improved_EE/cov/balance_%dvars_20blks_%dsamples.csv'%(10000,10000)), cov, delimiter=',')


for each in [[2500,2500,2500,2500],[3000,3000,2000,2000],[4000,2000,2000,2000],[5000,2000,2000,1000],
             [6000,2000,1000,1000],[7000,1000,1000,1000],[8000,1000,500,500],[9000,400,300,300],
             [9500,200,200,100],[9700,100,100,100],[9850,50,50,50],[9940,20,20,20],[9970,10,10,10],
             [9997,1,1,1]]:
    a,b,c,d=each
    mat=imBalance(a,b,c,d)
    print('Finished generating matrix')
    np.savetxt(str.format('/home/ubuntu/zjq/RFile/improved_EE/precision/imbalance_10000vars_(%d-%d-%d-%d).csv' % (a,b,c,d)), mat,delimiter=',')
    print('Finished writing matrix (%d-%d-%d-%d)'%(a,b,c,d))
    data = np.random.multivariate_normal(mean=np.zeros((1, 10000))[0], cov=np.linalg.inv(mat), size=20000, check_valid='ignore',tol=0.1)
    print('Finished generating data')
    rowNum, colNum = data.shape
    mean = data.mean(axis=0)
    data = data - mean
    cov = np.dot(data.T, data)
    cov /= np.max(abs(cov))
    print('Finished normalizing')
    np.savetxt(str.format('/home/ubuntu/zjq/RFile/improved_EE/cov/imbalance_10000vars_(%d-%d-%d-%d)_20000samples.csv' % (a,b,c,d)),cov,delimiter=',')
    # np.savetxt('samples.out',data, delimiter=',')
    print('Finished writing data')
    print("=========================================")
'''
# blk=20
# for p in [200]:
#     n=2*p
#     print('Finished generating matrix  ',p)
#     mat=weakGraph(p,blk = 4)

#     np.savetxt(str.format('./weak_%dvars_2blks.csv'%p), mat, delimiter=',')
#     print('Finished writing matrix  ',p)
#     data = np.random.multivariate_normal(mean=np.zeros((1, n))[0], cov=np.linalg.inv(mat), size=n, check_valid='ignore',tol=0.1)
#     print('Finished generating data  ',p)
#     rowNum, colNum = data.shape
#     mean = data.mean(axis=0)
#     data = data - mean
#     cov = np.dot(data.T, data)
#     cov /= np.max(abs(cov))
#     print('Finished normalizing')
#     np.savetxt(str.format('./weak_%dvars_2blks_%dsamples.csv'%(p,n)), cov, delimiter=',')
#     # np.savetxt('samples.out',data, delimiter=',')
#     print('Finished writing data  ',p)
#     print("=========================================")

