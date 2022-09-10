# %%
from random import shuffle
import numpy as np
import pandas as pd
import yaml
import argparse
import json
import os
import sys
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from scipy import stats
# import wandb
import json

config_dict = yaml.load(stream=open("config.yaml"),Loader=yaml.FullLoader)
"""
@Dscrp: Just load raw data
@Param: raw data file name
@Rtrns: name of sample(for label selection), content 
"""
def load_data(file_name = config_dict["FILE_PATH"]["RAW_DATA"],rtn_name = False):
    whl = np.array(pd.read_excel(file_name,skiprows=4))
    name = whl[0][1:]
    content = whl[1:,1:]
    if rtn_name:
        return name, content
    return content.T

def load_label(file_name = config_dict["FILE_PATH"]["LABEL"],rtn_name = False):
    name = np.array(pd.read_excel(file_name,skiprows=2)["idtext"])
    content = np.array(pd.read_excel(file_name,skiprows=2)["pCRtxt"])
    # label = np.array([1 if c=='pCR' else 0 for c in content]) 
    # if rtn_name:
    #     return name, label
    # return label

    pos_i = np.array([True if c==config_dict["POS_NAME"] else False for c in content])
    neg_i = np.array([True if c==config_dict["NEG_NAME"] else False for c in content])
    return pos_i, neg_i #分别选出pos样本 和neg样本



'''
Load data
'''
if config_dict["PRODUCE_DATA"]:
    label_tuple = load_label()
    raw_X= np.r_["0",load_data()[label_tuple[0]],load_data()[label_tuple[1]]]
    np.save("./process/raw_X.npy",raw_X)
    np.save("./process/label_tuple.npy",label_tuple)
else:
    raw_X = np.load("./process/raw_X.npy",allow_pickle=True)
    label_tuple = np.load("./process/label_tuple.npy",allow_pickle=True)



pos_num = np.count_nonzero(label_tuple[0])
neg_num = np.count_nonzero(label_tuple[1])
pos = raw_X[:pos_num]
neg = raw_X[-neg_num:]
label = np.r_["0",np.ones(pos_num),np.zeros(neg_num)]


# %%
"""
@Dscrp: Using ttest to find correct feature collumn idx
@Param: X_pos, X_neg, how much collumn to choose
@Rtrns: idx of features to select
"""

def find_p_value(a, b, top_num = config_dict["FEATURE"]["NUM_TTEST"]):
    p_value = stats.ttest_ind(a, b, axis=0)[1]
    # top_p_idx = np.argsort(p_value)[:top_num]
    top_p_idx = np.argsort(p_value)[12000:top_num+12000]
    print(f"we selected {top_num} collumn with the smallest p-values." )
    return top_p_idx

cnt = 0

shuffle_idx = np.random.permutation(label.shape[0])
fn = "shuffle"+str(cnt)+".txt"
with open(fn,"a+") as f:
    print(shuffle_idx,file = f)
    f.close()


# X = raw_X[:,find_p_value()][shuffle_idx]
# y = label[shuffle_idx]
X = raw_X[shuffle_idx]
y = label[shuffle_idx]

'''
解析变成json文件的数据结构，对于json而言并不知道 np.ndarray 是什么
'''
from json import JSONEncoder
class NpArrEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)

def save_json(file_name, dict):
    with open(file_name, "w") as f:
        o = json.dumps(dict, indent = "4", cls = NpArrEncoder)
        json.dump(dict, f, indent = 4, cls= NpArrEncoder)


cnt = 0
from tqdm import tqdm

from sklearn.model_selection import RepeatedStratifiedKFold
rskf = RepeatedStratifiedKFold(n_splits=8, n_repeats=1,random_state= 8)
for extra_idx, fast_idx in tqdm(rskf.split(X, y),desc=""):
    # print(f"EXTRA:{extra_idx.shape}, TEST:{test_idx.shape}")
    X_fast_raw = X[fast_idx]
    y_fast = y[fast_idx]
    print(f"fast num :{X_fast_raw.shape}")
    X_fast_raw_pos = X_fast_raw[y_fast.astype(bool)]
    X_fast_raw_neg = X_fast_raw[~(y_fast.astype(bool))]

    p_col = find_p_value(X_fast_raw_pos.tolist(),X_fast_raw_neg.tolist())
    extra_X = X[extra_idx]


    # print(f"TRAIN:{train_idx.shape}, FAST:{fast_idx.shape},TEST:{test_idx.shape}")
    rskf_extra = RepeatedStratifiedKFold(n_splits=8, n_repeats=1,random_state= 8)
    for train_idx, test_idx in rskf_extra.split(extra_X,y[extra_idx]):
        cnt += 1
        y_train = y[extra_idx][train_idx]
        X_train_raw = extra_X[train_idx]
        y_test = y[extra_idx][test_idx]
        X_test_raw = extra_X[test_idx]
        X_train = preprocessing.scale(X_train_raw[:,p_col])
        X_combine = preprocessing.scale(np.r_["0",X_fast_raw[:,p_col], X_test_raw[:,p_col]])
        X_fast = X_combine[:y_fast.shape[0]]
        X_test = X_combine[-y_test.shape[0]:]
        print(f"train num :{X_train.shape}")
        print(f"test num :{X_test.shape}")

        '''
        利用样本进行选择的时候，需要记得把int类型转化为bool。int会保留原大小.
        '''
        i_train_p = y_train.astype(bool)
        i_fast_p = y_fast.astype(bool)

        train_cov = dict(pos=np.cov(X_train[i_train_p],rowvar=False),neg = np.cov(X_train[~i_train_p],rowvar=False))
        fast_cov = dict(fast=np.cov(X_fast,rowvar=False))

        tn_cov_filepath = os.getcwd()+"\\process\\cov_train"+str(cnt)+".json"
        tn_smpl_filepath = os.getcwd()+"\\process\\sample_train"+str(cnt)+".npy"
        fst_cov_filepath = os.getcwd()+"\\process\\cov_fast"+str(cnt)+".json"
        fst_smpl_filepath = os.getcwd()+"\\process\\sample_fast"+str(cnt)+".npy"
        test_smpl_filepath = os.getcwd()+"\\process\\sample_test"+str(cnt)+".npy"

        # tn_cov_filepath = os.getcwd()+"\\process2\\cov_train"+str(cnt)+".json"
        # tn_smpl_filepath = os.getcwd()+"\\process2\\sample_train"+str(cnt)+".npy"
        # fst_cov_filepath = os.getcwd()+"\\process2\\cov_fast"+str(cnt)+".json"
        # fst_smpl_filepath = os.getcwd()+"\\process2\\sample_fast"+str(cnt)+".npy"
        # test_smpl_filepath  = os.getcwd()+"\\process2\\sample_test"+str(cnt)+".npy"

        save_json(tn_cov_filepath,train_cov)
        save_json(fst_cov_filepath,fast_cov)
        np.save(tn_smpl_filepath, zip(X_train, y_train))
        np.save(fst_smpl_filepath, zip(X_fast, y_fast))
        np.save(test_smpl_filepath,zip(X_test, y_test))



# %%


# current_path = os.path.dirname(os.path.realpath(sys.argv[0]))
# parser = argparse.ArgumentParser()

# parser.add_argument('--wv_model', type=str, default='enwiki_model/')
# parser.add_argument('--io_dir', type=str, default=os.path.join(current_path, 'T2D_IO'))
# parser.add_argument('--test_name', type=str, default='Limaye', help='T2D or Limaye or Wikipedia')
# parser.add_argument('--use_surrounding_columns', type=str, default='yes', help='yes or no')
# parser.add_argument('--use_property_vector', type=str, default='yes', help='yes or no')
# parser.add_argument('--prop2vec_dim', type=int, default=422)
# parser.add_argument('--algorithm', type=str, default='LR', help='LR or RF-n or MLP')
# parser.add_argument('--score_name', type=str, default='hnnc234', help='hnnc234, hnnc234r23, hnnc23')
# parser.add_argument('--micro_table_size', type=str, default='5,4')

# FLAGS, unparsed = parser.parse_known_args()
# print(FLAGS)

# # %%

# t_and_p = stats.ttest_ind([[1,2,3],[2,2,3]],[[0,0,8],[8,0,2]],axis=1)

# %%




    




# %%


# current_path = os.path.dirname(os.path.realpath(sys.argv[0]))
# parser = argparse.ArgumentParser()

# parser.add_argument('--wv_model', type=str, default='enwiki_model/')
# parser.add_argument('--io_dir', type=str, default=os.path.join(current_path, 'T2D_IO'))
# parser.add_argument('--test_name', type=str, default='Limaye', help='T2D or Limaye or Wikipedia')
# parser.add_argument('--use_surrounding_columns', type=str, default='yes', help='yes or no')
# parser.add_argument('--use_property_vector', type=str, default='yes', help='yes or no')
# parser.add_argument('--prop2vec_dim', type=int, default=422)
# parser.add_argument('--algorithm', type=str, default='LR', help='LR or RF-n or MLP')
# parser.add_argument('--score_name', type=str, default='hnnc234', help='hnnc234, hnnc234r23, hnnc23')
# parser.add_argument('--micro_table_size', type=str, default='5,4')

# FLAGS, unparsed = parser.parse_known_args()
# print(FLAGS)

# # %%

# t_and_p = stats.ttest_ind([[1,2,3],[2,2,3]],[[0,0,8],[8,0,2]],axis=1)

# %%
