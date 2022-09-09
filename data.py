# %%
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


        # wandb.init(
        #     project="pCR dataset",
        #     name=f"experiment_1",
        #     config={
        #         "ttest_feature": config_dict["FEATURE"]["NUM_TTEST"],
        #         "dataset": "MDA133"
        #     }
        # )
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
    label = np.array([1 if c=='pCR' else 0 for c in content]) 
    if rtn_name:
        return name, label
    return label

'''
Load data
'''
if config_dict["PRODUCE_DATA"]:
    raw_X = load_data()
    label = load_label()
    np.save("./process/raw_X.npy",raw_X)
    np.save("./process/label.npy",label)
else:
    raw_X = np.load("./process/raw_X.npy",allow_pickle=True)
    label = np.load("./process/label.npy",allow_pickle=True)


# wandb.login()


# %%
pos = raw_X[label.astype(bool)]
neg = raw_X[~label.astype(bool)]

"""
@Dscrp: Using ttest to find correct feature collumn idx
@Param: X_pos, X_neg, how much collumn to choose
@Rtrns: idx of features to select
"""
def find_p_value(a=pos.tolist(), b= neg.tolist(), top_num = config_dict["FEATURE"]["NUM_TTEST"]):
    p_value = stats.ttest_ind(a, b, axis=0)[1]
    top_p_idx = np.argsort(p_value)[:top_num]
    print(f"we selected {top_num} collumn with the smallest p-values." )
    return top_p_idx



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




# %%
from tqdm import tqdm
X = raw_X[:,find_p_value(top_num=config_dict["FEATURE"]["NUM_TTEST2"])]
y = label

cnt = 0

from sklearn.model_selection import RepeatedStratifiedKFold
rskf = RepeatedStratifiedKFold(n_splits=4, n_repeats=2,random_state= 3)
for train_idx, extra_idx in tqdm(rskf.split(X, y),desc=""):
    # print(f"EXTRA:{extra_idx.shape}, TEST:{test_idx.shape}")
    X_train = preprocessing.scale(X[train_idx])
    y_train = y[train_idx]
    extra_X = preprocessing.scale(X[extra_idx])

    # print(f"TRAIN:{train_idx.shape}, FAST:{fast_idx.shape},TEST:{test_idx.shape}")
    rskf_extra = RepeatedStratifiedKFold(n_splits=3, n_repeats=5,random_state= 8)
    for fast_idx, test_idx in rskf_extra.split(extra_X,y[extra_idx]):

        cnt += 1
        X_fast = extra_X[fast_idx]
        y_fast = y[extra_idx][fast_idx]
        X_test = extra_X[test_idx]
        y_test = y[extra_idx][test_idx]

        '''
        利用样本进行选择的时候，需要记得把int类型转化为bool。int会保留原大小.
        '''
        i_train_p = y_train.astype(bool)
        i_fast_p = y_fast.astype(bool)

        train_cov = dict(pos=np.cov(X_train[i_train_p],rowvar=False),neg = np.cov(X_train[~i_train_p],rowvar=False))
        fast_cov = dict(pos=np.cov(X_fast[i_fast_p],rowvar=False),neg = np.cov(X_fast[~i_fast_p],rowvar=False))

        # tn_cov_filepath = os.getcwd()+"\\process\\cov_train"+str(cnt)+".json"
        # tn_smpl_filepath = os.getcwd()+"\\process\\sample_train"+str(cnt)+".npy"
        # fst_cov_filepath = os.getcwd()+"\\process\\cov_fast"+str(cnt)+".json"
        # fst_smpl_filepath = os.getcwd()+"\\process\\sample_fast"+str(cnt)+".npy"
        # test_smpl_filepath  = os.getcwd()+"\\process\\sample_test"+str(cnt)+".npy"

        tn_cov_filepath = os.getcwd()+"\\process2\\cov_train"+str(cnt)+".json"
        tn_smpl_filepath = os.getcwd()+"\\process2\\sample_train"+str(cnt)+".npy"
        fst_cov_filepath = os.getcwd()+"\\process2\\cov_fast"+str(cnt)+".json"
        fst_smpl_filepath = os.getcwd()+"\\process2\\sample_fast"+str(cnt)+".npy"
        test_smpl_filepath  = os.getcwd()+"\\process2\\sample_test"+str(cnt)+".npy"

        save_json(tn_cov_filepath,train_cov)
        save_json(fst_cov_filepath,fast_cov)
        np.save(tn_smpl_filepath, zip(X_train, y_train))
        np.save(fst_smpl_filepath, zip(X_fast, y_fast))
        np.save(test_smpl_filepath,zip(X_test, y_test))













    




# %%


current_path = os.path.dirname(os.path.realpath(sys.argv[0]))
parser = argparse.ArgumentParser()

parser.add_argument('--wv_model', type=str, default='enwiki_model/')
parser.add_argument('--io_dir', type=str, default=os.path.join(current_path, 'T2D_IO'))
parser.add_argument('--test_name', type=str, default='Limaye', help='T2D or Limaye or Wikipedia')
parser.add_argument('--use_surrounding_columns', type=str, default='yes', help='yes or no')
parser.add_argument('--use_property_vector', type=str, default='yes', help='yes or no')
parser.add_argument('--prop2vec_dim', type=int, default=422)
parser.add_argument('--algorithm', type=str, default='LR', help='LR or RF-n or MLP')
parser.add_argument('--score_name', type=str, default='hnnc234', help='hnnc234, hnnc234r23, hnnc23')
parser.add_argument('--micro_table_size', type=str, default='5,4')

FLAGS, unparsed = parser.parse_known_args()
print(FLAGS)

# %%

t_and_p = stats.ttest_ind([[1,2,3],[2,2,3]],[[0,0,8],[8,0,2]],axis=1)

# %%
