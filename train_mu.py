# %%
# '''
# BRCA 300 PRAD 136 KIRC 146 LUAD 141 COAD 78
# 根据参考论文中num_pCR(pos sample) == 34, num_CR(neg sample) == 99,挑选数量差异稍大的两组分别作为正负例
# 此处我选取： LUAD : pos     BRCA : neg 
# '''
from re import X
import numpy as np
import json
import pandas as pd
from scipy.io import loadmat
from sklearn import preprocessing
import yaml

# config_dict = yaml.load(open("config.yaml","r"), Loader=yaml.FullLoader)
p_m = loadmat(file_name="fastmdmc_output/dimension100nonzero9984testing_output.mat")['X']
print(f"the shape of precision matrix is {p_m.shape}")

X_pos_fast = np.array(pd.read_csv("../processed_data/normalized_pos_fast_sample.csv"))
X_neg_fast = np.array(pd.read_csv("../processed_data/normalized_neg_fast_sample.csv"))

X_pos_test1 = np.array(pd.read_csv("../processed_data/normalized_pos_test_sample.csv"))
X_neg_test1 = np.array(pd.read_csv("../processed_data/normalized_neg_test_sample.csv"))

X_pos_train = np.array(pd.read_csv("../processed_data/normalized_pos_train_sample.csv"))
X_neg_train = np.array(pd.read_csv("../processed_data/normalized_neg_train_sample.csv"))

'''
要和baseline进行对比。把baseline方法放到baseline文件夹下
'''
p_m_pos_clime = np.array(pd.read_csv("../baseline/clime_result_pos_parameter0.01prediction.csv"))
p_m_neg_clime = np.array(pd.read_csv("../baseline/clime_result_neg_parameter0.01prediction.csv"))

p_m_pos_glasso = np.array(pd.read_csv("../baseline/glasso_result_pos_parameter0.01prediction.csv"))
p_m_neg_glasso = np.array(pd.read_csv("../baseline/glasso_result_neg_parameter0.01prediction.csv"))

p_m_pos_neigh = np.array(pd.read_csv("../baseline/neighborhood_selection_pos_prediction.csv",header=None))
p_m_neg_neigh = np.array(pd.read_csv("../baseline/neighborhood_selection_neg_prediction.csv",header=None))


print(f"the shape of X_pos and X_neg is {X_pos_fast.shape} and {X_neg_fast.shape}")
print(f"the shape of p_m_pos and p_m_neg is {p_m_pos_clime.shape} and {p_m_neg_clime.shape}")
print(f"the shape of p_m_pos and p_m_neg is {p_m_pos_glasso.shape} and {p_m_neg_glasso.shape}")
print(f"the shape of p_m_pos and p_m_neg is {p_m_pos_neigh.shape} and {p_m_neg_neigh.shape}")

pi_pos = X_pos_fast.shape[0]/(X_pos_fast.shape[0]+X_neg_fast.shape[0])
pi_neg = X_neg_fast.shape[0]/(X_pos_fast.shape[0]+X_neg_fast.shape[0])
# %%
class Deltas:
    def __init__(self,
                X_pos_test,
                X_neg_test,
                precision_matrix_pos,
                precision_matrix_neg,
                pi_pos = pi_pos,
                pi_neg = pi_neg,
                ):
        self.p_m_pos = precision_matrix_pos
        self.p_m_neg = precision_matrix_neg
        self.pi_pos = pi_pos
        self.mu_pos = None
        self.pi_neg = pi_neg
        self.mu_pos = None
        self.delta_pos_value = 0
        self.delta_neg_value = 0
        self.X_pos_test = X_pos_test
        self.X_neg_test = X_neg_test
        self.prediction = 0

    def fit_mu_k(self, X_train_pos, X_train_neg):
        self.mu_pos = np.sum(X_train_pos, axis=0)/X_train_pos.shape[0]
        self.mu_neg = np.sum(X_train_neg, axis=0)/X_train_neg.shape[0]
    
    def delta_pos(self, x_test):
        # skip log(pi_k) 
        # print(x_test.T.shape,self.p_m_pos.shape,self.mu_pos.shape,self.pi_pos)
        return x_test.T@self.p_m_pos@self.mu_pos - 0.5*self.mu_pos.T@self.p_m_pos@self.mu_pos + np.log10(self.pi_pos)
    
    def delta_neg(self, x_test):
        # skip log(pi_k) 
        return x_test.T@self.p_m_neg@self.mu_neg - 0.5*self.mu_neg.T@self.p_m_neg@self.mu_neg+ np.log10(self.pi_neg)
    
    
    def transform(self):
        X_test = np.r_['0',self.X_pos_test,self.X_neg_test]
        self.shuffle_ix = np.random.permutation(np.arange(X_test.shape[0]))
        self.X_test = X_test[self.shuffle_ix]

        num_sample = self.X_test.shape[0]
        self.delta_pos_value = np.array([self.delta_pos(self.X_test[i]) for i in range(num_sample)])
        self.delta_neg_value = np.array([self.delta_neg(self.X_test[i]) for i in range(num_sample)])
        '''
        predict 
        '''
        self.prediction = np.array([1 if self.delta_pos_value[i]>=self.delta_neg_value[i] else 0 for i in range(self.delta_pos_value.shape[0])])

    def accuracy(self):
        self.true_label = np.r_['0',np.ones(self.X_pos_test.shape[0]).tolist(), np.zeros(self.X_neg_test.shape[0]).tolist()][self.shuffle_ix]
        accurately_pred = self.true_label == self.prediction
        # print(f"shape of true_label and prediction is {true_label.shape},{self.prediction.shape}")
        acc = np.sum(accurately_pred)/accurately_pred.shape[0]
        return acc, accurately_pred
    
    # true_label = np.r_['0',np.ones(self.X_pos_test.shape[0]).tolist(), np.zeros(self.X_neg_test.shape[0]).tolist()]
    # 真实标签是按照顺序排列的，根据
    # 可知前 self.X_pos_test.shape[0] 个是正例，后 self.X_neg_test.shape[0] 个是负例,再进行random_shuffle
    def stack_evaluation_indicators(self):
        self.true_label = np.r_['0',np.ones(self.X_pos_test.shape[0]).tolist(), np.zeros(self.X_neg_test.shape[0]).tolist()][self.shuffle_ix]
        accurately_pred = self.true_label == self.prediction
        # print(f"If this is accurately pred")

        self.TP = np.count_nonzero(accurately_pred[self.true_label])
        self.FN = self.X_pos_test.shape[0] - self.TP
        self.TN = np.count_nonzero(accurately_pred[~self.true_label])
        self.FP = self.X_neg_test.shape[0] - self.TN


    # Specificity: TN/(TN+FP)
    # Sensitivity: TP/(TP+FN)
    # MCC: (TP*TN - FP*FN)/(sqrt(  (TP+FP)(TP+FN)(TN+FP)(TN+FN)  ))
    def Specificity_Sensitivity_MCC(self):
        self.stack_evaluation_indicators()
        return self.TN/(self.TN + self.FP), self.TP/(self.TP + self.FN), (self.TP*self.TN - self.FP*self.FN)/(np.sqrt((self.TP+self.FP)*(self.TP+self.FN)*(self.TN+self.FP)*(self.TN+self.FN)))
    
class COMPARE:
    def __init__(self,
                X_pos_test, 
                X_neg_test,
                X_pos_fast = X_pos_fast,
                X_neg_fast= X_neg_fast,
                p_m_pos_clime = p_m_pos_clime,
                p_m_neg_clime = p_m_neg_clime,
                p_m_pos_glasso = p_m_pos_glasso,
                p_m_neg_glasso=p_m_neg_glasso,
                p_m_pos_neigh=p_m_pos_neigh,
                p_m_neg_neigh=p_m_neg_neigh,
                p_m=p_m):
        self.X_pos_fast = X_pos_fast
        self.X_neg_fast = X_neg_fast
        self.clime = Deltas(precision_matrix_pos = p_m_pos_clime,precision_matrix_neg = p_m_neg_clime,X_pos_test=X_pos_test,X_neg_test=X_neg_test)
        self.glasso = Deltas(precision_matrix_pos = p_m_pos_glasso,precision_matrix_neg = p_m_neg_glasso,X_pos_test=X_pos_test,X_neg_test=X_neg_test)
        self.neigh = Deltas(precision_matrix_pos = p_m_pos_neigh,precision_matrix_neg = p_m_neg_neigh,X_pos_test=X_pos_test,X_neg_test=X_neg_test)
        self.simule = Deltas(precision_matrix_pos=p_m, precision_matrix_neg= p_m,X_pos_test=X_pos_test,X_neg_test=X_neg_test)

    def __fit_mu_k(self):
        self.clime.fit_mu_k(X_train_pos = self.X_pos_fast, X_train_neg = self.X_neg_fast)
        self.glasso.fit_mu_k(X_train_pos = self.X_pos_fast, X_train_neg = self.X_neg_fast)
        self.neigh.fit_mu_k(X_train_pos = self.X_pos_fast, X_train_neg = self.X_neg_fast)
        self.simule.fit_mu_k(X_train_pos = self.X_pos_fast, X_train_neg = self.X_neg_fast)
        self.clime.transform()
        self.glasso.transform()
        self.neigh.transform()
        self.simule.transform()

    def show_acc(self):
        self.__fit_mu_k()
        self.clime_score,_ = self.clime.accuracy()
        self.glasso_score,_ = self.glasso.accuracy()
        self.neigh_score,_ = self.neigh.accuracy()
        self.simule_score,_ = self.simule.accuracy()
        print(f"the accuracy of clime, glasso, neigh is {self.clime_score},{self.glasso_score},{self.neigh_score}")
        print(f"accuracy simule {self.simule_score}")
        print('-'*20)
        return self.clime_score,self.glasso_score,self.neigh_score,self.simule_score

    def clime_Speci_Sensi_MCC(self):
        self.clime_3scores = self.clime.Specificity_Sensitivity_MCC()
        return self.clime_3scores

    def glasso_Speci_Sensi_MCC(self):
        self.glasso_3scores = self.glasso.Specificity_Sensitivity_MCC()
        return self.glasso_3scores

    def neigh_Speci_Sensi_MCC(self):
        self.neigh_3scores = self.neigh.Specificity_Sensitivity_MCC()
        return self.neigh_3scores

    def simule_Speci_Sensi_MCC(self):
        self.simule_3scores = self.simule.Specificity_Sensitivity_MCC()
        return self.simule_3scores
    
    def show_3scores(self):
        self.__fit_mu_k()
        self.clime_Speci_Sensi_MCC(),self.glasso_Speci_Sensi_MCC(),self.neigh_Speci_Sensi_MCC(),self.simule_Speci_Sensi_MCC()
        print(f"the Specificity of self.clime, self.glasso, self.neigh is {self.clime_3scores[0]},{self.glasso_3scores[0]},{self.neigh_3scores[0]}")
        print(f"the speci of self.simule {self.simule_3scores[0]}")
        print('-'*20)
        print(f"the Sensitivity of self.clime, self.glasso, self.neigh is {self.clime_3scores[1]},{self.glasso_3scores[1]},{self.neigh_3scores[1]}")
        print(f"the sensi of self.simule {self.simule_3scores[1]}")
        print('-'*20)
        print(f"the MCC of self.clime, self.glasso, self.neigh is {self.clime_3scores[2]},{self.glasso_3scores[2]},{self.neigh_3scores[2]}")
        print(f"the of self.simule {self.simule_3scores[2]}")
        print('-'*20)
        print('-'*20)
        print(f"the speci, sensi, MCC of self.simule is {self.simule_3scores[0]},{self.simule_3scores[1]},{self.simule_3scores[2]}")


howmuch_sample_in_test = 4

if howmuch_sample_in_test == 0:
    X_pos_test = np.r_['0', X_pos_test1.tolist()]
    X_neg_test = np.r_['0', X_neg_test1.tolist()]
elif howmuch_sample_in_test == 1:
    X_pos_test = np.r_['0', X_pos_test1.tolist(), X_pos_train.tolist()]
    X_neg_test = np.r_['0', X_neg_test1.tolist(), X_neg_train.tolist()]
else:
    X_pos_test = np.r_['0', X_pos_test1.tolist(), X_pos_train.tolist(), X_pos_fast.tolist()]
    X_neg_test = np.r_['0', X_neg_test1.tolist(), X_neg_train.tolist(), X_neg_fast.tolist()]

# %%
a = COMPARE(X_pos_test=X_pos_test,X_neg_test=X_neg_test,X_pos_fast=X_pos_fast,X_neg_fast=X_neg_fast)
a.show_3scores()

# b = COMPARE(X_pos_test=X_pos_test,X_neg_test=X_neg_test,X_pos_fast=X_neg_fast,X_neg_fast=X_pos_fast)
# b.show_3scores()

# %%
