setwd('C:/Apiao/Projects/shared_graph_cancer_analysis')
library(simule)
library(rjson)
library(philentropy)
library(parallel)

graphics.off()
par(ask = F)
par(mfrow = c(1, 1))

### 导入数据，数据的格式为t*p*p，其中t是训练数据集的数量，p是特征数量
data = fromJSON(file = "processed_data/train_cov.json")
t = length(data) # 数据集数量
p = row_num = length(data[[1]]) # 特征数量
print(t)
print(p)

# 规范数据格式，方便计算
input_matrixs = list()
for (i in 1:t) {
  t = matrix(c(1:p * p), nrow = p, ncol = p)
  for (j in 1:p) {
    for (k in 1:p) {
      t[j, k] = data[[i]][[j]][[k]]
    }
  }
  input_matrixs[[i]] = t
}
out = 1
# epsilon和lambda是simule算法的参数
# 在没有true label的情况下，没法通过某个衡量指标判断参数是否合适，以下参数并不一定合适
epsilon = 0.3
lambda = 0.001
result = simule(input_matrixs, lambda, epsilon, covType = "cov", TRUE) # 训练
# result数据包含两部分内容
print(result[['share']])
print(result[['Graphs']])
# 保存数据
cat(toJSON(result), file = "graph_03_0001.json")

# epsilon和lambda是simule算法的参数
# 在没有true label的情况下，没法通过某个衡量指标判断参数是否合适，以下参数并不一定合适
epsilon = 0.4
lambda = 0.09
result = simule(input_matrixs, lambda, epsilon, covType = "cov", TRUE) # 训练
# result数据包含两部分内容
print(result[['share']])
print(result[['Graphs']])
# 保存数据
cat(toJSON(result), file = "graph_04_009.json")



