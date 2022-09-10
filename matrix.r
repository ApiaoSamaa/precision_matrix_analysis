setwd('C:/Apiao/Projects/precision_matrix_analysis')
# setwd('C:/Apiao/Projects/BRCA')
sink(file = "record0909.txt")
library(simule)
library(rjson)
library(philentropy)
library(parallel)
graphics.off()
par(ask = F)
par(mfrow = c(1, 1))
num = 1
### 导入数据，数据的格式为t*p*p，其中t是训练数据集的数量，p是特征数量
for (i in 10:11){
  loadFilePath = paste("./process/cov_train",i,".json",sep="")
  data = fromJSON(file = loadFilePath)
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
  
for (epsilon in c(0.3,0.4,0.01,0.015,0.02,0.03,0.05, 0.06)) {
#for (epsilon in c(0.0009, 0.001, 0.0015,0.002)) {
    print("=====================")
    for (lambda in c(0.09,0.001,0.01,0.0009,0.001,0.002,0.03)) {
      # 输出配置
      cat("epsilon:", epsilon, "lambda", lambda, "\n")
      
      result = simule(input_matrixs, lambda, epsilon, covType = "cov", TRUE) # 训练
      # result数据包含两部分内容
      # print(result[['share']])
      # print(result[['Graphs']])
      # 输出share的非零个数
      share = result[["share"]]
      cat("The number of nonzero entries:", sum(share!=0), "\n")
      
      # 保存数据
      filename = paste("./process/graphs/graph_",i,"_eps",epsilon, "lam", lambda, ".json", sep="")
      cat(toJSON(result), file = filename)
    }
  }
}
  
