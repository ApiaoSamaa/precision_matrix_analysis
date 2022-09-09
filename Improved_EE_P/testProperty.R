setwd('~/Documents/VscodeProjects/tx/Improved_EE_P')
library(simule)
library(rjson)
library(philentropy)
library(parallel)
# source(file = 'SIMULE.R')
### plotting window reset routine
graphics.off()
par(ask = F)
par(mfrow = c(1, 1))


### load  data 

data = fromJSON(file = "random_20dvars_40dsamples.json")
t = length(data)
n = row_num = length(data[[1]])
p = row_num = length(data[[1]][[1]])

print(t)
print(n)
print(p)

datasets = list()
for (i in 1:t) {
  t = matrix(c(1:n * p), nrow = n, ncol = p)
  for (j in 1:n) {
    for (k in 1:p) {
      t[j, k] = data[[i]][[j]][[k]]
    }
  }
  datasets[[i]] = t
}

input_matrixs = list()
for (i in 1:length(datasets)){
  input_matrixs[[i]] = cov(datasets[[i]])
}

# result = simule(input_matrixs, 0.2, 0.1, covType = "cov", TRUE)
# 
# share = result[["share"]]
# 
# graphs = result[["Graphs"]]
# 
# cat("lambda:", lambda, "  epsilon:", epsilon)
# heatmap(share, Rowv = NA, Colv = NA)

for (epsilon in c(0.2, 0.3, 0.4)) {
  print("=====================")
  for (lambda in seq(0.01, 0.1, 0.01)) {
    result = simule(input_matrixs, lambda, epsilon, covType = "cov", TRUE)

    share = result[["share"]]

    graphs = result[["Graphs"]]

    cat("lambda:", lambda, "  epsilon:", epsilon, "\n")
    heatmap(share, Rowv = NA, Colv = NA)

    
  }
}


cat(toJSON(share), file = "dimension20share_cov.json")





