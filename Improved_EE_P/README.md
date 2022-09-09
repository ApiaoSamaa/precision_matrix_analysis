# FST

## Project Structure:

主要的代码文件：

babelTest.py:real-world data实验以及绘制实验结果。主要对这个文件进行修改就可以。

其他代码文件：

    EE.py: Elementary Estimator - sGGM 的实现。
    evaluation.py: 计算accuracy的函数。
    generate.py: 生成 simulated dataset。
    GNTest.py: 随机生成一个 social-network。
    imageTest.py: 用FST预测real-world数据并绘制预测结果。
    improvedEE.py: FST的实现。在所有代码中，improvedEE就是指FST。
    plotFig.py:绘制实验结果图。
    benchmark.py:在simulated data上的实验。
    
    
## Data
 所有的数据来自于 openfMRI 对应项目的 BOLD fMRI 图像。data description 可以在对应项目网站中找到。
 
 [1] sub-01_func_sub-01_task-balloonanalogrisktask_run-01_bold.nii.gz: Balloon Analog Risk-taking Task (https://openneuro.org/datasets/ds000001/versions/00006)
 
 [2] sub-01_func_sub-01_task-probabilisticclassification_run-01_bold.nii.gz: Classification learning (https://openneuro.org/datasets/ds000002/versions/00002) 这个数据可以直接预测 precision matrix；更好的是考虑一下怎么改成一个 classification task 来解决。 
 
 [3] sub-01_func_sub-01_task-rhymejudgment_bold.nii.gz: Rhyme judgment (https://openneuro.org/datasets/ds000003/versions/57fed018cce88d000ac1757f)
 
 
## Dependencies:
    
    matplotlib 3.2.0
    nilearn 0.5.0
    numpy 1.17.3
    scikit-learn 0.21.3 
    
## Experioment Requirement:

比较 FST， EE， 和 GLasso 三种方法的时间。画出 sample covariance 以及 precison matrix 的图体现 FST 对于 social-network graph 的优势。

在实验解释部分，要解释清楚数据的作用，实验具体的应用，数据的说明。。。我这里找了一个写的还算详细的例子 （ECML-PKDD_2020_860.pdf），可以参照 4.2 节来写。
大致就是要写这些内容。