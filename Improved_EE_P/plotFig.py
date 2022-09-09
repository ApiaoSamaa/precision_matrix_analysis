import matplotlib.pyplot as plot
import random
import matplotlib.image as mpimg
from sklearn.linear_model import LinearRegression
import numpy as np

def plotTime():
    varList=[500,1000,2000,2500,3000,5000,10000,15000]
    tickList=[500,5000,10000,15000]
    # Random graph
    improve=[0.346,0.716,2.953,5.341,6.584,18.969,99.39,197.409]
    EE=[0.287,1.256,55.943,10.127,15.733,53.337,356.074,1037.103]
    GLASSO=[0.13,0.878,7.9165,16.92,29.6105,148.219,1227.967]
    QUIC=[0.233,1.8285,14.755,29.0445,50.2235,306.793]
    BIGQUIC=[0.996,6.193,36.3815,61.902,277.007,2008.729,13098.774]
    plot.figure(figsize=(16.5,12))
    plot.plot(varList, improve, 'ro-', label='FST', lw=8, ms=30)
    plot.plot([500,1000,2000,2500,5000], BIGQUIC[:-2], 'm>:', label='BIGQUIC', lw=8, ms=30)
    plot.plot(varList[0:7], GLASSO, 'k^-.', label='GLASSO', lw=8, ms=30)
    plot.plot(varList[0:6], QUIC, 'c<:', label='QUIC', lw=8, ms=30)
    plot.plot(varList, EE, 'b*--',label='EE',lw=8,ms=30)


    plot.plot([5000,5000],[0,1300],'k--',lw=1.5)
    plot.plot([10000, 10000], [0, 1300], 'k--',lw=1.5)
    plot.xlabel('Variable number (p)',fontsize=34)
    plot.ylabel('Time cost (sec)',fontsize=34)
    plot.xticks(tickList,fontsize=40)
    plot.yticks(np.arange(0,1400,step=200), fontsize=40)
    plot.legend(loc=2,fontsize=40)
    plot.title('p vs. Time -- Random Graph',fontsize=45)
    plot.show()


    # Circular graph
    varList = [500, 1000, 2000, 3000, 5000, 10000, 15000]
    improve = [0.263,0.985,3.009,8.052,31.76,104.318,211.99]
    EE = [0.277,1.655,6.125,14.896,68.511,373.35,1089.52]
    GLASSO = [0.184,0.8725,7.815,30.9635,136.452,1107.321]
    QUIC = [0.233,1.54,12.462,42.175,220.474,1883.597]
    BIGQUIC = [1.311-0.185,7.805-0.71,52.227-4.294,164.848-8.045,724.055-12,5612.665-15,19242.421-22.787]
    plot.figure(figsize=(16.5, 12))
    plot.plot(varList, improve, 'ro-', label='FST', lw=8, ms=30)
    plot.plot(varList[:-2], BIGQUIC[:-2], 'm>:', label='BIGQUIC', lw=8, ms=30)
    plot.plot(varList[0:6], GLASSO, 'k^-.', label='GLASSO', lw=8, ms=30)
    plot.plot(varList[0:6], QUIC, 'c<:', label='QUIC', lw=8, ms=30)
    plot.plot(varList, EE, 'b*--', label='EE', lw=8, ms=30)

    plot.plot([10000, 10000], [0, 1900], 'k--', lw=1.5)
    plot.plot([5000, 5000], [0, 1900], 'k--', lw=1.5)
    plot.xlabel('Variable number (p)', fontsize=34)
    plot.ylabel('Time cost (sec)', fontsize=34)
    plot.xticks(tickList, fontsize=38)
    plot.yticks(np.arange(0, 2000, step=200), fontsize=40)
    plot.legend(loc=2, fontsize=40)
    plot.title('p vs. Time -- Circular Graph', fontsize=45)
    plot.show()
    # Grid graph
    varList = [500, 1000, 2000, 3000, 5000, 10000, 15000]
    improve = [0.412,0.838,6,6.953,129.692,161.268,262.123]
    EE = [0.3,1.777,11.285,15.818,137.86,482.332,1182.057]
    GLASSO = [0.8145,2.5905,7.9525,29.834,134.601,1121.587]
    QUIC = [0.242,1.307,10.4325,25.2585,181.708,1466.93]
    BIGQUIC=[1.471-0.037,8.077-0.121,54.7-0.464,173.972-8.479,746.953-16.448,5784.293-18,19143.176-25]
    plot.figure(figsize=(16.5, 12))
    plot.plot(varList, improve, 'ro-', label='FST', lw=8, ms=30)
    plot.plot(varList[:-2], BIGQUIC[:-2], 'm>:', label='BIGQUIC', lw=8, ms=30)
    plot.plot(varList[0:6], GLASSO, 'k^-.', label='GLASSO', lw=8, ms=30)
    plot.plot(varList[0:6], QUIC, 'c<:', label='QUIC', lw=8, ms=30)
    plot.plot(varList, EE, 'b*--', label='EE', lw=8, ms=30)

    plot.plot([10000, 10000], [0, 1500], 'k--', lw=1.5)
    plot.plot([5000, 5000], [0, 1500], 'k--', lw=1.5)
    plot.xlabel('Variable number (p)', fontsize=34)
    plot.ylabel('Time cost (sec)', fontsize=34)
    plot.xticks(tickList, fontsize=40)
    plot.yticks(np.arange(0, 1500, step=200), fontsize=40)
    plot.legend(loc=2, fontsize=35)
    plot.title('p vs. Time -- Grid Graph', fontsize=45)
    plot.show()


def plotHighDim():
    samp=[300,600,1500,3000]
    #Improved EE
    im_TPR=[0.928,1,1,1]
    im_F_norm=[90.482,68.89,42.192,27.194]
    im_max_norm=[0.9878,0.842,0.602,0.467]
    #GLASSO
    ano_samp=[300,600,1500,2000,3000]
    g_TPR=[0.0199,0.0199,0.0199,0.0199,0.0199]
    g_F_norm=[72.7414,71.897,71.011,70.808,70.32]
    g_max_norm=[0.7,0.663,0.636,0.627,0.61]
    #Plot figures
    plot.subplot(1,3,1)
    plot.plot(samp,im_TPR,'r^-',label='FST')
    plot.plot(ano_samp,g_TPR,'bo--',label='GLASSO')
    plot.xlabel('Sample number')
    plot.ylabel('TPR')
    plot.legend(prop={'size':18})

    plot.subplot(1, 3, 2)
    plot.plot(samp, im_F_norm, 'r^-', label='FST')
    plot.plot(ano_samp, g_F_norm, 'bo--', label='GLASSO')
    plot.xlabel('Sample number')
    plot.ylabel('Frobenius norm')
    plot.legend(prop={'size':18})

    plot.subplot(1, 3, 3)
    plot.plot(samp, im_max_norm, 'r^-', label='FST')
    plot.plot(ano_samp, g_max_norm, 'bo--', label='GLASSO')
    plot.xlabel('Sample number')
    plot.ylabel('Max norm')
    plot.legend(prop={'size':18})

    plot.show()


def plotDiffThre():
    # Data
    varList=[100,500,1000,2000,2500,5000]
    #Random graph
    hard_F=[12.94395,17.66429,22.278,31.242,28.67595,43.55779]
    soft_F=[10.15013,28.4571,41.46295,58.86787,65.544,93.457]
    ada_F=[9.0788,25.4835,36.3336,51.293,57.225,80.95616]
    SCAD_F=[11.71639,30.64356,43.9413,62.159,68.9925,98.116]
    hard_max=[0.939,0.637,0.52,0.509,0.4127,0.44382]
    soft_max = [0.6846,0.644,0.62559,0.6,0.599,0.6]
    ada_max = [0.6637,0.589,0.5576,0.525,0.52,0.5129]
    SCAD_max = [0.7425,0.704,0.6965,0.678,0.678,0.677]
    # Plot figures

    #plot.subplot(1,2,1)
    plot.title('p vs. Error rate --[F-norm]', fontsize=35)
    plot.plot(varList,hard_F,'b^-',label='Hard',lw=8,ms=20)
    plot.plot(varList,soft_F,'rs--',label='Soft',lw=8,ms=20)
    plot.plot(varList,ada_F,'mo:',label='Ada. LASSO',lw=8,ms=20)
    plot.plot(varList,SCAD_F,'gD-.',label='SCAD',lw=8,ms=20)
    plot.xlabel('Variable number (p)',fontsize=35)
    plot.ylabel('F-norm',fontsize=35)
    plot.xticks([100, 1000, 2500, 5000], fontsize=30)
    plot.yticks(np.arange(0, 100, step=20), fontsize=30)
    plot.legend(fontsize=28,loc=2)
    plot.show()
    #plot.subplot(1, 2, 2)
    plot.title('p vs. Error rate --[max-norm]', fontsize=35)
    #plot.figure(figsize=(18,21))
    plot.plot(varList, hard_max, 'b^-', label='Hard', lw=8, ms=20)
    plot.plot(varList, soft_max, 'rs--', label='Soft', lw=8, ms=20)
    plot.plot(varList, ada_max, 'mo:', label='Ada. LASSO', lw=8, ms=20)
    plot.plot(varList, SCAD_max, 'gD-.', label='SCAD', lw=8, ms=20)
    plot.xlabel('Variable number (p)', fontsize=35)
    plot.xticks([100,1000,2500,5000], fontsize=30)
    plot.yticks(np.arange(0.4,1,step=0.2), fontsize=30)
    plot.ylabel('Max-norm', fontsize=35)
    plot.legend(prop={'size': 28})


    plot.show()

    '''
    # Circular graph
    varList = [500, 1000, 2000, 3000, 5000]
    hard_F = [4.718,6.291,6.2698,7.079,8.7623]
    soft_F = [11.02119,14.7008,20.138,24.67233,32.296]
    ada_F = [9.809,13.1161,17.769,22.015,28.458]
    SCAD_F = [11.10128,14.495,20.0514,24.457,32.005]
    hard_max = [0.3,0.27,0.197,0.172,0.17,]
    soft_max = [0.4021,0.3118,0.3,0.3,0.3]
    ada_max = [0.33,0.3,0.3,0.3,0.3]
    SCAD_max = [0.3745,0.3,0.3,0.3,0.3]
    # Plot figures
    plot.subplot(1, 2, 1)
    plot.plot(varList, hard_F, 'b^-', label='Hard', lw=3, ms=10)
    plot.plot(varList, soft_F, 'rs--', label='Soft', lw=3, ms=10)
    plot.plot(varList, ada_F, 'mo:', label='Ada. LASSO', lw=3, ms=10)
    plot.plot(varList, SCAD_F, 'gD-.', label='SCAD', lw=3, ms=10)
    plot.xlabel('Variable number', fontsize=20)
    plot.ylabel('Frobenius norm', fontsize=20)
    plot.legend(prop={'size': 18})
    plot.subplot(1, 2, 2)
    plot.plot(varList, hard_max, 'b^-', label='Hard', lw=3, ms=10)
    plot.plot(varList, soft_max, 'rs--', label='Soft', lw=3, ms=10)
    plot.plot(varList, ada_max, 'mo:', label='Ada. LASSO', lw=3, ms=10)
    plot.plot(varList, SCAD_max, 'gD-.', label='SCAD', lw=3, ms=10)
    plot.xlabel('Variable number', fontsize=20)
    plot.ylabel('Max norm', fontsize=20)
    plot.legend(prop={'size': 18})
    plot.show()

    # Grid graph
    varList = [1000, 2000, 3000, 5000]
    hard_F = [5.338,6.938,7.61,13.5666]
    soft_F = [7.104,9.752,11.408,14.64]
    ada_F = [6.806,9.492,11.19,14.493]
    SCAD_F = [6.907,9.574,11.263,14.539]
    hard_max = [0.214,0.16,0.132,0.1537]
    soft_max = [0.234,0.182,0.154,0.156]
    ada_max = [0.218,0.167,0.139,0.15]
    SCAD_max = [0.217,0.167,0.143,0.1496]
    # Plot figures
    plot.subplot(1, 2, 1)
    plot.plot(varList, hard_F, 'b^-', label='Hard', lw=3, ms=10)
    plot.plot(varList, soft_F, 'rs--', label='Soft', lw=3, ms=10)
    plot.plot(varList, ada_F, 'mo:', label='Ada. LASSO', lw=3, ms=10)
    plot.plot(varList, SCAD_F, 'gD-.', label='SCAD', lw=3, ms=10)
    plot.xlabel('Variable number', fontsize=20)
    plot.ylabel('Frobenius norm', fontsize=20)
    plot.legend(prop={'size': 18})
    plot.subplot(1, 2, 2)
    plot.plot(varList, hard_max, 'b^-', label='Hard',lw=3,ms=10)
    plot.plot(varList, soft_max, 'rs--', label='Soft',lw=3,ms=10)
    plot.plot(varList, ada_max, 'mo:', label='Ada. LASSO',lw=3,ms=10)
    plot.plot(varList, SCAD_max, 'gD-.', label='SCAD',lw=3,ms=10)
    plot.xlabel('Variable number',fontsize=20)
    plot.ylabel('Max norm',fontsize=20)
    plot.legend(prop={'size':18})
    plot.show()
    '''

def plotImbalance():
    IM=[2500,3000,4000,5000,6000,7000,8000,9000,9500,9850,9940,9970,9997]
    IM=[i/10000 for i in IM]
    improve=[172.63,161.96,176.803,227.417,295.698,457.47,505.408,673.239,768.631,839.736,848.168,921.34,874.9]
    EE=[515.772,540.153,545.941,563.611,585.422,671.16,735.148,797.495,847.093,888.969,890.378,919.754,913.536]
    plot.plot(IM,improve,'b^-',label='FST',lw=3,ms=10)
    plot.plot(IM, EE, 'rD-.',label='EE',lw=3,ms=10)
    plot.plot([0.25,0.25],[0,1000],'k--',lw=1.5)
    plot.xlabel('IM',fontsize=20)
    plot.ylabel('Time cost(sec)',fontsize=20)
    plot.legend(prop={'size':18},loc=9)
    plot.show()

def plotImbalanceBar():
    N = 5
    indices=[0,3,7,9,11]
    IM = np.array([2500, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 9500, 9850, 9940, 9970, 9997])
    IM = [i / 10000 for i in IM]
    improve = np.array([172.63, 161.96, 176.803, 227.417, 295.698, 457.47, 505.408, 673.239, 768.631, 839.736, 848.168, 921.34,
               874.9])
    EE = np.array([515.772, 540.153, 545.941, 563.611, 585.422, 671.16, 735.148, 797.495, 847.093, 888.969, 890.378, 919.754,
          913.536])
    ind = [2, 5, 8, 11, 14]  # the x locations for the groups
    width = 0.8  # the width of the bars: can also be len(x) sequence

    plot.figure(figsize=(16.5, 10))
    p1 = plot.bar(ind, improve[indices], width=width, align='edge')
    p2 = plot.bar([each-width for each in ind], EE[indices], width=width, align='edge')

    plot.ylabel('time cost (sec)', fontsize=34)
    plot.xlabel('IM', fontsize=34)
    plot.title('IM vs. time [p=10000, n=20000, K=4]', fontsize=34)
    plot.xticks(ind, ('0.25', '0.5', '0.9', '0.985', '0.997'), fontsize=30)
    plot.yticks([i for i in np.arange(100, 1001, 200)], fontsize=30)
    plot.legend((p1[0], p2[0]), ('FST', 'EE'), fontsize=45)
    plot.show()

def plotSplit():
    varNum=[2000,4000,5000,6000,10000,15000]
    splitEE=[4.206,19.668,34.151,53.607,242.336,747.56]
    improve = [3.981,22.714,42.625,70.931,290.024,866.266]
    EE = [5.448,44.355,82.756,138.614,589.077,1926.436]
    plot.plot(varNum, splitEE, 'ms-', label='FST for Social-Network Graph',lw=3,ms=10)
    plot.plot(varNum, improve, 'b^--', label='FST for Cluster Graph',lw=3,ms=10)
    plot.plot(varNum, EE, 'rD-.', label='EE',lw=3,ms=10)
    plot.xlabel('Variable number',{'fontsize':'xx-large'})
    plot.ylabel('Time cost(sec)',{'fontsize':'xx-large'})
    plot.legend(prop={'size': 18}, loc=2)
    plot.show()


def plotMulti():
    varNum = [1,2,5,10,20]
    time=[1.512,1,0.468,0.313,0.16]
    #plot.plot(varNum, excludeTime, 'rs--', label='Improved EE', lw=3, ms=10)
    plot.plot(varNum, time, 'bs-', label='FST', lw=3, ms=10)
    plot.plot([0,20], [2,2], 'r--', label='EE', lw=2.5)
    plot.xticks(varNum)
    time=[0.2,0.6,1.0,1.4,1.8,2.0]
    newTick=[str(i) for i in time]
    newTick[-2]='...'
    newTick[-1]='220'
    plot.yticks(time,newTick)
    plot.annotate('EE needs around 220 sec \nfor matrix inversion using \none CPU core', xy=(8,2.0), xytext=(10, 1.4), fontsize=12,
                arrowprops=dict(facecolor='red',
                            arrowstyle="->",
                            connectionstyle="arc3,rad=-0.3",
                                )
                )
    plot.xlabel('Core number', {'fontsize': 'xx-large'})
    plot.ylabel('Time cost (sec)', {'fontsize': 'xx-large'})
    plot.legend(prop={'size': 16})
    plot.show()

def plotMultiBar():
    N = 5
    f_time = [1.512,1,0.468,0.313,0.16]  # time for estimating F (step 2)
    ind = [2, 5, 8, 11, 14]  # the x locations for the groups
    width = 0.8  # the width of the bars: can also be len(x) sequence

    plot.figure(figsize=(16.5, 10))
    p1 = plot.bar(ind, f_time, width=width, align='center')
    p2 = plot.plot([each for each in ind],f_time,'cs-',lw=5, ms=20)

    plot.ylabel('time cost (sec)', fontsize=34)
    plot.xlabel('thread num', fontsize=34)
    plot.title('thread num vs. time [p=10000]', fontsize=34)
    plot.xticks(ind, ('1', '2', '5', '10', '20'), fontsize=30)
    plot.yticks([i * 0.1 for i in np.arange(0, 16, 3)], fontsize=30)
    plot.legend((p1[0], p2[0]), ('FST', 'FST'), fontsize=45)
    plot.show()

def plotBar():
    nd=np.arange(4)
    plot.figure(figsize=(14,12))
    plot.bar(nd, [197,11.8,24,114],width=[0.42,0.42,0.42,0.42],color=['r','r','g','b'])
    plot.xticks(nd, ('FST-1 core', 'FST', 'BIGQUIC', 'EE'),fontsize=40)
    plot.yticks(np.arange(10,230,step=40),fontsize=40)
    plot.ylabel('Time cost (h)',fontsize=40)
    labels = [str(each)+' h' for each in [197,11.8,24,114]]
    ax=plot.axes()
    rects = ax.patches
    for rect, label in zip(rects, labels):
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2, height + 1, label,
                ha='center', va='bottom',fontsize=25)
    plot.show()

def combinePlot():
    #plot.figure(1)
    fig,axs=plot.subplots(1,3)
    ax1=axs[0]
    ax2=axs[1]
    ax3=axs[2]

    #plot.subplot(1,3,1)
    varList = [500, 1000, 2000, 2500, 3000, 5000, 10000, 15000]
    tickList = [500, 5000, 10000, 15000]
    # Random graph
    improve = [0.346, 0.716, 2.953, 5.341, 6.584, 18.969, 99.39, 197.409]
    EE = [0.287, 1.256, 55.943, 10.127, 15.733, 53.337, 356.074, 1037.103]
    GLASSO = [0.13, 0.878, 7.9165, 16.92, 29.6105, 148.219, 1227.967]
    QUIC = [0.233, 1.8285, 14.755, 29.0445, 50.2235, 306.793]
    #plot.figure(figsize=(15, 15))
    ax1.plot(varList, improve, 'ro-', label='FST', lw=5, ms=20)
    ax1.plot(varList[0:6], QUIC, 'c<:', label='QUIC', lw=4, ms=16)
    ax1.plot(varList, EE, 'b*--', label='EE', lw=4, ms=16)
    ax1.plot(varList[0:7], GLASSO, 'k^-.', label='GLASSO', lw=4, ms=16)

    ax1.plot([5000, 5000], [0, 1300], 'k--', lw=1.5)
    ax1.plot([10000, 10000], [0, 1300], 'k--', lw=1.5)
    ax1.set_xlabel('Variable number', fontsize=35)
    ax1.set_ylabel('Time cost (sec)', fontsize=35)
    #ax1.set_xticks(tickList, fontsize=35)
    #ax1.set_yticks(np.arange(0, 1400, step=200), fontsize=35)
    ax1.legend(loc=2, fontsize=45)

    #plot.subplot(1,3,2)
    nd = np.arange(4)
    ax2.bar(nd, [197, 11.8, 24, 114], width=[0.9, 0.9, 0.9, 0.9], color=['r', 'r', 'g', 'b'])
    #ax2.set_xticks(nd, ('FST-1 core', 'FST', 'BIGQUIC', 'EE'), fontsize=30)
    #ax2.set_yticks(np.arange(10, 230, step=40), fontsize=30)
    ax2.set_ylabel('Time cost (h)', fontsize=35)
    labels = [str(each) + ' h' for each in [197, 11.8, 24, 114]]
    ax = plot.axes()
    rects = ax.patches
    for rect, label in zip(rects, labels):
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2, height + 1, label,
                ha='center', va='bottom', fontsize=25)

    plot.show()

if __name__ == '__main__':
    pass
    #plotTime()
    #plotHighDim()
    #plotDiffThre()
    #plotImbalance()
    #plotSplit()
    #plotMulti()
    #plotBar()
    #combinePlot()
    #plotMultiBar()
    plotImbalanceBar()


