import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
#下面两句代码防止中文显示成方块
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['SimHei']

plt.rcParams['figure.dpi'] = 600 #图片像素
# plt.rcParams['figure.figsize'] = (12.0, 4.0)
plt.rcParams['figure.figsize'] = (8.0, 4.0)

# x_bits = [12, 24, 32, 48]

def draw():
    # with PdfPages('figs/filternum_metric.pdf') as pdf:
        #cifar10-----------------------------------
        # plt.subplot(1,2, 1) #cifar部分画在子图1中
        # y_early_students=[[0.741551,0.76498,0.77793,0.77558],
        #                   [0.79482,0.818521,0.82154,0.827666],
        #                   [0.86727,0.89018,0.90012,0.900969],
        #                   [0.87354,0.8927,0.90919,0.905798]
        #                   ]
        # y_ours_students=[[0.76229,0.79393,0.78994,0.80432],
        #                  [0.80788,0.82604,0.84254,0.83907],
        #                  [0.87004,0.89094,0.90188,0.9048],
        #                  [0.87955,0.89755,0.91043,0.91302]]
    filter_nums = [5,10,20,30,40,50,60,70]
    dev_f1_micros = [91.27,91.58,92.02,92.05,91.76,92.28,92.21,92.26]
    dev_f1_macros = [86.07,87.25,88.22,88.50,88.19,89.37,89.07,89.13]
    dev_precisions = [93.64,92.83,92.48,92.63,92.15,92.60,92.34,92.42]
    dev_recalls = [89.02,90.36,91.56,91.48,91.38,91.95,92.07,92.11]
    
    test_f1_micros = [91.79,91.78,92.17,92.68,92.42,92.35,92.42,92.52]
    test_f1_macros = [86.27,87.01,87.71,88.68,88.59,89.00,88.41,89.02]
    test_precisions = [93.53,92.77,92.30,93.00,92.39,92.42,92.26,92.33]
    test_recalls = [90.11,90.80,92.05,92.36,92.45,92.29,92.57,92.71]
    
    y_dev_metrics = [np.array(dev_precisions)*0.01, np.array(dev_recalls)*0.01, np.array(dev_f1_micros)*0.01, np.array(dev_f1_macros)*0.01]
    y_test_metrics = [np.array(test_precisions)*0.01, np.array(test_recalls)*0.01, np.array(test_f1_micros)*0.01, np.array(test_f1_macros)*0.01]

    #开始绘制
    colors=['red','orange','blue','green']
    markers=['o','v','*','s']
    # labels_early=['student-1-early','student-2-early',
    #               'student-3-early','student-4-early']
    # labels_ours=['student-1-ours','student-2-ours',
    #              'student-3-ours','student-4-ours']
    labels_dev = ['val-precision', 'val-recall', 'val-f1_micro', 'val-f1_macro']
    labels_test = ['test-precision', 'test-recall', 'test-f1_micro', 'test-f1_macro']

    for i in range(4):
        color=colors[i]
        plt.plot(filter_nums,y_dev_metrics[i],color=color,
                 marker=markers[i],linestyle='--',label=labels_dev[i])
        plt.plot(filter_nums,y_test_metrics[i],color=color,
                 marker=markers[i],linestyle='-',label=labels_test[i])

    plt.xticks(filter_nums)  #横轴只有这四个刻度
    plt.ylim(0.82, 0.95)       #y坐标范围
    plt.title("卷积核个数与模型性能")
    plt.xlabel("卷积核个数")  # 作用为横坐标轴添加标签  fontsize=12
    plt.ylabel("模型性能")  # 作用为纵坐标轴添加标签

    '''
    #sun-----------------------------------
    plt.subplot(1,2, 2)#sun部分画在子图1中
    # 数据准备
    y_early_students = [[0.75348, 0.8166, 0.82471, 0.82749],
                        [0.78058, 0.83077, 0.83574, 0.84841],
                        [0.8249, 0.86114, 0.8627, 0.87235],
                        [0.83924, 0.86368, 0.8674, 0.87599]
                        ]
    y_ours_students = [[0.76123, 0.81592, 0.82724, 0.84166],
                       [0.78617, 0.83835, 0.84371, 0.85208],
                       [0.82594, 0.86230, 0.86595, 0.87225],
                       [0.83931, 0.86902, 0.87141, 0.87521]]

    # 开始绘制
    colors = ['red', 'orange', 'blue', 'green']
    markers = ['o', 'v', '*', 's']
    labels_early = ['student-1-early', 'student-2-early',
                    'student-3-early', 'student-4-early']
    labels_ours = ['student-1-ours', 'student-2-ours', 
                   'student-3-ours', 'student-4-ours']
    for i in range(4):
        color = colors[i]
        plt.plot(x_bits, y_early_students[i], color=color, 
                 marker=markers[i], linestyle='--', label=labels_early[i])
        plt.plot(x_bits, y_ours_students[i], color=color, 
                 marker=markers[i], linestyle='-', label=labels_ours[i])

    plt.xticks(x_bits)  # 横轴只有这四个刻度
    # plt.ylim(0.7, 0.9)       #y坐标范围
    plt.title("SUN")
    plt.xlabel("Number of bits")  # 作用为横坐标轴添加标签  fontsize=12
    plt.ylabel("MAP")  # 作用为纵坐标轴添加标签
    '''

    plt.legend(loc='lower center', prop = {'size':9.5}, ncol=4)
    # pdf.savefig()  
    plt.savefig('figs/filternum_metric.png')
    # plt.show()

draw()