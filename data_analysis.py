# refer: https://zhuanlan.zhihu.com/p/162749684
import os
import pandas as pd
# from transformers import BertTokenizer
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
# plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
# plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
# tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
font_en = 'DejaVu Sans'

def draw_hist(data, bins, xlabel, title, save_name):
    # plt.figure()
    fig, ax = plt.subplots()
    plt.hist(data, bins=bins)
    plt.axvline(x=400, ymin=0.0, ymax=0.97, color='k', alpha=0.7, linewidth=1, linestyle='--', label='word_count=400')
    plt.xlabel(xlabel)
    plt.legend(prop={'family': font_en})
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname(font_en) for label in labels]
    plt.title(title)

    plt.savefig(save_name)
    plt.show()

def draw_bar(x, y, xlabel, title, save_name):
    # plt.figure()
    fig, ax = plt.subplots()
    plt.bar(x, y, width=0.4)
    plt.xlim(0, max(x)+1)  # set the range of x axis
    plt.xticks(range(1, max(x)+1)) # set the ticks of x axis
    plt.xlabel(xlabel)
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname(font_en) for label in labels]
    plt.title(title)

    plt.savefig(save_name)
    plt.show()

def draw_barh(x, y, title, save_name):
    fig, ax = plt.subplots(figsize=(10,20))
    b = ax.barh(range(len(x)), y) #, color='#6699CC'
     
    #为横向水平的柱图右侧添加数据标签。
    for rect in b:
        w = rect.get_width()
        ax.text(w, rect.get_y()+rect.get_height()/2, '%d' %
                int(w), ha='left', va='center')
     
    #设置Y轴纵坐标上的刻度线标签。
    ax.set_yticks(range(len(x)))
    ax.set_yticklabels(x)
     
    #不要X横坐标上的label标签。
    plt.xticks(())

    plt.title(title)

    plt.savefig(save_name, dpi=600, bbox_inches ='tight')
    plt.show()

def get_subset_class_count(data_file):
    data = pd.read_csv(data_file, encoding="UTF-8", sep='\t')
    all_labels =[t for x in data['label'] for t in x.split()]
    all_labels_pd = pd.Series(all_labels)
    all_labels_count = all_labels_pd.value_counts()
    return {k:v for k,v in zip(list(all_labels_count.index), list(all_labels_count))}

def draw_barh_subset(labels):
    # refer https://blog.csdn.net/weixin_30552495/article/details/118791506
    plt.figure(figsize=(6.2,5.4))#(11,10)
    font_size = 7.5#13
    
    colors = ['#1f77b4', '#d62728', '#2ca02c']#['r', 'g', 'b']#['#3398DB', '#cd5c5c', '#3cb371']
    subsets = ['训练集', '验证集', '测试集']
    train_set = get_subset_class_count('../data/train.tsv')
    dev_set = get_subset_class_count('../data/dev.tsv')
    test_set = get_subset_class_count('../data/test.tsv')
    values = []
    for label in labels:
        values.append([train_set[label], dev_set[label], test_set[label]])
    values = np.array(values)
    lefts = np.insert(np.cumsum(values, axis=1),0,0, axis=1)[:, :-1]
    bottoms = np.arange(len(values))

    rect1 = [0.0, 0.0, 0.41, 1] # [左, 下, 宽, 高] 规定的矩形区域 （全部是0~1之间的数，表示比例）
    rect2 = [0.7, 0.0, 0.3, 1]
    ax1 = plt.axes(rect1)
    ax2 = plt.axes(rect2)
    # plt.subplot(121)
    start = 0
    end = int(len(values)/2)+1
    for subset_idx, color in zip(range(len(subsets)), colors):
        value = values[start:end, subset_idx]
        left = lefts[start:end, subset_idx]
        ax1.bar(left, height=0.8, width=value, bottom=bottoms[start:end], color=color, 
            orientation="horizontal", label=subsets[subset_idx], zorder=100) #left=

    ax1.set_yticks(bottoms[start:end]) #+0.4
    ax1.set_yticklabels(labels[start:end])
    ax1.tick_params(labelsize=font_size)
    ax1.legend(loc="best", bbox_to_anchor=(1.0, 1.00), prop={'size':font_size})
    # plt.subplots_adjust(right=0.85)
    ax1.grid(axis='x', zorder=0)

    # plt.subplot(122)
    
    start = end
    end = len(values)
    for subset_idx, color in zip(range(len(subsets)), colors):
        value = values[start:end, subset_idx]
        left = lefts[start:end, subset_idx]
        ax2.bar(left, height=0.8, width=value, bottom=bottoms[start:end], color=color, 
            orientation="horizontal", label=subsets[subset_idx], zorder=100) #left=

    # ax2.set_xticks(range())
    ax2.set_yticks(bottoms[start:end]) #+0.4
    ax2.set_yticklabels(labels[start:end])
    ax2.tick_params(labelsize=font_size)
    ax2.legend(loc="best", bbox_to_anchor=(1.0, 1.00), prop={'size':font_size})
    # plt.subplots_adjust(right=0.85)
    ax2.grid(axis='x', zorder=0)
    
    labels = ax1.get_xticklabels() + ax2.get_xticklabels()
    [label.set_fontname(font_en) for label in labels]

    plt.savefig('class_count_3.png', dpi=600, bbox_inches ='tight')
    plt.show()



def data_analyze(data_file='../data/all.tsv'):
    data = pd.read_csv(data_file, encoding="UTF-8", sep='\t')
    # prefix = os.path.basename(data_file).split('.')[0]+'_'
    
    # data['WORD_COUNT']=data['content'].apply(lambda x:len(x))#.split()
    # draw_hist(data['WORD_COUNT'], bins=50, xlabel='字数', title="试题字数统计", save_name=prefix+'word_count_2.png')
    # # data.hist('WORD_COUNT', bins=50)
    # print(data['WORD_COUNT'].describe())

    # data['LABELS_COUNT']=data['label'].apply(lambda x:len(x.split()))#.split()
    # # data.hist('LABELS_COUNT', bins=30)
    # # draw(data['LABELS_COUNT'], bins=18, xlabel='label count', title="Histogram of label count", save_name='label_count.png')
    # labels_count = data['LABELS_COUNT'].value_counts()
    # draw_bar(list(labels_count.index), list(labels_count), xlabel='知识点数目', title="试题对应知识点数目统计", save_name=prefix+'label_count_2.png')
    # print(data['LABELS_COUNT'].describe())
    
    # plt.figure()
    all_labels =[t for x in data['label'] for t in x.split()]
    all_labels_pd = pd.Series(all_labels)
    all_labels_count = all_labels_pd.value_counts()
    # draw_barh(list(all_labels_count.index), list(all_labels_count), title='知识点出现次数统计', save_name=prefix+'class_count.png')
    # print(all_labels_pd.describe())
    draw_barh_subset(all_labels_count.index)

    # len_list = []
    # for index, row in data.iterrows():
    #     tokenized_text = tokenizer.tokenize(row["content"])
    #     # print(tokenized_text)
    #     len_list.append(len(tokenized_text))
    #     # break
    
    # print("Min_len:", min(len_list))
    # print("Mean_len:", sum(len_list)/len(len_list))
    # print("Max_len:", max(len_list))

    # plt.hist(len_list, bins=50)
    
    return


data_analyze()
