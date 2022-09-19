import numpy as np
from glob import glob
import pickle
import pandas as pd
import random
from tqdm import tqdm
from collections import OrderedDict
from sklearn import metrics
from sklearn.metrics import auc
from dataset.dataset import Car_AutoEncoder


def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  
    np.random.seed(seed)  
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
def train(X_train,X_test,hidden_neurons,learning_rate,epochs,batch_size,contamination,drop_out,hidden_activation):
    same_seeds(42)
    clf_name = 'auto_encoder'
    clf = Car_AutoEncoder(hidden_neurons=hidden_neurons,  batch_size=batch_size, epochs=epochs,learning_rate=learning_rate,
                                    dropout_rate=drop_out,contamination=contamination,hidden_activation=hidden_activation)
    clf.fit(X_train)

    y_train_pred = clf.labels_ # 返回训练数据上的分类标签 (0: 正常值, 1: 异常值) # binary labels (0: inliers, 1: outliers)
    y_train_scores = clf.decision_scores_  # 返回训练数据上的异常值 (分值越大越异常)# raw outlier scores
    y_test_pred = clf.predict(X_test) # 返回未知数据上的分类标签 (0: 正常值, 1: 异常值) # outlier labels (0 or 1)
    y_test_scores = clf.decision_function(X_test)   #返回未知数据上的异常值 (分值越大越异常) # outlier scores
    return clf, y_test_scores

def evaluate(label,score):
    fpr, tpr, thresholds = metrics.roc_curve(label, score, pos_label=1)
    AUC = auc(fpr, tpr)
    return AUC

def  load_data(pkl_list,label=True):
    '''
    输入pkl的列表，进行文件加载
    label=True用来加载训练集
    label=False用来加载真正的测试集，真正的测试集无标签
    '''
    X = []
    y = []
    
    for  each_pkl in pkl_list:
        pic = open(each_pkl,'rb')
        item= pickle.load(pic)#下载pkl文件
        # 此处选取的是每个滑窗的最后一条数据，仅供参考，可以选择其他的方法，比如均值或者其他处理时序数据的网络
        # 此处选取了前7个特征，可以需求选取特征数量
        X.append(item[0][:,0:7][-1])
        if label:
            y.append(int(item[1]['label'][0]))
    X = np.vstack(X)
    if label:
        y = np.vstack(y)
    return X, y

def main():
    data_path='./data/Train'
    pkl_files = glob(data_path+'/*.pkl')
    ind_pkl_files = []
    ood_pkl_files = [] 
    for each_path in tqdm(pkl_files):
        pic = open(each_path,'rb')
        this_pkl_file= pickle.load(pic)
        if this_pkl_file[1]['label'] == '00':
            ind_pkl_files.append(each_path)
        else:
            ood_pkl_files.append(each_path)

    random.seed(0)
    random.shuffle(ind_pkl_files)
    random.shuffle(ood_pkl_files)
    
    train_pkl_files=[]
    for i in range(len(ind_pkl_files)//4):
        train_pkl_files.append(ind_pkl_files[i])
    # 选取训练集中正常片段的1/4作为训练集，正常片段的剩余3/4和异常片段作为测试集
    test_pkl_files=[]
    for j in range(len(ind_pkl_files)//4,len(ind_pkl_files)):
        test_pkl_files.append(ind_pkl_files[j])
    for item in ood_pkl_files:
        test_pkl_files.append(item)
    
    #参数设置
    hidden_neurons=[32,64,64,32]
    learning_rate=0.03
    epochs=15
    batch_size=640
    contamination=0.005
    drop_out=0.2
    hidden_activation='sigmoid'

    same_seeds(42)
    
    