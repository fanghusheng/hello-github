# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 15:29:22 2021

@author: fanghusheng
"""

import csv
import os
import numpy as np
import random
from sklearn import svm
import sklearn
from sklearn.metrics import accuracy_score
from sklearn import metrics


#读取数据
sample_data = []
filename=r'H:/fhs/paper_download/spatial_associate/experiment/PYTHON/circle3/circle33.csv'
with open(filename) as csvfile:
    csv_reader = csv.reader(csvfile)  # 使用csv.reader读取csvfile中的文件
    birth_header = next(csv_reader)  # 读取第一行每一列的标题
    for row in csv_reader:  # 将csv 文件中的数据保存到sample_data中
        sample_data.append(row)


sample_data = [[float(x) for x in row] for row in sample_data]  # 将数据从string形式转换为float形式


sample_data = np.array(sample_data)  # 将list数组转化成array数组便于查看数据结构
birth_header = np.array(birth_header)
# print(sample_data.shape,sample_data[0][10])  # 利用.shape查看结构。
# print(birth_header.shape)
#print(sample_data)


#SVM
#数据分为训练集与测试集
x, y = np.split(sample_data, (4,), axis=1)
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, random_state=1, train_size=0.5)

#训练svm分类器
clf = svm.SVC(C=0.6, kernel='rbf', gamma=20, decision_function_shape='ovr')
clf.fit(x_train, y_train.ravel())
#kernel='linear'时，为线性核，C越大分类效果越好，但有可能会过拟合（defaul C=1）。
#　　 kernel='rbf'时（default），为高斯核，gamma值越小，分类界面越连续；gamma值越大，分类界面越“散”，分类效果越好，但有可能会过拟合。
#　　decision_function_shape='ovr'时，为one v rest，即一个类别与其他类别进行划分，
#　　decision_function_shape='ovo'时，为one v one，即将类别两两之间进行划分，用二分类的方法模拟多分类的结果。

#计算svc分类器的准确率
print('训练精度',clf.score(x_train, y_train))  # 精度
y_hat_train = clf.predict(x_train)
#print('训练集',accuracy_score(y_hat_train, y_train))
print('测试精度',clf.score(x_test, y_test))
y_hat_test = clf.predict(x_test)
#print('测试集',accuracy_score(y_hat_test, y_test))
#查看决策函数，每列代表类间距离
print('decision_function:\n', clf.decision_function(x_train))
print('\npredict:\n', clf.predict(x_train))
#查看精度（分类性能）指标
print('test混淆矩阵：\n',metrics.confusion_matrix(y_test, y_hat_test))
print('test精度指标：\n',metrics.classification_report(y_test, y_hat_test))

# print('train混淆矩阵：\n',metrics.confusion_matrix(y_train, y_hat_train))
# print('train精度指标：',metrics.classification_report(y_train, y_hat_train))
