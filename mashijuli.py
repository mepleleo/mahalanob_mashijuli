# 马氏距离剔除异常值-python
#!/usr/bin/env bin
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from scipy.spatial import distance

# function[Xt, t, MahalaD] = Mahalanob(X, e)
def mahalanob(X, e):

    """
    parm X:(m x n)为光谱矩阵(row is samples,column is spectrum)
    parm e:(设定设置阈值范围)马氏距离标准差的权重系数
    return:
            Xt :预处理之后经过马氏距离筛选的光谱矩阵index
            MahalaD:为样品的马氏距离
    """
    data = X.copy() #避免因为拷贝问题出现计算问题
    if isinstance(data, pd.DataFrame):
        data = data.values
    # 计算样品集平均光谱
    data_mean = np.mean(data, axis=0)
    means = np.tile(data_mean, data.shape[0]).reshape((data.shape[0],
                                                       data.shape[1]), order='C')
    # 计算样品 与平均光谱的马氏距离
    #np.mat(np.cov()).I为协方差的逆矩阵
    # Minv = np.mat(np.cov(data)).I
    CenterX = data - means
    M = np.dot(CenterX.T, CenterX) / (data.shape[0] - 1)
    # M += np.eye(M.shape[0])
    # 样本数量小于维度 添加单位阵 不推荐使用
    Minv = np.linalg.inv(M)
    #print(Minv.shape,'----------------------')
    MahalaD = list(distance.mahalanobis(data[i, :], data_mean, Minv) \
                   for i in range(data.shape[0]))
    
    #print(MahalaD,'----------------------')
    #print(means,MahalaD)
    # 进行筛选
    MahalaDm = np.mean(MahalaD, axis=0)
    Dstd = np.std(MahalaD, axis=0)
    Xt = []
    for i in range(data.shape[0]):
        if MahalaD[i] < (MahalaDm + e * Dstd):
            Xt.append(i)

    return Xt, MahalaD



if __name__ == "__main__":
    from pandas import DataFrame
    clomuns = ['水分 %']
    data = pd.read_excel( r"G:\PAPER\data\2020.2019_apple_data.xlsx")
    colmuns_ = data.columns.values
    X = np.abs(data.drop(clomuns, axis=1))
    y = data.loc[:,'品种'].values

    X_ = X.values()
    print(X_.shape)
    col, ma = mahalanob(X_, 3.8)
    print(col)
    
