# -*- coding: utf-8 -*-
"""
Created on Fri Sep 21 10:24:23 2018

@author: fsxn2
"""

import numpy as np
import operator
def createDataSet():
    group=np.array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels=['A','A','B','B']
    return group,labels

def classify0(inX,dataSet,labels,k):
    dataSetSize=dataSet.shape[0]
   # print(dataSetSize)
    #计算距离
    diffMat=np.tile(inX,(dataSetSize,1))-dataSet
    #tile实际上是一个数据复制函数，将给出来的数组以指定模式填充
    #print(np.tile(inX,(dataSetSize,1)))
    #print(diffMat)
    sqDiffMat=diffMat**2
   # print(sqDiffMat)
    sqDistance=sqDiffMat.sum(axis=1)
    #按列对数据进行相加
    distances=sqDistance**0.5
    sortedDistIndicies = distances.argsort()
    #print(sortedDistIndicies)
    #argsort的含义是对dsitance中的元素进行排序！并且返回对应的索引，注意值返回索引
    classCount={}
    #选择最小的K点
    for i in range(k):
        voteIlabel=labels[sortedDistIndicies[i]]
        #print(voteIlabel)
        #print(classCount.get(voteIlabel,0))
        classCount[voteIlabel]=classCount.get(voteIlabel,0)+1
    sortedClassCount=sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
    print(sortedClassCount)
    return sortedClassCount[0][0]
    
        