# -*- coding: utf-8 -*-
"""
Created on Fri Sep 21 10:24:23 2018

@author: fsxn2
"""

import numpy as np
import operator
def datingClassTest():
    hoRatio = 0.50      #hold out 10%
    datingDataMat,datingLabels = file2matrix('datingTestSet2.txt')       #load data setfrom file
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m*hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i,:],normMat[numTestVecs:m,:],datingLabels[numTestVecs:m],3)
        print ("the classifier came back with: %d, the real answer is: %d" % (classifierResult, datingLabels[i]))
        if (classifierResult != datingLabels[i]): errorCount += 1.0
    print ("the total error rate is: %f" % (errorCount/float(numTestVecs)))
    print errorCount

def autoNorm(dataset):
    #归一化数据
    minVals=dataset.min(0)
    maxVals=dataset.max(0)
    ranges=maxVals-minVals
    normDataSet=np.zeros(np.shape(dataset))
    m=dataset.shape[0]
    normDataSet=dataset-np.tile(minVals,(m,1))
    normDataset = normDataSet/(np.tile(ranges,(m,1)))
    return normDataset,ranges,minVals
    #特征值相除
    


def file2matrix(filename):
    #读取文件，存在的问题为numpy的不能读取数字以外的内容
    fr=open(filename)
    arrayOLines=fr.readlines()
    numberOfLines=len(arrayOLines)
    returnMat=np.zeros((numberOfLines,3))
    classLabelVector=[]
    index=0
    for line in arrayOLines:
        line=line.strip()
        listFromLine=line.split(' ')
        #print(listFromLine)
        returnMat[index,:]=listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index +=1
    return returnMat,classLabelVector
        

def createDataSet():
    #初始化数据
    group=np.array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels=['A','A','B','B']
    return group,labels

def classify0(inX,dataSet,labels,k):
    #KNN的主函数计算函数
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
    #print(classCount)
    sortedClassCount=sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
    #operator.itemgetter是用来获取某一个域的数值
    return sortedClassCount[0][0]
    
        