# -*- coding: utf-8 -*-
"""
Created on Mon Sep 24 14:41:23 2018

@author: fsxn2
"""
import numpy as np
def loadDataSet():
    #定义数据集
    postingList=[['my','dog','has','flea','problem','help','please'],
                 ['maybe','not','take','him','to','dog','park','stupid'],
                 ['my','dalmation','is','so','cute','I','love','him'],
                 ['stop','posting','stupid','worthless','garbage'],
                 ['mr','licks','ate','my','steak','how','to','stop','him'],
                 ['quit','buying','worthless','dog','food','stupid']]
    classVec=[0,1,0,1,0,1]
    return postingList,classVec

def createVocabList(dataSet):
    #返回列表
    vocabSet=set([])
    for document in dataSet:
        vocabSet=vocabSet|set(document)
    return list(vocabSet)

def setOfWords2Vec(vocabList,inputSet):
    returnVec=[0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)]=1
        else:
            print(word," not in the vocabulary")
    return returnVec

def trainNB0(trainMatrix,trainCategory):
    numTrainDocs=len(trainMatrix)
    numWords=len(trainMatrix[0])
    pAbusive=sum(trainCategory)/float(numTrainDocs)
    p0Num=np.zeros(numWords)
    p1Num=np.zeros(numWords)
    p0Denom=0.0
    p1Denom=0.0
    for i in range(numTrainDocs):
        if trainCategory[i]==1:
            #进行向量相加
            p1Num+=trainMatrix[i]
            p1Denom+=sum(trainMatrix[i])
        else:
            p0Num+=trainMatrix[i]
            p0Denom+=sum(trainMatrix[i])
    p1Vect=p1Num/p1Denom
    #对每个元素做除法
    p0Vect=p0Num/p0Denom
    return p0Vect,p1Vect,pAbusive
            

if __name__=='__main__':
    listOPost,listClasses=loadDataSet()
    myVocabList=createVocabList(listOPost)      
    print(myVocabList)
    print(setOfWords2Vec(myVocabList,listOPost[0]))
    trainMat=[]
    for postinDoc in listOPost:
        trainMat.append(setOfWords2Vec(myVocabList,postinDoc))
    p0V,p1V,pAb=trainNB0(trainMat,listClasses)
    print(pAb)