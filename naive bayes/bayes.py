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
    #print(returnVec)
    return returnVec

def trainNB0(trainMatrix,trainCategory):
    #此处是生成朴素贝叶斯分类器的关键
    numTrainDocs=len(trainMatrix)
    numWords=len(trainMatrix[0])
    pAbusive=sum(trainCategory)/float(numTrainDocs)
    #求集合中出现1（侮辱词汇）的概率
    p0Num=np.ones(numWords)
    p1Num=np.ones(numWords)
    p0Denom=2.0
    p1Denom=2.0
    for i in range(numTrainDocs):
        if trainCategory[i]==1:
            #进行向量相加
            p1Num+=trainMatrix[i]
            p1Denom+=sum(trainMatrix[i])
            print(p1Num,p1Denom)
        else:
            p0Num+=trainMatrix[i]
            p0Denom+=sum(trainMatrix[i])
          
    p1Vect=np.log(p1Num/p1Denom)
   
    #这里得到是一个矩阵，关于每个单词的出现情况的
    #这里使用的log方法应该是numpy中的而不是math中的
    #对每个元素做除法
    p0Vect=np.log(p0Num/p0Denom)
    print(p1Vect,p0Vect)
    return p0Vect,p1Vect,pAbusive
            

def classifyNB(vec2Classify,p0Vec,p1Vec,pClass1):
    p1=sum(vec2Classify*p1Vec)+np.log(pClass1)
    p0=sum(vec2Classify*p0Vec)+np.log(1.0-pClass1)
    if p1>p0:
        return 1
    else:
        return 0

def testingNB():
    listOPosts,listClasses=loadDataSet()
    myVocabList=createVocabList(listOPosts)
    trainMat=[]
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList,postinDoc))
    p0V,p1V,pAb=trainNB0(np.array(trainMat),np.array(listClasses))
    testEntry=['love','my','dalmation']
    thisDoc=np.array(setOfWords2Vec(myVocabList,testEntry))
    print(testEntry,'calssified as:',classifyNB(thisDoc,p0V,p1V,pAb))
    testEntry=['stupid','garbage']
    thisDoc=np.array(setOfWords2Vec(myVocabList,testEntry))
    print(testEntry,'calssified as:',classifyNB(thisDoc,p0V,p1V,pAb))

def bagOfWords2VecMN(vocabList,inputSet):
    returnVec=[0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)]+=1
    return returnVec

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
    #testingNB()