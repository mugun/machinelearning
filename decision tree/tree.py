from math import log
import operator as op
def calcShannonEnt(dataSet):
    numEntries=len(dataSet)
    labelCounts={}
    for featVec in dataSet:
        currentLabel=featVec[-1]
       # print(currentLabel)
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel]=0
        labelCounts[currentLabel]+=1
    #print(labelCounts)
    shannonEnt=0.0
    for key in labelCounts:
        prob=float(labelCounts[key])/numEntries
        shannonEnt -=prob*log(prob,2)
    #H(x) = E[I(xi)] = E[ log(2,1/p(xi)) ] = -∑p(xi)log(2,p(xi)) (i=1,2,..n)
    #信息熵的公式，可以参考学习
    return shannonEnt

def createDataSet():
    dataSet=[[1,1,'yes'],[1,1,'yes'],[1,0,'no'],[0,1,'no'],[0,1,'no']]
    labels=['no surfacing','flippers']
    return dataSet,labels

def splitDataSet(dataSet,axis,value):
    retDataSet=[]
    for featVec in dataSet:
        if featVec[axis] == value:
                reducedFeatVec=featVec[:axis]
                reducedFeatVec.extend(featVec[axis+1:])
                retDataSet.append(reducedFeatVec)
    return retDataSet

def chooseBestFeatureToSplit(dataSet):
    numberFeatures=len(dataSet[0])-1
    baseEntropy=calcShannonEnt(dataSet)
    bestInfoGain=0.0
    bestFeature=-1
    for i in range(numberFeatures):
        #下列两行创建唯一的分类标签列表
        featList=[example[i]for example in dataSet]
        uniqueVals=set(featList)
        newEntropy=0.0
        #计算每种划分方式的信息熵
        for value in  uniqueVals:
            subDataSet=splitDataSet(dataSet,i,value)
            prob=len(subDataSet)/float(len(dataSet))
            newEntropy+=prob*calcShannonEnt(subDataSet)
        infoGain=baseEntropy-newEntropy
        if(infoGain>bestInfoGain):
            #计算最好的信息增益
            bestInfoGain=infoGain
            bestFeature=i
    return bestFeature

def majorityCnt(classList):
    classCount={}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote]=0
        classCount+=1
    sortedClassCount=sorted(classCount.items(),key=op.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]

def createTree(dataSet,labels):
    classList=[example[-1] for example in dataSet]
    #类别相同则停止划分
    if classList.count(classList[0])==len(classList):
        return classList[0]
    #遍历完后返回所有特征时返回出现次数最多的类别
    if len(dataSet[0])==1:
        return majorityCnt(classList)
    bestFeat=chooseBestFeatureToSplit(dataSet)
    bestFeatLabel=labels[bestFeat]
    myTree={bestFeatLabel:{}}
    #得到列表包含的所有属性值
    del(labels[bestFeat])
    featValues=[example[bestFeat]for example in dataSet]
    uniqueVals=set(featValues)
    for value in uniqueVals:
        subLabels=labels[:]
        myTree[bestFeatLabel][value]=createTree(splitDataSet(dataSet,bestFeat,value),subLabels)
    return myTree
    

if __name__ =='__main__':
    myDat,labels=createDataSet()
    #print(myDat)
    #print(calcShannonEnt(myDat))
    #print(splitDataSet(myDat,0,1))
    #print(splitDataSet(myDat,0,0))
    #print(chooseBestFeatureToSplit(myDat))
    myTree=createTree(myDat,labels)
    print(myTree)
        