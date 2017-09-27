# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 10:19:56 2017

@author: zzd
"""

import numpy as np
import scipy.io as sio

class kNN(object):
    
    def __init__(self,):
        self.trainSet=[]
        self.labels=[]
        self.testSet=[]
        self.k=[]

    def cosDist(self,v1,v2):
        v1=np.mat(v1)
        v2=np.mat(v2)
        if np.linalg.norm(v1)==0 or np.linalg.norm(v2)==0:
            return 1
        else:
            return (v1*v2.T)/(np.linalg.norm(v1)*np.linalg.norm(v2))


    def fit(self,trainSet,labels,k=3):
        self.trainSet=trainSet
        self.labels=labels
        if k>self.trainSet.shape[0]:
            print('k值设定的太大')
        else:
            self.k=k
        
    def predict(self,testSet):
        colTrain=self.trainSet.shape[0]
        colTest=testSet.shape[0]
        for i in range(colTest):
            dists=np.zeros([1,colTrain])
            for j in range(colTrain):
                dists[0,j]=self.cosDist(testSet[i],self.trainSet[j])
            
            #对距离从大到小排序，并获取排序索引
            sortedIndex=np.argsort(-dists)
            classDic={}
            print(sortedIndex)
            for z in range(self.k):
                classType=self.labels[sortedIndex[0][z]]
                #如果指定的键的值不存在，则默认为0
                classDic[classType]=classDic.get(classType,0)+1
            #对类别数目从大到小排序,采用字典排序方法
            #通过值降序排列
            sortedClass=sorted(classDic.items(),key=lambda x:x[1],reverse=True)
            print(sortedClass[0][0])

if __name__=='__main__':
    
    #学习测试案例一
    trainSet=np.array([[1,1.1],[1,1],[0,0],[0,0.1]])
    labels=['A','A','B','B']
    
    k=kNN()
    k.fit(trainSet,labels)
    testSet=np.array([[0,0.5],[1,1.2]])
    k.predict(testSet)

    #学习测试案例二        
    tfPath='tf.mat'
    labPath='labels.mat'
    tf=sio.loadmat(tfPath)['tf']
    labels=sio.loadmat(labPath)['labels'].tolist()[0]
    k2=kNN()
    k.fit(tf,labels)
    k.predict(np.array([tf[1]]))






