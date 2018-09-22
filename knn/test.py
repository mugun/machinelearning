# -*- coding: utf-8 -*-
"""
Created on Fri Sep 21 10:38:54 2018

@author: fsxn2
"""

import knn
group,labels=knn.createDataSet()
print(knn.classify0([0,0],group,labels,3))