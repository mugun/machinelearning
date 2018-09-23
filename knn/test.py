# -*- coding: utf-8 -*-
"""
Created on Fri Sep 21 10:38:54 2018

@author: fsxn2
"""

import knn
import matplotlib
import matplotlib.pyplot as plt
#group,labels=knn.createDataSet()
#print(knn.classify0([0,0],group,labels,3))
group,labels=knn.file2matrix("input.txt")
auto,ranges,minval=knn.autoNorm(group)
print(auto)
print(ranges)
print(minval)
#fig=plt.figure()
#ax=fig.add_subplot(111)
#ax.scatter(group[:,1],group[:2])
#plt.show()
print(knn.classify0([1,0,3],group,labels,3))