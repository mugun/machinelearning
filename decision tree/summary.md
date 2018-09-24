# ID3决策树算法
决策树算法也是一种常见的分类算法。其中ID3是最基础最经典的。  
决策树系列中基础理论就是信息论，而其中有几个概念是比较重要的，分别是信息熵和信息增益。  
*信息熵公式*  
H(x) = E[I(xi)] = E[ log(2,1/p(xi)) ] = -∑p(xi)log(2,p(xi)) (i=1,2,..n)  
*信息增益公式*  
IGain(S,G)=H(x)-∑p(xi)log(2,p(xi))  
其中H（x）表示为父节点中的总信息熵，IGain（S,G）中需要计算的是父节点按照子规则划分所产生的信息熵。  
在进行决策树生成的时候，我们总是需要倾向使用信息增益最大的划分方式来进行划分。而这在反映在建立决策树时，则是不断的使用当前父类分类所带来的最大信息增益来进行划分，本质上就是一个递归的过程。