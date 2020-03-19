'''
线性扫描法：
输入：训练数据集T={[x1,y1],[x2,y2],...,[xn,yn]} 
      待预测数据(x_test)
      k值
(1) 计算x_test和训练集中的实例xi的欧氏距离
(2) 欧氏距离排序
(3) 取前k个最小距离对应的训练数据点的类型y
(4) 对k个y值进行统计
(5) 返回频率出现最高的y值
'''
import numpy as np
from collections import Counter

class KNN:
    def __init__(self,X_train,Y_train,k):
        self.k=k
        self.X_train=X_train
        self.Y_train=Y_train
    

    def predict(self,X_new):
        ##计算欧氏距离,形式为[(d0,1),(d1,-1),...]
        list_dis=[(np.linalg.norm(X_new-self.X_train[i],ord=2),self.Y_train[i]) for i in range(self.X_train.shape[0])]
        # for i in range(self.X_train.shape[0]):
        #     list_dis.append(np.linalg.norm(X_new-self.X_train[i],ord=2),self.Y_train[i])

        ##排序,因为是元祖，所以排序时用lamda参数来规定按照哪一列来排序
        list_dis.sort(key=lambda x: x[0])

        ##取前k个值的标签,形式为[-1,1,-1,...]
        y_list=[]
        for i in range(self.k):
            y_list.append(list_dis[i][-1])

        ##对上述的k个点的分类进行统计
        y_count=Counter(y_list).most_common()

        return y_count[0][0]




def main():
    ##训练数据
    X_train=np.array([[5,4],[9,6],[4,7],[2,3],[8,1],[7,2]])
    Y_train=np.array([1,1,1,-1,-1,-1])

    ##测试数据
    X_new=np.array([[5,3]])

    ##取不同k值的结果
    for k in range(1,6,2):
        ##构建KNN实例
        knn=KNN(X_train,Y_train,k=k)
        ##对新数据进行预测
        y_predict=knn.predict(X_new)
        print("k={} 被分类为：{}".format(k,y_predict))


main()
