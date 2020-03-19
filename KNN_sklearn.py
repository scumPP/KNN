import numpy as np

from sklearn.neighbors import KNeighborsClassifier



def main():
    ##训练数据
    X_train=np.array([[5,4],[9,6],[4,7],[2,3],[8,1],[7,2]])
    Y_train=np.array([1,1,1,-1,-1,-1])

    ##测试数据
    X_new=np.array([[5,3]])

    ##取不同k值的结果
    for k in range(1,6,2):
        ##构建KNN实例
        knn=KNeighborsClassifier(n_neighbors=k,weights='distance',n_jobs=-1)
        '''
        KNeighborsClassifier中的参数解释：
        n_neighbors：k值
        weights：近邻的权重，默认为一样的（uniform），还可以邻居距离越近权重越大（distance）
        algorithm：共有三种（brute[暴力解决]，kd_tree，ball_tree）默认自动选择，当数据量很少时，默认只用暴力解决
                   数据维度小于20，kd_tree效果好，ball_tree一般解决高纬度数据查询问题
        leaf_size：叶子节点数量，当algorithm=kd_tree，ball_tree时起作用
        p：p范数，默认为2范数
        metric：距离度量准则，默认为minkowski距离（Lp距离）
        n_jobs：并行搜索，默认为none，即一个进程，-1为所有进程

        '''
        ##选择合适的算法
        knn.fit(X_train,Y_train)
        '''
        KNeighborsClassifier中的函数解释：
        fit：确定适用算法
        predict：对测试点进行分类
        predict_proba:对测试点属于不同分类的概率
        score：输入测试机，评价训练效果
        kneighbors：返回K临近点
        kneighbors_graph：返回k临近点
        '''
        ##对新数据进行预测
        y_predict=knn.predict(X_new)

        print(knn.predict_proba(X_new))

        #print("k={} 被分类为：{}".format(k,y_predict))

main()
'''
总结：
只考虑样本数N的影响：
线性扫描的时间复杂度：o(N)
kd_tree的时间复杂度：o(log2N)
当维度d接近N时，二者效率相当
'''
