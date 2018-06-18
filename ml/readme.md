# 1 Introduction
---
此文件夹为机器学习的相关笔记，主文件为`ml.ipynb`

# 2 contents
---
1. `ai.ipynb`:主要笔记文件
    - 机器学习的概况
    - 3 数据预处理
        - 3.1 移除平均
        - 3.2 范围缩放
        - 3.3 归一化
        - 3.4 二值化
        - 3.5 独热编码
        - 3.6 标记编码
    - 4 线性回归
    - 5 岭回归
    - 6 多项式回归
    - 7 决策树回归和自适应增强决策树回归
    - 8 简单分类器
    - 9 逻辑回归分类器
    - 10 朴素贝叶斯分类器
    - 11 划分训练集和测试集
    - 12 用交叉验证检验模型的准确性
    - 13 混淆矩阵 `sklearn.metrics.confusion_matrix()`
    - 14 性能报告 `sklearn.metrics.classfication_report()`
    - 15 汽车质量评估 `sklearn.ensemble.RandomForestClassifier()`
    - 16 验证曲线 `sklearn.model_selection.validation_curve()`
    - 17 学习曲线 `sklearn.model_selection.learning_curve()`
    - 18 完整分类学习过程 `sklearn.naive_bayes.GaussianNB()`
    - 19 svm线性分类器 `sklearn.svm.SVC(kernel='linear')`
    - 20 svm多项式非线性分类器 `sklearn.svm.SVC(kernel='poly')`
    - 21 svm径向基函数非线性分类器 `sklearn.svm.SVC(kernel='rbf')`
    - 22 解决类型数量不平衡问题 `sklearn.svm.SVC(kernel='linear', class_weight='balanced')`
    - 23 置信度 `model.predict_proba()`
    - 24 最优超参数 `sklearn.model_selection.GridSearchCV()`
    - 25 事件预测 ``
    - 26 估算交通流量
    - 27 k-means聚类
    - 28 利用聚类实现矢量量化
    - 29 均值漂移聚类
    - 30 凝聚层次聚类
    - 31 评价聚类算法的效果
    - 32 DBSCAN
    - 33
2. `data`:相关训练数据文件
3. `mymodel`:保存的训练好的模型