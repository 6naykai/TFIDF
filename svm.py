import time
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, average_precision_score, precision_recall_curve
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from score import score
from TFIDF import format_time, X_train, y_train, y_test, X_test, n_classes

# # one-hot多分类标签
# Y_test = label_binarize(y_test, classes=[0, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12])
# n_classes = Y_test.shape[1]  # 有几列，就是几分类！
# print(n_classes)
# Y_train = label_binarize(y_train, classes=[0, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12])


# X_train, X_test, y_train, y_test = train_test_split(tfidf_matrix, labels, test_size=0.1, random_state=0)


# 设置分类器，这里实用的是SVC支持向量机
# clf = SVC(C=0.2, gamma=0.2, kernel='linear', probability=True, random_state=42)
knn_classify = OneVsRestClassifier(DecisionTreeClassifier(criterion="gini",
                                                          min_samples_leaf=3, max_depth=15))
# knn_classify = OneVsRestClassifier(clf, n_jobs=-1)
t0 = time.time()  # 记录训练开始时间
knn_classify.fit(X_train, y_train)
t1 = time.time()  # 记录训练结束时间
training_time = t1 - t0  # 训练时间
training_time = format_time(training_time)
print(training_time)

# # 测试
# # pred = knn_classify.predict(X_test).argmax(axis=1)
# yy_pred = knn_classify.predict(X_test)  # 预测出[[0.4,0.45],[0.8,0.3],[0.6,0.71]]
# pred = np.argmax(yy_pred, axis=1)  # 选择max值进行输出0,或1

# (1) For each class
precision = dict()
recall = dict()
average_precision = dict()

# 概率分数y_score ，是一个shape为(测试集条数, 分类种数)的矩阵。
# 比如你测试集有200条数据，模型是5分类，那矩阵就是(200,5)。
# 矩阵的第(i,j)元素代表第i条数据是第j类的概率。
y_score = knn_classify.predict_proba(X_test)
# y_score = knn_classify.oob_decision_function_(X_test)

score(n_classes, y_test, y_score)

# # 计算分类结果信息并打印
# report = classification_report(y_test, pred)
# print("分类结果：")
# print(report)
# # 计算多分类混淆矩阵并打印
# confusion_mat = confusion_matrix(y_test, pred)
# print("混淆矩阵：")
# print(confusion_mat)
