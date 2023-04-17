# import numpy as np
# from sklearn.metrics import precision_recall_curve
#
#
# y_true = np.array([[0, 0, 1, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
# y_scores = np.array([[0.1, 0.4, 0.35, 0.8], [0.85, 0.64, 0.38, 0.5], [0.7, 0.4, 0.9, 0.1], [0.4, 0.25, 0.39, 0.46]])
# precision, recall, thresholds = precision_recall_curve(y_true, y_scores, 4)
#
# print(precision)
# print(recall)
# print(thresholds)
# # 只能2分类？？？

'''
Author: CloudSir
Date: 2021-08-01 13:40:50
LastEditTime: 2021-08-02 09:41:54
LastEditors: CloudSir
Description: Python拟合多项式
https://github.com/cloudsir
'''
# import matplotlib.pyplot as plt
# import numpy as np
#
# x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# y = [2.83, 9.53, 14.52, 21.57, 38.26, 53.92, 73.15, 101.56, 129.54, 169.75, 207.59]
# z1 = np.polyfit(x, y, 4)  # 用3次多项式拟合，输出系数从高到0
# p1 = np.poly1d(z1)  # 使用次数合成多项式
# y_pre = p1(x)
# print(p1(5))
# print(p1)
# print(y_pre)
# plt.plot(x, y, '.')
# plt.plot(x, y_pre)
# plt.show()

# import os
# os.mkdir('test')
