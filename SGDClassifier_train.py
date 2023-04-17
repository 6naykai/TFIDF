import time
import joblib
from sklearn.linear_model import SGDClassifier
from sklearn.multiclass import OneVsRestClassifier
from TFIDF import X_train, y_train, format_time, data_name
from score import mkdir

# 模型保存路径
save_path = 'saved_model/SGDClassifier-{}/'.format(data_name)
mkdir(save_path)


# 逻辑回归随机梯度下降
knn_classify = SGDClassifier(loss='log')
knn_classify = OneVsRestClassifier(knn_classify, n_jobs=-1)
t0 = time.time()  # 记录训练开始时间
knn_classify.fit(X_train, y_train)
t1 = time.time()  # 记录训练结束时间
training_time = t1 - t0  # 训练时间
training_time = format_time(training_time)
print(training_time)

# 保存模型参数
joblib.dump(knn_classify, save_path + 'knn_classify.pkl')
