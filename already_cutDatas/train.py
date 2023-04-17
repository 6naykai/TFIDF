import time
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from dataloader import X_train, y_train, format_time, test_data_name, train_data_name
from score import mkdir

data_name = train_data_name + '+' + test_data_name
# 模型保存路径
save_path = 'saved_model/RandomForest-{}/'.format(data_name)
mkdir(save_path)

# 随机森林
knn_classify = RandomForestClassifier(bootstrap=True, oob_score=True, n_estimators=50, criterion='gini', n_jobs=-1)
knn_classify = OneVsRestClassifier(knn_classify, n_jobs=-1)
t0 = time.time()  # 记录训练开始时间
knn_classify.fit(X_train, y_train)
t1 = time.time()  # 记录训练结束时间
training_time = t1 - t0  # 训练时间
training_time = format_time(training_time)
print(training_time)

# 保存模型参数
joblib.dump(knn_classify, save_path + 'knn_classify.pkl')
