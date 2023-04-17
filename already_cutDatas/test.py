import joblib
from dataloader import X_test, n_classes, y_test, classes, train_data_name, test_data_name
from score import score, model_result

data_name = train_data_name + '+' + test_data_name
# 模型保存路径
model_name = 'RandomForest'
save_path = 'saved_model/{}-{}/'.format(model_name, data_name)

# 加载模型
knn_classify = joblib.load(save_path + 'knn_classify.pkl')
# 计算score,并输出pr图
y_score = knn_classify.predict_proba(X_test)    # 样本预测概率
score(classes, n_classes, y_test, y_score, save_path)

# 测试(.argmax为从one-hot逆编码)
pred = (knn_classify.predict(X_test)).argmax(axis=1)
y_test = y_test.argmax(axis=1)
model_result(classes, pred, y_test, save_path)
