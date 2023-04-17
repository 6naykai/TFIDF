import joblib
from TFIDF import X_test, n_classes, y_test, data_name, classes
from score import score, model_result

# 模型保存路径
model_name = 'SGDClassifier'
save_path = 'saved_model/{}-{}/'.format(model_name, data_name)

# 加载模型
knn_classify = joblib.load(save_path + 'knn_classify.pkl')
# 计算score,并输出pr图
y_score = knn_classify.predict_proba(X_test)    # 样本预测概率
score(classes, n_classes, y_test, y_score, save_path)

# pred = dict()
# report = dict()
# confusion_mat = dict()
# for i in range(11):
#     pred[i] = (knn_classify.predict_proba(X_test)[:, i] >= 0.5).astype(bool)
#     report[i] = classification_report(y_test[:, i], pred[i], output_dict=True)
#     confusion_mat[i] = confusion_matrix(y_test[:, i], pred[i])
#     df = pd.DataFrame(report[i]).transpose()
#     df.to_csv("saved_model/RandomForest-total/test/r{}.csv".format(str(i)), index=True)
#     output = open("saved_model/RandomForest-total/test/c{}.xls".format(str(i)), 'w', encoding='gbk')
#     output.write('confuse_res20\n')
#     for k in range(len(confusion_mat[i])):
#         for j in range(len(confusion_mat[i][k])):
#             output.write(str(confusion_mat[i][k][j]))  # write函数不能写int类型的参数，所以使用str()转化
#             output.write('\t')
#         output.write('\n')  # 写完一行立马换行
#     output.close()


# 测试(.argmax为从one-hot逆编码)
pred = (knn_classify.predict(X_test)).argmax(axis=1)
y_test = y_test.argmax(axis=1)
model_result(classes, pred, y_test, save_path)
# # pred = (knn_classify.predict_proba(X_test)[:, 0] >= 0.5).astype(bool)
# # 计算分类结果信息并打印
# report = classification_report(y_test, pred, target_names=classes)
# # report = classification_report(y_test[:, 0], pred)
# print("分类结果：")
# print(report)
# # 计算多分类混淆矩阵并打印
# confusion_mat = confusion_matrix(y_test, pred)
# # confusion_mat = confusion_matrix(y_test[:, 0], pred)
# print("混淆矩阵：")
# print(confusion_mat)

# # 计算score(计算pr散点,拟合散点函数,输出对应结果)
# y_score = knn_classify.predict_proba(X_test)    # 样本预测概率
# for i in range(11):
#     pass
