import csv
import datetime
from collections import Counter

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import jieba
from sklearn.preprocessing import label_binarize
from score import count_labels, mkdir
import pickle


# 转化时间格式的函数
def format_time(time):
    elapsed_rounded = int(round(time))
    # 格式化为 hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


titles = []
labels = []
data_name = 'total'
data_path = 'data/{}.csv'.format(data_name)


# 写入数据集的函数
def im(path):
    with open(path, "rt", encoding='utf-8-sig') as csvfile:
        reader = csv.reader(csvfile)
        count = 0
        for row in reader:
            titles.append(row[0])
            labels.append(row[1])
            count += 1


im(data_path)


# 转化为向量模式
tfidf_vectorizer = TfidfVectorizer(binary=False, decode_error='ignore', tokenizer=jieba.cut)
tfidf_matrix = tfidf_vectorizer.fit_transform(titles)
print(tfidf_vectorizer.get_feature_names())
# # 保存词向量以便后续使用
# mkdir('data/vectorizer')
# # pickle.dump(tfidf_vectorizer, open("data/vectorizer/{}-vectorizer.pickle".format(data_name), "wb"))
# # joblib.dump(tfidf_vectorizer, 'data/vectorizer/{}-vectorizer.pkl'.format(data_name))
# tfidftransformer_path = 'data/vectorizer/{}-vectorizer.pkl'.format(data_name)
# with open(tfidftransformer_path, 'wb') as fw:
#     pickle.dump(tfidf_vectorizer, fw)


# label是str变量,转化为one-hot标签
classes = ['0', '2', '3', '4', '6', '7', '8', '9', '10', '11', '12']
labels_one_hot = label_binarize(labels, classes=classes)
n_classes = labels_one_hot.shape[1]

# 切分数据集
X_train, X_test, y_train, y_test = train_test_split(tfidf_matrix, labels_one_hot, test_size=0.2, random_state=42,
                                                    stratify=labels_one_hot)


# 统计样本数量,需转换为原来标签样式
count_labels(classes, y_train, y_test, labels)
# label_y_train = [one_label.tolist().index(1) for one_label in y_train]  # 找到下标是1的位置
# for i in range(len(label_y_train)):
#     label_y_train[i] = classes[label_y_train[i]]
# label_y_test = [one_label.tolist().index(1) for one_label in y_test]  # 找到下标是1的位置
# for i in range(len(label_y_test)):
#     label_y_test[i] = classes[label_y_test[i]]
# a = Counter(label_y_train)  # 训练集里面各个标签的出现次数
# b = Counter(label_y_test)  # 测试集里面各个标签的出现次数
# c = Counter(labels)  # 未切分前各个标签的出现次数
# print(a)
# print(b)
# print(c)
