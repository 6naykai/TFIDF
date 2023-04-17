import csv
import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
import jieba
from sklearn.preprocessing import label_binarize
from score import count_labels


# 转化时间格式的函数
def format_time(time):
    elapsed_rounded = int(round(time))
    # 格式化为 hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


titles = []
labels = []
train_data_name = 'increase_add'
train_data_path = 'data/{}.csv'.format(train_data_name)
test_data_name = 'test'
test_data_path = 'data/{}.csv'.format(test_data_name)


# 写入数据集的函数
def im(path):
    with open(path, "rt", encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        count = 0
        for row in reader:
            titles.append(row[0])
            labels.append(row[1])
            count += 1


im(train_data_path)
train_data_len = len(titles)
im(test_data_path)
test_data_len = len(titles) - train_data_len

# 转化为向量模式
tfidf_vectorizer = TfidfVectorizer(binary=False, decode_error='ignore', tokenizer=jieba.cut)
tfidf_matrix = tfidf_vectorizer.fit_transform(titles)
print(tfidf_vectorizer.get_feature_names())

# label是str变量,转化为one-hot标签
classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12']
labels_one_hot = label_binarize(labels, classes=classes)
n_classes = labels_one_hot.shape[1]

# 切分数据集
X_train = tfidf_matrix[0:train_data_len, :]
X_test = tfidf_matrix[train_data_len:train_data_len+test_data_len, :]
y_train = labels_one_hot[0:train_data_len, :]
y_test = labels_one_hot[train_data_len:train_data_len+test_data_len, :]

# 统计样本数量,需转换为原来标签样式
count_labels(classes, y_train, y_test, labels)

