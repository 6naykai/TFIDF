import os
from collections import Counter
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score, classification_report, confusion_matrix
# from TFIDF import classes


def count_labels(classes, y_train, y_test, labels):
    # 统计样本数量,需转换为原来标签样式
    label_y_train = [one_label.tolist().index(1) for one_label in y_train]  # 找到下标是1的位置
    for i in range(len(label_y_train)):
        label_y_train[i] = classes[label_y_train[i]]
    label_y_test = [one_label.tolist().index(1) for one_label in y_test]  # 找到下标是1的位置
    for i in range(len(label_y_test)):
        label_y_test[i] = classes[label_y_test[i]]
    a = Counter(label_y_train)  # 训练集里面各个标签的出现次数
    b = Counter(label_y_test)  # 测试集里面各个标签的出现次数
    c = Counter(labels)  # 未切分前各个标签的出现次数
    print(a)
    print(b)
    print(c)


def score(classes, n_classes, y_test, y_score, save_path):
    precision = dict()
    recall = dict()
    average_precision = dict()  # AP分数
    pr_scores = dict()       # 服务外包比赛分数

    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(y_test[:, i],
                                                            y_score[:, i])
        # average_precision[i] = average_precision_score(y_test[:, i],
        #                                                y_score[:, i])

    # (2) A "macro-average": quantifying score on all classes jointly
    precision["macro"], recall["macro"], _ = precision_recall_curve(y_test.ravel(),
                                                                    y_score.ravel())

    # average_precision["macro"] = average_precision_score(y_test, y_score,
    #                                                      average="macro")
    #
    # print('Average precision score, macro-averaged over all classes: {0:0.2f}'.format(average_precision["macro"]))

    # for i in range(n_classes):
    #     draw_pr(recall, precision, average_precision, i)
    # draw_pr(recall, precision, average_precision, 'macro')

    # 绘图:pr分数
    for i in range(n_classes):
        pr_scores[i] = pr_precision_score(recall[i], precision[i], numTo_tag_name(classes, i), save_path).round(3)
    pr_scores['macro'] = pr_precision_score(recall['macro'], precision['macro'], 'all', save_path).round(3)
    # 保存分数
    output = open(save_path + "score.csv", 'w', encoding='gbk')
    output.write('tag,score\n')
    for i in range(n_classes):
        output.write('{},{}\n'.format(classes[i], pr_scores[i]))
    output.write('all,{}\n'.format(pr_scores['macro']))
    output.close()


def draw_pr(recall, precision, average_precision, name):
    # 绘图
    plt.figure()
    # 阶梯图
    plt.step(recall[name], precision[name], where='post')
    # plt.plot(recall['macro'], precision['macro'], where='post')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.0])
    plt.xlim([0.0, 1.0])
    if name == 'macro':
        plt.title(
            'Average precision score, macro-averaged over all classes: AP={0:0.3f}'.format(average_precision[name]))
    else:
        plt.title('Average precision score, class {0}: AP={1:0.3f}'.format(str(name), average_precision[name]))
    plt.show()


def pr_precision_score(x, y, name, save_path):
    """
    采用拟合多项式的方式计算score
    score = 0.5×p{r==0.7} + 0.3×p{r==0.8} + 0.2×p{r==0.9}
    :return: score值
    """
    z1 = np.polyfit(x, y, 7)  # 用4次多项式拟合，输出系数从高到0
    p1 = np.poly1d(z1)  # 使用次数合成多项式
    y_pre = p1(x)
    p1_07_08_09 = [1, 1, 1]
    loss_07_08_09 = [1, 1, 1]
    for recalls in x:
        if abs(recalls - 0.7) < loss_07_08_09[0]:
            loss_07_08_09[0] = abs(recalls-0.7)
            p1_07_08_09[0] = p1(recalls)
        if abs(recalls - 0.8) < loss_07_08_09[1]:
            loss_07_08_09[1] = abs(recalls-0.8)
            p1_07_08_09[1] = p1(recalls)
        if abs(recalls - 0.9) < loss_07_08_09[2]:
            loss_07_08_09[2] = abs(recalls-0.9)
            p1_07_08_09[2] = p1(recalls)
    # print(type(p1_07_08_09[0]))
    for i in range(3):
        if p1_07_08_09[i] > 1:
            p1_07_08_09[i] = 1
    # # 转化成float64
    # p1_07_08_09 = p1_07_08_09.astype(np.float64)
    # pr_score = 0.5*p1(0.7) + 0.3*p1(0.8) + 0.2*p1(0.9)
    pr_score = 0.5*p1_07_08_09[0] + 0.3*p1_07_08_09[1] + 0.2*p1_07_08_09[2]
    pr_score = np.float64(pr_score)
    plot_pic(x, y, y_pre, pr_score, name, save_path)
    print('Class {0} score: {1:0.3f}'.format(str(name), pr_score))
    return pr_score


def plot_pic(x, y, y_pre, pr_score, name, save_path):
    plt.figure()
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.05])
    plt.plot(x, y, '.')
    plt.plot(x, y_pre)
    plt.title('Class {0} score: {1:0.3f}'.format(str(name), pr_score))
    mkdir(save_path + 'figures')
    plt.savefig(save_path + 'figures/' + str(name) + '.png')
    # plt.show()


def mkdir(path):
    """
    创建文件夹的函数
    :param path: 文件夹路径
    :return: none
    """
    folder = os.path.exists(path)
    if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(path)  # makedirs 创建文件时如果路径不存在会创建这个路径


def numTo_tag_name(classes, i):
    # classes = ['0', '2', '3', '4', '6', '7', '8', '9', '10', '11', '12']
    return classes[i]


# 传入非one-hot标签,输出多分类模型结果
def model_result(classes, pred, y_test, save_path):
    report = classification_report(y_test, pred, target_names=classes, output_dict=True)
    df = pd.DataFrame(report).transpose().round(2)  # .round(2)保留两位小数
    df.to_csv(save_path + "report.csv", index=True)
    confusion_mat = confusion_matrix(y_test, pred)
    output = open(save_path + "matrix.csv", 'w', encoding='gbk')
    for i in range(len(classes)):
        output.write(',pre{}'.format(classes[i]))
    output.write('\n')
    for k in range(len(confusion_mat)):
        output.write('tru{},'.format(classes[k]))
        for j in range(len(confusion_mat[k])):
            output.write(str(confusion_mat[k][j]))  # write函数不能写int类型的参数，所以使用str()转化
            if j != len(confusion_mat[k]) - 1:
                output.write(',')
        output.write('\n')  # 写完一行立马换行
    output.close()
