import csv
import time

print(csv.field_size_limit())
# 扩大字段大小
csv.field_size_limit(500 * 1024 * 1024)

# # 分片处理csv文件
# def chunk_csv(file1, chunk_size=500):
#     """
#     输入csv文件、一片包含的数量（可自定义）
#     """
#     chunk = []  # 存放500个数据，每个是一行，dict格式
#     id_list = []
#     time_start = time.time()
#     with open(file1, "r", newline="", encoding="utf-8") as csvfile:
#         csvreader = csv.reader(csvfile)
#         cnt = 0
#         fieldnames = ["name1", "name2"]  # 列名
#         for row in csvreader:
#             if row[0] == "name1":
#                 continue
#             try:
#                 tmpdict = {fieldnames[idx]: row[idx] for idx in range(len(fieldnames))}
#                 id_ = row[0]
#             except Exception as e:
#                 continue
#             chunk.append(tmpdict)
#             id_list.append(id_)
#             cnt += 1
#             if cnt == chunk_size:  # 满足一片的数量之后，开始处理
#                 cnt = 0
#                 # 处理获取feature
#                 a = time.time()
#                 # 可以加入你要对该片数据处理的函数，参数可以为chunk_size、id_list等
#                 read_data2(xxxx)
#                 b = time.time()
#                 print('500处理完成，耗时{}'.format(b - a))
#                 # response = smoke_test(chunk, port, debug)
#                 chunk = []
#         # 剩下的最后一片，如果不能均分的话
#         if chunk:
#             read_data2(curs, id_list)
#         end_time = time.time()
#         print('{}文件处理完成共耗时：{}'.format(file1, end_time - time_start))
#

all_data = []
all_data_count = 0
tag = {'0': 0, '1': 0, '2': 0, '3': 0, '4': 0, '5': 0, '6': 0, '7': 0, '8': 0, '9': 0, '10': 0, '11': 0, '12': 0}
train_text_path = "already_cutDatas/data/test.csv"
with open(train_text_path, "rt", encoding='utf-8-sig') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        # all_data.append(row[1])
        try:
            tag[row[1]] += 1
            all_data_count += 1
        except:
            print(row[0])
            print(row[1])
            print(row[2])
            print("数据总数：", all_data_count)
            print(tag)
        # if len(all_data) == 8524430:
        #     all_data_count += len(all_data)
        #     all_data = []
            # break
        # print("(", len(col1), ",", len(col2), ")")
    print("数据总数：", all_data_count)
    print(tag)
csvfile.close()
