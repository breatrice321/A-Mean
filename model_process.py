import os
import time

import numpy as np
import pandas as pd

path = './model_summary/'
name = os.listdir(path)
for name_i in name:
    if name_i == "sequential":
        model = os.listdir(path + name_i + "/")
        for model_i in model:
            begin_time = time.time()
            data = pd.read_table(path + name_i + '/' + model_i, header=None)
            data.drop(data.head(4).index, inplace=True)  # 从头去掉 n 行
            data.drop(data.tail(5).index, inplace=True)  # 从尾去掉 n 行

            # print(data)
            # print(".........................")
            data = np.array(data)
            num = len(data)
            # temp_data = []
            for i in range(num):
                # print(data[i])
                if i >= len(data) - 1:
                    break
                if data[i] == '_________________________________________________________________':
                    data = np.delete(data, int(i), 0)
                    num = len(data)
                    i = i - 1


            print(model_i + ":" + str(len(data)))

            data = pd.DataFrame(data).astype(str)

            data[0] = data[0].str.replace(r', ', ',', regex=True)
            data[[0, 1, 2, 3, 4]] = data[0].str.split(r'\s+', expand=True)
            data = data.drop(columns=[3, 4])

            # 查找包含特定字符的行，并修改另一列的值
            search_char_1 = 'conv_pw'  # 要查找的字符
            search_char_2 = 'Conv2D'  # 要查找的字符
            search_char_3 = 'global_average_pooling'

            for index, row in data.iterrows():
                if search_char_1 in row[0] and search_char_2 in row[1]:
                    data.at[index, 1] = 'PointwiseConv2D'  # 在这里设置新的值
                elif search_char_3 in row[0]:
                    data.at[index, 1] = 'Global_Average_Pooling'  # 在这里设置新的值

            data[0] = data[1].str.replace(r'[()\[\]]', '', regex=True)
            data[2] = data[2].str.replace(r'[()\[\]]', '', regex=True)
            data[[1, 2, 3, 4]] = data[2].str.split(',', expand=True)
            data = pd.concat([data, pd.DataFrame(columns=[5, 6, 7, 8, 9, 10, 11, 12, 13])], axis=1)
            # data.iloc[:, 5:13] = 0
            data[13] = 0

            # 2:w, 3:h, 4:c, 5:conv, 6:add, 7:sub, 8:mul, 9:div, 10:ReLu, 11:max-pooling, 12:softmax, 13:topology
            data = np.array(data)

            for col in range(2, 5):
                data[:, col] = [int(x) if x is not None and str(x) != "" else 1 for x in data[:, col]]

            # 部分需要手动加入的信息
            if model_i == "MobileNet.txt":
                conv_k = 3
                act = 1  # 有额外的ReLu
            if model_i == "VGG16.txt":
                conv_k = 3
                act = 0  # 没有额外的ReLu，和Conv集成在一起
            if model_i == "VGG19.txt":
                conv_k = 3
                act = 0  # 没有额外的ReLu，和Conv集成在一起

            for j in range(len(data)):
                if j == 0:
                    data[0][5:13] = 0
                elif data[j][0] == "ZeroPadding2D" or data[j][0] == "Reshape" or data[j][0] == "Dropout" or data[j][0] == "Activation" or data[j][0] == "Flatten":
                    data[j][5:13] = 0
                elif data[j][0] == "Conv2D" and act == 1:
                    if int(data[j-1][2] / data[j][2]) == 2 and data[j-1][2] % data[j][2] == 1:
                        padding = 0
                        stride = 2
                        data[j][7:13] = 0
                        data[j][5] = int(((data[j - 1][2] - conv_k + padding) / stride + 1) * (
                                    (data[j - 1][2] - conv_k + padding) / stride + 1) * data[j - 1][4] * data[j][4])
                        data[j][6] = data[j][2] * data[j][3] * data[j][4] * (data[j-1][4] - 1)
                    elif int(data[j-1][2] / data[j][2]) == 2 and data[j-1][2] % data[j][2] == 0:
                        padding = 1
                        stride = 2
                        data[j][7:13] = 0
                        data[j][5] = int(((data[j - 1][2] - conv_k + padding) / stride + 1) * (
                                    (data[j - 1][2] - conv_k + padding) / stride + 1) * data[j - 1][4] * data[j][4])
                        data[j][6] = data[j][2] * data[j][3] * data[j][4] * (data[j - 1][4] - 1)
                    elif int(data[j - 1][2] / data[j][2]) == 1 and data[j - 1][2] % data[j][2] == 0:
                        padding = 2
                        stride = 1
                        data[j][7:13] = 0
                        data[j][5] = int(((data[j - 1][2] - conv_k + padding) / stride + 1) * ((data[j - 1][2] - conv_k + padding) / stride + 1) * data[j - 1][4] * data[j][4])
                        data[j][6] = data[j][2] * data[j][3] * data[j][4] * (data[j - 1][4] - 1)
                elif data[j][0] == "Conv2D" and act == 0:
                    if int(data[j-1][2] / data[j][2]) == 2 and data[j-1][2] % data[j][2] == 1:
                        padding = 0
                        stride = 2
                        data[j][7:13] = 0
                        data[j][5] = int(((data[j - 1][2] - conv_k + padding) / stride + 1) * (
                                    (data[j - 1][2] - conv_k + padding) / stride + 1) * data[j - 1][4] * data[j][4])
                        data[j][6] = data[j][2] * data[j][3] * data[j][4] * (data[j-1][4] - 1)
                        data[j][10] = data[j][5]
                    elif int(data[j-1][2] / data[j][2]) == 2 and data[j-1][2] % data[j][2] == 0:
                        padding = 1
                        stride = 2
                        data[j][7:13] = 0
                        data[j][5] = int(((data[j - 1][2] - conv_k + padding) / stride + 1) * (
                                    (data[j - 1][2] - conv_k + padding) / stride + 1) * data[j - 1][4] * data[j][4])
                        data[j][6] = data[j][2] * data[j][3] * data[j][4] * (data[j - 1][4] - 1)
                        data[j][10] = data[j][5]
                    elif int(data[j - 1][2] / data[j][2]) == 1 and data[j - 1][2] % data[j][2] == 0:
                        padding = 2
                        stride = 1
                        data[j][7:13] = 0
                        data[j][5] = int(((data[j - 1][2] - conv_k + padding) / stride + 1) * (
                                (data[j - 1][2] - conv_k + padding) / stride + 1) * data[j - 1][4] * data[j][4])
                        data[j][6] = data[j][2] * data[j][3] * data[j][4] * (data[j - 1][4] - 1)
                        data[j][10] = data[j][5]
                elif data[j][0] == "DepthwiseConv2D":
                    data[j][6:13] = 0
                    data[j][5] = data[j - 1][2] * data[j - 1][3] * data[j - 1][4]
                elif data[j][0] == "PointwiseConv2D":
                    data[j][6:13] = 0
                    data[j][5] = data[j - 1][2] * data[j - 1][3] * data[j][4]
                elif "Batch" in data[j][0]:
                    data[j][5:13] = 0
                    data[j][7] = data[j][2] * data[j][3] * data[j][4]
                    data[j][9] = data[j][7]
                elif data[j][0] == "ReLU":
                    data[j][5:13] = 0
                    data[j][10] = data[j][2] * data[j][3] * data[j][4]
                elif data[j][0] == "Global_Average_Pooling":
                    data[j][5:13] = 0
                    data[j][12] = data[j][2]
                elif data[j][0] == "MaxPooling2D":
                    data[j][5:13] = 0
                    data[j][11] = int((data[j-1][2] / 2) * (data[j-1][2] / 2) * data[j-1][4])
                elif data[j][0] == "Dense" and act == 0:
                    data[j][5:13] = 0
                    data[j][6] = (data[j-1][2]-1) * data[j][2]
                    data[j][8] = data[j-1][2] * data[j][2]
                    data[j][10] = data[j][2]

            # 将DataFrame保存到Excel文件
            end_time = time.time()
            print(model_i[:-len(".txt")] + "_time:" + str(end_time - begin_time))
            data = pd.DataFrame(data)
            data = data.drop(data.columns[1], axis=1)
            excel_file_path = './model_processing/' + model_i[:-len(".txt")] + '.xlsx'
            data.to_excel(excel_file_path, index=False, sheet_name='Sheet1')
            # print(data)
            # print(".........................")

    # else:
    if name_i == "non_sequential":
        model = os.listdir(path + name_i + '/')
        for model_i in model:
            begin_time = time.time()
            data = pd.read_table(path + name_i + '/' + model_i, header=None)
            data.drop(data.head(4).index, inplace=True)  # 从头去掉 n 行
            data.drop(data.tail(5).index, inplace=True)  # 从尾去掉 n 行
            data = np.array(data)
            num = len(data)
            for i in range(num):
                # print(data[i])
                if data[i] == '__________________________________________________________________________________________________':
                    data = np.delete(data, int(i), 0)

                    num = len(data)
                    if i >= len(data) - 1 :
                        break
                    i = i - 1

            print(model_i + ":" + str(len(data)))


            data = pd.DataFrame(data).astype(str)

            data[0] = data[0].str.replace(r', ', ',', regex=True)
            data[[0, 1, 2, 3, 4, 5]] = data[0].str.split(r'\s+', expand=True)
            data = data.drop(columns=[3, 5])

            data = pd.concat([data, pd.DataFrame(columns=[13])], axis=1)
            data[13] = 0

            serach_char_2 = 'GlobalAveragePooling'
            search_char_3 = 'BN'
            search_char_4 = 'relu'
            search_char_5 = 'Depth'
            search_char_6 = 'average_pooling'

            for i, value in data.iterrows():
                if value[0] == '':
                    data.at[i, 4] = data.at[i, 1]
                    data.at[i, 1] = ''

            for index, row in data.iterrows():
                if search_char_3 in row[0]:
                    data.at[index, 1] = 'BatchNormalization'
                elif search_char_4 in row[0]:
                    data.at[index, 1] = 'ReLU'
                elif search_char_5 in row[1]:
                    data.at[index, 1] = 'DepthwiseConv2D'
                elif search_char_6 in row[0] or serach_char_2 in row[1]:
                    data.at[index, 1] = 'Global_Average_Pooling'

            data[4] = data[4].str.replace(r'\[0\]', '', regex=True)

            first_duplicate = data[data.duplicated([4], keep="last")]
            first_duplicate_index = first_duplicate.index.tolist()

            first_index = 0
            for i, value in data.iterrows():
                if value[0] == '':
                    temp = data.at[i-1, 4]
                    temp_1 = data.at[i, 4]

                    # 2个分支都要节点
                    if data.at[first_duplicate_index[first_index], 4] != temp and data.at[first_duplicate_index[first_index], 4] != temp_1:
                        begin = first_duplicate_index[first_index]
                        end = i-2
                        add_index = 1

                        branch = begin - 1
                        while branch < end + 1:
                            if data.at[branch, 0] == data.at[branch + add_index, 4]:
                                if add_index > 1 and data.at[branch, 13] == 2:
                                    data.at[branch + add_index, 13] = 2
                                    branch = branch + add_index
                                    add_index = 1
                                elif add_index == 1:
                                    data.at[branch + add_index, 13] = 2
                                    branch = branch + add_index
                                    add_index = 1
                            else:
                                add_index = add_index + 1

                        for branch_1 in range(begin, end + 1):
                            if data.at[branch_1, 13] == 0:
                                data.at[branch_1, 13] = 3

                        first_index = first_index + 1

                    # 1个节点有分支，一个节点没有
                    else:
                        current_index = i - 2
                        precious_index = first_duplicate_index[first_index]
                        data[precious_index:current_index + 1, 13] = 1
                        # print(data.iloc[precious_index:current_index + 1, 13])
                        first_index = first_index + 1

            for j, values in data.iterrows():
                if values[0] == '':
                    data.at[j-1, 13] = 0
                    data.drop([j], axis=0, inplace=True)

            data[0] = data[1].str.replace(r'[()\[\]]', '', regex=True)
            data[2] = data[2].str.replace(r'[()\[\]]', '', regex=True)
            data[[1, 2, 3, 4]] = data[2].str.split(',', expand=True)
            data = pd.concat([data, pd.DataFrame(columns=[5, 6, 7, 8, 9, 10, 11, 12])], axis=1)
            # data.iloc[:, 5:13] = 0
            new_column_order = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
            data = data[new_column_order]


            # 2:w, 3:h, 4:c, 5:conv, 6:add, 7:sub, 8:mul, 9:div, 10:ReLu, 11:max-pooling, 12:softmax, 13:topology
            data = np.array(data)

            for col in range(2, 5):
                data[:, col] = [int(x) if x is not None and str(x) != "" else 1 for x in data[:, col]]

            # 部分需要手动加入的信息
            if model_i == "MobileNetV2.txt":
                act = 1  # 有额外的ReLu
            if model_i == "ResNet50.txt":
                act = 1  # 没有额外的ReLu，和Conv集成在一起

            for j in range(len(data)):
                if j == 0:
                    data[0][5:13] = 0
                elif data[j][0] == "ZeroPadding2D" or data[j][0] == "Reshape" or data[j][0] == "Dropout" or data[j][0] == "Flatten":
                    data[j][5:13] = 0
                elif data[j][0] == "Conv2D" and act == 1:
                    if int(data[j - 1][2] / data[j][2]) == 2:
                        if data[j - 1][2] % data[j][2] == 1:
                            padding = 0
                            stride = 2
                            conv_k = data[j - 1][2] - (data[j][2] - 1) * stride + padding
                            data[j][7:13] = 0
                            data[j][5] = int(((data[j - 1][2] - conv_k + padding) / stride + 1) * (
                                    (data[j - 1][2] - conv_k + padding) / stride + 1) * data[j - 1][4] * data[j][4])
                            data[j][6] = data[j][2] * data[j][3] * data[j][4] * (data[j - 1][4] - 1)
                        elif data[j - 1][2] % data[j][2] == 0:
                            padding = 1
                            stride = 2
                            conv_k = data[j - 1][2] - (data[j][2] - 1) * stride + padding
                            data[j][7:13] = 0
                            data[j][5] = int(((data[j - 1][2] - conv_k + padding) / stride + 1) * (
                                    (data[j - 1][2] - conv_k + padding) / stride + 1) * data[j - 1][4] * data[j][4])
                            data[j][6] = data[j][2] * data[j][3] * data[j][4] * (data[j - 1][4] - 1)
                        elif data[j - 1][2] % data[j][2] > 1:
                            padding = 3
                            stride = 2
                            conv_k = 7
                            data[j][7:13] = 0
                            data[j][5] = int(((data[j - 1][2] - conv_k + padding) / stride + 1) * (
                                    (data[j - 1][2] - conv_k + padding) / stride + 1) * data[j - 1][4] * data[j][4])
                            data[j][6] = data[j][2] * data[j][3] * data[j][4] * (data[j - 1][4] - 1)

                    elif int(data[j - 1][2] / data[j][2]) == 1:
                        if data[j - 1][2] % data[j][2] == 0:
                            padding = 2
                            stride = 1
                            conv_k = data[j - 1][2] - (data[j][2] - 1) * stride + padding
                            data[j][7:13] = 0
                            data[j][5] = int(((data[j - 1][2] - conv_k + padding) / stride + 1) * (
                                    (data[j - 1][2] - conv_k + padding) / stride + 1) * data[j - 1][4] * data[j][4])
                            data[j][6] = data[j][2] * data[j][3] * data[j][4] * (data[j - 1][4] - 1)
                    # elif int()
                elif data[j][0] == "Conv2D" and act == 0:
                    if int(data[j - 1][2] / data[j][2]) == 2 and data[j - 1][2] % data[j][2] == 1:
                        padding = 0
                        stride = 2
                        conv_k = data[j - 1][2] - (data[j][2] - 1) * stride + padding
                        data[j][7:13] = 0
                        data[j][5] = int(((data[j - 1][2] - conv_k + padding) / stride + 1) * (
                                (data[j - 1][2] - conv_k + padding) / stride + 1) * data[j - 1][4] * data[j][4])
                        data[j][6] = data[j][2] * data[j][3] * data[j][4] * (data[j - 1][4] - 1)
                        data[j][10] = data[j][5]
                    elif int(data[j - 1][2] / data[j][2]) == 2 and data[j - 1][2] % data[j][2] == 0:
                        padding = 1
                        stride = 2
                        conv_k = data[j - 1][2] - (data[j][2] - 1) * stride + padding
                        data[j][7:13] = 0
                        data[j][5] = int(((data[j - 1][2] - conv_k + padding) / stride + 1) * (
                                (data[j - 1][2] - conv_k + padding) / stride + 1) * data[j - 1][4] * data[j][4])
                        data[j][6] = data[j][2] * data[j][3] * data[j][4] * (data[j - 1][4] - 1)
                        data[j][10] = data[j][5]
                    elif int(data[j - 1][2] / data[j][2]) == 1 and data[j - 1][2] % data[j][2] == 0:
                        padding = 2
                        stride = 1
                        conv_k = data[j - 1][2] - (data[j][2] - 1) * stride + padding
                        data[j][7:13] = 0
                        data[j][5] = int(((data[j - 1][2] - conv_k + padding) / stride + 1) * (
                                (data[j - 1][2] - conv_k + padding) / stride + 1) * data[j - 1][4] * data[j][4])
                        data[j][6] = data[j][2] * data[j][3] * data[j][4] * (data[j - 1][4] - 1)
                        data[j][10] = data[j][5]
                elif data[j][0] == "DepthwiseConv2D":
                    data[j][6:13] = 0
                    data[j][5] = data[j - 1][2] * data[j - 1][3] * data[j - 1][4]
                elif data[j][0] == "PointwiseConv2D":
                    data[j][6:13] = 0
                    data[j][5] = data[j - 1][2] * data[j - 1][3] * data[j][4]
                elif "Batch" in data[j][0]:
                    data[j][5:13] = 0
                    data[j][7] = data[j][2] * data[j][3] * data[j][4]
                    data[j][9] = data[j][7]
                elif data[j][0] == "ReLU" or data[j][0] == "Activation":
                    data[j][5:13] = 0
                    data[j][10] = data[j][2] * data[j][3] * data[j][4]
                elif data[j][0] == "Global_Average_Pooling":
                    data[j][5:13] = 0
                    data[j][12] = data[j][2]
                elif data[j][0] == "MaxPooling2D":
                    data[j][5:13] = 0
                    data[j][11] = int((data[j - 1][2] / 2) * (data[j - 1][2] / 2) * data[j - 1][4])
                elif data[j][0] == "Dense" and act == 0:
                    data[j][5:13] = 0
                    data[j][6] = (data[j - 1][2] - 1) * data[j][2]
                    data[j][8] = data[j - 1][2] * data[j][2]
                    data[j][10] = data[j][2]
                elif data[j][0] == "Dense" and act == 1:
                    data[j][5:13] = 0
                    data[j][6] = (data[j - 1][2] - 1) * data[j][2]
                    data[j][8] = data[j - 1][2] * data[j][2]
                elif data[j][0] == "Add":
                    data[j][5:13] = 0
                    data[j][6] = data[j][2] * data[j][3] * data[j][4]

            # 将DataFrame保存到Excel文件
            end_time = time.time()
            print(model_i[:-len(".txt")] + "_time:" + str(end_time - begin_time))
            data = pd.DataFrame(data)
            data = data.drop(data.columns[1], axis=1)
            excel_file_path = './model_processing/' + model_i[:-len(".txt")] + '.xlsx'
            data.to_excel(excel_file_path, index=False, sheet_name='Sheet1')
            # print(data)
            # print(".........................")



