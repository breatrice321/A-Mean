import math
import os
import time

import numpy as np
import pandas as pd

# 从键盘获取整数输入
dataset = int(input("请输入您的选择: "))

# 打印用户输入
print("please choose datasets，ImageNet for 1，cifar-100 for 2，cifar-10 for 3，stl-10 for 4：", dataset)

begin = time.time()
path = './model_processing/'
name = os.listdir(path)
for name_i in name:
    begin = time.time()
    Data = pd.read_excel(path+name_i)

    # Remove the first column using .iloc
    # data = data.iloc[:, 2:]
    data = Data.iloc[:, 1:-1]
    depth = Data.iloc[:, -1]
    # print(data)
    data = np.array(data)
    # print(data)
    weight = []
    SDC_op = []
    SDC_layer = []
    sum = 0
    SDC_basic = 1/8
    SDC_conv = SDC_basic * 2
    SDC_add = SDC_basic
    SDC_sub = SDC_basic
    SDC_mul = SDC_basic
    SDC_div = SDC_basic
    SDC_ReLu = (0.5 + SDC_basic * 2) / 2
    SDC_max_pool = SDC_basic * 2
    SDC_avg_pool = 1

    # SDC_conv = 0.22
    # SDC_add = 0.1
    # SDC_sub = 0.02
    # SDC_mul = 0.15
    # SDC_div = 0.15
    # SDC_ReLu = 0.3
    # SDC_max_pool = 0.35
    # SDC_avg_pool = 1


    rows, columns = data.shape
    for i in range(rows):
        weight_layer = int(data[i][0]) * int(data[i][1]) * int(data[i][2])
        weight.append(weight_layer)

        sum = sum + weight_layer
        op = int(data[i][3]) + int(data[i][4]) + int(data[i][5]) + int(data[i][6]) + int(data[i][7]) + int(
            data[i][8]) + int(data[i][9]) + int(data[i][10])
        if op == 0:
            conv = 0
            add = 0
            sub = 0
            mul = 0
            div = 0
            ReLu = 0
            Max_pooling = 0
            avg_pooling = 0
        else:
            conv = int(data[i][3]) / op
            add = int(data[i][4]) / op
            sub = int(data[i][5]) / op
            mul = int(data[i][6]) / op
            div = int(data[i][7]) / op
            ReLu = int(data[i][8]) / op
            Max_pooling = int(data[i][9]) / op
            avg_pooling = int(data[i][10]) / op

        # # different operations
        SDC_op_layer = SDC_conv * conv + SDC_add * add + SDC_sub * sub + SDC_mul * mul + SDC_div * div + SDC_ReLu * ReLu + SDC_max_pool * Max_pooling + SDC_avg_pool *avg_pooling
        # # one operation (conv)
        # SDC_op_layer = SDC_conv * (conv + add + sub + mul + div + ReLu + Max_pooling + avg_pooling)

        SDC_op.append(SDC_op_layer)

    for i in range(rows):
        if SDC_op[i] == 0:
            weight[i] = 0


    for i in range(rows):

        weight[i] = (weight[i] / sum) / (int(depth[i])-math.log(int(depth[i]))-0.5)


    # # no topology
    # for i in range(rows):
    #     SDC_layer.append(SDC_op[i])


    # topology
    temp_SDC_1 = 0
    temp_SDC_1_count = 0
    temp_SDC_2 = 0
    temp_SDC_2_count = 0
    for i in range(rows):
        topology = int(data[i][11])
        if i < rows - 1:
            topology_next = int(data[i + 1][11])

        if topology == 0:
            SDC_layer.append(SDC_op[i])

        elif topology == 1:

            if topology_next == 1:
                if SDC_op[i] > temp_SDC_1:
                    temp_SDC_1 = SDC_op[i]
                temp_SDC_1_count = temp_SDC_1_count + 1
            else:
                if SDC_op[i] > temp_SDC_1:
                    temp_SDC_1 = SDC_op[i]
                temp_SDC_1_count = temp_SDC_1_count + 1
                for j in range(temp_SDC_1_count):
                    SDC_layer.append(temp_SDC_1)
                temp_SDC_1 = 0
                temp_SDC_1_count = 0


        elif topology == 2:
            if SDC_op[i] > temp_SDC_2:
                temp_SDC_2 = SDC_op[i]
            temp_SDC_2_count = temp_SDC_2_count + 1

            # SDC_layer.append(SDC_op[i])
        elif topology == 3:
            if topology_next == 3:
                if SDC_op[i] > temp_SDC_2:
                    temp_SDC_2 = SDC_op[i]
                temp_SDC_2_count = temp_SDC_2_count + 1
            else:
                if SDC_op[i] > temp_SDC_2:
                    temp_SDC_2 = SDC_op[i]
                temp_SDC_2_count = temp_SDC_2_count + 1
                for j in range(temp_SDC_2_count):
                    SDC_layer.append(temp_SDC_2)
                temp_SDC_2 = 0
                temp_SDC_2_count = 0

            # SDC_layer.append(SDC_op[i])
        else:
            continue


    SDC_sum = 0
    for n in range(len(SDC_layer)):
        SDC_sum = SDC_sum + SDC_layer[n] * weight[n]

    if dataset == 1:
        if name_i == "VGG16.xlsx":
            VGG16_acc_no_FI = 0.7043
            VGG16_SCM_no_FI = 0.0211
            VGG16_nonSCM_no_FI = 1-VGG16_SCM_no_FI
            VGG16_SCM = ((VGG16_nonSCM_no_FI-VGG16_acc_no_FI)/2 + VGG16_acc_no_FI/(2+0.5)) * SDC_sum + VGG16_SCM_no_FI
            VGG16_acc = 1 - (VGG16_acc_no_FI * SDC_sum + (1-VGG16_acc_no_FI))
            print("VGG16_SDC:" + str(SDC_sum))
            print("VGG16_SCM:" + str(VGG16_SCM))
            print("VGG16_accuracy:" + str(VGG16_acc))
            end = time.time()
            print("VGG16_time:" + str(end - begin))
            print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        elif name_i == "VGG19.xlsx":
            VGG19_acc = 0.7118
            VGG19_SCM_no_FI = 0.0204
            VGG19_nonSCM_no_FI = 1-VGG19_SCM_no_FI
            VGG19_SCM = ((VGG19_nonSCM_no_FI-VGG19_acc)/2 + VGG19_acc/2.5) * SDC_sum + VGG19_SCM_no_FI
            print("VGG19_SDC:" + str(SDC_sum))
            print("VGG19_SCM:" + str(VGG19_SCM))
            print("VGG19_accuracy:" + str(1-(VGG19_acc * SDC_sum+(1-VGG19_acc))))
            end = time.time()
            print("VGG19_time:" + str(end - begin))
            print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        elif name_i == "MobileNet.xlsx":
            MobileNet_acc = 0.7028
            MobileNet_SCM_no_FI = 0.0243
            MobileNet_nonSCM_no_FI = 1-MobileNet_SCM_no_FI
            MobileNet_SCM = ((MobileNet_nonSCM_no_FI - MobileNet_acc)/ 2+ MobileNet_acc/2.5) * SDC_sum + MobileNet_SCM_no_FI
            print("MobileNet_SDC:" + str(SDC_sum))
            print("MobileNet_SCM:" + str(MobileNet_SCM))
            print("MobileNet_accuracy:" + str(1-(MobileNet_acc * SDC_sum+(1-MobileNet_acc))))
            end = time.time()
            print("MobileNet_time:" + str(end - begin))
            print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        elif name_i == "MobileNetV2.xlsx":
            MobileNetV2_acc = 0.7071
            MobileNetV2_SCM_no_FI = 0.0221
            MobileNetV2_nonSCM_no_FI = 1-MobileNetV2_SCM_no_FI
            MobileNetV2_SCM = ((MobileNetV2_nonSCM_no_FI - MobileNetV2_acc)/2 +MobileNetV2_acc/2.5) * SDC_sum + MobileNetV2_SCM_no_FI
            print("MobileNetV2_SDC:" + str(SDC_sum))
            print("MobileNetV2_SCM:" + str(MobileNetV2_SCM))
            print("MobileNetV2_accuracy:" + str(1-(MobileNetV2_acc * SDC_sum+(1-MobileNetV2_acc))))
            end = time.time()
            print("MobileNetV2_time:" + str(end - begin))
            print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        elif name_i == "ResNet50.xlsx":
            ResNet50_acc = 0.7458
            ResNet50_SCM_no_FI = 0.0186
            ResNet50_nonSCM_no_FI = 1-ResNet50_SCM_no_FI
            ResNet50_SCM = ((ResNet50_nonSCM_no_FI-ResNet50_acc)/2 + ResNet50_acc/2.5) * SDC_sum + ResNet50_SCM_no_FI
            print("ResNet50_SDC:" + str(SDC_sum))
            print("ResNet50_SCM:" + str(ResNet50_SCM))
            print("ResNet50_accuracy:" + str(1-(ResNet50_acc * SDC_sum+(1-ResNet50_acc))))
            end = time.time()
            print("ResNet50_time:" + str(end - begin))
            print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")


    elif dataset == 2:
        if name_i == "VGG16.xlsx":
            VGG16_acc = 0.6416
            VGG16_SCM_no_FI = 0.0681
            VGG16_nonSCM_no_FI = 1 - VGG16_SCM_no_FI
            VGG16_SCM = ((VGG16_nonSCM_no_FI-VGG16_acc)/2 + VGG16_acc/(2+0.5)) * SDC_sum + VGG16_SCM_no_FI
            print("VGG16_SDC:" + str(SDC_sum))
            print("VGG16_SCM:" + str(VGG16_SCM))
            print("VGG16_accuracy:" + str(1 - (VGG16_acc * SDC_sum + (1 - VGG16_acc))))
            end = time.time()
            print("VGG16_time:" + str(end - begin))
            print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        elif name_i == "VGG19.xlsx":
            VGG19_acc = 0.6382
            VGG19_SCM_no_FI = 0.0681
            VGG19_nonSCM_no_FI = 1 - VGG19_SCM_no_FI
            VGG19_SCM = ((VGG19_nonSCM_no_FI-VGG19_acc)/2 + VGG19_acc/2.5) * SDC_sum + VGG19_SCM_no_FI
            print("VGG19_SDC:" + str(SDC_sum))
            print("VGG19_SCM:" + str(VGG19_SCM))
            print("VGG19_accuracy:" + str(1 - (VGG19_acc * SDC_sum + (1 - VGG19_acc))))
            end = time.time()
            print("VGG19_time:" + str(end - begin))
            print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        elif name_i == "MobileNet.xlsx":
            MobileNet_acc = 0.596
            MobileNet_SCM_no_FI = 0.0828
            MobileNet_nonSCM_no_FI = 1 - MobileNet_SCM_no_FI
            MobileNet_SCM = ((MobileNet_nonSCM_no_FI - MobileNet_acc)/ 2+ MobileNet_acc/2.5) * SDC_sum + MobileNet_SCM_no_FI
            print("MobileNet_SDC:" + str(SDC_sum))
            print("MobileNet_SCM:" + str(MobileNet_SCM))
            print("MobileNet_accuracy:" + str(1 - (MobileNet_acc * SDC_sum + (1 - MobileNet_acc))))
            end = time.time()
            print("MobileNet_time:" + str(end - begin))
            print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        elif name_i == "MobileNetV2.xlsx":
            MobileNetV2_acc = 0.3179
            MobileNetV2_SCM_no_FI = 0.1579
            MobileNetV2_nonSCM_no_FI = 1 - MobileNetV2_SCM_no_FI
            MobileNetV2_SCM = ((MobileNetV2_nonSCM_no_FI - MobileNetV2_acc)/2 +MobileNetV2_acc/2.5) * SDC_sum + MobileNetV2_SCM_no_FI
            print("MobileNetV2_SDC:" + str(SDC_sum))
            print("MobileNetV2_SCM:" + str(MobileNetV2_SCM))
            print("MobileNetV2_accuracy:" + str(1 - (MobileNetV2_acc * SDC_sum + (1 - MobileNetV2_acc))))
            end = time.time()
            print("MobileNetV2_time:" + str(end - begin))
            print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        elif name_i == "ResNet50.xlsx":
            ResNet50_acc = 0.5472
            ResNet50_SCM_no_FI = 0.0984
            ResNet50_nonSCM_no_FI = 1 - ResNet50_SCM_no_FI
            ResNet50_SCM = ((ResNet50_nonSCM_no_FI-ResNet50_acc)/2 + ResNet50_acc/2.5) * SDC_sum + ResNet50_SCM_no_FI
            print("ResNet50_SDC:" + str(SDC_sum))
            print("ResNet50_SCM:" + str(ResNet50_SCM))
            print("ResNet50_accuracy:" + str(1 - (ResNet50_acc * SDC_sum + (1 - ResNet50_acc))))
            end = time.time()
            print("ResNet50_time:" + str(end - begin))
            print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")


    elif dataset == 3:
        if name_i == "VGG16.xlsx":
            VGG16_acc = 0.9028
            VGG16_SCM_no_FI = 0.0207
            VGG16_nonSCM_no_FI = 1 - VGG16_SCM_no_FI
            VGG16_SCM = ((VGG16_nonSCM_no_FI-VGG16_acc)/2 + VGG16_acc/(2+0.5)) * SDC_sum + VGG16_SCM_no_FI
            print("VGG16_SDC:" + str(SDC_sum))
            print("VGG16_SCM:" + str(VGG16_SCM))
            print("VGG16_accuracy:" + str(1 - (VGG16_acc * SDC_sum + (1 - VGG16_acc))))
            end = time.time()
            print("VGG16_time:" + str(end - begin))
            print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        elif name_i == "VGG19.xlsx":
            VGG19_acc = 0.912
            VGG19_SCM_no_FI = 0.0218
            VGG19_nonSCM_no_FI = 1 - VGG19_SCM_no_FI
            VGG19_SCM = ((VGG19_nonSCM_no_FI-VGG19_acc)/2 + VGG19_acc/2.5) * SDC_sum + VGG19_SCM_no_FI
            print("VGG19_SDC:" + str(SDC_sum))
            print("VGG19_SCM:" + str(VGG19_SCM))
            print("VGG19_accuracy:" + str(1 - (VGG19_acc * SDC_sum + (1 - VGG19_acc))))
            end = time.time()
            print("VGG19_time:" + str(end - begin))
            print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        elif name_i == "MobileNet.xlsx":
            MobileNet_acc = 0.8345
            MobileNet_SCM_no_FI = 0.0483
            MobileNet_nonSCM_no_FI = 1 - MobileNet_SCM_no_FI
            MobileNet_SCM = ((MobileNet_nonSCM_no_FI - MobileNet_acc)/ 2+ MobileNet_acc/2.5) * SDC_sum + MobileNet_SCM_no_FI
            print("MobileNet_SDC:" + str(SDC_sum))
            print("MobileNet_SCM:" + str(MobileNet_SCM))
            print("MobileNet_accuracy:" + str(1 - (MobileNet_acc * SDC_sum + (1 - MobileNet_acc))))
            end = time.time()
            print("MobileNet_time:" + str(end - begin))
            print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        elif name_i == "MobileNetV2.xlsx":
            MobileNetV2_acc = 0.8305
            MobileNetV2_SCM_no_FI = 0.04
            MobileNetV2_nonSCM_no_FI = 1 - MobileNetV2_SCM_no_FI
            MobileNetV2_SCM = ((MobileNetV2_nonSCM_no_FI - MobileNetV2_acc)/2 +MobileNetV2_acc/2.5) * SDC_sum + MobileNetV2_SCM_no_FI
            print("MobileNetV2_SDC:" + str(SDC_sum))
            print("MobileNetV2_SCM:" + str(MobileNetV2_SCM))
            print("MobileNetV2_accuracy:" + str(1 - (MobileNetV2_acc * SDC_sum + (1 - MobileNetV2_acc))))
            end = time.time()
            print("MobileNetV2_time:" + str(end - begin))
            print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        elif name_i == "ResNet50.xlsx":
            ResNet50_acc = 0.8765
            ResNet50_SCM_no_FI = 0.0319
            ResNet50_nonSCM_no_FI = 1 - ResNet50_SCM_no_FI
            ResNet50_SCM = ((ResNet50_nonSCM_no_FI - ResNet50_acc) / 2 + ResNet50_acc / 2.5) * SDC_sum + ResNet50_SCM_no_FI
            print("ResNet50_SDC:" + str(SDC_sum))
            print("ResNet50_SCM:" + str(ResNet50_SCM))
            print("ResNet50_accuracy:" + str(1 - (ResNet50_acc * SDC_sum + (1 - ResNet50_acc))))
            end = time.time()
            print("ResNet50_time:" + str(end - begin))
            print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")


    elif dataset == 4:
        if name_i == "VGG16.xlsx":
            VGG16_acc = 0.3905
            VGG16_SCM_no_FI = 0.279
            VGG16_nonSCM_no_FI = 1 - VGG16_SCM_no_FI
            VGG16_SCM = ((VGG16_nonSCM_no_FI-VGG16_acc)/2 + VGG16_acc/(2+0.5)) * SDC_sum + VGG16_SCM_no_FI
            print("VGG16_SDC:" + str(SDC_sum))
            print("VGG16_SCM:" + str(VGG16_SCM))
            print("VGG16_accuracy:" + str(1 - (VGG16_acc * SDC_sum + (1 - VGG16_acc))))
            end = time.time()
            print("VGG16_time:" + str(end - begin))
            print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        elif name_i == "VGG19.xlsx":
            VGG19_acc = 0.31475
            VGG19_SCM_no_FI = 0.3335
            VGG19_nonSCM_no_FI = 1 - VGG19_SCM_no_FI
            VGG19_SCM = ((VGG19_nonSCM_no_FI-VGG19_acc)/2 + VGG19_acc/2.5) * SDC_sum + VGG19_SCM_no_FI
            print("VGG19_SDC:" + str(SDC_sum))
            print("VGG19_SCM:" + str(VGG19_SCM))
            print("VGG19_accuracy:" + str(1 - (VGG19_acc * SDC_sum + (1 - VGG19_acc))))
            end = time.time()
            print("VGG19_time:" + str(end - begin))
            print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        elif name_i == "MobileNet.xlsx":
            MobileNet_acc = 0.32425
            MobileNet_SCM_no_FI = 0.311375
            MobileNet_nonSCM_no_FI = 1 - MobileNet_SCM_no_FI
            MobileNet_SCM = ((MobileNet_nonSCM_no_FI - MobileNet_acc)/ 2+ MobileNet_acc/2.5) * SDC_sum + MobileNet_SCM_no_FI
            print("MobileNet_SDC:" + str(SDC_sum))
            print("MobileNet_SCM:" + str(MobileNet_SCM))
            print("MobileNet_accuracy:" + str(1 - (MobileNet_acc * SDC_sum + (1 - MobileNet_acc))))
            end = time.time()
            print("MobileNet_time:" + str(end - begin))
            print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        elif name_i == "MobileNetV2.xlsx":
            MobileNetV2_acc = 0.3015
            MobileNetV2_SCM_no_FI = 0.28325
            MobileNetV2_nonSCM_no_FI = 1 - MobileNetV2_SCM_no_FI
            MobileNetV2_SCM = ((MobileNetV2_nonSCM_no_FI - MobileNetV2_acc)/2 +MobileNetV2_acc/2.5) * SDC_sum + MobileNetV2_SCM_no_FI
            print("MobileNetV2_SDC:" + str(SDC_sum))
            print("MobileNetV2_SCM:" + str(MobileNetV2_SCM))
            print("MobileNetV2_accuracy:" + str(1 - (MobileNetV2_acc * SDC_sum + (1 - MobileNetV2_acc))))
            end = time.time()
            print("MobileNetV2_time:" + str(end - begin))
            print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        elif name_i == "ResNet50.xlsx":
            ResNet50_acc = 0.3035
            ResNet50_SCM_no_FI = 0.34375
            ResNet50_nonSCM_no_FI = 1 - ResNet50_SCM_no_FI
            ResNet50_SCM = ((ResNet50_nonSCM_no_FI-ResNet50_acc)/2 + ResNet50_acc/2.5) * SDC_sum + ResNet50_SCM_no_FI
            print("ResNet50_SDC:" + str(SDC_sum))
            print("ResNet50_SCM:" + str(ResNet50_SCM))
            print("ResNet50_accuracy:" + str(1 - (ResNet50_acc * SDC_sum + (1 - ResNet50_acc))))
            end = time.time()
            print("ResNet50_time:" + str(end - begin))
            print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")

