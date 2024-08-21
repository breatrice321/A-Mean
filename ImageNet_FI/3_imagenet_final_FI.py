import os
import random
import re
import sys
import time
from glob import glob
from pathlib import Path

import numpy as np

from tensorflow.keras import backend as K
# from tensorflow.keras.applications import vgg16, vgg19, resnet, xception, nasnet, mobilenet, mobilenet_v2, inception_resnet_v2, inception_v3, densenet
# import keras_resnet as resnet
from tensorflow.keras.applications import vgg16, vgg19, resnet50, xception, nasnet, mobilenet, mobilenet_v2, inception_resnet_v2, inception_v3, densenet

from src import tensorfi_plus as tfi_batch
from src.utility import get_fault_injection_configs


def get_model_from_name(model_name):
    if model_name == "ResNet50":
        print(resnet50.ResNet50(weights='imagenet'))
        return resnet50.ResNet50(weights='imagenet')
    elif model_name == "VGG16":
        return vgg16.VGG16( include_top=True,
    weights="imagenet",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,)
    elif model_name == "VGG19":
        return vgg19.VGG19(weights='imagenet')
    elif model_name == "MobileNet":
        return mobilenet.MobileNet(weights='imagenet')
    elif model_name == "MobileNetV2":
        return mobilenet_v2.MobileNetV2(weights='imagenet')


def get_preprocessed_input_by_model_name(model_name, x_val):
    if model_name == "ResNet50" or model_name == "ResNet101" or model_name == "ResNet152":
        return resnet50.preprocess_input(x_val)
    elif model_name == "VGG16":
        return vgg16.preprocess_input(x_val)
    elif model_name == "VGG19":
        return vgg19.preprocess_input(x_val)
    elif model_name == "MobileNet":
        return mobilenet.preprocess_input(x_val)
    elif model_name == "MobileNetV2":
        return mobilenet_v2.preprocess_input(x_val)


def get_data_path_by_model_name(model_name, path_imagenet_val_dataset):
    if model_name == "ResNet50" or model_name == "VGG16" \
            or model_name == "VGG19" or model_name == "MobileNet" \
            or model_name == "MobileNetV2":
        return str(path_imagenet_val_dataset) + "/result/sampled_new_x_val_224_1.npy"

def get_statistics_without_fault_injection(original_label_list, predicted_label_list):
    correct_indices = []
    wrong_indices = []
    for i in range(len(original_label_list)):
        org_val = original_label_list[i]
        pred_val = predicted_label_list[i]
        if org_val != pred_val:
            wrong_indices.append(i)
        else:
            correct_indices.append(i)
    return correct_indices, wrong_indices


def main(model, process_no):
    model_name = model
    file_object = open('../dataset/imagenet/result/' + model_name + '/imagenet_final_log_' + model_name + '_' + str(process_no) + '.txt', 'a')
    path_imagenet_val_dataset = Path("../dataset/imagenet")  # path/to/data/
    y_val = np.load(str("../dataset/imagenet/result/y_val_sampled.npy"))
    x_val_path = get_data_path_by_model_name(model_name=model_name, path_imagenet_val_dataset=path_imagenet_val_dataset)

    file_object.write("Model name: " + model_name)
    # print("Model name: " + model_name)
    file_object.write("\n")
    file_object.flush()
    K.clear_session()

    low = 200 * process_no
    high = 200 * (process_no + 1)
    file_object.write("Low: " + str(low) + " , High: " + str(high))
    # print("Low: " + str(low) + " , High: " + str(high))
    file_object.write("\n")
    file_object.flush()

    model = get_model_from_name(model_name)
    with open('../model_summary/'+str(model_name)+'.txt', 'w') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))
    # print(model.summary())

    predicted_label_list = []
    x_val = np.load(x_val_path).astype('float32')
    x_val = get_preprocessed_input_by_model_name(model_name, x_val)
    data_count, _, _, _ = x_val.shape
    for i in range(data_count):
        if low <= i < high:
            img = x_val[i]
            img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))
            predicted_label = model.predict(img).argmax(axis=-1)[0]
            predicted_label_list.append(predicted_label)
            file_object.write(str(i) + " : " + str(y_val[i]) + " : " + str(predicted_label))
            # print(str(i) + " : " + str(y_val[i]) + " : " + str(predicted_label))
            file_object.write("\n")
            file_object.flush()
    correct_indices, wrong_indices = get_statistics_without_fault_injection(y_val[low:high], predicted_label_list)



    # begin to count fault injection data
    yaml_file = "../confFiles/sample1.yaml"

    model_graph, super_nodes = get_fault_injection_configs(model)
    # print(len(model.layers))
    count = 0
    for i in range(data_count):
        if low <= i < high:
            if count in correct_indices:
                for j in range(5):
                    img = x_val[i]
                    img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))
                    res = tfi_batch.inject(model=model, x_test=img, confFile=yaml_file,
                                           model_graph=model_graph, super_nodes=super_nodes)
                    faulty_prediction = res.final_label[0]
                    # print(str(i) + " : " + str(y_val[i]) + " : " + str(predicted_label_list[count]) + " : " + str(faulty_prediction))
                    file_object.write(str(i) + " : " + str(y_val[i]) + " : " + str(predicted_label_list[count]) + " : " + str(faulty_prediction))
                    file_object.write("\n")
                    file_object.flush()
            else:
                for j in range(30):
                    img = x_val[i]
                    img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))
                    res = tfi_batch.inject(model=model, x_test=img, confFile=yaml_file,
                                           model_graph=model_graph, super_nodes=super_nodes)
                    faulty_prediction = res.final_label[0]
                    # print(str(i) + " : " + str(y_val[i]) + " : " + str(predicted_label_list[count]) + " : " + str(faulty_prediction))
                    file_object.write(str(i) + " : " + str(y_val[i]) + " : " + str(predicted_label_list[count]) + " : " + str(faulty_prediction))
                    file_object.write("\n")
                    file_object.flush()
            count += 1
    file_object.close()


if __name__ == '__main__':

    model_name = ['VGG16', 'VGG19', 'MobileNet', 'ResNet50', 'MobileNetV2']
    file_model = open('model_time.txt', 'a')  # 使用'txt'扩展名的文件

    for name in model_name:
        time_start = time.time()
        for i in range(50):
            main(name, i)
        time_end = time.time()
        time_sum = time_end - time_start
        print(f"{name}: {time_sum}")
        # print(f"{name}: {time_sum}", file=file_model)

    file_model.close()  # 记得关闭文件
