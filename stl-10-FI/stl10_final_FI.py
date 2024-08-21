import os
import time
import sys

import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 获取当前脚本的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取 project 的根目录
project_root = os.path.abspath(os.path.join(current_dir, '..'))
# 添加 folder_b 到 sys.path
folder_b_path = os.path.join(project_root, 'src')
sys.path.insert(0, folder_b_path)
import tensorfi_plus as tfi_batch
from utility import get_fault_injection_configs

# from src import tensorfi_plus as tfi_batch
# from src.utility import get_fault_injection_configs

import json
# Check if GPU is available
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # 设置 GPU 显存占用为按需分配
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)

# 指定 CUDA 可见设备
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 使用 "0" 表示第一块 GPU 设备


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

def read_labels(path_to_labels, num_classes):
    """
    :param path_to_labels: path to the binary file containing labels from the STL-10 dataset
    :return: an array containing the labels
    """
    with open(path_to_labels, 'rb') as f:
        labels = np.fromfile(f, dtype=np.uint8)
        labels = labels.astype(np.int32)
        labels = labels - 1
        # Ensure labels are within the valid range
        labels = np.clip(labels, 0, num_classes - 1)
        return labels


def load_and_preprocess_label(label_path):
    label = read_labels(label_path, 10)

    return label

def read_all_images(path_to_data):
    """
    :param path_to_data: the file containing the binary images from the STL-10 dataset
    :return: an array containing all the images
    """

    with open(path_to_data, 'rb') as f:
        # read whole file in uint8 chunks
        everything = np.fromfile(f, dtype=np.uint8)

        images = np.reshape(everything, (-1, 3, 96, 96))
        images = np.transpose(images, (0, 3, 2, 1))

        # Convert images to float32 and normalize to [0, 1]
        images = images.astype(np.float32) / 255.0

        return images

def load_and_preprocess_image(image_path):
    image = read_all_images(image_path)
    test_datagen = ImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization=True
    )
    test_datagen.fit(image)

    return image



def main(model,process_no):
    model_name = model
    # model_name = "VGG19"
    # file_object = open('../dataset/stl-10/result/' + model_name + '/stl-10_final_log_' + model_name + '_' + str(process_no) + '.txt', 'a')
    inputs = load_and_preprocess_image("../dataset/stl-10/test_X.bin")
    labels = load_and_preprocess_label("../dataset/stl-10/test_y.bin")

    total = 8000

    print("Model name: " + model_name)
    # file_object.write("Model name: " + model_name)
    # file_object.write("\n")
    # file_object.flush()
    K.clear_session()

    # 加载模型配置
    path = '../stl-10/' + model_name + '.h5'
    print(path)

    model = tf.keras.models.load_model(path)


    low = 500 * process_no
    high = 500 * (process_no + 1)
    # print("Low: " + str(low) + " , High: " + str(high))
    # file_object.write("Low: " + str(low) + " , High: " + str(high))
    # file_object.write("\n")
    # file_object.flush()

    predicted_label_list = []

    for i in range(total):
        if low <= i < high:
            predicted_label = model.predict(tf.expand_dims(inputs[i], axis=0)).argmax(axis=-1)[0]

            # print(str(i) + " : " + str(labels[i]) + " : " + str(predicted_label))
    #         file_object.write(str(i) + " : " + str(labels[i]) + " : " + str(predicted_label))
    #         file_object.write("\n")
    #         file_object.flush()
    #         predicted_label_list.append(predicted_label)
    # correct_indices, wrong_indices = get_statistics_without_fault_injection(labels[low:high], predicted_label_list)

    # yaml_file = "../confFiles/sample1.yaml"
    #
    # model_graph, super_nodes = get_fault_injection_configs(model)
    #
    # count = 0
    #
    # for i in range(total):
    #     if low <= i < high:
    #         if count in correct_indices:
    #             for j in range(5):
    #                 temp_1 = tf.expand_dims(inputs[i], axis=0)
    #                 res = tfi_batch.inject(model=model, x_test=temp_1, confFile=yaml_file, model_graph=model_graph, super_nodes=super_nodes)
    #                 faulty_prediction = res.final_label[0]
    #                 # print(str(i) + " : " + str(labels[i]) + " : " + str(predicted_label_list[count]) + " : " + str(faulty_prediction))
    #                 file_object.write(str(i) + " : " + str(labels[i]) + " : " + str(predicted_label_list[count]) + " : " + str(faulty_prediction))
    #                 file_object.write("\n")
    #                 file_object.flush()
    #         else:
    #             for j in range(30):
    #                 temp_2 = tf.expand_dims(inputs[i], axis=0)
    #                 res = tfi_batch.inject(model=model, x_test=temp_2, confFile=yaml_file, model_graph=model_graph, super_nodes=super_nodes)
    #                 faulty_prediction = res.final_label[0]
    #                 # print(str(i) + " : " + str(labels[i]) + " : " + str(predicted_label_list[count]) + " : " + str(faulty_prediction))
    #                 file_object.write(str(i) + " : " + str(labels[i]) + " : " + str(predicted_label_list[count]) + " : " + str(faulty_prediction))
    #                 file_object.write("\n")
    #                 file_object.flush()
    #         count += 1
    #
    # file_object.close()


if __name__ == '__main__':
    # model_name = ['VGG16', 'VGG19', 'MobileNet', 'ResNet50', 'MobileNetV2']
    model_name = ['ResNet50']
    file_model = open('model_time.txt', 'a')  # 使用'txt'扩展名的文件

    for name in model_name:
        time_start = time.time()
        for i in range(16):
            main(name, i)
        time_end = time.time()
        time_sum = time_end - time_start
        print(f"{name}: {time_sum}")
        # print(f"{name}: {time_sum}", file=file_model)

    file_model.close()  # 记得关闭文件

