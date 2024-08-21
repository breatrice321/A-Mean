import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator

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
    correct_classification = 0
    misclassification = 0
    for i in range(len(original_label_list)):
        org_val = original_label_list[i]
        pred_val = predicted_label_list[i]
        if org_val != pred_val:
            wrong_indices.append(i)
            misclassification += 1
        else:
            correct_indices.append(i)
            correct_classification += 1
    return correct_indices, wrong_indices, correct_classification, misclassification

def read_labels(path_to_labels):
    """
    :param path_to_labels: path to the binary file containing labels from the STL-10 dataset
    :return: an array containing the labels
    """
    with open(path_to_labels, 'rb') as f:
        labels = np.fromfile(f, dtype=np.uint8)
        labels = labels.astype(np.int32)
        labels = labels - 1
        return labels

test_datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True
)
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

        test_datagen.fit(images)

        return images



def main(model):
    model_name = model
    inputs = read_all_images("./stl10_binary/test_X.bin")
    labels = read_labels("./stl10_binary/test_y.bin")

    total = 8000

    print("Model name: " + model_name)

    K.clear_session()


    # 加载模型配置
    path = model_name + '.h5'
    print(path)

    model = tf.keras.models.load_model(path)

    predicted_label_list = []

    for i in range(total):
        predicted_label = model.predict(tf.expand_dims(inputs[i], axis=0)).argmax(axis=-1)[0]
        predicted_label_list.append(predicted_label)

    correct_indices, wrong_indices, correct, wrong = get_statistics_without_fault_injection(labels, predicted_label_list)
    print(correct)
    print(wrong)
    print(correct/(correct+wrong))




if __name__ == '__main__':
    # model_name = ['VGG16', 'VGG19', 'MobileNet', 'ResNet50', 'MobileNetV2']
    main("VGG19")


