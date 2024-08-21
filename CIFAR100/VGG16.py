
import os
import numpy as np
import pickle
import tarfile
import tensorflow as tf
from tensorflow.keras.datasets import cifar100
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import LearningRateScheduler, ModelCheckpoint
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.callbacks import Callback

from tensorflow.keras.applications import VGG16


# Check if GPU is available
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # 设置 GPU 显存占用为按需分配
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

        # 在初始化之前，调用一次，否则可能会出错
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)]
        )

        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)

# 设置 GPU 设备
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Replace with the index of your GPU


# Define the data generators
train_datagen = ImageDataGenerator(
    horizontal_flip=True,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2,  # Add zoom augmentation
    shear_range=0.2  # Add shear augmentation
)

test_datagen = ImageDataGenerator()

# Path to the compressed archive file
archive_path = 'cifar-100-python.tar.gz'
extracted_folder = 'cifar-100-python'  # Use the folder name, not a specific file

# Check if the extracted folder is present in the current directory
if not os.path.exists(extracted_folder):
    # If the folder is not present, extract the contents from the archive
    with tarfile.open(archive_path, 'r:gz') as tar:
        tar.extractall()

# Check if the necessary files are present in the extracted folder
if not (os.path.exists(os.path.join(extracted_folder, 'train')) and os.path.exists(os.path.join(extracted_folder, 'test'))):
    print("Error: The required files are not present in the extracted folder.")
else:
    # Load the dataset from the extracted files with correct encoding
    with open(os.path.join(extracted_folder, 'train'), 'rb') as file:
        train_dataset = pickle.load(file, encoding='latin1')

    with open(os.path.join(extracted_folder, 'test'), 'rb') as file:
        test_dataset = pickle.load(file, encoding='latin1')

    # Unpack the dataset
    train_images, train_labels = train_dataset['data'], train_dataset['fine_labels']
    test_images, test_labels = test_dataset['data'], test_dataset['fine_labels']

    # Reshape the images to (num_samples, height, width, num_channels)
    train_images = train_images.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    test_images = test_images.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)

# (train_images, train_labels), (test_images, test_labels) = cifar100.load_data()

# Convert labels to one-hot encoding
train_labels = tf.keras.utils.to_categorical(train_labels, num_classes=100)
test_labels = tf.keras.utils.to_categorical(test_labels, num_classes=100)

# Normalize pixel values using mean and standard deviation
mean = np.mean(train_images, axis=(0, 1, 2, 3))
std = np.std(train_images, axis=(0, 1, 2, 3))
train_images = (train_images - mean) / (std + 1e-7)
test_images = (test_images - mean) / (std + 1e-7)

# Build the VGG16 model with weight initialization
def build_vgg16_model(input_shape=(32, 32, 3), num_classes=100):
    model = models.Sequential()

    # Conv1
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=input_shape, kernel_initializer=tf.keras.initializers.he_normal(),kernel_regularizer=tf.keras.regularizers.l2(0.0001)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0001)))
    model.add(layers.MaxPooling2D((2, 2), strides=(2, 2)))

    # Conv2
    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0001)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0001)))
    model.add(layers.MaxPooling2D((2, 2), strides=(2, 2)))

    # Conv3
    model.add(layers.Conv2D(256, (3, 3), activation='relu', padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0001)))
    model.add(layers.Conv2D(256, (3, 3), activation='relu', padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0001)))
    model.add(layers.Conv2D(256, (3, 3), activation='relu', padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0001)))
    model.add(layers.MaxPooling2D((2, 2), strides=(2, 2)))

    # Conv4
    model.add(layers.Conv2D(512, (3, 3), activation='relu', padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0001)))
    model.add(layers.Conv2D(512, (3, 3), activation='relu', padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0001)))
    model.add(layers.Conv2D(512, (3, 3), activation='relu', padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0001)))
    model.add(layers.MaxPooling2D((2, 2), strides=(2, 2)))

    # Conv5
    model.add(layers.Conv2D(512, (3, 3), activation='relu', padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0001)))
    model.add(layers.Conv2D(512, (3, 3), activation='relu', padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0001)))
    model.add(layers.Conv2D(512, (3, 3), activation='relu', padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0001)))
    model.add(layers.MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(layers.Flatten())
    model.add(layers.Dense(4096, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0001)))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(4096, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0001)))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(num_classes, activation='softmax'))

    return model

model = build_vgg16_model(input_shape=(32, 32, 3), num_classes=100)


class DynamicClipValue(Callback):
    def __init__(self, initial_clipvalue, update_interval):
        super(DynamicClipValue, self).__init__()
        self.initial_clipvalue = initial_clipvalue
        self.update_interval = update_interval

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.update_interval == 0 and epoch > 0:
            current_clipvalue = self.model.optimizer.clipvalue
            new_clipvalue = current_clipvalue * 0.9  # Adjust this factor as needed
            self.model.optimizer.clipvalue = new_clipvalue
            print(f'Updated clipvalue to {new_clipvalue} at epoch {epoch + 1}.')

learning_rate = 0.01
# lr_drop = 20
epoch = 100
batch_size = 128

# def lr_scheduler(epoch, lr):
#     print("learning rate:", lr)
#     return learning_rate * (0.5 ** (epoch // lr_drop))

optimizer = optimizers.SGD(learning_rate=learning_rate, momentum=0.9, decay=1e-6, nesterov=True, clipvalue=0.5)

# reduce_lr = LearningRateScheduler(lr_scheduler)

# 学习率调度
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7, verbose=1)

# 早停法
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)

# Callback to save the best model during training
model_checkpoint = ModelCheckpoint(
    'VGG16.h5',  # 文件名可以根据需要更改
    monitor='val_accuracy',  # 根据验证集准确率进行保存
    save_best_only=True,  # 只保存最好的模型
    mode='max',  # 保存模式：最大化验证集准确率
    verbose=1
)

model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Create an instance of the custom callback
clip_callback = DynamicClipValue(initial_clipvalue=0.5, update_interval=10)

# 在 model.fit 中添加回调
history = model.fit(
    train_datagen.flow(train_images, train_labels, batch_size=batch_size),
    epochs=epoch,
    validation_data=test_datagen.flow(test_images, test_labels, batch_size=batch_size),
    callbacks=[reduce_lr, early_stop, model_checkpoint, clip_callback],
    steps_per_epoch=len(train_images) // batch_size,
    validation_steps=len(test_images) // batch_size
)