import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras import layers, models, optimizers, regularizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint, Callback

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
    featurewise_center=True,
    featurewise_std_normalization=True,
    horizontal_flip=True,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2,  # Add zoom augmentation
    shear_range=0.2  # Add shear augmentation
)

test_datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True
)

(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

# Convert labels to one-hot encoding
train_labels = tf.keras.utils.to_categorical(train_labels, num_classes=10)
test_labels = tf.keras.utils.to_categorical(test_labels, num_classes=10)

# Fit the data generators to the training data
train_datagen.fit(train_images)
test_datagen.fit(train_images)


def VGG19_custom(input_shape=(32, 32, 3), num_classes=10):
    model = models.Sequential()

    # Block 1
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=input_shape))
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2), strides=(2, 2)))

    # Block 2
    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2), strides=(2, 2)))

    # Block 3
    model.add(layers.Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(layers.Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(layers.Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(layers.Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2), strides=(2, 2)))

    # Block 4
    model.add(layers.Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(layers.Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(layers.Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(layers.Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2), strides=(2, 2)))

    # Block 5
    model.add(layers.Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(layers.Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(layers.Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(layers.Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2), strides=(2, 2)))

    # Flatten and Fully Connected Layers
    model.add(layers.Flatten())
    model.add(layers.Dense(4096, activation='relu', kernel_regularizer=regularizers.l2(1e-4)))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(4096, activation='relu', kernel_regularizer=regularizers.l2(1e-4)))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(num_classes, activation='softmax', kernel_regularizer=regularizers.l2(1e-4)))

    return model


# Instantiate
input_shape = (32, 32, 3)
num_classes = 10
model = VGG19_custom(input_shape=input_shape, num_classes=num_classes)


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
epoch = 150
batch_size = 128

optimizer = optimizers.SGD(learning_rate=learning_rate, momentum=0.9, decay=1e-6, nesterov=True, clipvalue=0.5)

# 学习率调度
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7, verbose=1)

# 早停法
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)

# Callback to save the best model during training
model_checkpoint = ModelCheckpoint(
    'VGG19.h5',  # 文件名可以根据需要更改
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
