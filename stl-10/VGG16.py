import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from tensorflow.keras.utils import to_categorical
import os
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint, Callback


# GPU Configuration
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Set GPU memory growth
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)

# Set GPU device
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Replace with the index of your GPU


train_images_path = './stl10_binary/train_X.bin'
test_images_path = './stl10_binary/test_X.bin'
train_labels_path = './stl10_binary/train_y.bin'
test_labels_path = './stl10_binary/test_y.bin'

def read_labels(path_to_labels):
    """
    :param path_to_labels: path to the binary file containing labels from the STL-10 dataset
    :return: an array containing the labels
    """
    with open(path_to_labels, 'rb') as f:
        labels = np.fromfile(f, dtype=np.uint8)
        labels = labels.astype(np.float32)
        labels = labels - 1
        return labels


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

train_images = read_all_images(train_images_path)
train_labels = read_labels(train_labels_path)
test_images = read_all_images(test_images_path)
test_labels = read_labels(test_labels_path)

# Convert labels to one-hot encoding
train_labels = to_categorical(train_labels, 10)
test_labels = to_categorical(test_labels, 10)

# Create data augmentation generator for training
train_datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    horizontal_flip=True,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2,  # Add zoom augmentation
    shear_range=0.2
)

# Create data generator for validation (only rescaling)
test_datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True
)

# Fit the data generators to the training data
train_datagen.fit(train_images)
test_datagen.fit(test_images)

# Load VGG16 model without weights and add a custom top layer
def VGG16_custom(input_shape=(96, 96, 3), num_classes=10):
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
    model.add(layers.MaxPooling2D((2, 2), strides=(2, 2)))

    # Block 4
    model.add(layers.Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(layers.Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(layers.Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2), strides=(2, 2)))

    # Block 5
    model.add(layers.Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(layers.Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(layers.Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2), strides=(2, 2)))

    # Flatten and Fully Connected Layers
    model.add(layers.Flatten())
    model.add(layers.Dense(4096, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0001)))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(4096, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0001)))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(num_classes, activation='softmax'))

    return model


# Instantiate
input_shape = (96, 96, 3)
num_classes = 10
model = VGG16_custom(input_shape=input_shape, num_classes=num_classes)

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
epoch = 100
batch_size = 16

optimizer = optimizers.SGD(learning_rate=learning_rate, momentum=0.9, decay=1e-6, nesterov=True, clipvalue=0.5)

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
    validation_data=(test_images, test_labels),
    callbacks=[reduce_lr, early_stop, model_checkpoint, clip_callback],
    steps_per_epoch=len(train_images) // batch_size,
    # validation_steps=len(test_images) // batch_size
)
