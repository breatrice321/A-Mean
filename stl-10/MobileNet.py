import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import optimizers, regularizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.utils import to_categorical

# # GPU Configuration
# gpus = tf.config.list_physical_devices('GPU')
# if gpus:
#     try:
#         # Set GPU memory growth
#         for gpu in gpus:
#             tf.config.experimental.set_memory_growth(gpu, True)
#
#         logical_gpus = tf.config.list_logical_devices('GPU')
#         print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
#     except RuntimeError as e:
#         print(e)

# Set GPU device
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Replace with the index of your GPU



train_images_path= './stl10_binary/train_X.bin'
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
    zoom_range=0.2,
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

def depthwise_separable_conv_block(x, pointwise_filters, strides, block_id):
    """
    Depthwise separable convolution block as used in MobileNet.
    Args:
    - x: Input tensor
    - pointwise_filters: Number of filters in the pointwise convolution
    - strides: Stride for the depthwise convolution
    - block_id: Block identifier
    """
    x = layers.DepthwiseConv2D(kernel_size=3, padding='same', strides=strides, use_bias=False,
                               name=f'conv_dw_{block_id}')(x)
    x = layers.BatchNormalization(name=f'conv_dw_{block_id}_bn')(x)
    x = layers.ReLU(6., name=f'conv_dw_{block_id}_relu')(x)

    x = layers.Conv2D(pointwise_filters, kernel_size=1, padding='same', use_bias=False, name=f'conv_pw_{block_id}')(x)
    x = layers.BatchNormalization(name=f'conv_pw_{block_id}_bn')(x)
    x = layers.ReLU(6., name=f'conv_pw_{block_id}_relu')(x)

    return x


def MobileNet_custom(input_shape=(96, 96, 3), num_classes=10):
    inputs = layers.Input(shape=input_shape)

    # Initial convolution block
    x = layers.Conv2D(32, kernel_size=3, strides=(1, 1), padding='same', use_bias=False, name='conv1')(inputs)
    x = layers.BatchNormalization(name='conv1_bn')(x)
    x = layers.ReLU(6., name='conv1_relu')(x)

    # Depthwise separable convolutions
    x = depthwise_separable_conv_block(x, pointwise_filters=64, strides=1, block_id=1)
    x = depthwise_separable_conv_block(x, pointwise_filters=128, strides=2, block_id=2)
    x = depthwise_separable_conv_block(x, pointwise_filters=128, strides=1, block_id=3)
    x = depthwise_separable_conv_block(x, pointwise_filters=256, strides=2, block_id=4)
    x = depthwise_separable_conv_block(x, pointwise_filters=256, strides=1, block_id=5)
    x = depthwise_separable_conv_block(x, pointwise_filters=512, strides=2, block_id=6)

    for i in range(7, 13):  # Repeat 5 times with stride 1
        x = depthwise_separable_conv_block(x, pointwise_filters=512, strides=1, block_id=i)

    x = depthwise_separable_conv_block(x, pointwise_filters=1024, strides=2, block_id=13)
    x = depthwise_separable_conv_block(x, pointwise_filters=1024, strides=1, block_id=14)

    # Global Average Pooling
    x = layers.Dropout(0.5)(x)
    x = layers.GlobalAveragePooling2D(name='global_avg_pool')(x)
    x = layers.Dropout(0.5)(x)

    # Output layer
    outputs = layers.Dense(num_classes, activation='softmax', kernel_initializer=tf.keras.initializers.he_normal(),
                           kernel_regularizer=regularizers.l2(0.0001), name='predictions')(x)

    # Creating the model
    model = models.Model(inputs, outputs, name='mobilenet_custom')

    return model


# Instantiate
input_shape = (96, 96, 3)
num_classes = 10
model = MobileNet_custom(input_shape=input_shape, num_classes=num_classes)

learning_rate = 0.001
epoch = 100
batch_size = 16

# optimizer = optimizers.SGD(learning_rate=learning_rate, momentum=0.9, decay=1e-6, nesterov=True, clipvalue=0.5)
optimizer = optimizers.Adam(lr=learning_rate, clipvalue=0.5)
# 学习率调度
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7, verbose=1)

# 早停法
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)

# Callback to save the best model during training
model_checkpoint = ModelCheckpoint(
    'MobileNet.h5',
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)

model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(
    train_datagen.flow(train_images, train_labels, batch_size=batch_size, shuffle=True),
    epochs=epoch,
    # validation_data=test_datagen.flow(test_images, test_labels, batch_size=batch_size),
    validation_data=(test_images, test_labels),
    # callbacks=[reduce_lr, model_checkpoint, early_stop],
    callbacks=[reduce_lr, model_checkpoint],
    steps_per_epoch=len(train_images) // batch_size,
    # validation_steps=len(test_images) // batch_size
)
