import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, regularizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.utils import to_categorical

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
)

# Create data generator for validation (only rescaling)
test_datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True
)

# Fit the data generators to the training data
train_datagen.fit(train_images)
test_datagen.fit(test_images)


def identity_block(input_tensor, filters, kernel_size, stage, block):
    """
    The identity block has no convolutional layer at shortcut.
    Args:
    - input_tensor: input tensor
    - filters: list of integers, the filters of 3 conv layers in the main path
    - kernel_size: default 3, the kernel size of middle conv layer at main path
    - stage: integer, current stage label, used for generating layer names
    - block: 'a','b'..., current block label, used for generating layer names
    """
    filters1, filters2, filters3 = filters
    conv_name_base = f'res{stage}{block}_branch'
    bn_name_base = f'bn{stage}{block}_branch'

    x = layers.Conv2D(filters1, (1, 1), name=f'{conv_name_base}2a')(input_tensor)
    x = layers.BatchNormalization(name=f'{bn_name_base}2a')(x)
    x = layers.ReLU()(x)

    x = layers.Conv2D(filters2, kernel_size, padding='same', name=f'{conv_name_base}2b')(x)
    x = layers.BatchNormalization(name=f'{bn_name_base}2b')(x)
    x = layers.ReLU()(x)

    x = layers.Conv2D(filters3, (1, 1), name=f'{conv_name_base}2c')(x)
    x = layers.BatchNormalization(name=f'{bn_name_base}2c')(x)

    x = layers.add([x, input_tensor])
    x = layers.ReLU()(x)
    return x

def conv_block(input_tensor, filters, kernel_size, strides, stage, block):
    """
    The conv block has a conv layer at shortcut.
    Args:
    - input_tensor: input tensor
    - filters: list of integers, the filters of 3 conv layers in the main path
    - kernel_size: default 3, the kernel size of middle conv layer at main path
    - strides: strides for the first conv layer in the block
    - stage: integer, current stage label, used for generating layer names
    - block: 'a','b'..., current block label, used for generating layer names
    """
    filters1, filters2, filters3 = filters
    conv_name_base = f'res{stage}{block}_branch'
    bn_name_base = f'bn{stage}{block}_branch'

    x = layers.Conv2D(filters1, (1, 1), strides=strides, name=f'{conv_name_base}2a')(input_tensor)
    x = layers.BatchNormalization(name=f'{bn_name_base}2a')(x)
    x = layers.ReLU()(x)

    x = layers.Conv2D(filters2, kernel_size, padding='same', name=f'{conv_name_base}2b')(x)
    x = layers.BatchNormalization(name=f'{bn_name_base}2b')(x)
    x = layers.ReLU()(x)

    x = layers.Conv2D(filters3, (1, 1), name=f'{conv_name_base}2c')(x)
    x = layers.BatchNormalization(name=f'{bn_name_base}2c')(x)

    shortcut = layers.Conv2D(filters3, (1, 1), strides=strides, name=f'{conv_name_base}1')(input_tensor)
    shortcut = layers.BatchNormalization(name=f'{bn_name_base}1')(shortcut)

    x = layers.add([x, shortcut])
    x = layers.ReLU()(x)
    return x

def ResNet50_custom(input_shape=(96, 96, 3), num_classes=10):
    """
    Build a ResNet50 model from scratch with given input shape and number of classes.
    Args:
    - input_shape: tuple, shape of input images
    - num_classes: integer, number of classes for the output layer
    """
    inputs = layers.Input(shape=input_shape)

    # Initial Conv and Max Pool layers
    x = layers.Conv2D(64, (3, 3), strides=(1, 1), padding='same', use_bias=False, name='conv1')(inputs)
    x = layers.BatchNormalization(name='bn_conv1')(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

    # Stage 2
    x = conv_block(x, [64, 64, 256], kernel_size=3, strides=1, stage=2, block='a')
    x = identity_block(x, [64, 64, 256], kernel_size=3, stage=2, block='b')
    x = identity_block(x, [64, 64, 256], kernel_size=3, stage=2, block='c')

    # Stage 3
    x = conv_block(x, [128, 128, 512], kernel_size=3, strides=2, stage=3, block='a')
    x = identity_block(x, [128, 128, 512], kernel_size=3, stage=3, block='b')
    x = identity_block(x, [128, 128, 512], kernel_size=3, stage=3, block='c')
    x = identity_block(x, [128, 128, 512], kernel_size=3, stage=3, block='d')

    # Stage 4
    x = conv_block(x, [256, 256, 1024], kernel_size=3, strides=2, stage=4, block='a')
    x = identity_block(x, [256, 256, 1024], kernel_size=3, stage=4, block='b')
    x = identity_block(x, [256, 256, 1024], kernel_size=3, stage=4, block='c')
    x = identity_block(x, [256, 256, 1024], kernel_size=3, stage=4, block='d')
    x = identity_block(x, [256, 256, 1024], kernel_size=3, stage=4, block='e')
    x = identity_block(x, [256, 256, 1024], kernel_size=3, stage=4, block='f')

    # Stage 5
    x = conv_block(x, [512, 512, 2048], kernel_size=3, strides=2, stage=5, block='a')
    x = identity_block(x, [512, 512, 2048], kernel_size=3, stage=5, block='b')
    x = identity_block(x, [512, 512, 2048], kernel_size=3, stage=5, block='c')

    # Global Average Pooling
    x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
    x = layers.Dropout(0.5)(x)

    # Output layer
    outputs = layers.Dense(num_classes, activation='softmax', kernel_initializer=tf.keras.initializers.he_normal(), kernel_regularizer=regularizers.l2(1e-4), name='fc')(x)

    # Create model
    model = models.Model(inputs, outputs, name='resnet50_custom')

    return model

# Instantiate and compile the model
input_shape = (96, 96, 3)
num_classes = 10
model = ResNet50_custom(input_shape=input_shape, num_classes=num_classes)


learning_rate = 0.001
epoch = 150
batch_size = 16

optimizer = optimizers.Adam(lr=learning_rate, clipvalue=0.5)

# Learning rate scheduler
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7, verbose=1)

# Early stopping
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)

# Callback to save the best model during training
model_checkpoint = ModelCheckpoint(
    'ResNet50.h5',
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)

model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(
    train_datagen.flow(train_images, train_labels, batch_size=batch_size),
    epochs=epoch,
    # validation_data=test_datagen.flow(test_images, test_labels, batch_size=batch_size),
    validation_data=(test_images, test_labels),
    # callbacks=[reduce_lr, early_stop, model_checkpoint],
    callbacks=[reduce_lr, model_checkpoint],
    steps_per_epoch=len(train_images) // batch_size,
    # validation_steps=len(test_images) // batch_size
)
