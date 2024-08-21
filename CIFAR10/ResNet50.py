import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras import layers, models, optimizers, regularizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping


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

# Set GPU device
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Define the data generators
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
test_datagen.fit(train_images)  # Important: fit the test generator with training data stats

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

def ResNet50_custom(input_shape=(32, 32, 3), num_classes=10):
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
input_shape = (32, 32, 3)
num_classes = 10
model = ResNet50_custom(input_shape=input_shape, num_classes=num_classes)


learning_rate = 0.001
epoch = 150
batch_size = 32

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
    validation_data=test_datagen.flow(test_images, test_labels, batch_size=batch_size),
    callbacks=[reduce_lr, early_stop, model_checkpoint],
    steps_per_epoch=len(train_images) // batch_size,
    validation_steps=len(test_images) // batch_size
)
