
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, optimizers, regularizers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.callbacks import  EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical


# # GPU Configuration
# gpus = tf.config.list_physical_devices('GPU')
# if gpus:
#     try:
#         # Set GPU memory growth
#         for gpu in gpus:
#             tf.config.experimental.set_memory_growth(gpu, True)
#
#
#         logical_gpus = tf.config.list_logical_devices('GPU')
#         print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
#     except RuntimeError as e:
#         print(e)

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

def convert_labels_to_one_hot(labels, num_classes):
    """
    Convert labels to one-hot encoded format.

    :param labels: Array of labels to convert
    :param num_classes: Total number of classes
    :return: One-hot encoded labels
    """
    # Convert labels from 1-10 to 0-9
    labels = labels - 1
    # Ensure labels are within the valid range
    labels = np.clip(labels, 0, num_classes - 1)
    return to_categorical(labels, num_classes=num_classes)


train_images = read_all_images(train_images_path)
train_labels = read_labels(train_labels_path)
test_images = read_all_images(test_images_path)
test_labels = read_labels(test_labels_path)

# Convert labels to one-hot encoding
train_labels = convert_labels_to_one_hot(train_labels, 10)
test_labels = convert_labels_to_one_hot(test_labels, 10)

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


def inverted_res_block(x, expansion, filters, stride, block_id):
    in_channels = x.shape[-1]
    prefix = f'block_{block_id}_'

    # Expansion layer
    if expansion != 1:
        x_shortcut = x  # Store the input tensor for the skip connection

        x = layers.Conv2D(expansion * in_channels, kernel_size=1, padding='same', use_bias=False,
                          name=prefix + 'expand')(x)
        x = layers.BatchNormalization(epsilon=1e-3, momentum=0.999, name=prefix + 'expand_BN')(x)
        x = layers.ReLU(6., name=prefix + 'expand_relu')(x)
    else:
        x_shortcut = x  # No expansion for skip connection

    # Depthwise separable convolution
    x = layers.DepthwiseConv2D(kernel_size=3, strides=stride, padding='same', use_bias=False,
                               name=prefix + 'depthwise')(x)
    x = layers.BatchNormalization(epsilon=1e-3, momentum=0.999, name=prefix + 'depthwise_BN')(x)
    x = layers.ReLU(6., name=prefix + 'depthwise_relu')(x)

    # Project layer
    x = layers.Conv2D(filters, kernel_size=1, padding='same', use_bias=False,
                      name=prefix + 'project')(x)
    x = layers.BatchNormalization(epsilon=1e-3, momentum=0.999, name=prefix + 'project_BN')(x)

    # Skip connection
    if in_channels == filters and stride == 1:
        x = layers.Add(name=prefix + 'add')([x, x_shortcut])  # Apply the skip connection

    return x

def MobileNetV2_custom(input_shape=(96, 96, 3), num_classes=10):
    inputs = layers.Input(shape=input_shape)

    # Initial Conv layer
    x = layers.Conv2D(32, kernel_size=3, strides=(2, 2), padding='same', use_bias=False, name='Conv1')(inputs)
    x = layers.BatchNormalization(epsilon=1e-3, momentum=0.999, name='bn_Conv1')(x)
    x = layers.ReLU(6., name='Conv1_relu')(x)

    # Inverted Residual Blocks
    x = inverted_res_block(x, expansion=1, filters=16, stride=1, block_id=1)
    x = inverted_res_block(x, expansion=6, filters=24, stride=2, block_id=2)
    x = inverted_res_block(x, expansion=6, filters=24, stride=1, block_id=3)
    x = inverted_res_block(x, expansion=6, filters=32, stride=2, block_id=4)
    x = inverted_res_block(x, expansion=6, filters=32, stride=1, block_id=5)
    x = inverted_res_block(x, expansion=6, filters=32, stride=1, block_id=6)
    x = inverted_res_block(x, expansion=6, filters=64, stride=2, block_id=7)
    x = inverted_res_block(x, expansion=6, filters=64, stride=1, block_id=8)
    x = inverted_res_block(x, expansion=6, filters=64, stride=1, block_id=9)
    x = inverted_res_block(x, expansion=6, filters=64, stride=1, block_id=10)
    x = inverted_res_block(x, expansion=6, filters=96, stride=1, block_id=11)
    x = inverted_res_block(x, expansion=6, filters=96, stride=1, block_id=12)
    x = inverted_res_block(x, expansion=6, filters=96, stride=1, block_id=13)
    x = inverted_res_block(x, expansion=6, filters=160, stride=2, block_id=14)
    x = inverted_res_block(x, expansion=6, filters=160, stride=1, block_id=15)
    x = inverted_res_block(x, expansion=6, filters=160, stride=1, block_id=16)
    x = inverted_res_block(x, expansion=6, filters=320, stride=1, block_id=17)

    # Concluding Conv layer and pooling
    x = layers.Conv2D(1280, kernel_size=1, use_bias=False, name='Conv_1')(x)
    x = layers.BatchNormalization(epsilon=1e-3, momentum=0.999, name='Conv_1_bn')(x)
    x = layers.ReLU(6., name='out_relu')(x)

    # Pooling and Dropout
    x = layers.GlobalAveragePooling2D()(x)
    # x = layers.Dropout(0.4)(x)

    # Output layer
    # outputs = layers.Dense(num_classes, activation='softmax', kernel_initializer=tf.keras.initializers.he_normal(),
    #                        kernel_regularizer=regularizers.l2(1e-4))(x)
    # outputs = layers.Dense(num_classes, activation='softmax', kernel_initializer=tf.keras.initializers.he_normal())(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    # Creating the model
    model = models.Model(inputs, outputs)

    return model


# Instantiate
input_shape = (96, 96, 3)
num_classes = 10
model = MobileNetV2_custom(input_shape=input_shape, num_classes=num_classes)


learning_rate = 0.001
epoch = 200
batch_size = 16

optimizer = optimizers.SGD(learning_rate=learning_rate, momentum=0.999, decay=1e-6, nesterov=True, clipvalue=0.5)
# optimizer = optimizers.Adam(lr=learning_rate, clipvalue=0.5)
# 学习率调度
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7, verbose=1)

# 早停法
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)

# Callback to save the best model during training
model_checkpoint = ModelCheckpoint(
    'MobileNetV2.h5',
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