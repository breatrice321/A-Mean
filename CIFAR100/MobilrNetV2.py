import os
import pickle
import tarfile
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import LearningRateScheduler, ModelCheckpoint

from tensorflow.keras.callbacks import  EarlyStopping, ModelCheckpoint

from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2


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
    horizontal_flip=True,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2,
    shear_range=0.2
)

test_datagen = ImageDataGenerator()

# Load CIFAR-100 dataset
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

# def build_mobilenetv2_model(input_shape=(32, 32, 3), num_classes=100, alpha=1.0, weight_decay=1e-4):
#     input_tensor = layers.Input(shape=input_shape)
#
#     # Entry flow
#     x = _conv_block(input_tensor, 32, alpha, strides=(2, 2), weight_decay=weight_decay)
#     x = _inverted_residual_block(x, 16, alpha, 1, strides=(1, 1), block_id=0, weight_decay=weight_decay)
#     x = _inverted_residual_block(x, 24, alpha, 2, strides=(2, 2), block_id=1, weight_decay=weight_decay)
#     x = _inverted_residual_block(x, 24, alpha, 1, strides=(1, 1), block_id=2, weight_decay=weight_decay)
#     x = _inverted_residual_block(x, 32, alpha, 2, strides=(2, 2), block_id=3, weight_decay=weight_decay)
#     x = _inverted_residual_block(x, 32, alpha, 1, strides=(1, 1), block_id=4, weight_decay=weight_decay)
#     x = _inverted_residual_block(x, 32, alpha, 1, strides=(1, 1), block_id=5, weight_decay=weight_decay)
#     x = _inverted_residual_block(x, 64, alpha, 2, strides=(2, 2), block_id=6, weight_decay=weight_decay)
#     x = _inverted_residual_block(x, 64, alpha, 1, strides=(1, 1), block_id=7, weight_decay=weight_decay)
#     x = _inverted_residual_block(x, 64, alpha, 1, strides=(1, 1), block_id=8, weight_decay=weight_decay)
#     x = _inverted_residual_block(x, 64, alpha, 1, strides=(1, 1), block_id=9, weight_decay=weight_decay)
#
#     # Middle flow
#     x = _inverted_residual_block(x, 96, alpha, 1, strides=(1, 1), block_id=10, weight_decay=weight_decay)
#     x = _inverted_residual_block(x, 96, alpha, 1, strides=(1, 1), block_id=11, weight_decay=weight_decay)
#     x = _inverted_residual_block(x, 96, alpha, 1, strides=(1, 1), block_id=12, weight_decay=weight_decay)
#     x = _inverted_residual_block(x, 160, alpha, 2, strides=(2, 2), block_id=13, weight_decay=weight_decay)
#     x = _inverted_residual_block(x, 160, alpha, 1, strides=(1, 1), block_id=14, weight_decay=weight_decay)
#     x = _inverted_residual_block(x, 160, alpha, 1, strides=(1, 1), block_id=15, weight_decay=weight_decay)
#     x = _inverted_residual_block(x, 320, alpha, 1, strides=(1, 1), block_id=16, weight_decay=weight_decay)
#
#     # Exit flow
#     x = _conv_block(x, 1280, alpha, kernel_size=(1, 1), weight_decay=weight_decay)
#     x = layers.GlobalAveragePooling2D()(x)
#
#     # Fully Connected Layer
#     x = layers.Dense(num_classes, activation='softmax', kernel_initializer='he_normal', kernel_regularizer=tf.keras.regularizers.l2(weight_decay), name='Dense')(x)
#
#     model = models.Model(inputs=input_tensor, outputs=x, name='mobilenetv2')
#     return model
#
#
# def _conv_block(input_tensor, filters, alpha, kernel_size=(3, 3), strides=(1, 1), weight_decay=1e-4):
#     x = layers.Conv2D(int(filters * alpha), kernel_size, strides=strides, padding='same', use_bias=False, kernel_initializer='he_normal', kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(input_tensor)
#     x = layers.BatchNormalization()(x)
#     x = layers.ReLU(6.0)(x)
#     return x
#
# def _inverted_residual_block(input_tensor, filters, alpha, expansion, strides, weight_decay=1e-4, block_id=None):
#     in_channels = int(input_tensor.shape[-1])
#     pointwise_conv_filters = int(filters * alpha)
#     pointwise_filters = _make_divisible(pointwise_conv_filters, 8)
#     x = input_tensor
#     prefix = 'expanded_conv_'
#
#     if expansion > 1:
#         x = layers.Conv2D(expansion * in_channels, (1, 1), padding='same', use_bias=False, kernel_initializer='he_normal', kernel_regularizer=tf.keras.regularizers.l2(weight_decay), name=prefix + f'{block_id}_expand')(x)
#         x = layers.BatchNormalization(name=prefix + f'{block_id}_expand_BN')(x)
#         x = layers.ReLU(6.0, name=prefix + f'{block_id}_expand_relu')(x)
#
#     x = layers.DepthwiseConv2D((3, 3), padding='same', strides=strides, use_bias=False, kernel_initializer='he_normal', kernel_regularizer=tf.keras.regularizers.l2(weight_decay), name=prefix + f'{block_id}_depthwise')(x)
#     x = layers.BatchNormalization(name=prefix + f'{block_id}_depthwise_BN')(x)
#     x = layers.ReLU(6.0, name=prefix + f'{block_id}_depthwise_relu')(x)
#
#     x = layers.Conv2D(pointwise_filters, (1, 1), padding='same', use_bias=False, kernel_initializer='he_normal', kernel_regularizer=tf.keras.regularizers.l2(weight_decay), name=prefix + f'{block_id}_project')(x)
#     x = layers.BatchNormalization(name=prefix + f'{block_id}_project_BN')(x)
#
#     if strides == 1 and in_channels == pointwise_filters:
#         x = layers.Add(name=prefix + f'{block_id}_add')([input_tensor, x])
#
#     return x
#
#
# def _make_divisible(v, divisor, min_value=None):
#
#     if min_value is None:
#         min_value = divisor
#     new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
#     if new_v < 0.9 * v:
#         new_v += divisor
#     return new_v
#
#
# # Build MobileNetV2 model with regularization and weight initialization
# model = build_mobilenetv2_model(input_shape=(32, 32, 3), num_classes=100)


base_model = MobileNetV2(input_shape=[32, 32, 3], include_top=False, weights='imagenet')
x = base_model.output
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(1024, activation='relu')(x)
predictions = layers.Dense(100, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)

for layer in base_model.layers:
    layer.trainable = False

learning_rate = 0.01
lr_drop = 20
epoch = 200
batch_size = 32

def lr_scheduler(epoch, lr):
    print("learning rate:", lr)
    return learning_rate * (0.5 ** (epoch // lr_drop))


optimizer = optimizers.Adam(lr=learning_rate, clipvalue=0.5)
# optimizer = optimizers.Adam(lr=learning_rate)

reduce_lr = LearningRateScheduler(lr_scheduler)


# 早停法
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)

# Compile the model
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Callback to save the best model during training based on test accuracy
model_checkpoint = ModelCheckpoint(
    'MobileNetV2.h5',  # Adjust the filename as needed
    monitor='val_accuracy',  # Monitor validation accuracy (test accuracy in this case)
    save_best_only=True,
    mode='max',  # Save the model based on the maximum validation accuracy
    verbose=1
)

# Train the model
history = model.fit(
    train_datagen.flow(train_images, train_labels, batch_size=batch_size),
    epochs=epoch,
    validation_data=(test_images, test_labels),
    callbacks=[reduce_lr, model_checkpoint, early_stop],
    # callbacks=[reduce_lr, model_checkpoint],
    steps_per_epoch=len(train_images)//batch_size,
    validation_steps=len(test_images)//batch_size
)