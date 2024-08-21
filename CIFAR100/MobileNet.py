import os
import pickle
import tarfile
import numpy as np
import tensorflow as tf
from tensorflow.keras import optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.callbacks import LearningRateScheduler, ModelCheckpoint, EarlyStopping
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, DepthwiseConv2D, BatchNormalization, ReLU, GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.callbacks import Callback



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

# Build the MobileNet model
def depthwise_separable_conv_block(x, filters, kernel_size, strides, weight_decay=1e-4, dropout_rate=0.15):
    # Depthwise Convolution
    x = DepthwiseConv2D(kernel_size=kernel_size, strides=strides, padding='same', depthwise_regularizer=regularizers.l2(weight_decay))(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    # Pointwise Convolution
    x = Conv2D(filters, kernel_size=(1, 1), strides=(1, 1), padding='same', kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    # Dropout
    if dropout_rate > 0:
        x = Dropout(dropout_rate)(x)

    return x

def build_mobilenet_model(input_shape=(32, 32, 3), num_classes=100):
    input_tensor = Input(shape=input_shape)

    x = Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_initializer=tf.keras.initializers.he_normal(), kernel_regularizer=regularizers.l2(1e-4))(input_tensor)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = depthwise_separable_conv_block(x, filters=64, kernel_size=(3, 3), strides=(1, 1))
    x = depthwise_separable_conv_block(x, filters=128, kernel_size=(3, 3), strides=(2, 2))
    x = depthwise_separable_conv_block(x, filters=128, kernel_size=(3, 3), strides=(1, 1))
    x = depthwise_separable_conv_block(x, filters=256, kernel_size=(3, 3), strides=(2, 2))
    x = depthwise_separable_conv_block(x, filters=256, kernel_size=(3, 3), strides=(1, 1))
    x = depthwise_separable_conv_block(x, filters=512, kernel_size=(3, 3), strides=(2, 2))

    for _ in range(5):
        x = depthwise_separable_conv_block(x, filters=512, kernel_size=(3, 3), strides=(1, 1))

    x = depthwise_separable_conv_block(x, filters=1024, kernel_size=(3, 3), strides=(2, 2))
    x = depthwise_separable_conv_block(x, filters=1024, kernel_size=(3, 3), strides=(1, 1))

    x = GlobalAveragePooling2D()(x)

    x = Dropout(0.5)(x)  # Additional dropout layer

    output_tensor = Dense(num_classes, activation='softmax', kernel_initializer=tf.keras.initializers.he_normal(), kernel_regularizer=regularizers.l2(1e-4))(x)

    model = Model(inputs=input_tensor, outputs=output_tensor)

    return model


model = build_mobilenet_model(input_shape=(32, 32, 3), num_classes=100)


learning_rate = 0.1
# lr_drop = 20
epoch = 100
batch_size = 128


optimizer = optimizers.SGD(learning_rate=learning_rate, momentum=0.9, decay=1e-6, nesterov=True, clipvalue=0.5)

# 学习率调度
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7, verbose=1)

# 早停法
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)

# reduce_lr = LearningRateScheduler(lr_scheduler)

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
    validation_data=test_datagen.flow(test_images, test_labels, batch_size=batch_size),
    # callbacks=[reduce_lr, model_checkpoint, early_stop],
    callbacks=[reduce_lr, model_checkpoint],
    steps_per_epoch=len(train_images) // batch_size,
    validation_steps=len(test_images) // batch_size
)
