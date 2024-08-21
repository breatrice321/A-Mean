import os
import pickle
import tarfile
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
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
        train_dataset = pickle.load(file, encoding='latin1', fix_imports=False, errors='ignore')

    with open(os.path.join(extracted_folder, 'test'), 'rb') as file:
        test_dataset = pickle.load(file, encoding='latin1', fix_imports=False, errors='ignore')

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

# Build the ResNet50 model
def build_resnet50_model(input_shape=(32, 32, 3), num_classes=100, weight_decay=1e-4):
    # Input layer
    input_tensor = layers.Input(shape=input_shape)

    # Initial Convolutional Layer
    x = layers.Conv2D(64, (7, 7), strides=(2, 2), padding='same', kernel_initializer=tf.keras.initializers.he_normal(), kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(input_tensor)
    x = layers.BatchNormalization(epsilon=1.001e-5)(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

    # Residual Blocks
    x = _residual_block(x, filters=[64, 64, 256], block_name='block1', strides=(1, 1), weight_decay=weight_decay)
    x = _residual_block(x, filters=[128, 128, 512], block_name='block2', weight_decay=weight_decay)
    x = _residual_block(x, filters=[256, 256, 1024], block_name='block3', weight_decay=weight_decay)
    x = _residual_block(x, filters=[512, 512, 2048], block_name='block4', weight_decay=weight_decay)

    # Global Average Pooling
    x = layers.GlobalAveragePooling2D()(x)

    # Dropout Layer
    x = layers.Dropout(rate=0.5)(x)

    # Fully Connected Layer
    x = layers.Dense(1000, kernel_initializer=tf.keras.initializers.he_normal(), kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(x)
    x = layers.BatchNormalization(epsilon=1.001e-5)(x)
    x = layers.Activation('relu')(x)

    # Dropout Layer
    x = layers.Dropout(rate=0.5)(x)

    # Output Layer
    output_tensor = layers.Dense(num_classes, activation='softmax', kernel_initializer=tf.keras.initializers.he_normal(), kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(x)

    # Create model
    model = models.Model(inputs=input_tensor, outputs=output_tensor)

    return model

def _residual_block(input_tensor, filters, block_name, strides=(2, 2), weight_decay=1e-4):
    # Shortcut connection
    shortcut = layers.Conv2D(filters[2], (1, 1), strides=strides, kernel_initializer=tf.keras.initializers.he_normal(), kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(input_tensor)
    shortcut = layers.BatchNormalization(epsilon=1.001e-5)(shortcut)

    # Main path
    x = layers.Conv2D(filters[0], (1, 1), strides=strides, kernel_initializer=tf.keras.initializers.he_normal(), kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(input_tensor)
    x = layers.BatchNormalization(epsilon=1.001e-5)(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters[1], (3, 3), padding='same', kernel_initializer=tf.keras.initializers.he_normal(), kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(x)
    x = layers.BatchNormalization(epsilon=1.001e-5)(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters[2], (1, 1), kernel_initializer=tf.keras.initializers.he_normal(), kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(x)
    x = layers.BatchNormalization(epsilon=1.001e-5)(x)

    # Add shortcut to main path
    x = layers.Add()([x, shortcut])
    x = layers.Activation('relu')(x)

    return x

model = build_resnet50_model(input_shape=(32, 32, 3), num_classes=100)


learning_rate = 0.001
epoch = 150
batch_size = 32


optimizer = optimizers.Adam(lr=learning_rate, clipvalue=0.5)


# 学习率调度
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7, verbose=1)

# 早停法
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


# 在 model.fit 中添加回调
history = model.fit(
    train_datagen.flow(train_images, train_labels, batch_size=batch_size),
    epochs=epoch,
    validation_data=test_datagen.flow(test_images, test_labels, batch_size=batch_size),
    callbacks=[reduce_lr, early_stop, model_checkpoint],
    steps_per_epoch=len(train_images) // batch_size,
    validation_steps=len(test_images) // batch_size
)

