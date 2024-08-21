import os
import tensorflow as tf
from tensorflow.keras import optimizers, regularizers
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models, regularizers
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


def MobileNet_custom(input_shape=(32, 32, 3), num_classes=10):
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
    x = layers.Dropout(0.4)(x)
    x = layers.GlobalAveragePooling2D(name='global_avg_pool')(x)
    x = layers.Dropout(0.4)(x)

    # Output layer
    outputs = layers.Dense(num_classes, activation='softmax', kernel_initializer=tf.keras.initializers.he_normal(),
                           kernel_regularizer=regularizers.l2(1e-4), name='predictions')(x)

    # Creating the model
    model = models.Model(inputs, outputs, name='mobilenet_custom')

    return model


# Instantiate
input_shape = (32, 32, 3)
num_classes = 10
model = MobileNet_custom(input_shape=input_shape, num_classes=num_classes)

learning_rate = 0.01
epoch = 100
batch_size = 128

optimizer = optimizers.SGD(learning_rate=learning_rate, momentum=0.9, decay=1e-6, nesterov=True, clipvalue=0.5)
# optimizer = optimizers.Adam(lr=learning_rate, clipvalue=0.5)
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
    validation_data=test_datagen.flow(test_images, test_labels, batch_size=batch_size),
    callbacks=[reduce_lr, model_checkpoint, early_stop],
    steps_per_epoch=len(train_images) // batch_size,
    validation_steps=len(test_images) // batch_size
)
