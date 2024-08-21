
from tensorflow.keras.datasets import cifar10
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, optimizers, regularizers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.models import Model

from tensorflow.keras.callbacks import  EarlyStopping, ModelCheckpoint



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


def inverted_res_block(x, expansion, filters, stride, block_id):
    in_channels = x.shape[-1]
    prefix = f'block_{block_id}_'

    if block_id:
        x_shortcut = x  # Store the input tensor for the skip connection

        x = layers.Conv2D(expansion * in_channels, kernel_size=1, padding='same', use_bias=False,
                          name=prefix + 'expand')(x)
        x = layers.BatchNormalization(epsilon=1e-3, momentum=0.999, name=prefix + 'expand_BN')(x)
        x = layers.ReLU(6., name=prefix + 'expand_relu')(x)

    x = layers.DepthwiseConv2D(kernel_size=3, strides=stride, activation=None, use_bias=False, padding='same',
                               name=prefix + 'depthwise')(x)
    x = layers.BatchNormalization(epsilon=1e-3, momentum=0.999, name=prefix + 'depthwise_BN')(x)
    x = layers.ReLU(6., name=prefix + 'depthwise_relu')(x)

    x = layers.Conv2D(filters, kernel_size=1, padding='same', use_bias=False, activation=None, name=prefix + 'project')(
        x)
    x = layers.BatchNormalization(epsilon=1e-3, momentum=0.999, name=prefix + 'project_BN')(x)

    if in_channels == filters and stride == 1:
        x = layers.Add(name=prefix + 'add')([x, x_shortcut])  # Apply the skip connection

    return x



def MobileNetV2_custom(input_shape=(32, 32, 3), num_classes=10):
    inputs = layers.Input(shape=input_shape)

    x = layers.Conv2D(32, kernel_size=3, strides=(1, 1), padding='same', use_bias=False, name='Conv1')(inputs)
    x = layers.BatchNormalization(epsilon=1e-3, momentum=0.999, name='bn_Conv1')(x)
    x = layers.ReLU(6., name='Conv1_relu')(x)

    # Add Inverted Residual Blocks
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

    # Concluding with Conv layer and pooling
    x = layers.Conv2D(1280, kernel_size=1, use_bias=False, name='Conv_1')(x)
    x = layers.BatchNormalization(epsilon=1e-3, momentum=0.999, name='Conv_1_bn')(x)
    x = layers.ReLU(6., name='out_relu')(x)

    # Pooling and Dropout
    x = layers.Dropout(0.5)(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.5)(x)

    # Output layer
    outputs = layers.Dense(num_classes, activation='softmax', kernel_initializer=tf.keras.initializers.he_normal(),
                           kernel_regularizer=regularizers.l2(1e-4))(x)

    # Creating the model
    model = models.Model(inputs, outputs)

    return model


# Instantiate
input_shape = (32, 32, 3)
num_classes = 10
model = MobileNetV2_custom(input_shape=input_shape, num_classes=num_classes)


learning_rate = 0.01
lr_drop = 20
epoch = 200
batch_size = 128


def lr_scheduler(epoch, lr):
    print("Learning rate:", lr)
    return learning_rate * (0.5 ** (epoch // lr_drop))

optimizer = optimizers.SGD(learning_rate=learning_rate, momentum=0.9, decay=1e-6, nesterov=True, clipvalue=0.5)
# optimizer = optimizers.Adam(lr=learning_rate, clipvalue=0.5)
reduce_lr = LearningRateScheduler(lr_scheduler)
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)

# Callback to save the best model during training based on test accuracy
model_checkpoint = ModelCheckpoint(
    'MobileNetV2.h5',  # Adjust the filename as needed
    monitor='val_accuracy',  # Monitor validation accuracy (test accuracy in this case)
    save_best_only=True,
    mode='max',  # Save the model based on the maximum validation accuracy
    verbose=1
)

# Compile the model
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(
    train_datagen.flow(train_images, train_labels, batch_size=batch_size),
    epochs=epoch,
    validation_data=test_datagen.flow(test_images, test_labels, batch_size=batch_size),
    callbacks=[reduce_lr, model_checkpoint, early_stop],
    steps_per_epoch=len(train_images) // batch_size,
    validation_steps=len(test_images) // batch_size
)
