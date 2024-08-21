import numpy as np
from tensorflow.keras.datasets import cifar100
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Initialize ImageDataGenerator for preprocessing
test_datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True
)

# Load CIFAR-10 dataset
(train_images, train_labels), (test_images, test_labels) = cifar100.load_data()

# Convert images to float32
test_images = test_images.astype('float32')

# Flatten the test_labels to a 1D array (from shape (10000, 1) to shape (10000,))
test_labels = test_labels.flatten()

# Fit the ImageDataGenerator to the training images for normalization
test_datagen.fit(test_images)

# Standardize the test images using the fitted parameters
test_images = test_datagen.standardize(test_images)

# Save the standardized test images and integer labels as .npy files
np.save('../dataset/CIFAR100/cifar100_test_images.npy', test_images)
np.save('../dataset/CIFAR100/cifar100_test_labels.npy', test_labels)

print("测试数据和标签已保存为 cifar10_test_images.npy 和 cifar10_test_labels.npy")
