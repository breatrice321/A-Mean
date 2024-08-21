import os
from glob import glob
from pathlib import Path

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import cv2


def load_images(image_paths, image_dim, returned_shard, n_shards=5):
    ratio = 1.14286
    resize_dim = int(ratio * image_dim)
    print("Resize dim: " + str(resize_dim))
    assert 0 <= returned_shard < n_shards, "The argument returned_shard must be between 0 and n_shards"

    shard_size = len(image_paths) // n_shards
    sharded_image_paths = image_paths[returned_shard * shard_size:(
                                           returned_shard + 1) * shard_size] if returned_shard < n_shards - 1 \
        else image_paths[returned_shard * shard_size:]

    images_list = np.zeros((len(sharded_image_paths), image_dim, image_dim, 3), dtype=np.uint8)

    for i, image_path in enumerate(sharded_image_paths):
        # Load (in BGR channel order)
        image = cv2.imread(image_path)

        # Resize
        height, width, _ = image.shape
        new_height = height * resize_dim // min(image.shape[:2])
        new_width = width * resize_dim // min(image.shape[:2])
        image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)

        # Crop
        height, width, _ = image.shape
        startx = width // 2 - (image_dim // 2)
        starty = height // 2 - (image_dim // 2)
        image = image[starty:starty + image_dim, startx:startx + image_dim]
        assert image.shape[0] == image_dim and image.shape[1] == image_dim, (image.shape, height, width)
        images_list[i, ...] = image[..., ::-1]

    return images_list


def process_labels(path_meta, path_synset_words, path_labels):
    meta = scipy.io.loadmat(str(path_meta))
    original_idx_to_synset = {}
    synset_to_name = {}

    for i in range(1000):

        ilsvrc2012_id = int(meta["synsets"][i, 0][0][0][0])
        synset = meta["synsets"][i, 0][1][0]
        name = meta["synsets"][i, 0][2][0]
        original_idx_to_synset[ilsvrc2012_id] = synset
        synset_to_name[synset] = name

    synset_to_keras_idx = {}
    keras_idx_to_name = {}
    with open(str(path_synset_words), "r") as f:
        for idx, line in enumerate(f):
            parts = line.split(" ")
            synset_to_keras_idx[parts[0]] = idx
            keras_idx_to_name[idx] = " ".join(parts[1:])

    convert_original_idx_to_keras_idx = lambda idx: synset_to_keras_idx[original_idx_to_synset[idx]]
    print(convert_original_idx_to_keras_idx)

    with open(str(path_labels), "r") as f:
        y_val = f.read().strip().split("\n")
        y_val = np.array([convert_original_idx_to_keras_idx(int(idx)) for idx in y_val])

    np.save(str(path_imagenet_val_dataset / "result/y_val_sampled.npy"), y_val)
    return keras_idx_to_name, y_val


if __name__ == '__main__':
    path_imagenet_val_dataset = Path("../dataset/imagenet/")  # path/to/data/
    dir_images = Path("../dataset/imagenet/sampled_val")  # path/to/images/directory
    path_labels = Path("./sampled_validation_ground_truth.txt")
    path_synset_words = Path("./synset_words.txt")
    path_meta = Path("./imagenet/meta.mat")

    image_paths = sorted(glob(str(dir_images / "*")))
    n_images = len(image_paths)
    print("Total image count : " + str(n_images))

    keras_idx_to_name, y_val = process_labels(path_meta, path_synset_words, path_labels)

    image_dim_list = [224, 299, 331]
    n_shards = 1
    for image_dim in image_dim_list:
        print("Image dim: " + str(image_dim))
        for i in range(n_shards):
            images = load_images(image_paths, image_dim=image_dim, returned_shard=i, n_shards=n_shards)
            np.save(str(path_imagenet_val_dataset / "result/sampled_new_x_val_{}_{}.npy".format(image_dim, (i + 1))), images)
            if (i + 1) * 100 / n_shards % 1 == 0:
                print("{:.0f}% Completed.".format((i + 1) / n_shards * 100))
            images = None

        idx_shard = 1
        x_val = np.load(str(path_imagenet_val_dataset / "result/sampled_new_x_val_{}_{}.npy").format(image_dim, idx_shard))

        n_images2show = 15
        n_cols = 3
        n_rows = 15 // n_cols
        figsize = (20, 20)

        indices = np.random.choice(x_val.shape[0], size=n_images2show, replace=False)
        images = x_val[indices] / 255.

        fig, ax = plt.subplots(figsize=figsize, nrows=n_rows, ncols=n_cols)
        for i, axi in enumerate(ax.flat):
            axi.imshow(images[i])
            label_index = (idx_shard - 1) * (n_images // n_shards) + indices[i]
            axi.set_title(keras_idx_to_name[y_val[label_index]], y=.9, fontdict={'fontweight': 'bold'}, pad=10)
            axi.set_axis_off()

        plt.savefig('../dataset/imagenet/result/imagenet_sample_new_' + str(image_dim) + '.png')
