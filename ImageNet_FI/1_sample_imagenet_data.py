# sample picture from val
import os
import random
import shutil
import sys
from glob import glob
from pathlib import Path

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

if __name__ == '__main__':
    try:
        sampled_image_count = int(sys.argv[1])
    except:
        sampled_image_count = 10000
    print("Trying to sample " + str(sampled_image_count) + " images")
    original_image_dir = Path("../dataset/imagenet/val")
    original_path_labels = Path("./ILSVRC2012_validation_ground_truth.txt")
    sampled_image_dir = Path("../dataset/imagenet/sampled_val/")
    sampled_path_labels = Path("./sampled_validation_ground_truth.txt")

    image_paths = sorted(glob(str(original_image_dir / "*")))
    n_images = len(image_paths)

    total_index = list(range(n_images))
    sampled_index = random.sample(total_index, sampled_image_count)
    sampled_index.sort()
    sampled_images = [image_paths[i] for i in sampled_index]

    processed = 0
    for image in sampled_images:
        shutil.copy(image, sampled_image_dir)
        processed = processed + 1
        if processed % 100 == 0:
            print("Processed " + str(processed) + " images")

    with open(str(original_path_labels), "r") as f:
        y_val = f.read().strip().split("\n")
        sampled_y_val = [y_val[i] for i in sampled_index]
        textfile = open(sampled_path_labels, "w")
        for element in sampled_y_val:
            textfile.write(element + "\n")
        textfile.close()
