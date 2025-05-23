
import os
import random

def split_dataset(image_dir, mask_dir):
    image_files = sorted(os.listdir(image_dir))
    mask_files = sorted(os.listdir(mask_dir))

    image_paths = [os.path.join(image_dir, f) for f in image_files]
    mask_paths = [os.path.join(mask_dir, f) for f in mask_files]

    paired = list(zip(image_paths, mask_paths))
    random.shuffle(paired)

    total = len(paired)
    train_len = int(total * 0.7)
    val_len = int(total * 0.15)
    test_len = total - train_len - val_len

    train = paired[:train_len]
    val = paired[train_len:train_len + val_len]
    test = paired[train_len + val_len:]

    return train, val, test
