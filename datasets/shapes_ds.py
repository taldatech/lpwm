"""
Simple Random Colored Shapes Dataset
"""
# imports
import numpy as np
from skimage.draw import random_shapes
from tqdm.auto import tqdm
import torch


def generate_shape_dataset(img_size=64, min_shapes=2, max_shapes=5, min_size=10, max_size=12, allow_overlap=False,
                           num_images=10_000):
    images = []
    for i in tqdm(range(num_images)):
        img, _ = random_shapes((img_size, img_size), min_shapes=min_shapes, max_shapes=max_shapes,
                               intensity_range=((0, 200),), min_size=min_size, max_size=max_size,
                               allow_overlap=allow_overlap, num_trials=100)
        img[:, :, 0][img[:, :, 0] == 255] = 0
        img[:, :, 1][img[:, :, 1] == 255] = 255
        img[:, :, 2][img[:, :, 2] == 255] = 255
        img = img / 255.0
        images.append(img)
    images = np.stack(images, axis=0)  # [num_mages, H, W, 3]
    return images


def generate_shape_dataset_torch(img_size=64, min_shapes=2, max_shapes=5, min_size=11, max_size=13, allow_overlap=False,
                                 num_images=10_000):
    images = generate_shape_dataset(img_size=img_size, min_shapes=min_shapes, max_shapes=max_shapes, min_size=min_size,
                                    max_size=max_size,
                                    allow_overlap=allow_overlap, num_images=num_images)
    # create torch dataset
    img_data_torch = images.transpose(0, 3, 1, 2)  # [num_images, 3, H, W]
    img_ds = torch.utils.data.TensorDataset(torch.tensor(img_data_torch, dtype=torch.float), torch.arange(num_images))
    return img_ds