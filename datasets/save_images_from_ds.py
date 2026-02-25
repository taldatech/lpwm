from get_dataset import get_image_dataset, get_video_dataset
import os
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import ToPILImage
from torchvision.io.image import write_png
from PIL import Image
from tqdm import tqdm


def save_images_from_ds(ds_name, root, num_samples, target_dir, image_size):
    ds = get_image_dataset(ds_name, root, image_size=image_size)
    batch_size = num_samples
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0)
    batch = next(iter(dl))
    images = batch[0]
    for i in tqdm(range(images.shape[0])):
        img = images[i]
        img_pil = ToPILImage()(img)
        ex_dir = os.path.join(target_dir, f'{i}')
        os.makedirs(ex_dir, exist_ok=True)
        img_pil.save(os.path.join(ex_dir, f'{i}.png'))
    print(f'saved {batch_size} images in {target_dir}')


if __name__ == '__main__':
    # ds_name = 'cifar10'
    # target_dir = '/home/tal/projects/gdlp/assets/cifar10'
    root = '/media/newhd/data/cifar10'
    # image_size = 32
    ds_name = 'shapes'
    target_dir = '/home/tal/projects/gdlp/assets/shapes'
    image_size = 64
    num_samples = 20
    save_images_from_ds(ds_name, root, num_samples, target_dir, image_size)
