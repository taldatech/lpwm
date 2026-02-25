"""
functions and classes to process the Traffic dataset
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm


def list_images_in_dir(path):
    valid_images = [".jpg", ".gif", ".png"]
    img_list = []
    for f in os.listdir(path):
        ext = os.path.splitext(f)[1]
        if ext.lower() not in valid_images:
            continue
        img_list.append(os.path.join(path, f))
    return img_list


def prepare_numpy_file(path_to_image_dir, image_size=128, frameskip=1):
    # path_to_image_dir = '/media/newhd/data/traffic_data/rimon_frames/'
    img_list = list_images_in_dir(path_to_image_dir)
    img_list = sorted(img_list, key=lambda x: int(x.split('/')[-1].split('.')[0]))
    print(f'img_list: {len(img_list)}, 0: {img_list[0]}, -1: {img_list[-1]}')
    img_np_list = []
    for i in tqdm(range(len(img_list))):
        if i % frameskip != 0:
            continue
        img = Image.open(img_list[i])
        img = img.convert('RGB')
        img = img.crop((60, 0, 480, 420))
        img = img.resize((image_size, image_size), Image.BICUBIC)
        img_np = np.asarray(img)
        img_np_list.append(img_np)
    img_np_array = np.stack(img_np_list, axis=0)
    print(f'img_np_array: {img_np_array.shape}')
    save_path = os.path.join(path_to_image_dir, f'img{image_size}np_fs{frameskip}.npy')
    np.save(save_path, img_np_array)
    print(f'file save at @ {save_path}')


class TrafficDataset(Dataset):
    def __init__(self, path_to_npy, mode, ep_len=50, sample_length=20, image_size=128, transform=None):
        super(TrafficDataset, self).__init__()
        assert mode in ['train', 'val', 'valid', 'test']
        if mode == 'valid':
            mode = 'val'
        self.mode = mode
        self.horizon = sample_length
        self.ep_len = ep_len
        data = np.load(path_to_npy)
        train_size = int(0.8 * data.shape[0])
        valid_size = int(0.1 * data.shape[0])
        test_size = int(0.1 * data.shape[0])
        if mode == 'train':
            print(f'loaded data with shape: {data.shape}, train_size: {train_size}, valid_size: {valid_size}')
            self.data = data[:train_size]
        elif mode == 'val':
            self.data = data[train_size:train_size + valid_size]
        elif mode == 'test':
            self.data = data[train_size + valid_size:]
        else:
            raise SystemError('unrecognized ds mode: {mode}')
        self.image_size = image_size
        self.num_episodes = len(self.data) // self.ep_len
        if transform is None:
            self.input_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(image_size),
                transforms.ToTensor()
            ])
        else:
            self.input_transform = transform

    def __getitem__(self, index):
        images = []
        if self.mode == 'train':
            length = self.data.shape[0]
            horizon = self.horizon if self.mode == 'train' else self.ep_len
            if (index + horizon) >= length:
                slack = index + horizon - length
                index = index - slack
            for i in range(horizon):
                t = index + i
                images.append(self.input_transform(self.data[t]))
        else:
            # episode i, get the starting index
            first_frame = index * self.ep_len
            length = self.data.shape[0]
            horizon = self.ep_len
            if (first_frame + horizon) >= length:
                slack = first_frame + horizon - length
                first_frame = first_frame - slack
            for i in range(horizon):
                t = first_frame + i
                images.append(self.input_transform(self.data[t]))

        images = torch.stack(images, dim=0)
        pos = torch.zeros(0)
        size = torch.zeros(0)
        id = torch.zeros(0)
        in_camera = torch.zeros(0)
        return images, pos, size, id, in_camera

    def __len__(self):
        if self.mode == 'train':
            return self.data.shape[0]
        else:
            return self.num_episodes


class TrafficDatasetImage(Dataset):
    def __init__(self, path_to_npy, mode, ep_len=50, sample_length=20, image_size=128, transform=None):
        super(TrafficDatasetImage, self).__init__()
        assert mode in ['train', 'val', 'valid', 'test']
        if mode == 'valid':
            mode = 'val'
        self.mode = mode
        self.horizon = sample_length
        self.ep_len = ep_len
        data = np.load(path_to_npy)
        train_size = int(0.8 * data.shape[0])
        valid_size = int(0.1 * data.shape[0])
        test_size = int(0.1 * data.shape[0])
        if mode == 'train':
            print(f'loaded data with shape: {data.shape}, train_size: {train_size}, valid_size: {valid_size}')
            self.data = data[:train_size]
        elif mode == 'val':
            self.data = data[train_size:train_size + valid_size]
        elif mode == 'test':
            self.data = data[train_size + valid_size:]
        else:
            raise SystemError('unrecognized ds mode: {mode}')
        self.image_size = image_size
        self.num_episodes = len(self.data) // self.ep_len
        if transform is None:
            self.input_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(image_size),
                transforms.ToTensor()
            ])
        else:
            self.input_transform = transform

    def __getitem__(self, index):
        images = []
        length = self.data.shape[0]
        horizon = self.horizon
        if (index + horizon) >= length:
            slack = index + horizon - length
            index = index - slack
        for i in range(horizon):
            t = index + i
            images.append(self.input_transform(self.data[t]))
        images = torch.stack(images, dim=0)
        pos = torch.zeros(0)
        size = torch.zeros(0)
        id = torch.zeros(0)
        in_camera = torch.zeros(0)
        return images, pos, size, id, in_camera

    def __len__(self):
        return self.data.shape[0]
