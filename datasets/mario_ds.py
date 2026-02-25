import os
import os.path as osp
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.utils as vutils
import matplotlib.pyplot as plt
from tqdm import tqdm
import glob
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


def list_images_in_dir(path):
    valid_images = [".jpg", ".gif", ".png"]
    img_list = []
    for f in os.listdir(path):
        ext = os.path.splitext(f)[1]
        if ext.lower() not in valid_images:
            continue
        img_list.append(os.path.join(path, f))
    return img_list


def prepare_numpy_file(path_to_image_dir, image_size=128, frameskip=1, prefix=""):
    img_list = list_images_in_dir(path_to_image_dir)
    img_list = sorted(img_list, key=lambda x: int(x.split('/')[-1].split('.')[0].split('t')[-1]))
    print(f'img_list: {len(img_list)}, 0: {img_list[0]}, -1: {img_list[-1]}')
    img_np_list = []
    for i in tqdm(range(len(img_list))):
        if i % frameskip != 0:
            continue
        img = Image.open(img_list[i])
        img = img.convert('RGB')
        w, h = img.size
        img = img.crop((0, 0, w, h - 30))
        img = img.resize((image_size, image_size), Image.BICUBIC)
        img_np = np.asarray(img)
        img_np_list.append(img_np)
    img_np_array = np.stack(img_np_list, axis=0)
    print(f'img_np_array: {img_np_array.shape}')
    save_path = os.path.join(path_to_image_dir, f'mario{prefix}_img{image_size}np_fs{frameskip}.npy')
    np.save(save_path, img_np_array)
    print(f'file save at @ {save_path}')
    
    
class MarioVideo(Dataset):
    def __init__(self, root, mode, ep_len=100, sample_length=20, image_size=128):
        # path = os.path.join(root, mode)
        assert mode in ['train', 'val', 'valid', 'test']
        if mode == 'valid' or mode == 'test':
            mode = 'val'
        self.root = os.path.join(root, mode)
        self.image_size = image_size

        self.mode = mode
        self.sample_length = sample_length

        # Get all numbers
        self.folders = []
        for file in os.listdir(self.root):
            try:
                self.folders.append(int(file))
            except ValueError:
                continue
        self.folders.sort()

        self.episodes = []
        self.EP_LEN = ep_len
        self.seq_per_episode = self.EP_LEN - self.sample_length + 1

        for f in self.folders:
            dir_name = os.path.join(self.root, str(f))
            paths = list(glob.glob(osp.join(dir_name, '*.png')))
            # if len(paths) != self.EP_LEN:
            #     continue
            # assert len(paths) == self.EP_LEN, 'len(paths): {}'.format(len(paths))
            get_num = lambda x: int(osp.splitext(osp.basename(x))[0])
            paths.sort(key=get_num)
            self.episodes.append(paths)

    def __getitem__(self, index):

        imgs = []
        if self.mode == 'train':
            # Implement continuous indexing
            ep = index // self.seq_per_episode
            offset = index % self.seq_per_episode
            end = offset + self.sample_length

            e = self.episodes[ep]
            for image_index in range(offset, end):
                img = Image.open(osp.join(e[image_index]))
                img = img.resize((self.image_size, self.image_size))
                img = transforms.ToTensor()(img)[:3]
                imgs.append(img)
        else:
            for path in self.episodes[index]:
                img = Image.open(path)
                img = img.resize((self.image_size, self.image_size))
                img = transforms.ToTensor()(img)[:3]
                imgs.append(img)

        img = torch.stack(imgs, dim=0).float()
        pos = torch.zeros(0)
        size = torch.zeros(0)
        id = torch.zeros(0)
        in_camera = torch.zeros(0)

        return img, pos, size, id, in_camera

    def __len__(self):
        length = len(self.episodes)
        if self.mode == 'train':
            return length * self.seq_per_episode
        else:
            return length
        
        
class MarioImage(Dataset):
    def __init__(self, root, mode, ep_len=100, sample_length=1, image_size=128):
        # path = os.path.join(root, mode)
        assert mode in ['train', 'val', 'valid', 'test']
        if mode == 'valid' or mode == 'test':
            mode = 'val'
        self.root = os.path.join(root, mode)
        self.image_size = image_size

        self.mode = mode
        self.sample_length = sample_length

        # Get all numbers
        self.folders = []
        for file in os.listdir(self.root):
            try:
                self.folders.append(int(file))
            except ValueError:
                continue
        self.folders.sort()

        self.episodes = []
        self.EP_LEN = ep_len
        self.seq_per_episode = self.EP_LEN - self.sample_length + 1

        for f in self.folders:
            dir_name = os.path.join(self.root, str(f))
            paths = list(glob.glob(osp.join(dir_name, '*.png')))
            # if len(paths) != self.EP_LEN:
            #     continue
            # assert len(paths) == self.EP_LEN, 'len(paths): {}'.format(len(paths))
            get_num = lambda x: int(osp.splitext(osp.basename(x))[0])
            paths.sort(key=get_num)
            self.episodes.append(paths)

    def __getitem__(self, index):
        imgs = []
        if self.mode == 'train':
            # Implement continuous indexing
            ep = index // self.seq_per_episode
            offset = index % self.seq_per_episode
            end = offset + self.sample_length

            e = self.episodes[ep]
            for image_index in range(offset, end):
                img = Image.open(osp.join(e[image_index]))
                img = img.resize((self.image_size, self.image_size))
                img = transforms.ToTensor()(img)[:3]
                imgs.append(img)
        else:
            for path in self.episodes[index]:
                img = Image.open(path)
                img = img.resize((self.image_size, self.image_size))
                img = transforms.ToTensor()(img)[:3]
                imgs.append(img)

        img = torch.stack(imgs, dim=0).float()
        pos = torch.zeros(0)
        size = torch.zeros(0)
        id = torch.zeros(0)
        in_camera = torch.zeros(0)

        return img, pos, size, id, in_camera

    def __len__(self):
        length = len(self.episodes)
        if self.mode == 'train':
            return length * self.seq_per_episode
        else:
            return length


if __name__ == '__main__':
    # prepare data
    # image_size = 128
    # path_to_image_dir = '/media/newhd/data/mario/blue'
    # debug
    # path_to_image = os.path.join(path_to_image_dir, '2317.png')
    # img = Image.open(path_to_image)
    # img = img.convert('RGB')
    # w, h = img.size
    # img = img.crop((0, 0, w, h - 30))
    # img.show()
    # img = img.resize((image_size, image_size), Image.BICUBIC)

    # prefix = '_blue'
    # frameskip = 1
    # # prepare_numpy_file(path_to_image_dir, image_size=image_size, frameskip=frameskip, prefix=prefix)
    #
    test_epochs = True
    # # load data
    # path_to_npy = f'/media/newhd/data/mario/mario{prefix}_img{image_size}np_fs{frameskip}.npy'
    # mode = 'single'
    # horizon = 4
    # train = True
    # mario_ds = MarioDataset(path_to_npy, mode=mode, train=train, horizon=horizon)
    # mario_dl = DataLoader(mario_ds, shuffle=True, pin_memory=True, batch_size=5)
    # batch = next(iter(mario_dl))
    # if mode == 'single':
    #     im1 = batch[0]
    # elif mode == 'frames' or mode == 'tps':
    #     im1 = batch[0][0]
    #     im2 = batch[1][0]
    #
    # if mode == 'single':
    #     print(im1.shape)
    #     img_np = im1.permute(1, 2, 0).data.cpu().numpy()
    #     fig = plt.figure(figsize=(5, 5))
    #     ax = fig.add_subplot(111)
    #     ax.imshow(img_np)
    # elif mode == 'horizon':
    #     print(f'batch shape: {batch.shape}')
    #     images = batch[0]
    #     print(f'images shape: {images.shape}')
    #     fig = plt.figure(figsize=(8, 8))
    #     for i in range(images.shape[0]):
    #         ax = fig.add_subplot(1, horizon, i + 1)
    #         im = images[i]
    #         im_np = im.permute(1, 2, 0).data.cpu().numpy()
    #         ax.imshow(im_np)
    #         ax.set_title(f'im {i + 1}')
    # else:
    #     print(f'im1: {im1.shape}, im2: {im2.shape}')
    #     im1_np = im1.permute(1, 2, 0).data.cpu().numpy()
    #     im2_np = im2.permute(1, 2, 0).data.cpu().numpy()
    #     fig = plt.figure(figsize=(8, 8))
    #     ax = fig.add_subplot(1, 2, 1)
    #     ax.imshow(im1_np)
    #     ax.set_title('im1')
    #
    #     ax = fig.add_subplot(1, 2, 2)
    #     ax.imshow(im2_np)
    #     ax.set_title('im2 [t-1] or [tps]')
    # plt.show()
    # if test_epochs:
    #     # from tqdm import tqdm
    #     pbar = tqdm(iterable=mario_dl)
    #     for batch in pbar:
    #         pass
    #     pbar.close()

    # --- episodic setting --- #
    root = '/media/newhd/data/mario'
    mario_ds = MarioVideo(root=root, ep_len=100, sample_length=10, mode='train')
    mario_dl = DataLoader(mario_ds, shuffle=True, pin_memory=True, batch_size=32)
    batch = next(iter(mario_dl))
    im = batch[0][0][0]
    print(im.shape)
    img_np = im.permute(1, 2, 0).data.cpu().numpy()
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111)
    ax.imshow(img_np)
    plt.show()

    if test_epochs:
        from tqdm import tqdm

        pbar = tqdm(iterable=mario_dl)
        for batch in pbar:
            pass
        pbar.close()