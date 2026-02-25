from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import glob
import os
import os.path as osp
import torch
import numpy as np
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


class BAIR64Video(Dataset):
    def __init__(self, root, mode, ep_len=30, sample_length=17, image_size=64):
        # path = os.path.join(root, mode)
        if mode == 'val' or mode == 'valid':
            mode = 'test'
        assert mode in ['train', 'test']
        self.root = os.path.join(root, mode)
        self.image_size = image_size
        self.mode = mode
        self.sample_length = sample_length

        # Get all numbers
        self.folders = []
        for file in os.listdir(self.root):
            try:
                self.folders.append(file)
            except ValueError:
                continue
        self.folders.sort(key=lambda x: int(x.split('_')[1]))  # 'traj_x_to_y'

        self.episodes = []
        self.actions_path = []
        self.EP_LEN = ep_len
        self.seq_per_episode = self.EP_LEN - self.sample_length + 1

        for f in self.folders:
            trajectories = list(os.listdir(os.path.join(self.root, f)))
            trajectories.sort(key=lambda x: int(x))
            for f_t in trajectories:
                dir_name = os.path.join(self.root, f, f_t)
                paths = list(glob.glob(osp.join(dir_name, '*.png')))
                # if len(paths) != self.EP_LEN:
                #     continue
                # assert len(paths) == self.EP_LEN, 'len(paths): {}'.format(len(paths))
                get_num = lambda x: int(osp.splitext(osp.basename(x))[0])
                paths.sort(key=get_num)
                self.episodes.append(paths)
                self.actions_path.append(osp.join(dir_name, 'endeffector_positions.csv'))
            #if len(paths) != ep_len:
            #    print(dir_name)
            #    print(len(paths))
        #print(len(self.episodes))
        # if self.mode == 'val':
        #     self.episodes = self.episodes[:200]

    def __getitem__(self, index):
        #print(index)
        imgs = []
        # actions = []
        # dones = []
        # rewards = []
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

            act = np.genfromtxt(self.actions_path[ep], delimiter=',', dtype=float)
            actions = torch.tensor(act[offset:end])

        else:
            for path in self.episodes[index]:
                img = Image.open(path)
                img = img.resize((self.image_size, self.image_size))
                img = transforms.ToTensor()(img)[:3]
                imgs.append(img)

            act = np.genfromtxt(self.actions_path[index], delimiter=',', dtype=float)
            actions = torch.tensor(act)


        img = torch.stack(imgs, dim=0).float()
        pos = torch.zeros(0)
        size = torch.zeros(0)
        id = torch.zeros(0)
        in_camera = torch.zeros(0)

        return img, actions, size, id, in_camera

    def __len__(self):
        length = len(self.episodes)
        if self.mode == 'train':
            return length * self.seq_per_episode
        else:
            return length


class BAIR64Image(Dataset):
    def __init__(self, root, mode, ep_len=30, sample_length=1, image_size=64):
        # path = os.path.join(root, mode)
        if mode == 'val' or mode == 'valid':
            mode = 'test'
        assert mode in ['train', 'test']
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
            trajectories = list(os.listdir(os.path.join(self.root, f)))
            trajectories.sort(key=lambda x: int(x))
            for f_t in trajectories:
                dir_name = os.path.join(self.root, f, f_t)
                paths = list(glob.glob(osp.join(dir_name, '*.png')))
                # if len(paths) != self.EP_LEN:
                #     continue
                # assert len(paths) == self.EP_LEN, 'len(paths): {}'.format(len(paths))
                get_num = lambda x: int(osp.splitext(osp.basename(x))[0])
                paths.sort(key=get_num)
                self.episodes.append(paths)

    def __getitem__(self, index):
        imgs = []
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

        img = torch.stack(imgs, dim=0).float()
        pos = torch.zeros(0)
        size = torch.zeros(0)
        id = torch.zeros(0)
        in_camera = torch.zeros(0)

        return img, pos, size, id, in_camera

    def __len__(self):
        length = len(self.episodes)
        return length * self.seq_per_episode


if __name__ == '__main__':
    test_epochs = True
    plot = False
    # --- episodic setting --- #
    root = '/data/bair_256_ours'
    ds = BAIR64Video(root=root, ep_len=17, sample_length=12, mode='train', image_size=128)
    dl = DataLoader(ds, shuffle=True, pin_memory=False, batch_size=4, num_workers=0)
    batch = next(iter(dl))
    im = batch[0]
    actions = batch[1]
    print(im.shape)
    print(f'actions: {actions.shape}, action[0]: {actions[0]}')

    if plot:
        import matplotlib.pyplot as plt

        img_np = im[0, 0].permute(1, 2, 0).data.cpu().numpy()
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(111)
        ax.imshow(img_np)
        plt.show()

    if test_epochs:
        from tqdm import tqdm

        pbar = tqdm(iterable=dl)
        for batch in pbar:
            pass
        pbar.close()