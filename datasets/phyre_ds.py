"""
Dataset class for the PHYRE dataset
"""

import os
import os.path as osp
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
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


# --- new preprocessing functions for the episodic setting --- #
class PhyreDataset(Dataset):
    def __init__(self, root, mode, ep_len=100, sample_length=20, image_size=128, start_idx=0, fps=10):
        # path = os.path.join(root, mode)
        # assume 10 frames-per-second
        # the data is generated such that a task is completed if the completion condition is met for 3 seconds or more.
        # that means that we can cut off 3 seconds (=30 frames) from the end of the episode)
        assert mode in ['train', 'val', 'valid', 'test']
        if mode == 'val':
            mode = 'valid'
        self.root = os.path.join(root, mode)
        self.image_size = image_size
        self.start_idx = start_idx
        self.fps = fps
        self.cutoff = 3 * self.fps  # 3 seconds off the end is just idle stuff

        self.mode = mode
        self.sample_length = sample_length

        # Get all numbers
        get_dir_num = lambda x: int(x)

        self.folders = [d for d in os.listdir(self.root) if osp.isdir(osp.join(self.root, d))]
        self.folders.sort(key=get_dir_num)

        self.episodes = []
        self.episodes_len = []
        self.EP_LEN = ep_len
        self.seq_per_episode = self.EP_LEN - self.sample_length + 1

        for f in self.folders:
            dir_name = os.path.join(self.root, str(f))
            paths = list(glob.glob(osp.join(dir_name, '*.png')))
            get_num = lambda x: int(osp.splitext(osp.basename(x))[0])
            paths.sort(key=get_num)
            paths = paths[self.start_idx:-self.cutoff]
            if len(paths) < self.EP_LEN:
                # continue
                self.episodes_len.append(len(paths))
            else:
                self.episodes_len.append(self.EP_LEN)
            while len(paths) < self.EP_LEN:
                paths.append(paths[-1])
            self.episodes.append(paths[:self.EP_LEN])
        # print(f'episodes: {len(self.episodes)}, min: {min(self.episodes_len)}, max: {max(self.episodes_len)}')

    def __getitem__(self, index):

        imgs = []
        if self.mode == 'train':
            # Implement continuous indexing
            ep = index // self.seq_per_episode
            # ep = np.argmax((index < self.episodes_len_cumsum))
            offset = index % self.seq_per_episode
            # offset = index % self.seq_per_episode[ep]
            end = offset + self.sample_length
            # if `end` is after the episode ended, move backwards
            ep_len = self.episodes_len[ep]
            if end > ep_len:
                # print(f'before: offset: {offset}, end: {end}, ep_len: {ep_len}')
                if self.sample_length > ep_len:
                    offset = 0
                    end = offset + self.sample_length
                else:
                    offset = ep_len - self.sample_length
                    end = ep_len
                # print(f'after: offset: {offset}, end: {end}, ep_len: {ep_len}')

            e = self.episodes[ep]
            for image_index in range(offset, end):
                img = Image.open(osp.join(e[image_index]))
                # img.point(lambda x: 215.0 if x >= 253 else x)
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
        # invert colors
        img = 1.0 - img
        pos = torch.zeros(0)
        size = torch.zeros(0)
        id = torch.zeros(0)
        in_camera = torch.zeros(0)

        return img, pos, size, id, in_camera

    def __len__(self):
        length = len(self.episodes)
        if self.mode == 'train':
            return length * self.seq_per_episode
            # return sum(self.episodes_len)
        else:
            return length


class PhyreDatasetImage(Dataset):
    def __init__(self, root, mode, ep_len=100, sample_length=20, image_size=128, start_idx=0, fps=10):
        # path = os.path.join(root, mode)
        # assume 10 frames-per-second
        # the data is generated such that a task is completed if the completion condition is met for 3 seconds or more.
        # that means that we can cut off 3 seconds (=30 frames) from the end of the episode)
        assert mode in ['train', 'val', 'valid', 'test']
        if mode == 'val':
            mode = 'valid'
        self.root = os.path.join(root, mode)
        self.image_size = image_size
        self.start_idx = start_idx
        self.fps = fps
        self.cutoff = 3 * self.fps  # 3 seconds off the end is just idle stuff

        self.mode = mode
        self.sample_length = sample_length

        # Get all numbers
        get_dir_num = lambda x: int(x)

        self.folders = [d for d in os.listdir(self.root) if osp.isdir(osp.join(self.root, d))]
        self.folders.sort(key=get_dir_num)

        self.episodes = []
        self.episodes_len = []
        self.EP_LEN = ep_len
        self.seq_per_episode = self.EP_LEN - self.sample_length + 1

        for f in self.folders:
            dir_name = os.path.join(self.root, str(f))
            paths = list(glob.glob(osp.join(dir_name, '*.png')))
            get_num = lambda x: int(osp.splitext(osp.basename(x))[0])
            paths.sort(key=get_num)
            paths = paths[self.start_idx:-self.cutoff]
            if len(paths) < self.EP_LEN:
                # continue
                self.episodes_len.append(len(paths))
            else:
                self.episodes_len.append(self.EP_LEN)
            while len(paths) < self.EP_LEN:
                paths.append(paths[-1])
            self.episodes.append(paths[:self.EP_LEN])
        # print(f'episodes: {len(self.episodes)}, min: {min(self.episodes_len)}, max: {max(self.episodes_len)}')

    def __getitem__(self, index):

        imgs = []
        # Implement continuous indexing
        ep = index // self.seq_per_episode
        # ep = np.argmax((index < self.episodes_len_cumsum))
        offset = index % self.seq_per_episode
        # offset = index % self.seq_per_episode[ep]
        end = offset + self.sample_length
        # if `end` is after the episode ended, move backwards
        ep_len = self.episodes_len[ep]
        if end > ep_len:
            # print(f'before: offset: {offset}, end: {end}, ep_len: {ep_len}')
            if self.sample_length > ep_len:
                offset = 0
                end = offset + self.sample_length
            else:
                offset = ep_len - self.sample_length
                end = ep_len
            # print(f'after: offset: {offset}, end: {end}, ep_len: {ep_len}')

        e = self.episodes[ep]
        for image_index in range(offset, end):
            img = Image.open(osp.join(e[image_index]))
            # img.point(lambda x: 215.0 if x >= 253 else x)
            img = img.resize((self.image_size, self.image_size))
            img = transforms.ToTensor()(img)[:3]
            imgs.append(img)

        img = torch.stack(imgs, dim=0).float()
        # invert colors
        img = 1.0 - img
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
    # --- episodic setting --- #
    root = '/media/newhd/data/phyre'
    # root = '/mnt/data/tal/phyre'
    phyre_ds = PhyreDataset(root=root, ep_len=100, sample_length=10, mode='train', image_size=128, start_idx=0)
    phyre_dl = DataLoader(phyre_ds, shuffle=True, pin_memory=False, batch_size=32, num_workers=4)
    batch = next(iter(phyre_dl))
    im = batch[0]
    print(im.shape)
    # img_np = im.permute(1, 2, 0).data.cpu().numpy()
    # fig = plt.figure(figsize=(5, 5))
    # ax = fig.add_subplot(111)
    # ax.imshow(img_np)
    # plt.show()

    if test_epochs:
        from tqdm import tqdm

        pbar = tqdm(iterable=phyre_dl)
        for batch in pbar:
            pass
        pbar.close()
