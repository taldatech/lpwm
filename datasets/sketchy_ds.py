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
class SketchyVideoDataset(Dataset):
    def __init__(self, root, mode, ep_len=70, sample_length=20, image_size=128):
        # path = os.path.join(root, mode)
        assert mode in ['train', 'val', 'valid', 'test']
        if mode == 'val':
            mode = 'valid'
        self.root = os.path.join(root, mode)
        self.image_size = image_size

        self.mode = mode
        self.sample_length = sample_length

        # Get all numbers
        # data is separated to demos and rollouts
        get_dir_num = lambda x: int(x[2:])

        demos_dir = osp.join(self.root, 'demos')
        demos_folders = [d for d in os.listdir(demos_dir) if osp.isdir(osp.join(demos_dir, d))]
        demos_folders.sort(key=get_dir_num)
        demos_folders = [osp.join('demos', d) for d in demos_folders]

        rollouts_dir = osp.join(self.root, 'rollouts')
        rollouts_folders = [d for d in os.listdir(rollouts_dir) if osp.isdir(osp.join(rollouts_dir, d))]
        rollouts_folders.sort(key=get_dir_num)
        rollouts_folders = [osp.join('rollouts', d) for d in rollouts_folders]

        self.folders = demos_folders + rollouts_folders
        # print(f'folders: {len(self.folders)}')

        self.episodes = []
        self.episodes_actions = []
        self.EP_LEN = ep_len
        self.seq_per_episode = self.EP_LEN - self.sample_length + 1

        for f in self.folders:
            dir_name = os.path.join(self.root, f)
            paths = list(glob.glob(osp.join(dir_name, '*.png')))
            if len(paths) < self.EP_LEN:
                continue
            # assert len(paths) == self.EP_LEN, 'len(paths): {}'.format(len(paths))
            get_num = lambda x: int(osp.splitext(osp.basename(x))[0].split('_')[1][1:])
            paths.sort(key=get_num)
            self.episodes.append(paths[:self.EP_LEN])
            actions_paths = list(glob.glob(osp.join(dir_name, '*.pt')))
            actions_paths.sort(key=get_num)
            self.episodes_actions.append(actions_paths[:self.EP_LEN])
        # print(f'episodes: {len(self.episodes)}')

    def __getitem__(self, index):
        imgs = []
        actions = []
        # Implement continuous indexing
        ep = index // self.seq_per_episode
        offset = index % self.seq_per_episode
        end = offset + self.sample_length

        e = self.episodes[ep]
        e_act = self.episodes_actions[ep]
        for image_index in range(offset, end):
            img = Image.open(osp.join(e[image_index]))
            img = img.resize((self.image_size, self.image_size))
            img = transforms.ToTensor()(img)[:3]
            imgs.append(img)

            # actions
            act = torch.load(e_act[image_index])['actions']
            actions.append(act)

        img = torch.stack(imgs, dim=0).float()
        actions = torch.stack(actions, dim=0).float()
        pos = torch.zeros(0)
        size = torch.zeros(0)
        id = torch.zeros(0)
        in_camera = torch.zeros(0)

        return img, actions, size, id, in_camera

    def __len__(self):
        length = len(self.episodes)
        return length * self.seq_per_episode


class SketchyImageDataset(Dataset):
    def __init__(self, root, mode, ep_len=70, sample_length=1, image_size=128):
        # path = os.path.join(root, mode)
        assert mode in ['train', 'val', 'valid', 'test']
        if mode == 'val':
            mode = 'valid'
        self.root = os.path.join(root, mode)
        self.image_size = image_size

        self.mode = mode
        self.sample_length = sample_length

        # Get all numbers
        # data is separated to demos and rollouts
        get_dir_num = lambda x: int(x[2:])

        demos_dir = osp.join(self.root, 'demos')
        demos_folders = [d for d in os.listdir(demos_dir) if osp.isdir(osp.join(demos_dir, d))]
        demos_folders.sort(key=get_dir_num)
        demos_folders = [osp.join('demos', d) for d in demos_folders]

        rollouts_dir = osp.join(self.root, 'rollouts')
        rollouts_folders = [d for d in os.listdir(rollouts_dir) if osp.isdir(osp.join(rollouts_dir, d))]
        rollouts_folders.sort(key=get_dir_num)
        rollouts_folders = [osp.join('rollouts', d) for d in rollouts_folders]

        self.folders = demos_folders + rollouts_folders
        # print(f'folders: {len(self.folders)}')

        self.episodes = []
        self.episodes_actions = []
        self.EP_LEN = ep_len
        self.seq_per_episode = self.EP_LEN - self.sample_length + 1

        for f in self.folders:
            dir_name = os.path.join(self.root, f)
            paths = list(glob.glob(osp.join(dir_name, '*.png')))
            if len(paths) < self.EP_LEN:
                continue
            # assert len(paths) == self.EP_LEN, 'len(paths): {}'.format(len(paths))
            get_num = lambda x: int(osp.splitext(osp.basename(x))[0].split('_')[1][1:])
            paths.sort(key=get_num)
            self.episodes.append(paths[:self.EP_LEN])
            actions_paths = list(glob.glob(osp.join(dir_name, '*.pt')))
            actions_paths.sort(key=get_num)
            self.episodes_actions.append(actions_paths[:self.EP_LEN])
        # print(f'episodes: {len(self.episodes)}')

    def __getitem__(self, index):

        imgs = []
        actions = []
        # Implement continuous indexing
        ep = index // self.seq_per_episode
        offset = index % self.seq_per_episode
        end = offset + self.sample_length

        e = self.episodes[ep]
        e_act = self.episodes_actions[ep]
        for image_index in range(offset, end):
            img = Image.open(osp.join(e[image_index]))
            img = img.resize((self.image_size, self.image_size))
            img = transforms.ToTensor()(img)[:3]
            imgs.append(img)

            # actions
            act = torch.load(e_act[image_index])['actions']
            actions.append(act)

        img = torch.stack(imgs, dim=0).float()
        actions = torch.stack(actions, dim=0).float()
        pos = torch.zeros(0)
        size = torch.zeros(0)
        id = torch.zeros(0)
        in_camera = torch.zeros(0)

        return img, actions, size, id, in_camera

    def __len__(self):
        length = len(self.episodes)
        return length * self.seq_per_episode


if __name__ == '__main__':
    test_epochs = True
    # --- episodic setting --- #
    root = '/media/newhd/data/sketchy/data'
    sketchy_ds = SketchyVideoDataset(root=root, ep_len=70, sample_length=10, mode='train', image_size=128)
    sketchy_dl = DataLoader(sketchy_ds, shuffle=True, pin_memory=False, batch_size=32, num_workers=0)
    batch = next(iter(sketchy_dl))
    im = batch[0][0][0]
    print(im.shape)
    # img_np = im.permute(1, 2, 0).data.cpu().numpy()
    # fig = plt.figure(figsize=(5, 5))
    # ax = fig.add_subplot(111)
    # ax.imshow(img_np)
    # plt.show()

    if test_epochs:
        from tqdm import tqdm

        pbar = tqdm(iterable=sketchy_dl)
        for batch in pbar:
            pass
        pbar.close()
