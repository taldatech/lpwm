"""
Dataset class for the OG-Bench Dataset
"""

import os
import os.path as osp
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import glob
from PIL import Image, ImageFile
import json
import random
import numpy as np

ImageFile.LOAD_TRUNCATED_IMAGES = True

def get_task_id_from_path(p):
    tasks = ['cube', 'scene']
    for t in tasks:
        if t in p:
            return t

def list_images_in_dir(path):
    valid_images = [".jpg", ".gif", ".png"]
    img_list = []
    for f in os.listdir(path):
        ext = os.path.splitext(f)[1]
        if ext.lower() not in valid_images:
            continue
        img_list.append(os.path.join(path, f))
    return img_list

class OGBenchDataset(Dataset):
    def __init__(self, root, mode, ep_len=1001, sample_length=10, image_size=64, dense=False, clips_per_video=50,
                 tasks=("cube", "scene"), force_cache_rebuild=False, use_cache=True, max_goal_dist=70):
        # ep_len is max episode length
        assert mode in ['train', 'val', 'valid', 'validation', 'test']
        if mode in ["val", "valid", "validation", "test"]:
            mode = 'valid'
        self.mode = mode
        self.dense = dense
        self.clips_per_video = clips_per_video if not dense else 1
        self.root = root
        self.image_size = image_size
        self.sample_length = sample_length
        self.tasks = tasks
        self.task_folders = []
        self.max_goal_dist = max_goal_dist
        if "cube" in self.tasks:
            self.task_folders.append(osp.join(root, "cube", self.mode))

        if "scene" in self.tasks:
            self.task_folders.append(osp.join(root, "scene", self.mode))
        # Get all numbers
        self.folders = []
        for d in self.task_folders:
            episodes = os.listdir(d)
            episodes.sort(key=lambda x: int(x[-4:]))
            self.folders.extend(episodes)

        cache_file = os.path.join(self.root, f"cached_index_{self.mode}.pt")

        self.episodes = []
        self.episodes_metadata = []
        # self.episodes_instruction = []
        self.episodes_len = []

        self.EP_LEN = max(ep_len, sample_length)
        self.seq_per_episode = self.EP_LEN - self.sample_length + 1  # dense mode

        if use_cache and os.path.exists(cache_file) and not force_cache_rebuild:
            print(f"[Dataset] Loading dataset index from cache: {cache_file}")
            cache = torch.load(cache_file)
            self.episodes = cache['episodes']
            self.episodes_metadata = cache['episodes_metadata']
            # self.episodes_instruction = cache['episodes_instruction']
            self.episodes_len = cache['episodes_len']
        else:
            if use_cache:
                print(f"[Dataset] Building dataset index from scratch (force_rebuild={force_cache_rebuild})...")
            for d in self.task_folders:
                episodes = os.listdir(d)
                episodes.sort(key=lambda x: int(x[-4:]))
                # task_id = get_task_id_from_path(d)
                # get files
                for f in episodes:
                    dir_name = os.path.join(d, str(f))
                    paths = list(glob.glob(osp.join(dir_name, '*.png')))
                    paths = [p for p in paths if 'goal' not in p]
                    get_num = lambda x: int(osp.basename(x).split('.')[0][-4:])
                    paths.sort(key=get_num)
                    paths = paths[:self.EP_LEN]
                    self.episodes_len.append(len(paths))
                    while len(paths) < self.EP_LEN:
                        paths.append(paths[-1])
                    self.episodes.append(paths)

                    metadata_paths = list(glob.glob(osp.join(dir_name, 'metadata*.pt')))
                    metadata_paths.sort(key=get_num)
                    metadata_paths = metadata_paths[:self.EP_LEN]
                    while len(metadata_paths) < self.EP_LEN:
                        metadata_paths.append(metadata_paths[-1])
                    self.episodes_metadata.append(metadata_paths)

            if use_cache:
                print(f"[Dataset] Saving dataset index to cache: {cache_file}")
                torch.save({
                    'episodes': self.episodes,
                    'episodes_metadata': self.episodes_metadata,
                    # 'episodes_instruction': self.episodes_instruction,
                    'episodes_len': self.episodes_len,
                }, cache_file)

        self.transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.CenterCrop(self.image_size),
            transforms.ToTensor()
        ])

    def __getitem__(self, index):
        imgs = []
        actions = []
        dones = []

        if self.mode == 'train':
            if self.dense:
                # DENSE MODE
                ep = index // self.seq_per_episode
                offset = index % self.seq_per_episode
                end = offset + self.sample_length

                ep_len = self.episodes_len[ep]
                ep_path = self.episodes[ep]
                e_act = self.episodes_metadata[ep]
                # e_inst = self.episodes_instruction[ep]

                if end > ep_len:
                    if self.sample_length > ep_len:
                        # offset = 0  # original
                        # max_offset = max(1, ep_len - self.sample_length)
                        max_offset = max(1, ep_len - self.sample_length // 2)
                        # To make clip selection more deterministic per clip_num, optionally:
                        #   seed = (ep * self.clips_per_video + clip_num)
                        #   rng = random.Random(seed)
                        #   offset = rng.randint(0, max_offset)
                        offset = random.randint(0, max_offset)
                        end = offset + self.sample_length
                    else:
                        offset = ep_len - self.sample_length  # original
                        end = ep_len

                for image_index in range(offset, end):
                    if image_index >= ep_len:
                        imgs.append(imgs[-1])
                        actions.append(torch.zeros_like(actions[-1]))
                        dones.append(torch.tensor(0, dtype=torch.int))
                    else:
                        img = Image.open(ep_path[image_index])
                        img = self.transform(img)[:3]
                        imgs.append(img)

                        meta_i = torch.load(e_act[image_index])
                        # actions
                        act = meta_i['actions']
                        actions.append(act)

                        # dones
                        done_ie = (image_index < ep_len) and (not meta_i['terminals'])
                        # done_i = torch.tensor((image_index < ep_len),
                        #                       dtype=torch.int)  # episode_mask: 1 if valid else 0, after end of episode
                        # done_i = torch.tensor((not meta_i['is_terminal']), dtype=torch.int)
                        done_i = torch.tensor(done_ie, dtype=torch.int)
                        dones.append(done_i)

                if end < ep_len:
                    goal_idx = random.randint(end, min(end + self.max_goal_dist, ep_len - 1))
                else:
                    goal_idx = ep_len - 1
                try:
                    goal_img = Image.open(ep_path[goal_idx])
                except IndexError:
                    print(f'ep_len: {ep_len}, goal_idx: {goal_idx}, end: {end}')
                    raise SystemExit
                goal_img = self.transform(goal_img)[:3]
                # # instructions
                # inst = torch.load(e_inst)
                # instruction = inst['raw_instruction']
                # # instructions embeddings
                # instruction_embedding = inst['instruction_embedding'].detach()

            else:
                # SPARSE MODE
                ep = index // self.clips_per_video
                clip_num = index % self.clips_per_video
                ep_path = self.episodes[ep]
                e_act = self.episodes_metadata[ep]
                # e_inst = self.episodes_instruction[ep]
                ep_len = self.episodes_len[ep]

                # ensure coverage by considering clip_num
                # ep_end = max(1, ep_len - self.sample_length // 2)
                ep_end = ep_len if (self.sample_length > ep_len) else (ep_len - self.sample_length // 2)
                segment_size = ep_end // self.clips_per_video  # divide the clip to segments
                min_offset = clip_num * segment_size  # minimum index to start from
                max_offset = min((clip_num + 1) * segment_size, ep_len - 1)
                offset = random.randint(min_offset, max_offset)

                # max_offset = max(1, ep_len - self.sample_length)  # original
                # max_offset = max(1, ep_len - self.sample_length // 2)
                # To make clip selection more deterministic per clip_num, optionally:
                #   seed = (ep * self.clips_per_video + clip_num)
                #   rng = random.Random(seed)
                #   offset = rng.randint(0, max_offset)
                # offset = random.randint(0, max_offset)

                end = offset + self.sample_length

                for image_index in range(offset, end):
                    if image_index >= ep_len:
                        imgs.append(imgs[-1])
                        actions.append(torch.zeros_like(actions[-1]))
                        dones.append(torch.tensor(0, dtype=torch.int))
                    else:
                        img = Image.open(ep_path[image_index])
                        img = self.transform(img)[:3]
                        imgs.append(img)

                        try:
                            meta_i = torch.load(e_act[image_index])
                        except IndexError:
                            print(f'image index: {image_index}, len(e_act): {len(e_act)}, ep len: {ep_len}, ep: {ep}')
                            print(ep_path)
                            print(e_act)
                            raise SystemExit
                        # actions
                        act = meta_i['actions']
                        actions.append(act)

                        # dones
                        done_ie = (image_index < ep_len) and (not meta_i['terminals'])
                        # done_i = torch.tensor((image_index < ep_len),
                        #                       dtype=torch.int)  # episode_mask: 1 if valid else 0, after end of episode
                        # done_i = torch.tensor((not meta_i['is_terminal']), dtype=torch.int)
                        done_i = torch.tensor(done_ie, dtype=torch.int)
                        dones.append(done_i)

                if end < ep_len:
                    goal_idx = random.randint(end, min(end + self.max_goal_dist, ep_len - 1))
                else:
                    goal_idx = ep_len - 1
                goal_img = Image.open(ep_path[goal_idx])
                goal_img = self.transform(goal_img)[:3]

                # instructions
                # inst = torch.load(e_inst)
                # instruction = inst['raw_instruction']
                # # instructions embeddings
                # instruction_embedding = inst['instruction_embedding'].detach()

        else:
            # EVAL MODE — always return full video (padded if needed)
            paths = self.episodes[index]
            action_paths = self.episodes_metadata[index]
            # e_inst = self.episodes_instruction[index]
            ep_len = self.episodes_len[index]
            # cut to maximal ep_len
            paths = paths[:self.EP_LEN]
            paths = paths[:self.sample_length]

            for pi, path in enumerate(paths):
                img = Image.open(path)
                img = self.transform(img)[:3]
                imgs.append(img)

                meta_i = torch.load(action_paths[pi])
                # actions
                act = meta_i['actions']
                actions.append(act)

                # dones
                done_ie = (pi < ep_len) and (not meta_i['terminals'])
                # done_i = torch.tensor((pi < ep_len),
                #                       dtype=torch.int)  # episode_mask: 1 if valid else 0, after end of episode
                # done_i = torch.tensor((not meta_i['is_terminal']), dtype=torch.int)
                done_i = torch.tensor(done_ie, dtype=torch.int)
                dones.append(done_i)

            while len(imgs) < self.sample_length:
                imgs.append(imgs[-1])
                actions.append(torch.zeros_like(actions[-1]))
                dones.append(torch.tensor(0, dtype=torch.int))

            goal_img = Image.open(paths[-1])
            goal_img = self.transform(goal_img)[:3]

            # instructions
            # inst = torch.load(e_inst)
            # instruction = inst['raw_instruction']
            # # instructions embeddings
            # instruction_embedding = inst['instruction_embedding'].detach()

        img = torch.stack(imgs, dim=0).float()
        actions = torch.stack(actions, dim=0).float()
        # dones = torch.stack(dones, dim=0).bool()
        dones = torch.stack(dones, dim=0).int()

        # Placeholder meta fields
        # pos = torch.zeros(0)
        size = torch.zeros(0)
        # id = torch.zeros(0)
        # in_camera = torch.zeros(0)

        return img, actions, size, goal_img, dones

    def __len__(self):
        length = len(self.folders)
        if self.mode == 'train':
            if self.dense:
                return length * self.seq_per_episode
            else:
                return length * self.clips_per_video
        else:
            return length


class OGBenchDatasetOLD(Dataset):
    def __init__(self, root, mode, ep_len=1001, sample_length=20, image_size=64, dense=True, clips_per_video=1):
        assert mode in ['train', 'val', 'valid', 'validation', 'test']
        if mode in ["val", "valid", "test"]:
            mode = 'valid'
        self.mode = mode
        self.dense = dense
        self.clips_per_video = clips_per_video if not dense else 1

        self.n_max_train = 50_000
        self.n_max_valid = 1000
        self.root = os.path.join(root, self.mode)
        self.image_size = image_size
        self.sample_length = sample_length

        get_dir_num = lambda x: int(x)
        self.get_num = lambda x: int(osp.splitext(osp.basename(x))[0])

        # Get all numbers
        self.folders = []
        for file in os.listdir(self.root):
            try:
                self.folders.append(int(file))
            except ValueError:
                continue
        self.folders.sort()

        if self.mode == 'train' and self.n_max_train > 0:
            self.folders = self.folders[:self.n_max_train]
        elif self.mode == 'valid' and self.n_max_valid > 0:
            self.folders = self.folders[:self.n_max_valid]

        self.episodes = []
        self.episodes_len = []
        self.ep_actions = []
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
            while len(paths) < self.EP_LEN:
                paths.append(paths[-1])
            self.episodes.append(paths)
            self.episodes_len.append(len(paths))
            self.ep_actions.append(os.path.join(dir_name, 'actions.npz'))

        self.transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.CenterCrop(self.image_size),
            transforms.ToTensor()
        ])

    def __getitem__(self, index):
        imgs = []

        if self.mode == 'train':
            if self.dense:
                # DENSE MODE
                ep = index // self.seq_per_episode
                offset = index % self.seq_per_episode
                end = offset + self.sample_length

                ep_len = self.episodes_len[ep]
                ep_path = self.episodes[ep]
                with np.load(self.ep_actions[ep], allow_pickle=True) as f:
                    actions = f['actions']

                if end > ep_len:
                    if self.sample_length > ep_len:
                        offset = 0
                        end = offset + self.sample_length
                    else:
                        offset = ep_len - self.sample_length
                        end = ep_len

                for image_index in range(offset, end):
                    img = Image.open(ep_path[image_index])
                    img = self.transform(img)[:3]
                    imgs.append(img)

                actions = actions[offset:end - 1]

            else:
                # SPARSE MODE
                ep = index // self.clips_per_video
                clip_num = index % self.clips_per_video
                ep_path = self.episodes[ep]
                ep_len = self.episodes_len[ep]
                with np.load(self.ep_actions[ep], allow_pickle=True) as f:
                    actions = f['actions']

                max_offset = max(1, ep_len - self.sample_length)
                # To make clip selection more deterministic per clip_num, optionally:
                #   seed = (ep * self.clips_per_video + clip_num)
                #   rng = random.Random(seed)
                #   offset = rng.randint(0, max_offset)
                offset = random.randint(0, max_offset)
                end = offset + self.sample_length

                for image_index in range(offset, end):
                    img = Image.open(ep_path[image_index])
                    img = self.transform(img)[:3]
                    imgs.append(img)

                actions = actions[offset:end - 1]

        else:
            # EVAL MODE — always return full video (padded if needed)
            paths = self.episodes[index]
            with np.load(self.ep_actions[index], allow_pickle=True) as f:
                actions = f['actions']
            for path in paths:
                img = Image.open(path)
                img = self.transform(img)[:3]
                imgs.append(img)

        img = torch.stack(imgs, dim=0).float()

        # Placeholder meta fields
        # pos = torch.zeros(0)
        size = torch.zeros(0)
        actions = torch.tensor(actions)
        id = torch.zeros(0)
        in_camera = torch.zeros(0)

        return img, actions, size, id, in_camera

    def __len__(self):
        length = len(self.folders)
        if self.mode == 'train':
            if self.dense:
                return length * self.seq_per_episode
            else:
                return length * self.clips_per_video
        else:
            return length


class OGBenchDatasetImage(Dataset):
    def __init__(self, root, mode, ep_len=1001, sample_length=1, image_size=128):
        assert mode in ['train', 'val', 'valid', 'validation', 'test']
        if mode in ["val", "valid", "test"]:
            mode = 'valid'
        self.mode = mode

        self.n_max_train = 50_000
        self.n_max_valid = 1000
        self.root = os.path.join(root, self.mode)
        self.image_size = image_size
        self.sample_length = sample_length

        get_dir_num = lambda x: int(x)
        self.get_num = lambda x: int(osp.splitext(osp.basename(x))[0])

        # Get all numbers
        self.folders = []
        for file in os.listdir(self.root):
            try:
                self.folders.append(int(file))
            except ValueError:
                continue
        self.folders.sort()

        if self.mode == 'train' and self.n_max_train > 0:
            self.folders = self.folders[:self.n_max_train]
        elif self.mode == 'valid' and self.n_max_valid > 0:
            self.folders = self.folders[:self.n_max_valid]

        self.episodes = []
        self.episodes_len = []
        self.ep_actions = []
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
            while len(paths) < self.EP_LEN:
                paths.append(paths[-1])
            self.episodes.append(paths)
            self.episodes_len.append(len(paths))
            self.ep_actions.append(os.path.join(dir_name, 'actions.npz'))

        self.transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.CenterCrop(self.image_size),
            transforms.ToTensor()
        ])

    def __getitem__(self, index):
        imgs = []
        # DENSE MODE
        ep = index // self.seq_per_episode
        offset = index % self.seq_per_episode
        end = offset + self.sample_length

        ep_len = self.episodes_len[ep]
        ep_path = self.episodes[ep]
        with np.load(self.ep_actions[ep], allow_pickle=True) as f:
            actions = f['actions']

        if end > ep_len:
            if self.sample_length > ep_len:
                offset = 0
                end = offset + self.sample_length
            else:
                offset = ep_len - self.sample_length
                end = ep_len

        for image_index in range(offset, end):
            img = Image.open(ep_path[image_index])
            img = self.transform(img)[:3]
            imgs.append(img)

        actions = actions[offset:end - 1]

        img = torch.stack(imgs, dim=0).float()
        actions = torch.tesnor(actions)
        size = torch.zeros(0)
        id = torch.zeros(0)
        in_camera = torch.zeros(0)

        return img, actions, size, id, in_camera

    def __len__(self):
        length = len(self.folders)
        return length * self.seq_per_episode


if __name__ == '__main__':
    test_epochs = True
    plot = False
    # --- episodic setting --- #
    root = './ogbench_ds'
    ds = OGBenchDataset(root=root, ep_len=1001, sample_length=10, mode='train', image_size=64, dense=True)
    dl = DataLoader(ds, shuffle=True, pin_memory=False, batch_size=4, num_workers=0)
    batch = next(iter(dl))
    im = batch[0]
    actions = batch[1]
    im_goal = batch[3]
    dones = batch[-1]
    print(im.shape)
    print(f'actions: {actions.shape}, action[0]: {actions[0]}')
    print(f'goal image: {im_goal.shape}')
    print(f'dones: {dones}')

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
