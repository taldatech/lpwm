"""
Dataset class for the LangugaeTable Dataset
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
class LanguageTableDataset(Dataset):
    def __init__(self, root, mode, ep_len=49, sample_length=10, image_size=128, dense=False, clips_per_video=4,
                 force_cache_rebuild=False):
        assert mode in ['train', 'val', 'valid', 'validation', 'test']
        if mode in ["val", "valid", "validation"]:
            mode = 'valid'
        self.mode = mode
        self.dense = dense
        self.clips_per_video = clips_per_video if not dense else 1
        # self.n_max_train = 150_000
        # self.n_max_train = 80_000
        self.n_max_train = -1
        # self.n_max_valid = 1000
        self.n_max_valid = 100
        self.root = os.path.join(root, self.mode)
        self.image_size = image_size
        self.sample_length = sample_length

        # get_dir_num = lambda x: int(x)
        # self.get_num = lambda x: int(osp.splitext(osp.basename(x))[0])

        # Get all numbers
        self.folders = []
        # for file in os.listdir(self.root):
        #     try:
        #         if osp.isdir(osp.join(self.root, file)):
        #             self.folders.append(file)
        #     except ValueError:
        #         continue
        # self.folders.sort()
        #
        # if self.mode == 'train' and self.n_max_train > 0:
        #     self.folders = self.folders[:self.n_max_train]
        # elif self.mode == 'valid' and self.n_max_valid > 0:
        #     self.folders = self.folders[:self.n_max_valid]

        cache_file = os.path.join(self.root, f"cached_index_{self.mode}.pt")

        self.episodes = []
        self.episodes_metadata = []
        self.episodes_instruction = []
        self.episodes_len = []
        self.EP_LEN = max(ep_len, sample_length)
        self.seq_per_episode = self.EP_LEN - self.sample_length + 1  # dense mode

        if os.path.exists(cache_file) and not force_cache_rebuild:
            print(f"[Dataset] Loading dataset index from cache: {cache_file}")
            cache = torch.load(cache_file, weights_only=False)
            self.folders = cache['folders']
            self.episodes = cache['episodes']
            self.episodes_metadata = cache['episodes_metadata']
            self.episodes_instruction = cache['episodes_instruction']
            self.episodes_len = cache['episodes_len']
        else:
            print(f"[Dataset] Building dataset index from scratch (force_rebuild={force_cache_rebuild})...")
            for file in os.listdir(self.root):
                try:
                    if osp.isdir(osp.join(self.root, file)):
                        self.folders.append(file)
                except ValueError:
                    continue
            self.folders.sort()

            if self.mode == 'train' and self.n_max_train > 0:
                self.folders = self.folders[:self.n_max_train]
            elif self.mode == 'valid' and self.n_max_valid > 0:
                self.folders = self.folders[:self.n_max_valid]
            for f in self.folders:
                dir_name = os.path.join(self.root, str(f))
                paths = list(glob.glob(osp.join(dir_name, '*.png')))
                get_num = lambda x: int(osp.basename(x).split('.')[0].split('_')[-1])
                paths.sort(key=get_num)
                self.episodes_len.append(len(paths))
                while len(paths) < self.EP_LEN:
                    paths.append(paths[-1])
                self.episodes.append(paths)
                metadata_paths = list(glob.glob(osp.join(dir_name, 'metadata_*.pt')))
                metadata_paths.sort(key=get_num)
                while len(metadata_paths) < self.EP_LEN:
                    metadata_paths.append(metadata_paths[-1])
                self.episodes_metadata.append(metadata_paths)
                self.episodes_instruction.append(osp.join(dir_name, 'instruction.pt'))

            print(f"[Dataset] Saving dataset index to cache: {cache_file}")
            torch.save({
                'folders': self.folders,
                'episodes': self.episodes,
                'episodes_metadata': self.episodes_metadata,
                'episodes_instruction': self.episodes_instruction,
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
                e_inst = self.episodes_instruction[ep]

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
                    else:
                        img = Image.open(ep_path[image_index])
                        img = self.transform(img)[:3]
                        imgs.append(img)

                        meta_i = torch.load(e_act[image_index], weights_only=False)
                        # actions
                        act = meta_i['action']
                        actions.append(act)

                    # dones
                    done_i = torch.tensor((image_index < ep_len),
                                          dtype=torch.int)  # episode_mask: 1 if valid else 0, after end of episode
                    # done_i = meta_i['is_last']
                    dones.append(done_i)

                # instructions
                inst = torch.load(e_inst, weights_only=False)
                instruction = inst['raw_instruction']
                # instructions embeddings
                instruction_embedding = inst['instruction_embedding'].detach()

            else:
                # SPARSE MODE
                ep = index // self.clips_per_video
                clip_num = index % self.clips_per_video
                ep_path = self.episodes[ep]
                e_act = self.episodes_metadata[ep]
                e_inst = self.episodes_instruction[ep]
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
                    else:
                        img = Image.open(ep_path[image_index])
                        img = self.transform(img)[:3]
                        imgs.append(img)

                        try:
                            meta_i = torch.load(e_act[image_index], weights_only=False)
                        except IndexError:
                            print(f'image index: {image_index}, len(e_act): {len(e_act)}, ep len: {ep_len}, ep: {ep}')
                            print(ep_path)
                            print(e_act)
                            raise SystemExit
                        # actions
                        act = meta_i['action']
                        actions.append(act)

                    # dones
                    done_i = torch.tensor((image_index < ep_len),
                                          dtype=torch.int)  # episode_mask: 1 if valid else 0, after end of episode
                    # done_i = meta_i['is_last']
                    dones.append(done_i)

                # instructions
                inst = torch.load(e_inst, weights_only=False)
                instruction = inst['raw_instruction']
                # instructions embeddings
                instruction_embedding = inst['instruction_embedding'].detach()

        else:
            # EVAL MODE — always return full video (padded if needed)
            paths = self.episodes[index]
            action_paths = self.episodes_metadata[index]
            e_inst = self.episodes_instruction[index]
            ep_len = self.episodes_len[index]
            for pi, path in enumerate(paths):
                img = Image.open(path)
                img = self.transform(img)[:3]
                imgs.append(img)

                meta_i = torch.load(action_paths[pi], weights_only=False)
                # actions
                act = meta_i['action']
                actions.append(act)

                # dones
                done_i = torch.tensor((pi < ep_len),
                                      dtype=torch.int)  # episode_mask: 1 if valid else 0, after end of episode
                # done_i = meta_i['is_last']
                dones.append(done_i)

                if pi == self.sample_length - 1:
                    break

            while len(imgs) < self.sample_length:
                imgs.append(imgs[-1])
                actions.append(torch.zeros_like(actions[-1]))
                dones.append(torch.tensor(0, dtype=torch.int))
            # instructions
            inst = torch.load(e_inst, weights_only=False)
            instruction = inst['raw_instruction']
            # instructions embeddings
            instruction_embedding = inst['instruction_embedding'].detach()

        img = torch.stack(imgs, dim=0).float()
        actions = torch.stack(actions, dim=0).float()
        # dones = torch.stack(dones, dim=0).bool()
        dones = torch.stack(dones, dim=0).int()

        # Placeholder meta fields
        # pos = torch.zeros(0)
        # size = torch.zeros(0)
        # id = torch.zeros(0)
        # in_camera = torch.zeros(0)

        return img, actions, instruction, instruction_embedding, dones

    def __len__(self):
        length = len(self.folders)
        if self.mode == 'train':
            if self.dense:
                return length * self.seq_per_episode
            else:
                return length * self.clips_per_video
        else:
            return length


class LanguageTableDatasetImage(Dataset):
    def __init__(self, root, mode, ep_len=24, sample_length=1, image_size=128, force_cache_rebuild=False):
        assert mode in ['train', 'val', 'valid', 'validation', 'test']
        if mode in ["val", "valid"]:
            mode = 'validation'
        self.mode = mode
        self.n_max_train = 150_000
        self.n_max_valid = 1000
        self.root = os.path.join(root, self.mode)
        self.image_size = image_size
        self.sample_length = sample_length

        # get_dir_num = lambda x: int(x)
        # self.get_num = lambda x: int(osp.splitext(osp.basename(x))[0])

        # Get all numbers
        self.folders = []
        for file in os.listdir(self.root):
            try:
                if osp.isdir(osp.join(self.root, file)):
                    self.folders.append(file)
            except ValueError:
                continue
        self.folders.sort()

        if self.mode == 'train' and self.n_max_train > 0:
            self.folders = self.folders[:self.n_max_train]
        elif self.mode == 'valid' and self.n_max_valid > 0:
            self.folders = self.folders[:self.n_max_valid]

        cache_file = os.path.join(self.root, f"cached_index_{self.mode}.pt")

        self.episodes = []
        self.episodes_metadata = []
        self.episodes_instruction = []
        self.episodes_len = []
        self.EP_LEN = ep_len
        self.seq_per_episode = self.EP_LEN - self.sample_length + 1

        if os.path.exists(cache_file) and not force_cache_rebuild:
            print(f"[Dataset] Loading dataset index from cache: {cache_file}")
            cache = torch.load(cache_file, weights_only=False)
            self.episodes = cache['episodes']
            self.episodes_metadata = cache['episodes_metadata']
            self.episodes_instruction = cache['episodes_instruction']
            self.episodes_len = cache['episodes_len']
        else:
            print(f"[Dataset] Building dataset index from scratch (force_rebuild={force_cache_rebuild})...")
            for f in self.folders:
                dir_name = os.path.join(self.root, str(f))
                paths = list(glob.glob(osp.join(dir_name, '*.png')))
                get_num = lambda x: int(osp.basename(x).split('.')[0].split('_')[-1])
                paths.sort(key=get_num)
                self.episodes_len.append(len(paths))
                while len(paths) < self.EP_LEN:
                    paths.append(paths[-1])
                self.episodes.append(paths)
                metadata_paths = list(glob.glob(osp.join(dir_name, 'metadata_*.pt')))
                metadata_paths.sort(key=get_num)
                while len(metadata_paths) < self.EP_LEN:
                    metadata_paths.append(metadata_paths[-1])
                self.episodes_metadata.append(metadata_paths)
                self.episodes_instruction.append(osp.join(dir_name, 'instruction.pt'))

            print(f"[Dataset] Saving dataset index to cache: {cache_file}")
            torch.save({
                'episodes': self.episodes,
                'episodes_metadata': self.episodes_metadata,
                'episodes_instruction': self.episodes_instruction,
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
        # DENSE MODE
        ep = index // self.seq_per_episode
        offset = index % self.seq_per_episode
        end = offset + self.sample_length

        ep_len = self.episodes_len[ep]
        ep_path = self.episodes[ep]
        e_act = self.episodes_metadata[ep]
        e_inst = self.episodes_instruction[ep]

        if end > ep_len:
            if self.sample_length > ep_len:
                # offset = 0  # original
                # max_offset = max(1, ep_len - self.sample_length)
                max_offset = max(1, ep_len - self.sample_length // 2)
                offset = random.randint(0, max_offset)
                end = offset + self.sample_length
            else:
                offset = ep_len - self.sample_length
                end = ep_len

        for image_index in range(offset, end):
            if image_index >= ep_len:
                imgs.append(imgs[-1])
                actions.append(torch.zeros_like(actions[-1]))
            else:
                img = Image.open(ep_path[image_index])
                img = self.transform(img)[:3]
                imgs.append(img)

                meta_i = torch.load(e_act[image_index], weights_only=False)
                # actions
                act = meta_i['action']
                actions.append(act)

            done_i = torch.tensor((image_index < ep_len),
                                  dtype=torch.int)  # episode_mask: 1 if valid else 0, after end of episode
            # done_i = meta_i['is_last']
            dones.append(done_i)

        # instructions
        inst = torch.load(e_inst, weights_only=False)
        instruction = inst['raw_instruction']
        # instructions embeddings
        instruction_embedding = inst['instruction_embedding'].detach()

        img = torch.stack(imgs, dim=0).float()
        actions = torch.stack(actions, dim=0).float()
        # dones = torch.stack(dones, dim=0).bool()
        dones = torch.stack(dones, dim=0).int()

        # Placeholder meta fields
        # pos = torch.zeros(0)
        # size = torch.zeros(0)
        # id = torch.zeros(0)
        # in_camera = torch.zeros(0)

        return img, actions, instruction, instruction_embedding, dones

    def __len__(self):
        length = len(self.folders)
        return length * self.seq_per_episode


if __name__ == '__main__':
    test_epochs = True
    plot = False
    # --- episodic setting --- #
    root = '/data/lang_table'
    ds = LanguageTableDataset(root=root, ep_len=50, sample_length=12, mode='train', image_size=128, dense=True)
    dl = DataLoader(ds, shuffle=True, pin_memory=False, batch_size=4, num_workers=0)
    batch = next(iter(dl))
    im = batch[0]
    actions = batch[1]
    instruction = batch[2]
    instruction_embedding = batch[3]
    print(im.shape)
    print(f'actions: {actions.shape}, action[0]: {actions[0]}')
    print(f'instructions: {len(instruction)}, instruction[0]: {instruction[0]}')
    print(f'instructions embed: {instruction_embedding.shape}')

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
