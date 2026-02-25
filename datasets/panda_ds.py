from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import random
import glob
import os
import os.path as osp
import torch
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

def get_task_id_from_path(p):
    tasks = ['1C', '2C', '3C', '1T', '2T', '3T']
    for t in tasks:
        if t in p:
            return t


class PandaPushVideo(Dataset):
    def __init__(self, root, mode, ep_len=50, sample_length=10, image_size=128, dense=True, clips_per_video=4,
                 tasks=("C", "T"), force_cache_rebuild=False, use_cache=True, multiview=True, random_goal=True):
        # ep_len is max episode length
        assert mode in ['train', 'val', 'valid', 'validation', 'test']
        if mode in ["val", "valid", "validation"]:
            mode = 'valid'
        self.mode = mode
        self.dense = dense
        self.clips_per_video = clips_per_video if not dense else 1
        self.multiview = multiview
        self.random_goal = random_goal
        # self.n_max_train = 80_000
        # self.n_max_valid = 1000
        self.valid_ratio = 0.01
        # self.root = os.path.join(root, self.mode)
        self.root = root
        self.image_size = image_size
        self.sample_length = sample_length
        self.tasks = tasks
        self.task_folders = []
        self.task_folders_view2 = []
        if "C" in self.tasks:
            if self.multiview:
                self.task_folders.extend([osp.join(root, "1C", "view1"),
                                          osp.join(root, "2C", "view1"),
                                          osp.join(root, "3C", "view1"),
                                          osp.join(root, "3C_rand", "view1")])

                self.task_folders_view2.extend([osp.join(root, "1C", "view2"),
                                                osp.join(root, "2C", "view2"),
                                                osp.join(root, "3C", "view2"),
                                                osp.join(root, "3C_rand", "view2")])
            else:
                self.task_folders.extend([osp.join(root, "1C", "view1"),
                                          osp.join(root, "1C", "view2"),
                                          osp.join(root, "2C", "view1"),
                                          osp.join(root, "2C", "view2"),
                                          osp.join(root, "3C", "view1"),
                                          osp.join(root, "3C", "view2"),
                                          osp.join(root, "3C_rand", "view1"),
                                          osp.join(root, "3C_rand", "view2")])
        if "T" in self.tasks:
            if self.multiview:
                self.task_folders.extend([osp.join(root, "1T", "view1"),
                                          osp.join(root, "2T", "view1"),
                                          osp.join(root, "3T", "view1")])

                self.task_folders_view2.extend([osp.join(root, "1T", "view2"),
                                                osp.join(root, "2T", "view2"),
                                                osp.join(root, "3T", "view2")])
            else:
                self.task_folders.extend([osp.join(root, "1T", "view1"),
                                          osp.join(root, "1T", "view2"),
                                          osp.join(root, "2T", "view1"),
                                          osp.join(root, "2T", "view2"),
                                          osp.join(root, "3T", "view1"),
                                          osp.join(root, "3T", "view2")])

        # get_dir_num = lambda x: int(x)
        # self.get_num = lambda x: int(osp.splitext(osp.basename(x))[0])

        # Cubes frames: 1C: 30  | 2C: 50  | 3C: 100
        # T frames: 1T: 50  | 2T: 100  | 3T: 150
        task_valid_frames = {'1C':  25, '2C': 35, '3C': 50, '1T': 20, '2T': 45, '3T': 50}

        # Get all numbers
        self.folders = []
        self.folders_view2 = []
        for d in self.task_folders:
            episodes = os.listdir(d)
            episodes.sort(key=lambda x: int(x[-4:]))
            n_valid = int(self.valid_ratio * len(episodes))
            if self.mode == 'train':
                episodes = episodes[:-n_valid]
            else:
                episodes = episodes[-n_valid:]
            self.folders.extend(episodes)

        if self.multiview:
            for d in self.task_folders_view2:
                episodes_2 = os.listdir(d)
                episodes_2.sort(key=lambda x: int(x[-4:]))
                n_valid = int(self.valid_ratio * len(episodes_2))
                if self.mode == 'train':
                    episodes_2 = episodes_2[:-n_valid]
                else:
                    episodes_2 = episodes_2[-n_valid:]
                self.folders_view2.extend(episodes_2)

        # for file in os.listdir(self.root):
        #     try:
        #         self.folders.append(file)
        #     except ValueError:
        #         continue
        # self.folders.sort()

        # if self.mode == 'train' and self.n_max_train > 0:
        #     self.folders = self.folders[:self.n_max_train]
        # elif self.mode == 'valid' and self.n_max_valid > 0:
        #     self.folders = self.folders[:self.n_max_valid]

        if self.multiview:
            cache_file = os.path.join(self.root, f"cached_index_{self.mode}_mv.pt")
        else:
            cache_file = os.path.join(self.root, f"cached_index_{self.mode}.pt")

        self.episodes = []
        self.episodes_metadata = []
        # self.episodes_instruction = []
        self.episodes_len = []

        self.episodes_view2 = []
        self.episodes_metadata_view2 = []
        self.episodes_len_view2 = []
        self.EP_LEN = max(ep_len, sample_length)
        self.seq_per_episode = self.EP_LEN - self.sample_length + 1  # dense mode

        if use_cache and os.path.exists(cache_file) and not force_cache_rebuild:
            print(f"[Dataset] Loading dataset index from cache: {cache_file}")
            cache = torch.load(cache_file)
            self.episodes = cache['episodes']
            self.episodes_metadata = cache['episodes_metadata']
            # self.episodes_instruction = cache['episodes_instruction']
            self.episodes_len = cache['episodes_len']
            if self.multiview:
                self.episodes_view2 = cache['episodes_view2']
                self.episodes_metadata_view2 = cache['episodes_metadata_view2']
                self.episodes_len_view2 = cache['episodes_len_view2']
        else:
            if use_cache:
                print(f"[Dataset] Building dataset index from scratch (force_rebuild={force_cache_rebuild})...")
            for d in self.task_folders:
                episodes = os.listdir(d)
                episodes.sort(key=lambda x: int(x[-4:]))
                n_valid = int(self.valid_ratio * len(episodes))
                if self.mode == 'train':
                    episodes = episodes[:-n_valid]
                else:
                    episodes = episodes[-n_valid:]
                task_id = get_task_id_from_path(d)
                # get files
                for f in episodes:
                    dir_name = os.path.join(d, str(f))
                    paths = list(glob.glob(osp.join(dir_name, '*.png')))
                    paths = [p for p in paths if 'goal' not in p]
                    get_num = lambda x: int(osp.basename(x).split('.')[0][-4:])
                    paths.sort(key=get_num)
                    paths = paths[:task_valid_frames[task_id]]
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
            if self.multiview:
                for d in self.task_folders_view2:
                    episodes = os.listdir(d)
                    episodes.sort(key=lambda x: int(x[-4:]))
                    n_valid = int(self.valid_ratio * len(episodes))
                    if self.mode == 'train':
                        episodes = episodes[:-n_valid]
                    else:
                        episodes = episodes[-n_valid:]
                    task_id = get_task_id_from_path(d)
                    # get files
                    for f in episodes:
                        dir_name = os.path.join(d, str(f))
                        paths = list(glob.glob(osp.join(dir_name, '*.png')))
                        paths = [p for p in paths if 'goal' not in p]
                        get_num = lambda x: int(osp.basename(x).split('.')[0][-4:])
                        paths.sort(key=get_num)
                        paths = paths[:task_valid_frames[task_id]]
                        paths = paths[:self.EP_LEN]
                        self.episodes_len_view2.append(len(paths))
                        while len(paths) < self.EP_LEN:
                            paths.append(paths[-1])
                        self.episodes_view2.append(paths)

                        metadata_paths = list(glob.glob(osp.join(dir_name, 'metadata*.pt')))
                        metadata_paths.sort(key=get_num)
                        metadata_paths = metadata_paths[:self.EP_LEN]
                        while len(metadata_paths) < self.EP_LEN:
                            metadata_paths.append(metadata_paths[-1])
                        self.episodes_metadata_view2.append(metadata_paths)

            if use_cache:
                print(f"[Dataset] Saving dataset index to cache: {cache_file}")
                if self.multiview:
                    torch.save({
                        'episodes': self.episodes,
                        'episodes_metadata': self.episodes_metadata,
                        # 'episodes_instruction': self.episodes_instruction,
                        'episodes_len': self.episodes_len,
                        'episodes_view2': self.episodes_view2,
                        'episodes_metadata_view2': self.episodes_metadata_view2,
                        'episodes_len_view2': self.episodes_len_view2,
                    }, cache_file)
                else:
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

        imgs_2 = []
        actions_2 = []
        dones_2 = []

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

                if self.multiview:
                    ep_len_2 = self.episodes_len_view2[ep]
                    ep_path_2 = self.episodes_view2[ep]
                    e_act_2 = self.episodes_metadata_view2[ep]

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

                        if self.multiview:
                            imgs_2.append(imgs_2[-1])
                            actions_2.append(torch.zeros_like(actions_2[-1]))
                            dones_2.append(torch.tensor(0, dtype=torch.int))
                    else:
                        img = Image.open(ep_path[image_index])
                        img = self.transform(img)[:3]
                        imgs.append(img)

                        meta_i = torch.load(e_act[image_index])
                        # actions
                        act = meta_i['action']
                        actions.append(act)

                        # dones
                        done_ie = (image_index < ep_len) and (not meta_i['is_terminal'])
                        # done_i = torch.tensor((image_index < ep_len),
                        #                       dtype=torch.int)  # episode_mask: 1 if valid else 0, after end of episode
                        # done_i = torch.tensor((not meta_i['is_terminal']), dtype=torch.int)
                        done_i = torch.tensor(done_ie, dtype=torch.int)
                        dones.append(done_i)

                        if self.multiview:
                            img = Image.open(ep_path_2[image_index])
                            img = self.transform(img)[:3]
                            imgs_2.append(img)

                            meta_i = torch.load(e_act_2[image_index])
                            # actions
                            act = meta_i['action']
                            actions_2.append(act)

                            # dones
                            done_ie = (image_index < ep_len_2) and (not meta_i['is_terminal'])
                            # done_i = torch.tensor((image_index < ep_len),
                            #                       dtype=torch.int)  # episode_mask: 1 if valid else 0, after end of episode
                            # done_i = torch.tensor((not meta_i['is_terminal']), dtype=torch.int)
                            done_i = torch.tensor(done_ie, dtype=torch.int)
                            dones_2.append(done_i)

                if end < ep_len and self.random_goal:
                    goal_idx = random.randint(end, ep_len - 1)
                else:
                    goal_idx = ep_len - 1
                try:
                    goal_img = Image.open(ep_path[goal_idx])
                except IndexError:
                    print(f'ep_len: {ep_len}, goal_idx: {goal_idx}, end: {end}')
                    raise SystemExit
                goal_img = self.transform(goal_img)[:3]
                if self.multiview:
                    goal_idx_2 = goal_idx
                    try:
                        goal_img_2 = Image.open(ep_path_2[goal_idx_2])
                    except IndexError:
                        print(f'ep_len: {ep_len}, goal_idx: {goal_idx}, end: {end}')
                        raise SystemExit
                    goal_img_2 = self.transform(goal_img_2)[:3]

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

                if self.multiview:
                    ep_len_2 = self.episodes_len_view2[ep]
                    ep_path_2 = self.episodes_view2[ep]
                    e_act_2 = self.episodes_metadata_view2[ep]

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

                        if self.multiview:
                            imgs_2.append(imgs_2[-1])
                            actions_2.append(torch.zeros_like(actions_2[-1]))
                            dones_2.append(torch.tensor(0, dtype=torch.int))
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
                        act = meta_i['action']
                        actions.append(act)

                        # dones
                        done_ie = (image_index < ep_len) and (not meta_i['is_terminal'])
                        # done_i = torch.tensor((image_index < ep_len),
                        #                       dtype=torch.int)  # episode_mask: 1 if valid else 0, after end of episode
                        # done_i = torch.tensor((not meta_i['is_terminal']), dtype=torch.int)
                        done_i = torch.tensor(done_ie, dtype=torch.int)
                        dones.append(done_i)

                        if self.multiview:
                            img = Image.open(ep_path_2[image_index])
                            img = self.transform(img)[:3]
                            imgs_2.append(img)

                            meta_i = torch.load(e_act_2[image_index])
                            # actions
                            act = meta_i['action']
                            actions_2.append(act)

                            # dones
                            done_ie = (image_index < ep_len_2) and (not meta_i['is_terminal'])
                            # done_i = torch.tensor((image_index < ep_len),
                            #                       dtype=torch.int)  # episode_mask: 1 if valid else 0, after end of episode
                            # done_i = torch.tensor((not meta_i['is_terminal']), dtype=torch.int)
                            done_i = torch.tensor(done_ie, dtype=torch.int)
                            dones_2.append(done_i)

                if end < ep_len and self.random_goal:
                    goal_idx = random.randint(end, ep_len - 1)
                else:
                    goal_idx = ep_len - 1
                goal_img = Image.open(ep_path[goal_idx])
                goal_img = self.transform(goal_img)[:3]

                if self.multiview:
                    goal_idx_2 = goal_idx
                    try:
                        goal_img_2 = Image.open(ep_path_2[goal_idx_2])
                    except IndexError:
                        print(f'ep_len: {ep_len}, goal_idx: {goal_idx}, end: {end}')
                        raise SystemExit
                    goal_img_2 = self.transform(goal_img_2)[:3]

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

            if self.multiview:
                paths_2 = self.episodes_view2[index]
                action_paths_2 = self.episodes_metadata_view2[index]
                ep_len_2 = self.episodes_len_view2[index]
                # cut to maximal ep_len
                paths_2 = paths_2[:self.EP_LEN]

            for pi, path in enumerate(paths):
                img = Image.open(path)
                img = self.transform(img)[:3]
                imgs.append(img)

                meta_i = torch.load(action_paths[pi])
                # actions
                act = meta_i['action']
                actions.append(act)

                # dones
                done_ie = (pi < ep_len) and (not meta_i['is_terminal'])
                # done_i = torch.tensor((pi < ep_len),
                #                       dtype=torch.int)  # episode_mask: 1 if valid else 0, after end of episode
                # done_i = torch.tensor((not meta_i['is_terminal']), dtype=torch.int)
                done_i = torch.tensor(done_ie, dtype=torch.int)
                dones.append(done_i)

                if self.multiview:
                    path = paths_2[pi]
                    img = Image.open(path)
                    img = self.transform(img)[:3]
                    imgs_2.append(img)

                    meta_i = torch.load(action_paths_2[pi])
                    # actions
                    act = meta_i['action']
                    actions_2.append(act)

                    # dones
                    done_ie = (pi < ep_len_2) and (not meta_i['is_terminal'])
                    # done_i = torch.tensor((pi < ep_len),
                    #                       dtype=torch.int)  # episode_mask: 1 if valid else 0, after end of episode
                    # done_i = torch.tensor((not meta_i['is_terminal']), dtype=torch.int)
                    done_i = torch.tensor(done_ie, dtype=torch.int)
                    dones_2.append(done_i)

            while len(imgs) < self.sample_length:
                imgs.append(imgs[-1])
                actions.append(torch.zeros_like(actions[-1]))
                dones.append(torch.tensor(0, dtype=torch.int))

                if self.multiview:
                    imgs_2.append(imgs_2[-1])
                    actions_2.append(torch.zeros_like(actions_2[-1]))
                    dones_2.append(torch.tensor(0, dtype=torch.int))

            goal_img = Image.open(paths[-1])
            goal_img = self.transform(goal_img)[:3]

            if self.multiview:
                goal_img_2 = Image.open(paths_2[-1])
                goal_img_2 = self.transform(goal_img_2)[:3]

            # instructions
            # inst = torch.load(e_inst)
            # instruction = inst['raw_instruction']
            # # instructions embeddings
            # instruction_embedding = inst['instruction_embedding'].detach()

        img = torch.stack(imgs, dim=0).float()
        actions = torch.stack(actions, dim=0).float()
        # dones = torch.stack(dones, dim=0).bool()
        dones = torch.stack(dones, dim=0).int()

        if self.multiview:
            img_2 = torch.stack(imgs_2, dim=0).float()
            actions_2 = torch.stack(actions_2, dim=0).float()
            dones_2 = torch.stack(dones_2, dim=0).int()

            img = torch.stack([img, img_2], dim=1)  # [T, 2, ...]
            actions = torch.stack([actions, actions_2], dim=1)  # [T, 2, ...]
            dones = torch.stack([dones, dones_2], dim=1)  # [T, 2, ...]

            goal_img = torch.stack([goal_img, goal_img_2], dim=0)  # [2, ....]

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


class PandaPushImage(Dataset):
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
            cache = torch.load(cache_file)
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
                max_offset = max(1, ep_len)
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

                meta_i = torch.load(e_act[image_index])
                # actions
                act = meta_i['action']
                actions.append(act)

            done_i = torch.tensor((image_index < ep_len),
                                  dtype=torch.int)  # episode_mask: 1 if valid else 0, after end of episode
            # done_i = meta_i['is_terminal']
            dones.append(done_i)

        # instructions
        inst = torch.load(e_inst)
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


class PandaPushVideoOLD(Dataset):
    def __init__(self, root, mode, ep_len=50, sample_length=20, image_size=128):
        # path = os.path.join(root, mode)
        if mode == 'val':
            mode = 'valid'
        assert mode in ['train', 'valid', 'test']
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
        if self.mode == 'valid':
            self.episodes = self.episodes[:200]

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


class PandaPushImageOLD(Dataset):
    def __init__(self, root, mode, ep_len=50, sample_length=1, image_size=128):
        # path = os.path.join(root, mode)
        if mode == 'val':
            mode = 'valid'
        assert mode in ['train', 'valid', 'test']
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
    root = 'C:/Users/tabad/Downloads/panda_ds'
    ds = PandaPushVideo(root=root, ep_len=50, sample_length=20, mode='valid', image_size=128, dense=True)
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
        img_goal_np = im_goal[0].permute(1, 2, 0).data.cpu().numpy()

        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(121)
        ax.imshow(img_np)

        ax = fig.add_subplot(122)
        ax.imshow(img_goal_np)
        plt.show()

    if test_epochs:
        from tqdm import tqdm

        pbar = tqdm(iterable=dl)
        for batch in pbar:
            pass
        pbar.close()
