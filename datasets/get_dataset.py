# datasets
from datasets.traffic_ds import TrafficDataset, TrafficDatasetImage
from datasets.shapes_ds import generate_shape_dataset_torch
from datasets.balls_ds import Balls, BallsImage
from datasets.obj3d_ds import Obj3D, Obj3DImage
from datasets.phyre_ds import PhyreDataset, PhyreDatasetImage
from datasets.langtable_ds import LanguageTableDataset, LanguageTableDatasetImage
from datasets.bridge_ds import BridgeDataset, BridgeDatasetImage
from datasets.panda_ds import PandaPushVideo, PandaPushImage
from datasets.mario_ds import MarioVideo, MarioImage
from datasets.bair_ds import BAIRVideo, BAIRImage
from datasets.bair64_ds import BAIR64Video, BAIR64Image
from datasets.sketchy_ds import SketchyVideoDataset, SketchyImageDataset
from datasets.ogbench_ds import OGBenchDataset, OGBenchDatasetImage
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor


def get_video_dataset(ds, root, seq_len=1, mode='train', image_size=128):
    # load data
    if ds == "traffic":
        dataset = TrafficDataset(path_to_npy=root, image_size=image_size, mode=mode, sample_length=seq_len)
    elif ds == 'balls':
        dataset = Balls(root=root, mode=mode, sample_length=seq_len)
    elif ds == 'obj3d':
        dataset = Obj3D(root=root, mode=mode, sample_length=seq_len)
    elif ds == 'obj3d128':
        image_size = 128
        dataset = Obj3D(root=root, mode=mode, sample_length=seq_len, res=image_size)
    elif ds == 'phyre':
        dataset = PhyreDataset(root=root, mode=mode, sample_length=seq_len, image_size=image_size)
    elif ds == 'bair':
        dataset = BAIRVideo(root=root, mode=mode, sample_length=seq_len, image_size=image_size)
    elif ds == 'bair64':
        dataset = BAIR64Video(root=root, mode=mode, sample_length=seq_len, image_size=image_size)
    elif ds == 'langtable':
        dataset = LanguageTableDataset(root=root, mode=mode, sample_length=seq_len, image_size=image_size)
    elif ds == 'bridge':
        dataset = BridgeDataset(root=root, mode=mode, sample_length=seq_len, image_size=image_size)
    elif ds == "panda":
        dataset = PandaPushVideo(root=root, image_size=image_size, mode=mode, sample_length=seq_len)
    elif ds == "mario":
        dataset = MarioVideo(root=root, image_size=image_size, mode=mode, sample_length=seq_len)
    elif ds == "sketchy":
        dataset = SketchyVideoDataset(root=root, mode=mode, sample_length=seq_len, image_size=image_size)
    elif ds == "ogbench":
        dataset = OGBenchDataset(root=root, mode=mode, sample_length=seq_len, image_size=image_size)
    else:
        raise NotImplementedError
    return dataset


def get_image_dataset(ds, root, mode='train', image_size=128, seq_len=1):
    # set seq_len > 1 when training with use_tracking
    # load data
    if ds == "traffic":
        dataset = TrafficDatasetImage(path_to_npy=root, image_size=image_size, mode=mode, sample_length=seq_len)
    elif ds == 'balls':
        dataset = BallsImage(root=root, mode=mode, sample_length=seq_len)
    elif ds == 'obj3d':
        dataset = Obj3DImage(root=root, mode=mode, sample_length=seq_len)
    elif ds == 'obj3d128':
        image_size = 128
        dataset = Obj3DImage(root=root, mode=mode, sample_length=seq_len, res=image_size)
    elif ds == 'phyre':
        dataset = PhyreDatasetImage(root=root, mode=mode, sample_length=seq_len, image_size=image_size)
    elif ds == "sketchy":
        dataset = SketchyImageDataset(root=root, mode=mode, sample_length=seq_len, image_size=image_size)
    elif ds == 'bair':
        dataset = BAIRImage(root=root, mode=mode, sample_length=seq_len, image_size=image_size)
    elif ds == 'bair64':
        dataset = BAIR64Image(root=root, mode=mode, sample_length=seq_len, image_size=image_size)
    elif ds == 'shapes':
        if mode == 'train':
            dataset = generate_shape_dataset_torch(img_size=image_size, num_images=40_000)
        else:
            dataset = generate_shape_dataset_torch(img_size=image_size, num_images=2_000)
    elif ds == 'langtable':
        dataset = LanguageTableDatasetImage(root=root, mode=mode, sample_length=seq_len, image_size=image_size)
    elif ds == 'bridge':
        dataset = BridgeDatasetImage(root=root, mode=mode, sample_length=seq_len, image_size=image_size)
    elif ds == 'cifar10':
        dataset = CIFAR10(root=root, train=(mode == 'train'), download=True, transform=ToTensor())
    elif ds == "panda":
        dataset = PandaPushImage(root=root, image_size=image_size, mode=mode, sample_length=seq_len)
    elif ds == "mario":
        dataset = MarioImage(root=root, image_size=image_size, mode=mode, sample_length=seq_len)
    elif ds == "ogbench":
        dataset = OGBenchDatasetImage(root=root, mode=mode, sample_length=seq_len, image_size=image_size)
    else:
        raise NotImplementedError
    return dataset
