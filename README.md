# lpwm

<p align="center">
  <img src="https://img.shields.io/badge/conference-ICLR%202026-orange" />
  <img src="https://img.shields.io/badge/pytorch-%E2%89%A52.6-blue" />
  <img src="https://img.shields.io/badge/license-MIT-green" />
  <img src="https://img.shields.io/badge/arXiv-2401.xxxxx-b31b1b" />
</p>

Official PyTorch implementation of the paper "Latent Particle World Models: Self-supervised Object-centric Stochastic
Dynamics Modeling".

<h1 align="center">
  <br>
	Latent Particle World Models: Self-supervised Object-centric Stochastic Dynamics Modeling
  <br>
</h1>
<h2 align="center">
  ICLR 2026 Oral
  <br>
</h2>
  <h3 align="center">
    <a href="https://taldatech.github.io">Tal Daniel</a> •
    <a href="https://carl-qi.github.io/">Carl Qi</a> •
    <a href="https://danhrmti.github.io/">Dan Haramati</a> •
    <a href="https://lambda.ai/research">Amir Zadeh</a> •
    <a href="https://lambda.ai/research">Chuan Li</a> •
    <a href="https://avivt.github.io/avivt/">Aviv Tamar</a> •
    <a href="https://www.cs.cmu.edu/~dpathak/">Deepak Pathak</a> •
    <a href="https://r-pad.github.io/">David Held</a>

  </h3>
<h3 align="center">Official repository of Deep Latent Particles v3 (DLPv3) & LPWM</h3>

<h4 align="center"><a href="https://arxiv.org/abs/2603.04553">Arxiv</a> • <a href="https://taldatech.github.io/lpwm-web">Project
Website</a> • <a href="https://youtu.be/aZeaCyXJjYI">Video</a> • <a href="https://openreview.net/forum?id=lTaPtGiUUc">OpenReview</a></h4>

<h4 align="center">
    <a href="https://colab.research.google.com/github/taldatech/lpwm"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
</h4>


<p align="center">
  <img src="https://github.com/taldatech/lpwm-web/blob/main/assets/sketchy_particlegrid.gif?raw=true" height="150"><br>
  <img src="https://github.com/taldatech/lpwm-web/blob/main/assets/langtable_476_lpwm_gen_comp.gif?raw=true" height="130">
  <img src="https://github.com/taldatech/lpwm-web/blob/main/assets/bridge_313_lang_comp.gif?raw=true" height="130"><br>
</p>

# Latent Particle World Models: Self-supervised Object-centric Stochastic Dynamics Modeling

> **Latent Particle World Models: Self-supervised Object-centric Stochastic Dynamics Modeling**<br>
> Tal Daniel , Carl Qi, Dan Haramati, Amir Zadeh, Chuan Li, Aviv Tamar, Deepak Pathak, David Held<br>
>
> **Abstract:** *We introduce Latent Particle World Model (LPWM),
> a self-supervised object-centric world model scaled to real-world multi-object datasets and applicable in
decision-making.
> LPWM autonomously discovers keypoints, bounding boxes, and object masks directly from video data,
> enabling it to learn rich scene decompositions without supervision.
> Our architecture is trained end-to-end purely from videos and supports
> flexible conditioning on actions, language, and image goals. LPWM models stochastic particle dynamics via
> a novel latent action module and achieves state-of-the-art results on diverse real-world and synthetic datasets.
> Beyond stochastic video modeling, LPWM is readily applicable to decision-making,
> including goal-conditioned imitation learning, as we demonstrate in the paper.*

## Citation

Daniel, T., Qi, C., Haramati, D., Zadeh, A., Li, C., Tamar, A., Pathak, D., & Held, D. (2026).
Latent particle world models: Self-supervised object-centric stochastic dynamics modeling.
In The Fourteenth International Conference on Learning Representations (ICLR
2026). https://openreview.net/forum?id=lTaPtGiUUc

    @inproceedings{
    daniel2026latent,
    title={Latent Particle World Models: Self-supervised Object-centric Stochastic Dynamics Modeling},
    author={Tal Daniel and Carl Qi and Dan Haramati and Amir Zadeh and Chuan Li and Aviv Tamar and Deepak Pathak and David Held},
    booktitle={The Fourteenth International Conference on Learning Representations},
    year={2026},
    url={https://openreview.net/forum?id=lTaPtGiUUc}
    }

## Quickstart

```bash
# Install environment
conda env create -f environment.yml
conda activate dlp

# Train LPWM on Sketchy
python train_lpwm.py --dataset sketchy

# Generate videos with a pretrained model
python generate_lpwm_video_prediction.py --help
```

- [lpwm](#lpwm)
- [Latent Particle World Models: Self-supervised Object-centric Stochastic Dynamics Modeling](#latent-particle-world-models--self-supervised-object-centric-stochastic-dynamics-modeling)
    * [Citation](#citation)
    * [Prerequisites](#prerequisites)
    * [Model Zoo - Pretrained Models](#model-zoo---pretrained-models)
    * [Datasets](#datasets)
    * [Conditioning Types, Multi-View and Examples](#conditioning-types--multi-view-and-examples)
    * [Attribute Variable Names](#attribute-variable-names)
    * [DLPv3 and LPWM - Training](#dlpv3-and-lpwm---training)
    * [Training Logs, Progress Bar and Saved Images/Videos](#training-logs--progress-bar-and-saved-images-videos)
    * [LPWM - Evaluation](#lpwm---evaluation)
        + [PSNR, SSIM, LPIPS](#psnr--ssim--lpips)
        + [FVD](#fvd)
    * [LPWM - Video Prediction and Generation with Pre-trained Models](#lpwm---video-prediction-and-generation-with-pre-trained-models)
    * [Example Usage, Documentation and Notebooks](#example-usage--documentation-and-notebooks)
        + [DLPv3 and LPWM Walkthrough Tutorial Notebook](#dlpv3-and-lpwm-walkthrough-tutorial-notebook)
    * [Repository Organization](#repository-organization)

## Prerequisites

We provide an `environment.yml` file which installs the required packages in a `conda` environment named `torch`.
Alternatively, you can use `pip` to install `requirements.txt`.

- Create the environment with: `conda env create -f environment.yml`.

| Library           | Version      | Notes                                           |
|-------------------|--------------|-------------------------------------------------|
| `Python`          | `> = 3.9`    | -                                               |
| `torch`           | > = `2.6.0`  | -                                               |
| `torchvision`     | > = `0.21`   | -                                               |
| `matplotlib`      | > = `3.10.0` | -                                               |
| `numpy`           | > = `1.24.3` | -                                               |
| `h5py`            | > = `3.13.0` | Some datasets (e.g., Balls) use H5/HDF          |
| `py-opencv`       | > = `4.11`   | For plotting                                    |
| `tqdm`            | > = `4.67.0` | -                                               |
| `scipy`           | > = `1.15`   | -                                               |
| `scikit-image`    | > = `0.25.2` | Required to generate the "Shapes" dataset       |
| `ffmpeg`          | = `4.2.2`    | Required to generate video files                |
| `accelerate`      | > = `1.5.0`  | For multi-GPU training                          |
| `imageio`         | > = `2.6.1`  | For creating video GIFs                         |
| `piqa`            | > = `1.3.1`  | For image evaluation metrics: LPIPS, SSIM, PSNR |
| `einops`          | > = `0.81`   | -                                               |
| `huggingface-hub` | > = `0.29`   | Downloading checkpoint and datasets             |
| `notebook`        | > = `6.5.4`  | To run Jupyter Notebooks                        |

For a manual installation guide, see `docs/installation.md`.

## Model Zoo - Pretrained Models

* We provide pre-trained checkpoints for datasets used in the paper.
* All model checkpoints should be placed inside the `/checkpoints` directory.

The following table lists the available pre-trained checkpoints and where to download them.

| Model Type    | Dataset                 | Link                                                                                 |
|---------------|-------------------------|--------------------------------------------------------------------------------------|
| LPWM          | Sketchy (128x128)       | [MEGA.nz](https://mega.nz/file/YVMnyZYJ#ndHMuT0ifI3xEFG_PCa6AX0BP_e8vM8hRJHZ_lBNQE8) |
| LPWM-Action   | Sketchy (128x128)       | [MEGA.nz](https://mega.nz/file/1Uk1GLxb#DUdgscdLvw8NZdPGT6cs6Htxgjj-ep_gRtc_mfkkgGo) |
| LPWM          | BAIR (128x128)          | [MEGA.nz](https://mega.nz/file/pdtHDKZI#RFwaweMdTsiHKPOkmpM4sQHX8ZPorag7r3jSM4CgiNk) |
| LPWM-Language | LangaugeTable (128x128) | [MEGA.nz](https://mega.nz/file/BM0QiZRQ#-Px0DJmnypDK4Wq8O31VRzR_kJMnrQGaSSOlGkDUUGI) |
| LPWM-Language | Bridge (128x128)        | [MEGA.nz](https://mega.nz/file/1MNSnSLD#BINQh4WozzCr_3JhF9ctRtrlcNf6ayRQbxn2SzcFJM0) |

## Datasets

| Dataset        | Notes                                                                                                                                                                                                                                                                | Link                                                                                                                                                     |
|----------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------|
| Sketchy        | Original dataset from [Deepmind](https://github.com/google-deepmind/deepmind-research/tree/master/sketchy). We use a subset and provide the pre-processed data on HF.                                                                                                | [HF](https://huggingface.co/datasets/taldatech/sketchy_128/tree/main)                                                                                    |
| Bridge         | Pre-processed videos from the BRIDGE dataset with T5-large embeddings for the text instructions. If you want to prepare the data yourself, download from [Open-X](https://github.com/google-deepmind/open_x_embodiment) and follow `datasets/bridge_preparation.py`. | [HF](https://huggingface.co/datasets/taldatech/bridge_256)                                                                                               |
| BAIR           | We use a high-resolution version of the [BAIR](https://sites.google.com/berkeley.edu/robotic-interaction-datasets/home) dataset, courtesy of [PVG](https://github.com/willi-menapace/PlayableVideoGeneration).                                                       | [HF](https://huggingface.co/datasets/taldatech/bair_256)                                                                                                 |
| LanguageTable  | Follow download and pre-processing instructions in `datasets/langtable_preparation.py`. It requires `tensorflow` and the `transformers` libraries.                                                                                                                   | `datasets/langtable_preparation.py`, [Open-X](https://github.com/google-deepmind/open_x_embodiment)                                                      |
| Panda          | Expert imitation trajectories for several manipualation tasks from IsaacsGym simulator. Used in [EC-Diffuser](https://github.com/carl-qi/EC-Diffuser).                                                                                                               | [HF](https://huggingface.co/datasets/taldatech/panda_ds/tree/main)                                                                                       |
| OGBench        | Offline RL trajectories from [OGBench](https://github.com/seohongpark/ogbench) of `scene` and `cube` tasks.                                                                                                                                                          | [HF](https://huggingface.co/datasets/taldatech/ogbench_data/tree/main)                                                                                   |
| Mario          | A small dataset of Mario gameplay videos divided to 100-frame episodes.                                                                                                                                                                                              | [HF](https://huggingface.co/datasets/taldatech/mario_gameplay/tree/main)                                                                                 |
| OBJ3D          | Courtesy of [G-SWM](https://github.com/zhixuan-lin/G-SWM)                                                                                                                                                                                                            | [Google Drive](https://drive.google.com/file/d/1XSLW3qBtcxxvV-5oiRruVTlDlQ_Yatzm/view) , [HF](https://huggingface.co/datasets/taldatech/OBJ3D/tree/main) |
| PHYRE          | See `datasets/phyre_preparation.py` to generate new data or download the data we generated                                                                                                                                                                           | [MEGA.nz](https://mega.nz/file/lIkDnZZD#Ym4vlAqd3egCljoujya33KHaua7AwmusmbUw27OdIHE)                                                                     |
| Balls          | Synthetic, courtesy of [G-SWM](https://github.com/zhixuan-lin/G-SWM), see [this link](https://github.com/zhixuan-lin/G-SWM/blob/master/scripts/gen_data_balls.sh) to generate or download data we generated                                                          | [MEGA.nz](https://mega.nz/file/4cUR1b5a#RwFFzCiESeeQb8rYgt7PK2_D8b_69-K85RV3jlaphTo)                                                                     |
| Shapes         | Synthetic, generated on-the-fly                                                                                                                                                                                                                                      | see `generate_shape_dataset_torch()` in `datasets/shapes_ds.py`                                                                                          |
| Custom Dataset | 1. Implement a `Dataset` (see examples in `/datasets`).<br>2. Add it to `get_image_dataset()` and `get_video_dataset()` in `/datasets/get_dataset.py`.<br> 3. Prepare a `json` config file with the hyperparameters and place it in`/configs`.                       | -                                                                                                                                                        |

## Conditioning Types, Multi-View and Examples

| Condition Type             | Required Flags                                                                                                 | `Dataset` Example                                      | `config.json` Example                                            |
|----------------------------|----------------------------------------------------------------------------------------------------------------|--------------------------------------------------------|------------------------------------------------------------------|
| None (only latent actions) | -                                                                                                              | `/datasets/bair_ds.py`, `/datasets/mario_ds.py`        | `/configs/bair.json`, `/configs/mario.json`                      |
| Action                     | `"action_condition": true`, `"action_dim": 7`                                                                  | `/datasets/sketchy_ds.py`, `/datasets/langtable_ds.py` | `/configs/sketchy_action.json`, `/configs/langtable_action.json` |
| Language                   | `"language_condition": true`, `language_embed_dim": 1024` (T5), `"language_max_len": 32` (max language tokens) | `/datasets/langtable_ds.py`, `/datasets/bridge_ds.py`  | `/configs/bridge.json`, `/configs/langtable.json`                |
| Image                      | `"image_goal_condition": true`                                                                                 | `/datasets/ogbench_ds.py`, `/datasets/panda_ds.py`     | `/configs/ogbench.json`, `/configs/panda.json`                   |
| --                         | --                                                                                                             | --                                                     | --                                                               |
| Multi-view                 | `"n_views": 2`                                                                                                 | `/datasets/panda_ds.py`                                | `/configs/panda.json`                                            |

## Attribute Variable Names

Most latent attributes variables names in the code follow the same convention from the paper (e.g., `z_scale`,
`z_depth`, ...).

**Position (KP)**: `z`,
**Transparency**: `z_obj_on` / `obj_on`,
**Visual features**: `z_features`

## DLPv3 and LPWM - Training

You can train the models on single-GPU machines and multi-GPU machines. For multi-GPU training We use
[HuggingFace Accelerate](https://huggingface.co/docs/accelerate/index): `pip install accelerate`.

1. Set visible GPUs under: `os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3"` (`NUM_GPUS=4`)
2. Set "num_processes": NUM_GPUS in `accel_conf.yml` (e.g. `"num_processes":4`
   if `os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3"`).


* Single-GPU machines: `python train_dlp.py -d {dataset}` / `python train_lpwm.py -d {dataset}`
* Multi-GPU machines: `accelerate launch --config_file ./accel_conf.yml train_dlp_accelerate.py -d {dataset}` /
  `accelerate launch --config_file ./accel_conf.yml train_lpwm_accelerate.py -d {dataset}`

Config files for the datasets are located in `/configs/{ds_name}.json`. You can modify hyperparameters in these files.
To generate a config file for a new datasets you can copy one of the files in `/confgis` or use the
`/configs/generate_config_file.py` script.

**Hyperparameters**: See `/docs/hyperparamters.md` for extended details and recommended values.

The scripts `train_dlp.py`/`train_lpwm.py` or `train_dlp_accelerate.py`/`train_lpwm_accelerate.py` are using the config
file `/configs/{ds_name}.json`

Examples:

### Single-GPU

- `python train_dlp.py --dataset shapes`
- `python train_dlp.py --dataset obj3d_img`
- `python train_lpwm.py --dataset mario`
- `python train_lpwm.py --dataset bridge`

### Multi-GPU

- `accelerate launch --config_file ./accel_conf.yml train_dlp_accelerate.py --dataset obj3d_img`
- `accelerate launch --config_file ./accel_conf.yml train_lpwm_accelerate.py --dataset obj3d128`

* Note: if you want multiple multi-GPU runs, each run should have a different accelerate config file (
  e.g., `accel_conf.yml`, `accel_conf_2.yml`, etc..). The only difference between the files should be
  the `main_process_port` field (e.g., for the second config file, set `main_process_port: 81231`).

## Training Logs, Progress Bar and Saved Images/Videos

During training, the script saves (locally) a log file with the metrics output and images in the following structure:

<p align="center">
  <img src="https://github.com/taldatech/lpwm-web/blob/main/assets/langtable_dlp.jpg?raw=true" height="350">
</p>

where the columns are different images in the batch (DLPv3) or an image sequence (LPWM) and the rows correspond to:

| Row | Image Meaning                                                                              |
|-----|--------------------------------------------------------------------------------------------|
| 1   | Ground-truth (GT) original input image                                                     |
| 2   | GT image + *all* `L` (DLP)/`M` (LPWM) posterior particles                                  |
| 3   | Reconstruction of the entire scene                                                         |
| 4   | GT image + *all* `M` prior keypoints (proposals)                                           |
| 5   | GT image + top-`K` posterior particle filtered by their uncertainty                        |
| 6   | Foreground reconstruction (decoded glimpses and masks of the particles)                    |
| 7   | GT image + bounding boxes based on the scale attribute `z_s` (+ non-maximal suppression)   |
| 8   | GT image + bounding boxes based on the decoded particles masks (+ non-maximal suppression) |
| 9   | Colored segementation masks (each particle's alpha map is colored differently)             |
| 10  | Backgroung reconstruction                                                                  |

During training, the script saves animation videos of rollouts with 3 panels:

<p align="center">
  <img src="https://github.com/taldatech/lpwm-web/blob/main/assets/sketchy_video.gif?raw=true" height="120">
  <img src="https://github.com/taldatech/lpwm-web/blob/main/assets/langtable_video.gif?raw=true" height="120">
</p>

| Ground Truth                 | Latent-actioned Conditioned                                                                                           | Sampling                                       |
|------------------------------|-----------------------------------------------------------------------------------------------------------------------|------------------------------------------------|
| The real video from the data | latent actions are extracted from GT video and used to condition the dynamics module with the first frame of GT video | Latent actions are sampled from te first frame |

* For image conditioning: the image will be plotted in the animation as "Goal".
* For language conditioning: the text will be added as title.

More examples are [here](https://github.com/taldatech/lpwm-web/tree/main/assets).

**Progress Bar**: In addition to the losses (reconstruction and KLs), the progress bar displays:

* `on_l1` - the average number of visible particles (the average `z_t`). This is useful to diagnose whether objects are
  not detected (`=0`) or if the model is saturated (`=MAX_PARTICELS`) and all the information is in the FG (e.g., BG is
  not learned).
* `a, b` - the average values of the concentration parameters of the Beta distribution (`Beta(a, b)`) for the
  transparency `z_t` (`z_obj_on`). They need to be balanced and not too large/small.
* `smu` - the average `mu` of the scale attribute `z_scale` (also the bounding boxes). Need to make sure not too large.

## LPWM - Evaluation

The evaluation protocol measures the reconstruction quality via 3 metrics: LPIPS, SSIM and PSNR and video generation via
FVD.

### PSNR, SSIM, LPIPS

We use the open-source `piqa` (`pip install piqa`) to compute the metrics. If `eval_im_metrics=True` in the config file,
the metrics will be computed every evaluation epoch on the *validation* set. The code saves the best model based
on the LPIPS metric.

To evaluate a pre-trained LPWM model (on the *test* set) on video prediction, we provide a script:

`python eval/eval_gen_metrics.py --help`

For example, to evaluate a model saved in `/checkpoints/sketchy/sketchy_gddlp.pth` on the test set, with 6 conditional
frames and
a generation horizon of 50 frames:

`python eval/eval_gen_metrics.py -d sketchy -p ./checkpoints/sketchy --checkpoint ./checkpoints/sketchy/sketchy.pth --sample -b 10 -c 6 --horizon 50 --prefix "" --ctx`

where `b` is the batch size and `ctx` specifies to use latent actions to generate the video (meaning that the script
will first extract the latent actions from the GT video and use them to condition the dynamics module).

The script will load the config file `hparams.json` from `./checkpoints/sketchy/hparams.json` and the model
checkpoint from `./checkpoints/sketchy/sketchy.pth` and use it to generate video
predictions. Make sure the path to the dataset is defined correctly in `hparams.json`.

For more options, see `python eval/eval_gen_metrics.py --help`.

For a similar evaluation of DLPv3 in the single-image setting, see `eval_dlp_im_metric()` in
`/eval/eval_gen_metrics.py`.

### FVD

For FVD, we use: https://github.com/JunyaoHu/common_metrics_on_video_quality

Please download the networks checkpoints from the repository before evaluating the FVD (placed under
`/eval/fvd/fvd/styleganv/i3d_torchscript.pt`).

To evaluate a pre-trained LPWM model on video generation, we provide a script:

`python eval/eval_fvd.py --help`

For example, to evaluate a model saved in `/checkpoints/sketchy/sketchy_gddlp.pth` with 6 conditional frames and
a generation horizon of 50 frames:

`python eval/eval_fvd.py -d sketchy -p ./checkpoints/sketchy --checkpoint ./checkpoints/sketchy/sketchy.pth --sample -b 4 -c 6 --horizon 50 --prefix "" --n_videos_per_clip 1`

where `b` is the batch size and `n_videos_per_clip` specifies how many video to generate per one video in the dataset (
for the `bair64` benchmark, this number is set to 100).

The script will load the config file `hparams.json` from `./checkpoints/sketchy/hparams.json` and the model
checkpoint from `./checkpoints/sketchy/sketchy.pth` and use it to generate video
predictions. Make sure the path to the dataset is defined correctly in `hparams.json`.

For more options, see `python eval/eval_fvd.py --help`.

## LPWM - Video Prediction and Generation with Pre-trained Models

To generate video predictions using a pre-trained model use `generate_lpwm_video_prediction.py` as follows:

`python generate_lpwm_video_prediciton.py -d sketchy -p ./checkpoints/sketchy --checkpoint ./checkpoints/sketchy/sketchy.pth --sample -n 4 -c 6 --horizon 50 --prefix ""`

The script will load the config file `hparams.json` from `./checkpoints/sketchy/hparams.json` and the model
checkpoint from `./checkpoints/sketchy/sketchy.pth` and use it to generate `n` video
predictions,
based on 6 conditional input frames, and a final video length of 50 frames. In the example above, four (`n=4`)
videos
will be generated and saved within a `videos` directory (will be created if it doesn't exist) under the checkpoint
directory. Make sure the path to the dataset is defined correctly in `hparams.json`.

For more options, see `python generate_lpwm_video_prediction.py --help`.

## Example Usage, Documentation and Notebooks

For your convenience, we provide more documentation in `/docs` and more examples of using the models in `/notebooks`.

| File                                                                      | Content                                                                                                       |
|---------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------|
| `docs/installation.md`                                                    | Manual instructions to install packages with `conda`                                                          |
| `docs/hyperparameters.md`                                                 | Explanations of the various hyperparameters of the models and recommended values                              |
| `docs/example_usage.py`                                                   | Overview of the models functionality: forward output, loss calculation and sampling                           |
| `notebooks/dlpv3_lpwm_walkthrough_tutorial.ipynb`                         | Tutorial and walkthrough of DLPv3 and LPWM, where we train and evaluate a DLPv3 model on the `shapes` dataset |
| `notebooks/lpwm_gen_sketchy.ipynb` / `notebooks/lpwm_gen_langtable.ipynb` | Generating videos with pre-trained LPWMs and creating figures                                                 |

### DLPv3 and LPWM Walkthrough Tutorial Notebook

Please see our tutorial if you are you new to Deep Latent Particles, where you can train the model online:
`notebooks/dlpv3_lpwm_walkthrough_tutorial.ipynb`
<p align="center">
  <img src="https://github.com/taldatech/lpwm-web/blob/main/assets/tutorial.png?raw=true" height="150">
</p>

[Open in Colab](https://colab.research.google.com/github/taldatech/lpwm)

## Repository Organization

| File name                                            | Content                                                                                      |
|------------------------------------------------------|----------------------------------------------------------------------------------------------|
| `/checkpoints`                                       | directory for pre-trained checkpoints                                                        |
| `/assets`                                            | directory containing sample images                                                           |
| `/datasets`                                          | directory containing data loading classes for the various datasets                           |
| `/configs`                                           | directory containing config files for the various datasets                                   |
| `/docs`                                              | various documentation files                                                                  |
| `/notebooks`                                         | various Jupyter Notebook examples of DLPv3 and LPWM                                          |
| `/eval/eval_model.py`                                | evaluation functions such as evaluating the ELBO                                             |
| `/eval/eval_gen_metrics.py`                          | evaluation functions for image metrics (LPIPS, PSNR, SSIM)                                   |
| `/modules/modules.py`                                | basic neural network blocks used to implement DLPv3                                          |
| `/utils/loss_functions.py`                           | loss functions used to optimize the model such as Chamfer-KL and perceptual (VGG/LPIPS) loss |
| `/utils/util_func.py`                                | utility functions such as logging and plotting functions, Spatial Transformer Network (STN)  |
| `models.py`                                          | implementation of DLPv3 and LPWM                                                             |
| `train_dlp.py`/`train_lpwm.py`                       | training function of DLP/LPWM for single-GPU machines                                        |
| `train_dlp_accelerate.py`/`train_lpwm_accelerate.py` | training function of DLP/LPWM for multi-GPU machines                                         |
| `environment.yml`                                    | Anaconda environment file to install the required dependencies                               |
| `requirements.txt`                                   | requirements file for `pip`                                                                  |
| `accel_conf.yml`                                     | configuration file for `accelerate` to run training on multiple GPUs                         |
