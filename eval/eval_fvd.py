"""
Evaluate video generation metric - FVD
FVD tools from https://github.com/JunyaoHu/common_metrics_on_video_quality
"""

# set workdir
import os
import sys

sys.path.append(os.getcwd())
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
import argparse
import json
from tqdm import tqdm
from models import DLP
from utils.util_func import get_config
# datasets
from datasets.get_dataset import get_video_dataset, get_image_dataset

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from fvd.calculate_fvd import calculate_fvd


def load_dlp_from_config(conf_path, ckpt_path=None):
    # load config
    try:
        config = get_config(conf_path)
    except FileNotFoundError:
        raise SystemExit("config file not found")
    # hparams = config  # to save a copy of the hyper-parameters
    ch = config['ch']  # image channels
    image_size = config['image_size']
    n_views = config.get('n_views', 1)
    # model
    timestep_horizon = config['timestep_horizon']
    num_static_frames = config['num_static_frames']
    pad_mode = config['pad_mode']
    n_kp_per_patch = config['n_kp_per_patch']  # kp per patch in prior, best to leave at 1
    n_kp_prior = config['n_kp_prior']  # number of prior kp to filter for the kl
    n_kp_enc = config['n_kp_enc']  # total posterior kp
    patch_size = config['patch_size']  # prior patch size
    anchor_s = config['anchor_s']  # posterior patch/glimpse ratio of image size
    # visual latent features
    features_dist = config.get('features_dist', 'gauss')
    learned_feature_dim = config['learned_feature_dim']
    learned_bg_feature_dim = config.get('learned_bg_feature_dim', learned_feature_dim)
    n_fg_categories = config.get('n_fg_categories', 8)  # Number of foreground feature categories (if categorical)
    n_fg_classes = config.get('n_fg_classes', 4)  # Number of foreground feature classes per category
    n_bg_categories = config.get('n_bg_categories', 4)  # Number of background feature categories
    n_bg_classes = config.get('n_bg_classes', 4)
    # latent context
    context_dist = config.get('context_dist', 'gauss')
    context_dim = config['context_dim']
    ctx_pool_mode = config.get("ctx_pool_mode", "none")
    n_ctx_categories = config.get('n_ctx_categories', 8)  # Number of context feature categories (if categorical)
    n_ctx_classes = config.get('n_ctx_classes', 4)  # Number of context feature classes per category
    dropout = config['dropout']
    use_resblock = config['use_resblock']
    # priors
    scale_std = config['scale_std']
    offset_std = config['offset_std']
    obj_on_alpha = config['obj_on_alpha']  # transparency beta distribution "a"
    obj_on_beta = config['obj_on_beta']  # transparency beta distribution "b"
    # transformer - PINT
    pint_enc_layers = config['pint_enc_layers']
    pint_enc_heads = config['pint_enc_heads']
    pint_ctx_layers = config['pint_ctx_layers']
    pint_ctx_heads = config['pint_ctx_heads']
    pint_dyn_layers = config['pint_dyn_layers']
    pint_dyn_heads = config['pint_dyn_heads']
    pint_dim = config['pint_dim']

    predict_delta = config['predict_delta']  # dynamics module predicts the delta from previous step

    normalize_rgb = config['normalize_rgb']
    obj_res_from_fc = config["obj_res_from_fc"]
    obj_ch_mult = config["obj_ch_mult"]
    obj_ch_mult_prior = config.get("obj_ch_mult_prior", obj_ch_mult)
    obj_base_ch = config["obj_base_ch"]
    obj_final_cnn_ch = config["obj_final_cnn_ch"]
    bg_res_from_fc = config["bg_res_from_fc"]
    bg_ch_mult = config["bg_ch_mult"]
    bg_base_ch = config["bg_base_ch"]
    bg_final_cnn_ch = config["bg_final_cnn_ch"]
    num_res_blocks = config["num_res_blocks"]
    cnn_mid_blocks = config.get('cnn_mid_blocks', False)
    mlp_hidden_dim = config.get('mlp_hidden_dim', 256)

    # actions
    action_condition = config.get('action_condition', False)
    action_dim = config.get('action_dim', 0)
    null_action_embed = config.get('null_action_embed', False)

    random_action_condition = config.get('random_action_condition', False)
    random_action_dim = config.get('random_action_dim', 0)

    # language
    language_condition = config.get('language_condition', False)
    language_embed_dim = config.get('language_embed_dim', 0)
    language_max_len = config.get('language_max_len', 32)

    # image goal condition
    img_goal_condition = config.get('image_goal_condition', False)

    # model
    model = DLP(cdim=ch,  # Number of input image channels
                image_size=image_size,  # Input image size (assumed square)
                normalize_rgb=normalize_rgb,  # If True, normalize RGB to [-1, 1], else keep [0, 1]
                n_views=n_views,  # number of input views (e.g., multiple cameras)

                # Keypoint and patch configuration
                n_kp_per_patch=n_kp_per_patch,  # Number of proposal/prior keypoints to extract per patch
                patch_size=patch_size,  # Size of patches for keypoint proposal network
                anchor_s=anchor_s,  # Glimpse size ratio relative to image size
                n_kp_enc=n_kp_enc,  # Number of posterior keypoints to learn
                n_kp_prior=n_kp_prior,  # Number of keypoints to filter from prior proposals

                # Network configuration
                pad_mode=pad_mode,  # Padding mode for CNNs ('zeros' or 'replicate')
                dropout=dropout,  # Dropout rate for transformers

                # Feature representation
                features_dist=features_dist,  # Distribution type for features ('gauss' or 'categorical')
                learned_feature_dim=learned_feature_dim,  # Dimension of learned visual features
                learned_bg_feature_dim=learned_bg_feature_dim,
                # Background feature dimension (if None, equals learned_feature_dim)
                n_fg_categories=n_fg_categories,  # Number of foreground feature categories (if categorical)
                n_fg_classes=n_fg_classes,  # Number of foreground feature classes per category
                n_bg_categories=n_bg_categories,  # Number of background feature categories
                n_bg_classes=n_bg_classes,  # Number of background feature classes per category

                # Prior distributions parameters
                scale_std=scale_std,  # Prior standard deviation for scale
                offset_std=offset_std,  # Prior standard deviation for offset
                obj_on_alpha=obj_on_alpha,  # Alpha parameter for transparency Beta distribution
                obj_on_beta=obj_on_beta,  # Beta parameter for transparency Beta distribution

                # Object decoder architecture
                obj_res_from_fc=obj_res_from_fc,  # Initial resolution for object encoder-decoder
                obj_ch_mult_prior=obj_ch_mult_prior,  # Channel multipliers for prior patch encoder (kp proposals)
                obj_ch_mult=obj_ch_mult,  # Channel multipliers for object encoder-decoder
                obj_base_ch=obj_base_ch,  # Base channels for object encoder-decoder
                obj_final_cnn_ch=obj_final_cnn_ch,  # Final CNN channels for object encoder-decoder

                # Background decoder architecture
                bg_res_from_fc=bg_res_from_fc,  # Initial resolution for background encoder-decoder
                bg_ch_mult=bg_ch_mult,  # Channel multipliers for background encoder-decoder
                bg_base_ch=bg_base_ch,  # Base channels for background encoder-decoder
                bg_final_cnn_ch=bg_final_cnn_ch,  # Final CNN channels for background encoder-decoder

                # Network architecture options
                use_resblock=use_resblock,  # Use residual blocks in encoders-decoders
                num_res_blocks=num_res_blocks,  # Number of residual blocks per resolution
                cnn_mid_blocks=cnn_mid_blocks,  # Use middle blocks in CNN
                mlp_hidden_dim=mlp_hidden_dim,  # Hidden dimension for MLPs

                # Particle interaction transformer (PINT) configuration
                pint_enc_layers=pint_enc_layers,  # Number of PINT encoder layers
                pint_enc_heads=pint_enc_heads,  # Number of PINT encoder attention heads

                # Dynamics configuration
                timestep_horizon=timestep_horizon,  # Number of timesteps to predict ahead
                n_static_frames=num_static_frames,  # Number of initial frames for static KL optimization
                predict_delta=predict_delta,  # Predict position deltas instead of absolute positions
                context_dim=context_dim,  # Context latent dimension (if None, equals learned_feature_dim)
                ctx_dist=context_dist,  # Context distribution type ('gauss' or 'categorical')
                n_ctx_categories=n_ctx_categories,  # Number of context categories (if categorical)
                n_ctx_classes=n_ctx_classes,  # Number of context classes per category
                ctx_pool_mode=ctx_pool_mode,  # Context pooling mode ('none' = per-particle context)

                # Context and dynamics transformer configuration
                pint_dyn_layers=pint_dyn_layers,  # Number of dynamics transformer layers
                pint_dyn_heads=pint_dyn_heads,  # Number of dynamics transformer heads
                pint_dim=pint_dim,  # Hidden dimension for PINT
                pint_ctx_layers=pint_ctx_layers,  # Number of context transformer layers
                pint_ctx_heads=pint_ctx_heads,

                # external conditioning
                action_condition=action_condition,  # condition on actions
                action_dim=action_dim,  # dimension of input actions
                null_action_embed=null_action_embed,
                random_action_condition=random_action_condition,
                random_action_dim=random_action_dim,
                # learn a "no-input-action" embedding, to learn on action-free videos as well
                language_condition=language_condition,  # condition on language embedding
                language_embed_dim=language_embed_dim,  # embedding dimension for each token
                language_max_len=language_max_len,  # maximum tokens per prompt
                img_goal_condition=img_goal_condition,  # condition the future on image goal
                )
    if ckpt_path is not None:
        try:
            model.load_state_dict(torch.load(ckpt_path, map_location=torch.device('cpu'), weights_only=False))
            print("loaded dlp model from checkpoint")
        except:
            print("dlp model checkpoint not found")

    return model


def eval_lpwm_fvd(model, device, config, timestep_horizon=16, val_mode='test', eval_dir='./',
                  cond_steps=1, use_all_ctx=False, batch_size=10, accelerator=None,
                  deterministic=False, n_videos_per_clip=1):
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    model.eval()
    ds = config['ds']
    ch = config['ch']  # image channels
    image_size = config['image_size']
    n_views = config.get('n_views', 1)
    root = config['root']  # dataset root
    action_condition = config.get('action_condition', False)
    language_condition = config.get('language_condition', False)
    img_goal_condition = config.get('image_goal_condition', False)

    dataset = get_video_dataset(ds, root, seq_len=timestep_horizon, mode=val_mode, image_size=image_size)

    dataloader = DataLoader(dataset, shuffle=False, batch_size=batch_size, num_workers=0, drop_last=False)
    model_timestep_horizon = model.timestep_horizon
    cond_steps = model_timestep_horizon if cond_steps is None else cond_steps

    if accelerator is not None and not accelerator.is_main_process:
        disable_pbar = True
    else:
        disable_pbar = False

    # pixel value should be in [0, 1]!
    # real videos
    real_videos = []
    for i, batch in enumerate(tqdm(dataloader, disable=disable_pbar)):
        x = batch[0][:, :timestep_horizon]
        real_videos.append(x)
    real_videos = torch.cat(real_videos, dim=0)  # [n_real_videos, ts, ch, h, w]
    real_videos = real_videos.repeat(n_videos_per_clip, 1, 1, 1, 1)
    print(f'real videos: {real_videos.shape}')

    # generate videos
    generated_videos = []
    for k in range(n_videos_per_clip):
        for i, batch in enumerate(tqdm(dataloader, disable=disable_pbar)):
            x = batch[0][:, :timestep_horizon].to(device)
            actions = None if not action_condition else batch[1][:, :timestep_horizon].to(device)
            lang_str = None if not language_condition else batch[2]
            lang_embed = None if not language_condition else batch[3].to(device).to(device)
            x_goal = None if not img_goal_condition else batch[3].to(device)
            if n_views > 1:
                # expect: [bs, T, n_views, ...]
                x = x.permute(0, 2, 1, 3, 4, 5)
                x = x.reshape(-1, *x.shape[2:])  # [bs * n_views, T, ...]
                if x_goal is not None:
                    x_goal = x_goal.reshape(-1, *x_goal.shape[2:])  # [bs * n_views, ...]
                if actions is not None:
                    actions = actions.permute(0, 2, 1, 3)
                    actions = actions.reshape(-1, *actions.shape[2:])
            with torch.no_grad():
                generated = model.sample_from_x(x, cond_steps=cond_steps, num_steps=timestep_horizon - cond_steps,
                                                use_all_ctx=use_all_ctx, actions=actions, lang_embed=lang_embed,
                                                x_goal=x_goal, deterministic=deterministic)
                generated = generated.clamp(0, 1)
                assert x.shape[1] == generated.shape[1], "prediction and gt frames shape don't match"
            generated_videos.append(generated.cpu())
    generated_videos = torch.cat(generated_videos, dim=0)
    print(f'generated videos: {generated_videos.shape}')

    fvd = calculate_fvd(real_videos, generated_videos, device, method='styleganv', only_final=True)
    results = {'fvd': fvd}

    # save results
    if accelerator is not None:
        save_metric = accelerator.is_main_process
    else:
        save_metric = True
    if save_metric:
        path_to_conf = os.path.join(eval_dir, f'last_{val_mode}_fvd.json')
        with open(path_to_conf, "w") as outfile:
            json.dump(results, outfile, indent=2)

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="LPWM Video Prediction Evaluation")
    parser.add_argument("-d", "--dataset", type=str, default='balls',
                        help="dataset to use: ['balls', 'traffic', 'clevrer', 'obj3d128', ...]")
    parser.add_argument("-p", "--path", type=str,
                        help="path to model directory, e.g. ./310822_141959_balls_ddlp")
    parser.add_argument("--checkpoint", type=str,
                        help="direct path to model checkpoint, e.g. ./checkpoints/ddlp-obj3d128/obj3d_ddlp.pth",
                        default="")
    parser.add_argument("--use_last", action='store_true',
                        help="use the last checkpoint instead of best")
    parser.add_argument("--use_train", action='store_true',
                        help="use the train set for the predictions")
    parser.add_argument("--sample", action='store_true',
                        help="use stochastic (non-deterministic) predictions")
    parser.add_argument("--cpu", action='store_true',
                        help="use cpu for inference")
    parser.add_argument("--ctx", action='store_true',
                        help="use context for inference")
    parser.add_argument("-c", "--cond_steps", type=int, help="the initial number of frames for predictions", default=-1)
    parser.add_argument("-b", "--batch_size", type=int, help="batch size", default=10)
    parser.add_argument("--horizon", type=int, help="timestep horizon for prediction", default=50)
    parser.add_argument("--prefix", type=str, default='',
                        help="prefix used for model saving")
    parser.add_argument("--n_videos_per_clip", type=int, help="n_videos to generate per data sample", default=1)
    args = parser.parse_args()
    # parse input
    dir_path = args.path
    checkpoint_path = args.checkpoint
    # ds = args.dataset
    use_train = args.use_train
    cond_steps = args.cond_steps
    timestep_horizon = args.horizon
    batch_size = args.batch_size
    use_cpu = args.cpu
    use_ctx = args.ctx
    deterministic = not args.sample
    prefix = args.prefix
    n_videos_per_clip = args.n_videos_per_clip
    # load model config
    pref = 'gddlp'

    conf_path = os.path.join(dir_path, 'hparams.json')
    with open(conf_path, 'r') as f:
        config = json.load(f)
    if use_cpu:
        device = torch.device("cpu")
    else:
        device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

    ds = config['ds']

    model_ckpt_name = f'{ds}_{pref}{prefix}.pth'
    # model_best_ckpt_name = f'{ds}_ddlp{prefix}_best.pth'
    model_best_ckpt_name = f'{ds}_{pref}{prefix}_best_lpips.pth'
    use_last = args.use_last if os.path.exists(os.path.join(dir_path, f'saves/{model_best_ckpt_name}')) else True

    if checkpoint_path.endswith('.pth'):
        ckpt_path = checkpoint_path
    else:
        ckpt_path = os.path.join(dir_path, f'saves/{model_ckpt_name if use_last else model_best_ckpt_name}')

    print(f'checkpoint path: {ckpt_path}')

    model = load_dlp_from_config(conf_path, ckpt_path)
    model = model.to(device)
    model.eval()

    # create dir for results
    pred_dir = os.path.join(dir_path, 'eval')
    os.makedirs(pred_dir, exist_ok=True)

    # conditional frames
    cond_steps = cond_steps if cond_steps > 0 else config['timestep_horizon']
    val_mode = 'train' if use_train else 'test'
    results = eval_lpwm_fvd(model, device, config=config, timestep_horizon=timestep_horizon, val_mode=val_mode,
                            eval_dir=pred_dir,
                            cond_steps=cond_steps, use_all_ctx=use_ctx, batch_size=batch_size, accelerator=None,
                            deterministic=deterministic, n_videos_per_clip=n_videos_per_clip)
    print(f'results: {results}')
