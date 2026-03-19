"""
Single-GPU training of LPWM
"""
# imports
import numpy as np
import os
from tqdm import tqdm
import matplotlib
import argparse
# torch
import torch
from utils.loss_functions import calc_reconstruction_loss, LossLPIPS
from torch.utils.data import DataLoader
import torchvision.utils as vutils
import torch.optim as optim
# modules
from models import DLP
# datasets
from datasets.get_dataset import get_video_dataset
# util functions
from utils.util_func import plot_keypoints_on_image_batch, prepare_logdir, save_config, log_line, \
    plot_bb_on_image_batch_from_z_scale_nms, plot_bb_on_image_batch_from_masks_nms, create_segmentation_map, get_config, \
    LinearWithWarmupScheduler, format_epoch_summary, plot_training_metrics, save_metrics_data, save_code_backup
from eval.eval_model import evaluate_validation_elbo_dyn, animate_trajectory_lpwm
from eval.eval_gen_metrics import eval_lpwm_im_metric

matplotlib.use("Agg")
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


def train_ddlp(config_path='./configs/balls.json'):
    # load config
    try:
        config = get_config(config_path)
    except FileNotFoundError:
        raise SystemExit("config file not found")
    hparams = config  # to save a copy of the hyper-parameters
    device = config['device']
    if 'cuda' in device:
        device = torch.device(f'{device}' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')

    # data and general
    ds = config['ds']
    ch = config['ch']  # image channels
    image_size = config['image_size']
    n_views = config.get('n_views', 1)
    root = config['root']  # dataset root
    run_prefix = config['run_prefix']
    load_model = config['load_model']
    pretrained_path = config['pretrained_path']  # path of pretrained model to load, if None, train from scratch

    # model
    timestep_horizon = config['timestep_horizon']
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

    # optimization
    batch_size = config['batch_size']
    lr = config['lr']
    num_epochs = config['num_epochs']
    start_epoch = config.get('start_epoch', 0)
    weight_decay = config['weight_decay']
    adam_betas = config['adam_betas']
    adam_eps = config['adam_eps']
    use_scheduler = config['use_scheduler']
    scheduler_gamma = config['scheduler_gamma']
    warmup_epoch = config['warmup_epoch']
    recon_loss_type = config['recon_loss_type']
    beta_kl = config['beta_kl']
    beta_dyn = config['beta_dyn']
    beta_rec = config['beta_rec']
    beta_dyn_rec = config['beta_dyn_rec']
    beta_obj = config.get('beta_obj', 0.0)  # beta_reg in the paper
    kl_balance = config['kl_balance']  # balance between visual features and the other particle attributes
    num_static_frames = config['num_static_frames']  # frames for which kl is calculated w.r.t constant prior params

    # priors
    scale_std = config['scale_std']
    offset_std = config['offset_std']
    obj_on_alpha = config['obj_on_alpha']  # transparency beta distribution "a"
    obj_on_beta = config['obj_on_beta']  # transparency beta distribution "b"

    # evaluation
    eval_epoch_freq = config['eval_epoch_freq']
    eval_im_metrics = config['eval_im_metrics']
    cond_steps = config['cond_steps']  # conditional frames for the dynamics module during inference
    ctx_for_eval = config.get('ctx_for_eval', False)

    # visualization
    iou_thresh = config['iou_thresh']  # threshold for NMS for plotting bounding boxes
    topk = min(config['topk'], config['n_kp_enc'])  # top-k particles to plot
    animation_horizon = config['animation_horizon']

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
    use_ep_done_mask = config.get('ep_done_mask', False)  # original
    # use_ep_done_mask = config.get('use_ep_done_mask', False)  # correct

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

    # load data
    dataset = get_video_dataset(ds, root, seq_len=timestep_horizon + 1, mode='train', image_size=image_size)
    dataloader = DataLoader(dataset, shuffle=True, batch_size=batch_size, num_workers=4, pin_memory=True,
                            drop_last=True)
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
                ).to(device)
    model_info = model.info()
    print(model.info())

    # prepare saving location
    run_name = f'{ds}_gddlp' + run_prefix
    log_dir = prepare_logdir(runname=run_name, src_dir='./')
    fig_dir = os.path.join(log_dir, 'figures')
    save_dir = os.path.join(log_dir, 'saves')
    save_config(log_dir, hparams)
    log_line(log_dir, model_info)
    # save a backup of the code for this run
    backup_info = save_code_backup('.', backup_dir=os.path.join(log_dir, 'saves', 'code_backup'))
    log_line(log_dir, backup_info)
    print(backup_info)

    # get the range of the keypoints, it is [-1, 1] by default
    kp_range = model.kp_range
    # prepare loss functions
    if recon_loss_type == "vgg":
        recon_loss_func = LossLPIPS(normalized_rgb=normalize_rgb).to(device)
    else:
        recon_loss_func = calc_reconstruction_loss

    # optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=adam_betas, eps=adam_eps, weight_decay=weight_decay)
    # accelerate baking
    if use_scheduler:
        scheduler = LinearWithWarmupScheduler(optimizer, gamma=scheduler_gamma, verbose=False,
                                              steps=(max(warmup_epoch, 1), max(warmup_epoch, 1) + 1),
                                              factors=(1.0, 1.0, 1.0 * scheduler_gamma))
    else:
        scheduler = None

    if load_model and pretrained_path is not None:
        try:
            model.load_state_dict(torch.load(pretrained_path, map_location=device, weights_only=False))
            print("loaded model from checkpoint")
        except:
            print("model checkpoint not found")

    # log statistics
    losses = []
    losses_rec = []
    losses_kl = []
    losses_kl_kp = []
    losses_kl_feat = []
    losses_kl_dyn = []
    losses_kl_scale = []
    losses_kl_depth = []
    losses_kl_obj_on = []
    losses_kl_context = []

    # initialize validation statistics
    valid_loss = best_valid_loss = 1e8
    valid_losses = []
    best_valid_epoch = 0

    # save PSNR values of the reconstruction
    psnrs = []

    # image metrics
    if eval_im_metrics:
        val_lpipss = []
        best_val_lpips_epoch = 0
        val_lpips = best_val_lpips = 1e8

    # iteration counter for discounting, optional
    iter_per_epoch = 1 * len(dataloader)
    iteration = 0  # initialize iterations counter
    warmup_iteration = 0
    max_warmup_iterations = int(0.8 * iter_per_epoch)

    for epoch in range(start_epoch, num_epochs):
        model.train()
        batch_losses = []
        batch_losses_rec = []
        batch_losses_kl = []
        batch_losses_kl_kp = []
        batch_losses_kl_feat = []
        batch_losses_kl_dyn = []
        batch_losses_kl_scale = []
        batch_losses_kl_depth = []
        batch_losses_kl_obj_on = []
        batch_losses_kl_context = []
        batch_psnrs = []

        pbar = tqdm(iterable=dataloader)
        for batch in pbar:
            x = batch[0].to(device)
            actions = None if not action_condition else batch[1].to(device)
            lang_str = None if not language_condition else batch[2]
            lang_embed = None if not language_condition else batch[3].to(device)
            ep_done_mask = None if not use_ep_done_mask else batch[-1].to(device)
            x_goal = None if not img_goal_condition else batch[3].to(device)
            warmup = (epoch < warmup_epoch)
            discount = None
            if n_views > 1:
                # expect: [bs, T, n_views, ...]
                x = x.permute(0, 2, 1, 3, 4, 5)
                x = x.reshape(-1, *x.shape[2:])  # [bs * n_views, T, ...]
                if x_goal is not None:
                    x_goal = x_goal.reshape(-1, *x_goal.shape[2:]) # [bs * n_views, ...]
                if actions is not None:
                    actions = actions.permute(0, 2, 1, 3)
                    actions = actions.reshape(-1, *actions.shape[2:])
                if ep_done_mask is not None:
                    ep_done_mask = ep_done_mask.permute(0, 2, 1)
                    ep_done_mask - ep_done_mask.reshape(-1, *ep_done_mask.shape[2:])
            model_output = model(x, actions=actions, lang_embed=lang_embed, warmup=warmup, with_loss=True,
                                 beta_kl=beta_kl,
                                 beta_dyn=beta_dyn, beta_rec=beta_rec, kl_balance=kl_balance,
                                 dynamic_discount=discount, recon_loss_type=recon_loss_type,
                                 recon_loss_func=recon_loss_func, beta_dyn_rec=beta_dyn_rec, beta_obj=beta_obj,
                                 done_mask=ep_done_mask, x_goal=x_goal)
            # calculate loss
            all_losses = model_output['loss_dict']
            iteration += 1

            loss = all_losses['loss']
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # output for logging and plotting
            mu_p = model_output['kp_p']
            z_base = model_output['z_base']
            mu_offset = model_output['mu_offset']
            logvar_offset = model_output['logvar_offset']
            rec_x = model_output['rec_rgb']
            mu_scale = model_output['mu_scale']
            mu_depth = model_output['mu_depth']
            # object stuff
            dec_objects_original = model_output['dec_objects_original']
            cropped_objects_original = model_output['cropped_objects_original']
            obj_on = model_output['obj_on']  # [batch_size, n_kp]
            alpha_masks = model_output['alpha_masks']  # [batch_size, n_kp, 1, h, w]

            psnr = all_losses['psnr']
            obj_on_l1 = all_losses['obj_on_l1']

            loss_kl = all_losses['kl']
            loss_kl_dyn = all_losses['kl_dyn']
            loss_rec = all_losses['loss_rec']
            loss_kl_kp = all_losses['loss_kl_kp']
            loss_kl_feat = all_losses['loss_kl_feat']
            loss_kl_scale = all_losses['loss_kl_scale']
            loss_kl_depth = all_losses['loss_kl_depth']
            loss_kl_obj_on = all_losses['loss_kl_obj_on']
            loss_kl_context = all_losses['loss_kl_context']

            # for plotting, confidence calculation
            mu_tot = z_base + mu_offset
            mu_tot = mu_tot.view(-1, *mu_tot.shape[2:])
            logvar_tot = logvar_offset
            logvar_tot = logvar_tot.view(-1, *logvar_tot.shape[2:])

            # for progress bar
            a_mean = model_output['obj_on_a'].mean()  # the mean value of the "a" param in transparency Beta(a,b) dist
            b_mean = model_output['obj_on_b'].mean()  # the mean value of the "b" param in transparency Beta(a,b) dist
            mu_scale_mean = torch.sigmoid(model_output['mu_scale']).mean()  # the mean bounding-box size

            # log
            batch_psnrs.append(psnr.data.cpu().item())
            batch_losses.append(loss.data.cpu().item())
            batch_losses_rec.append(loss_rec.data.cpu().item())
            batch_losses_kl.append(loss_kl.data.cpu().item())
            batch_losses_kl_kp.append(loss_kl_kp.data.cpu().item())
            batch_losses_kl_feat.append(loss_kl_feat.data.cpu().item())
            batch_losses_kl_dyn.append(loss_kl_dyn.data.cpu().item())
            batch_losses_kl_scale.append(loss_kl_scale.data.cpu().item())
            batch_losses_kl_depth.append(loss_kl_depth.data.cpu().item())
            batch_losses_kl_obj_on.append(loss_kl_obj_on.data.cpu().item())
            batch_losses_kl_context.append(loss_kl_context.data.cpu().item())
            # progress bar
            if epoch < warmup_epoch:
                pbar.set_description_str(f'epoch #{epoch} (warmup)')
            else:
                pbar.set_description_str(f'epoch #{epoch}')
            pbar.set_postfix(loss=loss.data.cpu().item(), rec=loss_rec.data.cpu().item(),
                             kl=loss_kl.data.cpu().item(), on_l1=obj_on_l1.cpu().item(),
                             kl_dyn=loss_kl_dyn.data.cpu().item(),
                             a=a_mean.data.cpu().item(), b=b_mean.data.cpu().item(),
                             smu=mu_scale_mean.data.cpu().item())
            if warmup:
                warmup_iteration += 1
                if warmup_iteration > max_warmup_iterations:
                    warmup_iteration = 0
                    break
            # break  # for debug
        pbar.close()
        losses.append(np.mean(batch_losses))
        losses_rec.append(np.mean(batch_losses_rec))
        losses_kl.append(np.mean(batch_losses_kl))
        losses_kl_kp.append(np.mean(batch_losses_kl_kp))
        losses_kl_feat.append(np.mean(batch_losses_kl_feat))
        losses_kl_dyn.append(np.mean(batch_losses_kl_dyn))
        losses_kl_scale.append(np.mean(batch_losses_kl_scale))
        losses_kl_depth.append(np.mean(batch_losses_kl_depth))
        losses_kl_obj_on.append(np.mean(batch_losses_kl_obj_on))
        losses_kl_context.append(np.mean(batch_losses_kl_context))
        if len(batch_psnrs) > 0:
            psnrs.append(np.mean(batch_psnrs))
        # scheduler
        if use_scheduler:
            scheduler.step()
            curr_lr = scheduler.get_lr()
            lr_str = f'learning rate: {curr_lr}'
            print(curr_lr)
            log_line(log_dir, lr_str)

        # epoch summary
        log_str = format_epoch_summary(
            epoch=epoch,
            loss=losses[-1],
            loss_rec=losses_rec[-1],
            loss_kl=losses_kl[-1],
            kl_balance=kl_balance,
            loss_kl_kp=losses_kl_kp[-1],
            loss_kl_feat=losses_kl_feat[-1],
            loss_kl_scale=losses_kl_scale[-1],
            loss_kl_depth=losses_kl_depth[-1],
            loss_kl_obj_on=losses_kl_obj_on[-1],
            mu_tot=mu_tot,
            mu_offset=mu_offset,
            valid_loss=valid_loss,
            best_valid_loss=best_valid_loss,
            best_valid_epoch=best_valid_epoch,
            obj_on=obj_on,
            mu_scale=mu_scale,
            mu_depth=mu_depth,
            eval_epoch_freq=eval_epoch_freq,
            val_lpips=val_lpips if eval_im_metrics else None,
            best_val_lpips=best_val_lpips if eval_im_metrics else None,
            best_val_lpips_epoch=best_val_lpips_epoch if eval_im_metrics else None,
            psnr=psnrs[-1] if len(psnrs) > 0 else None,
            loss_kl_dyn=losses_kl_dyn[-1],
            loss_kl_context=losses_kl_context[-1]
        )
        print(log_str)
        log_line(log_dir, log_str)

        if epoch % eval_epoch_freq == 0 or epoch == num_epochs - 1:
            x = x.view(-1, *x.shape[2:])
            # for plotting purposes
            mu_plot = mu_tot.clamp(min=kp_range[0], max=kp_range[1])
            max_imgs = 8
            img_with_kp = plot_keypoints_on_image_batch(mu_plot, x, radius=3,
                                                        thickness=1, max_imgs=max_imgs, kp_range=kp_range)
            img_with_kp_p = plot_keypoints_on_image_batch(mu_p, x, radius=3, thickness=1, max_imgs=max_imgs,
                                                          kp_range=kp_range)
            # top-k
            with torch.no_grad():
                z_base_var = model_output['z_base_var']
                z_base_var = z_base_var.view(-1, *z_base_var.shape[2:])
                logvar_sum = z_base_var.sum(-1) * obj_on.view(-1, *obj_on.shape[2:]).squeeze(-1)  # [bs, n_kp]
                logvar_topk = torch.topk(logvar_sum, k=topk, dim=-1, largest=False)
                indices = logvar_topk[1]  # [batch_size, topk]
                batch_indices = torch.arange(mu_tot.shape[0]).view(-1, 1).to(mu_tot.device)
                topk_kp = mu_tot[batch_indices, indices]
                # bounding boxes
                bb_scores = -1 * logvar_sum
                hard_threshold = None

            kp_batch = mu_plot
            scale_batch = mu_scale.view(-1, *mu_scale.shape[2:])
            img_with_masks_nms, nms_ind = plot_bb_on_image_batch_from_z_scale_nms(kp_batch, scale_batch, x,
                                                                                  scores=bb_scores,
                                                                                  iou_thresh=iou_thresh,
                                                                                  thickness=1, max_imgs=max_imgs,
                                                                                  hard_thresh=hard_threshold)
            alpha_masks = torch.where(alpha_masks < 0.05, 0.0, 1.0)
            if alpha_masks.shape[1] != bb_scores.shape[1]:
                bb_scores = -1 * torch.topk(logvar_sum, k=alpha_masks.shape[1], dim=-1, largest=False)[0]
            img_with_masks_alpha_nms, _ = plot_bb_on_image_batch_from_masks_nms(alpha_masks, x, scores=bb_scores,
                                                                                iou_thresh=iou_thresh, thickness=1,
                                                                                max_imgs=max_imgs,
                                                                                hard_thresh=hard_threshold)
            img_with_seg_maps = create_segmentation_map(x=x, masks=alpha_masks, scores=bb_scores, alpha=0.7)
            # hard_thresh: a general threshold for bb scores (set None to not use it)
            bb_str = f'\nbb scores: max: {bb_scores.max():.2f}, min: {bb_scores.min():.2f},' \
                     f' mean: {bb_scores.mean():.2f}\n'
            print(bb_str)
            log_line(log_dir, bb_str)
            img_with_kp_topk = plot_keypoints_on_image_batch(topk_kp.clamp(min=kp_range[0], max=kp_range[1]), x,
                                                             radius=3, thickness=1, max_imgs=max_imgs,
                                                             kp_range=kp_range)
            dec_objects = model_output['dec_objects']
            bg = model_output['bg_rgb']
            vutils.save_image(torch.cat([x[:max_imgs, -3:], img_with_kp[:max_imgs, -3:].to(device),
                                         rec_x[:max_imgs, -3:], img_with_kp_p[:max_imgs, -3:].to(device),
                                         img_with_kp_topk[:max_imgs, -3:].to(device),
                                         dec_objects[:max_imgs, -3:],
                                         img_with_masks_nms[:max_imgs, -3:].to(device),
                                         img_with_masks_alpha_nms[:max_imgs, -3:].to(device),
                                         img_with_seg_maps[:max_imgs, -3:],
                                         bg[:max_imgs, -3:]],
                                        dim=0).data.cpu(), '{}/image_{}.jpg'.format(fig_dir, epoch),
                              nrow=8, pad_value=1)

            torch.save(model.state_dict(), os.path.join(save_dir, f'{ds}_gddlp{run_prefix}.pth'))
            animate_trajectory_lpwm(model, config, epoch, device=device, fig_dir=fig_dir,
                                    timestep_horizon=animation_horizon, num_trajetories=1,
                                    train=True, cond_steps=cond_steps)
            print("validation step...")
            valid_loss = evaluate_validation_elbo_dyn(model, config, epoch, batch_size=batch_size,
                                                      recon_loss_type=recon_loss_type, device=device,
                                                      save_image=True, fig_dir=fig_dir, topk=topk,
                                                      recon_loss_func=recon_loss_func, beta_rec=beta_rec,
                                                      beta_dyn=beta_dyn, iou_thresh=iou_thresh,
                                                      timestep_horizon=timestep_horizon, beta_dyn_rec=beta_dyn_rec,
                                                      beta_kl=beta_kl, kl_balance=kl_balance,
                                                      animation_horizon=animation_horizon, beta_obj=beta_obj)
            log_str = f'validation loss: {valid_loss:.3f}\n'
            print(log_str)
            log_line(log_dir, log_str)
            if best_valid_loss > valid_loss:
                log_str = f'validation loss updated: {best_valid_loss:.3f} -> {valid_loss:.3f}\n'
                print(log_str)
                log_line(log_dir, log_str)
                best_valid_loss = valid_loss
                best_valid_epoch = epoch
                torch.save(model.state_dict(),
                           os.path.join(save_dir,
                                        f'{ds}_gddlp{run_prefix}_best.pth'))
            torch.cuda.empty_cache()
            if eval_im_metrics and epoch > 0:
                valid_imm_results = eval_lpwm_im_metric(model, device, config,
                                                        timestep_horizon=animation_horizon, val_mode='val',
                                                        eval_dir=log_dir, use_all_ctx=ctx_for_eval,
                                                        cond_steps=cond_steps, batch_size=batch_size)
                log_str = f'validation: lpips: {valid_imm_results["lpips"]:.3f}, '
                log_str += f'psnr: {valid_imm_results["psnr"]:.3f}, ssim: {valid_imm_results["ssim"]:.3f}\n'
                val_lpips = valid_imm_results['lpips']
                print(log_str)
                log_line(log_dir, log_str)
                if (not torch.isinf(torch.tensor(val_lpips))) and (best_val_lpips > val_lpips):
                    log_str = f'validation lpips updated: {best_val_lpips:.3f} -> {val_lpips:.3f}\n'
                    print(log_str)
                    log_line(log_dir, log_str)
                    best_val_lpips = val_lpips
                    best_val_lpips_epoch = epoch
                    torch.save(model.state_dict(),
                               os.path.join(save_dir, f'{ds}_gddlp{run_prefix}_best_lpips.pth'))
                torch.cuda.empty_cache()
        valid_losses.append(valid_loss)
        if eval_im_metrics:
            val_lpipss.append(val_lpips)
        # plot graphs
        if epoch > start_epoch:
            metrics_data = [
                (losses[1:], "Total Loss", "#2d72bc", True),
                (losses_kl[1:], "KL Loss", "#c92a2a", True),
                (losses_rec[1:], "Reconstruction Loss", "#087f5b", True),
                (valid_losses[1:], "Validation Loss", "#862e9c", True),
            ]
            save_metrics_data(metrics_data, run_name, save_dir=os.path.join(save_dir, 'metrics'))
            plot_training_metrics(metrics_data, run_name, fig_dir, max_plots_per_figure=4)
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LPWM Single-GPU Training")
    parser.add_argument("-d", "--dataset", type=str, default='balls_occlusion',
                        help="dataset of to train the model on: ['traffic', 'clevrer', 'obj3d128', 'phyre']")
    args = parser.parse_args()
    ds = args.dataset
    if ds.endswith('json'):
        conf_path = ds
    else:
        conf_path = os.path.join('./configs', f'{ds}.json')

    train_ddlp(conf_path)
