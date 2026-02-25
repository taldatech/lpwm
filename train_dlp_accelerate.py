"""
Main DLPv3 training function for multi-GPU machines.
We use HuggingFace Accelerate: https://huggingface.co/docs/accelerate/index
1. Set visible GPUs under: `os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3"`
2. Set "num_processes": NUM_GPUS in `accel_conf.json/yaml`
NUM_GPUS = len(GPUS)
"""
GPUS = ["0", "1", "2", "3"]
# GPUS = ["0", "1", "2", "3", "4", "5", "6", "7"]
# imports
import os

# os.environ["NCCL_P2P_LEVEL"] = "NVL"  # uncomment if torch-distributed/accelerate hangs because old linux kernel
os.environ["CUDA_VISIBLE_DEVICES"] = ", ".join(GPUS)  # "0, 1, 2, 3"
# imports
import numpy as np
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
from datasets.get_dataset import get_image_dataset
# util functions
from utils.util_func import (plot_keypoints_on_image_batch, prepare_logdir, save_config, log_line,
                             plot_bb_on_image_batch_from_z_scale_nms, plot_bb_on_image_batch_from_masks_nms,
                             create_segmentation_map, get_config, LinearWithWarmupScheduler, format_epoch_summary,
                             plot_training_metrics, save_metrics_data, save_code_backup)
from eval.eval_model import evaluate_validation_elbo
from eval.eval_gen_metrics import eval_dlp_im_metric
from accelerate import Accelerator, DistributedDataParallelKwargs

matplotlib.use("Agg")
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


def train_dlp(config_path='./configs/shapes.json'):
    # load config
    try:
        config = get_config(config_path)
    except FileNotFoundError:
        raise SystemExit("config file not found")
    find_unused_parameters = False
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=find_unused_parameters)
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
    # in conf: "num_processes": num_visible_gpus
    hparams = config  # to save a copy of the hyper-parameters
    # data and general
    ds = config['ds']
    ch = config['ch']  # image channels
    image_size = config['image_size']
    root = config['root']  # dataset root

    run_prefix = config['run_prefix']
    load_model = config['load_model']
    pretrained_path = config['pretrained_path']  # path of pretrained model to load, if None, train from scratch

    # model
    pad_mode = config['pad_mode']
    n_kp_per_patch = config['n_kp_per_patch']  # kp per patch in prior, best to leave at 1
    n_kp_prior = config['n_kp_prior']  # number of prior kp to filter for the kl
    n_kp_enc = config['n_kp_enc']  # total posterior kp
    patch_size = config['patch_size']  # prior patch size
    anchor_s = config['anchor_s']  # posterior patch/glimpse ratio of image size

    features_dist = config.get('features_dist', 'gauss')
    learned_feature_dim = config['learned_feature_dim']
    learned_bg_feature_dim = config.get('learned_bg_feature_dim', learned_feature_dim)
    n_fg_categories = config.get('n_fg_categories', 8)  # Number of foreground feature categories (if categorical)
    n_fg_classes = config.get('n_fg_classes', 4)  # Number of foreground feature classes per category
    n_bg_categories = config.get('n_bg_categories', 4)  # Number of background feature categories
    n_bg_classes = config.get('n_bg_classes', 4)

    dropout = config['dropout']
    use_resblock = config['use_resblock']

    pint_enc_layers = config['pint_enc_layers']
    pint_enc_heads = config['pint_enc_heads']

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
    beta_rec = config['beta_rec']
    beta_obj = config.get('beta_obj', 0.0)
    kl_balance = config['kl_balance']  # balance between visual features and the other particle attributes

    # priors
    scale_std = config['scale_std']
    offset_std = config['offset_std']
    obj_on_alpha = config['obj_on_alpha']  # transparency beta distribution "a"
    obj_on_beta = config['obj_on_beta']  # transparency beta distribution "b"

    # evaluation
    eval_epoch_freq = config['eval_epoch_freq']
    eval_im_metrics = config['eval_im_metrics']

    # visualization
    topk = min(config['topk'], config['n_kp_enc'])  # top-k particles to plot
    iou_thresh = config['iou_thresh']  # threshold for NMS for plotting bounding boxes

    # load data
    dataset = get_image_dataset(ds, root, mode='train', image_size=image_size)
    dataloader = DataLoader(dataset, shuffle=True, batch_size=batch_size, num_workers=4, pin_memory=True,
                            drop_last=True)
    # model
    model = DLP(
        cdim=ch,  # Number of input image channels
        image_size=image_size,  # Input image size (assumed square)
        normalize_rgb=normalize_rgb,  # If True, normalize RGB to [-1, 1], else keep [0, 1]

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
        timestep_horizon=1)
    model_info = model.info()
    if accelerator.is_main_process:
        print(model_info)
    # prepare saving location
    run_name = f'{ds}_gdlp' + run_prefix
    log_dir = prepare_logdir(runname=run_name, src_dir='./')
    fig_dir = os.path.join(log_dir, 'figures')
    save_dir = os.path.join(log_dir, 'saves')
    if accelerator.is_main_process:
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
        recon_loss_func = LossLPIPS(normalized_rgb=normalize_rgb).to(accelerator.device)
    else:
        recon_loss_func = calc_reconstruction_loss

    # optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=adam_betas, eps=adam_eps, weight_decay=weight_decay)
    # accelerate baking
    # verbose = accelerator.is_local_main_process
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
    if use_scheduler:
        scheduler = LinearWithWarmupScheduler(optimizer, gamma=scheduler_gamma, verbose=False,
                                              steps=(max(warmup_epoch, 1), max(warmup_epoch, 1) + 1),
                                              factors=(1.0, 1.0, 1.0 * scheduler_gamma))
    else:
        scheduler = None

    if load_model and pretrained_path is not None:
        try:
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.load_state_dict(torch.load(pretrained_path, map_location=accelerator.device))
            print("loaded model from checkpoint")
        except:
            print("model checkpoint not found")

    # log statistics
    losses = []
    losses_rec = []
    losses_kl = []
    losses_kl_kp = []
    losses_kl_feat = []
    losses_kl_scale = []
    losses_kl_depth = []
    losses_kl_obj_on = []

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
    else:
        best_val_lpips_epoch = None
        val_lpips = best_val_lpips = None

    # iteration counter
    iteration = 0

    for epoch in range(start_epoch, num_epochs):
        model.train()
        batch_losses = []
        batch_losses_rec = []
        batch_losses_kl = []
        batch_losses_kl_kp = []
        batch_losses_kl_feat = []
        batch_losses_kl_scale = []
        batch_losses_kl_depth = []
        batch_losses_kl_obj_on = []
        batch_psnrs = []

        pbar = tqdm(iterable=dataloader, disable=not accelerator.is_local_main_process)
        for batch in pbar:
            x = batch[0].to(accelerator.device)
            if len(x.shape) == 4:
                # [bs, ch, h, w]
                x = x.unsqueeze(1)
            warmup = (epoch < warmup_epoch)
            # forward pass
            model_output = model(x, warmup=warmup, with_loss=True,
                                 beta_kl=beta_kl,
                                 beta_rec=beta_rec, kl_balance=kl_balance,
                                 recon_loss_type=recon_loss_type,
                                 recon_loss_func=recon_loss_func,
                                 beta_obj=beta_obj)
            # calculate loss
            all_losses = model_output['loss_dict']
            loss = all_losses['loss']

            optimizer.zero_grad()
            accelerator.backward(loss)
            optimizer.step()

            iteration += 1

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
            loss_rec = all_losses['loss_rec']
            loss_kl_kp = all_losses['loss_kl_kp']
            loss_kl_feat = all_losses['loss_kl_feat']
            loss_kl_scale = all_losses['loss_kl_scale']
            loss_kl_depth = all_losses['loss_kl_depth']
            loss_kl_obj_on = all_losses['loss_kl_obj_on']

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
            batch_losses_kl_scale.append(loss_kl_scale.data.cpu().item())
            batch_losses_kl_depth.append(loss_kl_depth.data.cpu().item())
            batch_losses_kl_obj_on.append(loss_kl_obj_on.data.cpu().item())
            # progress bar
            if epoch < warmup_epoch:
                pbar.set_description_str(f'epoch #{epoch} (warmup)')
            else:
                pbar.set_description_str(f'epoch #{epoch}')

            pbar.set_postfix(loss=loss.data.cpu().item(), rec=loss_rec.data.cpu().item(),
                             kl=loss_kl.data.cpu().item(), on_l1=obj_on_l1.cpu().item(),
                             a=a_mean.data.cpu().item(), b=b_mean.data.cpu().item(),
                             smu=mu_scale_mean.data.cpu().item())
            # break  # for debug
        pbar.close()
        losses.append(np.mean(batch_losses))
        losses_rec.append(np.mean(batch_losses_rec))
        losses_kl.append(np.mean(batch_losses_kl))
        losses_kl_kp.append(np.mean(batch_losses_kl_kp))
        losses_kl_feat.append(np.mean(batch_losses_kl_feat))
        losses_kl_scale.append(np.mean(batch_losses_kl_scale))
        losses_kl_depth.append(np.mean(batch_losses_kl_depth))
        losses_kl_obj_on.append(np.mean(batch_losses_kl_obj_on))
        if len(batch_psnrs) > 0:
            psnrs.append(np.mean(batch_psnrs))
        # scheduler
        if use_scheduler:
            scheduler.step()
            if accelerator.is_main_process:
                curr_lr = scheduler.get_lr()
                lr_str = f'learning rate: {curr_lr}'
                accelerator.print(curr_lr)
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
            psnr=psnrs[-1] if len(psnrs) > 0 else None
        )
        accelerator.print(log_str)
        if accelerator.is_main_process:
            log_line(log_dir, log_str)

        # wait an unwrap model
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)

        if epoch % eval_epoch_freq == 0 or epoch == num_epochs - 1:
            if accelerator.is_main_process:
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
                accelerator.print(bb_str)
                log_line(log_dir, bb_str)
                img_with_kp_topk = plot_keypoints_on_image_batch(topk_kp.clamp(min=kp_range[0], max=kp_range[1]), x,
                                                                 radius=3, thickness=1, max_imgs=max_imgs,
                                                                 kp_range=kp_range)
                dec_objects = model_output['dec_objects']
                bg = model_output['bg_rgb']
                device = accelerator.device
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

            accelerator.save(unwrapped_model.state_dict(), os.path.join(save_dir, f'{ds}_gddlp{run_prefix}.pth'))
            print("validation step...")
            eval_model = unwrapped_model
            valid_loss = evaluate_validation_elbo(eval_model, config, epoch, batch_size=batch_size,
                                                  recon_loss_type=recon_loss_type, device=accelerator.device,
                                                  save_image=True, fig_dir=fig_dir, topk=topk,
                                                  recon_loss_func=recon_loss_func, beta_rec=beta_rec,
                                                  iou_thresh=iou_thresh, accelerator=accelerator,
                                                  beta_kl=beta_kl, kl_balance=kl_balance, beta_obj=beta_obj)
            log_str = f'validation loss: {valid_loss:.3f}\n'
            accelerator.print(log_str)
            if accelerator.is_main_process:
                log_line(log_dir, log_str)
            if best_valid_loss > valid_loss:
                log_str = f'validation loss updated: {best_valid_loss:.3f} -> {valid_loss:.3f}\n'
                accelerator.print(log_str)
                if accelerator.is_main_process:
                    log_line(log_dir, log_str)
                best_valid_loss = valid_loss
                best_valid_epoch = epoch
                accelerator.save(unwrapped_model.state_dict(),
                                 os.path.join(save_dir, f'{ds}_gddlp{run_prefix}_best.pth'))
            accelerator.wait_for_everyone()
            torch.cuda.empty_cache()
            if eval_im_metrics and epoch > 0:
                valid_imm_results = eval_dlp_im_metric(unwrapped_model, accelerator.device, config,
                                                       val_mode='val',
                                                       eval_dir=log_dir,
                                                       batch_size=batch_size,
                                                       accelerator=accelerator)
                log_str = f'validation: lpips: {valid_imm_results["lpips"]:.3f}, '
                log_str += f'psnr: {valid_imm_results["psnr"]:.3f}, ssim: {valid_imm_results["ssim"]:.3f}\n'
                val_lpips = valid_imm_results['lpips']
                accelerator.print(log_str)
                if accelerator.is_main_process:
                    log_line(log_dir, log_str)
                if (not torch.isinf(torch.tensor(val_lpips))) and (best_val_lpips > val_lpips):
                    log_str = f'validation lpips updated: {best_val_lpips:.3f} -> {val_lpips:.3f}\n'
                    accelerator.print(log_str)
                    if accelerator.is_main_process:
                        log_line(log_dir, log_str)
                    best_val_lpips = val_lpips
                    best_val_lpips_epoch = epoch
                    accelerator.save(unwrapped_model.state_dict(),
                                     os.path.join(save_dir, f'{ds}_gddlp{run_prefix}_best_lpips.pth'))
                accelerator.wait_for_everyone()
                torch.cuda.empty_cache()
        valid_losses.append(valid_loss)
        if eval_im_metrics:
            val_lpipss.append(val_lpips)
        # plot graphs
        if epoch > start_epoch and accelerator.is_main_process:
            metrics_data = [
                (losses[1:], "Total Loss", "#2d72bc", True),
                (losses_kl[1:], "KL Loss", "#c92a2a", True),
                (losses_rec[1:], "Reconstruction Loss", "#087f5b", True),
                (valid_losses[1:], "Validation Loss", "#862e9c", True),
            ]
            save_metrics_data(metrics_data, run_name, save_dir=os.path.join(save_dir, 'metrics'))
            plot_training_metrics(metrics_data, run_name, fig_dir, max_plots_per_figure=4)
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    return unwrapped_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DLPv3 Multi-GPU Training")
    parser.add_argument("-d", "--dataset", type=str, default='shapes',
                        help="dataset of to train the model on: ['traffic', 'clevrer', 'obj3d128', 'phyre']")
    args = parser.parse_args()
    ds = args.dataset
    if ds.endswith('json'):
        conf_path = ds
    else:
        conf_path = os.path.join('./configs', f'{ds}.json')

    train_dlp(conf_path)
