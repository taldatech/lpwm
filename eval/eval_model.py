"""
Evaluation of the ELBO on the validation set
"""
# imports
import numpy as np
import os
# torch
import torch
import torch.nn.functional as F
from utils.loss_functions import calc_reconstruction_loss, LossLPIPS
from torch.utils.data import DataLoader
import torchvision.utils as vutils
# datasets
from datasets.get_dataset import get_video_dataset, get_image_dataset
# util functions
from utils.util_func import plot_keypoints_on_image_batch, animate_trajectories, \
    plot_bb_on_image_batch_from_z_scale_nms, plot_bb_on_image_batch_from_masks_nms, create_segmentation_map


def evaluate_validation_elbo(model, config, epoch, batch_size=100, recon_loss_type="vgg", device=torch.device('cpu'),
                             save_image=False, fig_dir='./', topk=5, recon_loss_func=None, beta_rec=1.0, beta_kl=1.0,
                             kl_balance=0.001, accelerator=None, iou_thresh=0.2, beta_obj=0.0):
    model.eval()
    kp_range = model.kp_range
    ds = config['ds']
    ch = config['ch']  # image channels
    image_size = config['image_size']
    root = config['root']  # dataset root
    dataset = get_image_dataset(ds, root, mode='valid', image_size=image_size)
    dataloader = DataLoader(dataset, shuffle=True, batch_size=batch_size, num_workers=4, drop_last=False)
    if recon_loss_func is None:
        if recon_loss_type == "vgg":
            recon_loss_func = LossLPIPS().to(device)
        else:
            recon_loss_func = calc_reconstruction_loss

    elbos = []
    for batch in dataloader:
        x = batch[0].to(device)
        if len(x.shape) == 4:
            # [bs, ch, h, w]
            x = x.unsqueeze(1)
        # forward pass
        with torch.no_grad():
            model_output = model(x, with_loss=True, beta_kl=beta_kl,
                                 beta_rec=beta_rec, kl_balance=kl_balance,
                                 recon_loss_type=recon_loss_type,
                                 recon_loss_func=recon_loss_func, beta_obj=beta_obj)
        all_losses = model_output['loss_dict']
        loss = all_losses['loss']

        mu_p = model_output['kp_p']
        mu = model_output['mu_anchor']
        z_base = model_output['z_base']
        mu_offset = model_output['mu_offset']
        logvar_offset = model_output['logvar_offset']
        rec_x = model_output['rec_rgb']
        mu_scale = model_output['mu_scale']
        # object stuff
        dec_objects_original = model_output['dec_objects_original']
        cropped_objects_original = model_output['cropped_objects_original']
        obj_on = model_output['obj_on']  # [batch_size, n_kp]
        alpha_masks = model_output['alpha_masks']  # [batch_size, n_kp, 1, h, w]

        # for plotting, confidence calculation
        mu_tot = z_base + mu_offset
        mu_tot = mu_tot.view(-1, *mu_tot.shape[2:])
        logvar_tot = logvar_offset
        logvar_tot = logvar_tot.view(-1, *logvar_tot.shape[2:])

        elbo = loss
        elbos.append(elbo.data.cpu().numpy())
    if save_image:
        x = x.view(-1, *x.shape[2:])
        max_imgs = 8
        mu_plot = mu_tot.clamp(min=kp_range[0], max=kp_range[1])
        img_with_kp = plot_keypoints_on_image_batch(mu_plot, x, radius=3,
                                                    thickness=1, max_imgs=max_imgs, kp_range=model.kp_range)
        img_with_kp_p = plot_keypoints_on_image_batch(mu_p, x, radius=3, thickness=1, max_imgs=max_imgs,
                                                      kp_range=model.kp_range)
        with torch.no_grad():
            # top-k
            logvar_sum = logvar_tot.sum(-1) * obj_on.view(-1, *obj_on.shape[2:]).squeeze(-1)  # [bs, n_kp]
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
                                                                                  thickness=1,
                                                                                  max_imgs=max_imgs,
                                                                                  hard_thresh=hard_threshold)
            alpha_masks = torch.where(alpha_masks < 0.05, 0.0, 1.0)
            if alpha_masks.shape[1] != bb_scores.shape[1]:
                bb_scores = -1 * torch.topk(logvar_sum, k=alpha_masks.shape[1], dim=-1, largest=False)[0]
            img_with_masks_alpha_nms, _ = plot_bb_on_image_batch_from_masks_nms(alpha_masks, x,
                                                                                scores=bb_scores,
                                                                                iou_thresh=iou_thresh,
                                                                                thickness=1,
                                                                                max_imgs=max_imgs,
                                                                                hard_thresh=hard_threshold)
            img_with_seg_maps = create_segmentation_map(x=x, masks=alpha_masks, scores=bb_scores, alpha=0.7)
        img_with_kp_topk = plot_keypoints_on_image_batch(topk_kp.clamp(min=kp_range[0], max=kp_range[1]), x,
                                                         radius=3, thickness=1, max_imgs=max_imgs,
                                                         kp_range=kp_range)
        dec_objects = model_output['dec_objects']
        bg = model_output['bg_rgb']
        if accelerator is not None:
            if accelerator.is_main_process:
                vutils.save_image(torch.cat([x[:max_imgs, -3:], img_with_kp[:max_imgs, -3:].to(accelerator.device),
                                             rec_x[:max_imgs, -3:],
                                             img_with_kp_p[:max_imgs, -3:].to(accelerator.device),
                                             img_with_kp_topk[:max_imgs, -3:].to(accelerator.device),
                                             dec_objects[:max_imgs, -3:],
                                             img_with_masks_nms[:max_imgs, -3:].to(accelerator.device),
                                             img_with_masks_alpha_nms[:max_imgs, -3:].to(accelerator.device),
                                             img_with_seg_maps[:max_imgs, -3:],
                                             bg[:max_imgs, -3:]],
                                            dim=0).data.cpu(), '{}/image_valid_{}.jpg'.format(fig_dir, epoch),
                                  nrow=8, pad_value=1)
        else:
            vutils.save_image(torch.cat([x[:max_imgs, -3:], img_with_kp[:max_imgs, -3:].to(device),
                                         rec_x[:max_imgs, -3:],
                                         img_with_kp_p[:max_imgs, -3:].to(device),
                                         img_with_kp_topk[:max_imgs, -3:].to(device),
                                         dec_objects[:max_imgs, -3:],
                                         img_with_masks_nms[:max_imgs, -3:].to(device),
                                         img_with_masks_alpha_nms[:max_imgs, -3:].to(device),
                                         img_with_seg_maps[:max_imgs, -3:],
                                         bg[:max_imgs, -3:]],
                                        dim=0).data.cpu(), '{}/image_valid_{}.jpg'.format(fig_dir, epoch),
                              nrow=8, pad_value=1)
    return np.mean(elbos)


def evaluate_validation_elbo_dyn(model, config, epoch, batch_size=100, recon_loss_type="vgg",
                                 device=torch.device('cpu'),
                                 save_image=False, fig_dir='./', topk=5, recon_loss_func=None, beta_rec=1.0,
                                 beta_kl=1.0, beta_dyn=1.0, iou_thresh=0.2, beta_dyn_rec=1.0,
                                 kl_balance=1.0, accelerator=None, timestep_horizon=10,
                                 animation_horizon=50, beta_obj=0.0):
    model.eval()
    kp_range = model.kp_range
    # load data
    ds = config['ds']
    ch = config['ch']  # image channels
    image_size = config['image_size']
    n_views = config.get('n_views', 1)
    root = config['root']  # dataset root
    cond_steps = config['cond_steps']  # dataset root
    action_condition = config.get('action_condition', False)
    language_condition = config.get('language_condition', False)
    use_ep_done_mask = config.get('ep_done_mask', False)
    img_goal_condition = config.get('image_goal_condition', False)
    dataset = get_video_dataset(ds, root, seq_len=timestep_horizon + 1, mode='valid', image_size=image_size)

    dataloader = DataLoader(dataset, shuffle=True, batch_size=batch_size, num_workers=4, drop_last=False)

    elbos = []
    for batch in dataloader:
        x = batch[0][:, :timestep_horizon + 1].to(device)
        actions = None if not action_condition else batch[1][:, :timestep_horizon + 1].to(device)
        lang_str = None if not language_condition else batch[2]
        lang_embed = None if not language_condition else batch[3].to(device)
        ep_done_mask = None if not use_ep_done_mask else batch[-1].to(device)
        x_goal = None if not img_goal_condition else batch[3].to(device)
        # x_prior = x
        if n_views > 1:
            # expect: [bs, T, n_views, ...]
            x = x.permute(0, 2, 1, 3, 4, 5)
            x = x.reshape(-1, *x.shape[2:])  # [bs * n_views, T, ...]
            if x_goal is not None:
                x_goal = x_goal.reshape(-1, *x_goal.shape[2:])  # [bs * n_views, ...]
            if actions is not None:
                actions = actions.permute(0, 2, 1, 3)
                actions = actions.reshape(-1, *actions.shape[2:])
            if ep_done_mask is not None:
                ep_done_mask = ep_done_mask.permute(0, 2, 1)
                ep_done_mask - ep_done_mask.reshape(-1, *ep_done_mask.shape[2:])
        with torch.no_grad():
            model_output = model(x, actions=actions, lang_embed=lang_embed, with_loss=True, beta_kl=beta_kl,
                                 beta_dyn=beta_dyn, beta_rec=beta_rec, kl_balance=kl_balance,
                                 recon_loss_type=recon_loss_type, recon_loss_func=recon_loss_func,
                                 beta_dyn_rec=beta_dyn_rec, beta_obj=beta_obj, done_mask=ep_done_mask, x_goal=x_goal)
        all_losses = model_output['loss_dict']
        loss = all_losses['loss']

        mu_p = model_output['kp_p']
        mu = model_output['mu_anchor']
        z_base = model_output['z_base']
        mu_offset = model_output['mu_offset']
        logvar_offset = model_output['logvar_offset']
        rec_x = model_output['rec_rgb']
        mu_scale = model_output['mu_scale']
        # object stuff
        dec_objects_original = model_output['dec_objects_original']
        cropped_objects_original = model_output['cropped_objects_original']
        obj_on = model_output['obj_on']  # [batch_size, n_kp]
        alpha_masks = model_output['alpha_masks']  # [batch_size, n_kp, 1, h, w]
        x = x.reshape(-1, *x.shape[2:])
        # x_prior = x_prior.view(-1, *x_prior.shape[2:])
        # for plotting, confidence calculation
        mu_tot = z_base + mu_offset
        mu_tot = mu_tot.view(-1, *mu_tot.shape[2:])
        logvar_tot = logvar_offset
        logvar_tot = logvar_tot.view(-1, *logvar_tot.shape[2:])

        elbo = loss
        elbos.append(elbo.data.cpu().numpy())
    if save_image:
        max_imgs = 8
        mu_plot = mu_tot.clamp(min=kp_range[0], max=kp_range[1])
        img_with_kp = plot_keypoints_on_image_batch(mu_plot, x, radius=3,
                                                    thickness=1, max_imgs=max_imgs, kp_range=model.kp_range)
        img_with_kp_p = plot_keypoints_on_image_batch(mu_p, x, radius=3, thickness=1, max_imgs=max_imgs,
                                                      kp_range=model.kp_range)
        # top-k
        with torch.no_grad():
            logvar_sum = logvar_tot.sum(-1) * obj_on.view(-1, *obj_on.shape[2:]).squeeze(-1)  # [bs, n_kp]
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
                                                                                  thickness=1,
                                                                                  max_imgs=max_imgs,
                                                                                  hard_thresh=hard_threshold)
            alpha_masks = torch.where(alpha_masks < 0.05, 0.0, 1.0)
            if alpha_masks.shape[1] != bb_scores.shape[1]:
                bb_scores = -1 * torch.topk(logvar_sum, k=alpha_masks.shape[1], dim=-1, largest=False)[0]
            img_with_masks_alpha_nms, _ = plot_bb_on_image_batch_from_masks_nms(alpha_masks, x,
                                                                                scores=bb_scores,
                                                                                iou_thresh=iou_thresh,
                                                                                thickness=1,
                                                                                max_imgs=max_imgs,
                                                                                hard_thresh=hard_threshold)
            img_with_seg_maps = create_segmentation_map(x=x, masks=alpha_masks, scores=bb_scores, alpha=0.7)
        img_with_kp_topk = plot_keypoints_on_image_batch(topk_kp.clamp(min=kp_range[0], max=kp_range[1]), x,
                                                         radius=3, thickness=1, max_imgs=max_imgs,
                                                         kp_range=kp_range)
        dec_objects = model_output['dec_objects']
        bg = model_output['bg_rgb']
        if accelerator is not None:
            if accelerator.is_main_process:
                vutils.save_image(torch.cat([x[:max_imgs, -3:], img_with_kp[:max_imgs, -3:].to(accelerator.device),
                                             rec_x[:max_imgs, -3:],
                                             img_with_kp_p[:max_imgs, -3:].to(accelerator.device),
                                             img_with_kp_topk[:max_imgs, -3:].to(accelerator.device),
                                             dec_objects[:max_imgs, -3:],
                                             img_with_masks_nms[:max_imgs, -3:].to(accelerator.device),
                                             img_with_masks_alpha_nms[:max_imgs, -3:].to(accelerator.device),
                                             img_with_seg_maps[:max_imgs, -3:],
                                             bg[:max_imgs, -3:]],
                                            dim=0).data.cpu(), '{}/image_valid_{}.jpg'.format(fig_dir, epoch),
                                  nrow=8, pad_value=1)
        else:
            vutils.save_image(torch.cat([x[:max_imgs, -3:], img_with_kp[:max_imgs, -3:].to(device),
                                         rec_x[:max_imgs, -3:],
                                         img_with_kp_p[:max_imgs, -3:].to(device),
                                         img_with_kp_topk[:max_imgs, -3:].to(device),
                                         dec_objects[:max_imgs, -3:],
                                         img_with_masks_nms[:max_imgs, -3:].to(device),
                                         img_with_masks_alpha_nms[:max_imgs, -3:].to(device),
                                         img_with_seg_maps[:max_imgs, -3:],
                                         bg[:max_imgs, -3:]],
                                        dim=0).data.cpu(), '{}/image_valid_{}.jpg'.format(fig_dir, epoch),
                              nrow=8, pad_value=1)
        animate_trajectory_lpwm(model, config, epoch, device=device, fig_dir=fig_dir, prefix='valid_',
                                timestep_horizon=animation_horizon, num_trajetories=1,
                                accelerator=accelerator, train=False, cond_steps=cond_steps)
    return np.mean(elbos)


def animate_trajectory_lpwm(model, config, epoch, device=torch.device('cpu'), fig_dir='./', timestep_horizon=3,
                            num_trajetories=5, accelerator=None, train=False, prefix='', cond_steps=None,
                            deterministic=True, det_and_stoch=True, use_all_ctx=True):
    # load data
    ds = config['ds']
    ch = config['ch']  # image channels
    image_size = config['image_size']
    n_views = config.get('n_views', 1)
    root = config['root']  # dataset root
    duration = config['animation_fps']
    action_condition = config.get('action_condition', False)
    language_condition = config.get('language_condition', False)
    img_goal_condition = config.get('image_goal_condition', False)

    mode = 'train' if train else "valid"
    dataset = get_video_dataset(ds, root, seq_len=timestep_horizon, mode=mode, image_size=image_size)

    batch_size = max(1, num_trajetories)
    dataloader = DataLoader(dataset, shuffle=True, batch_size=batch_size, num_workers=4, drop_last=False)
    batch = next(iter(dataloader))
    model_timestep_horizon = model.timestep_horizon
    cond_steps = model_timestep_horizon if cond_steps is None else cond_steps
    model.eval()
    x_horizon = batch[0][:, :timestep_horizon].to(device)
    actions_horizon = None if not action_condition else batch[1][:, :timestep_horizon].to(device)
    lang_str = None if not language_condition else batch[2]
    lang_embed = None if not language_condition else batch[3].to(device)
    x_goal = None if not img_goal_condition else batch[3].to(device)
    if n_views > 1:
        # expect: [bs, T, n_views, ...]
        x_horizon = x_horizon.permute(0, 2, 1, 3, 4, 5)
        x_horizon = x_horizon.reshape(-1, *x_horizon.shape[2:])  # [bs * n_views, T, ...]
        if x_goal is not None:
            x_goal = x_goal.reshape(-1, *x_goal.shape[2:])  # [bs * n_views, ...]
        if actions_horizon is not None:
            actions_horizon = actions_horizon.permute(0, 2, 1, 3)
            actions_horizon = actions_horizon.reshape(-1, *actions_horizon.shape[2:])
    # forward pass
    with torch.no_grad():
        if det_and_stoch:
            preds_1 = model.sample_from_x(x_horizon, num_steps=timestep_horizon - cond_steps, deterministic=True,
                                          cond_steps=cond_steps, use_all_ctx=use_all_ctx, actions=actions_horizon,
                                          lang_embed=lang_embed, x_goal=x_goal)
            preds_2 = model.sample_from_x(x_horizon, num_steps=timestep_horizon - cond_steps, deterministic=False,
                                          cond_steps=cond_steps, actions=actions_horizon, lang_embed=lang_embed,
                                          x_goal=x_goal)
        else:
            preds_1 = model.sample_from_x(x_horizon, num_steps=timestep_horizon - cond_steps,
                                          deterministic=deterministic,
                                          cond_steps=cond_steps, actions=actions_horizon, lang_embed=lang_embed,
                                          x_goal=x_goal)
            preds_2 = None
        # preds: [bs, timestep_horizon, 3, im_size, im_size]
    for i in range(num_trajetories):
        if n_views > 1:
            x_horizon = x_horizon.reshape(-1, n_views, *x_horizon.shape[1:])
            x_preds_1 = preds_1.reshape(-1, n_views, *preds_1.shape[1:])

            gt_traj = x_horizon[i, 0].permute(0, 2, 3, 1).data.cpu().numpy()
            pred_traj = x_preds_1[i, 0].permute(0, 2, 3, 1).data.cpu().numpy()

            gt_traj_12 = x_horizon[i, 1].permute(0, 2, 3, 1).data.cpu().numpy()
            pred_traj_12 = x_preds_1[i, 1].permute(0, 2, 3, 1).data.cpu().numpy()
        else:
            gt_traj = x_horizon[i].permute(0, 2, 3, 1).data.cpu().numpy()
            pred_traj = preds_1[i].permute(0, 2, 3, 1).data.cpu().numpy()

            gt_traj_12 = pred_traj_12 = None
        lang_str_i = None if not language_condition else lang_str[i]
        if img_goal_condition:
            if n_views > 1:
                x_goal = x_goal.reshape(-1, n_views, *x_goal.shape[1:])
                x_goal_i = x_goal[i, 0]
                x_goal_i2 = x_goal[i, 1]
            else:
                x_goal_i = x_goal[i]
                x_goal_i2 = None
            if len(x_goal_i.shape) == 4:
                # [1, ch, h, w]
                x_goal_i = x_goal_i[0]
                if x_goal_i2 is not None:
                    x_goal_i2 = x_goal_i2[0]
            x_goal_i = x_goal_i.permute(1, 2, 0).data.cpu().numpy()  # [h, w, ch]
            if x_goal_i2 is not None:
                x_goal_i2 = x_goal_i2.permute(1, 2, 0).data.cpu().numpy()  # [h, w, ch]
        else:
            x_goal_i = x_goal_i2 = None
        if det_and_stoch:
            if n_views > 1:
                x_preds_2 = preds_2.reshape(-1, n_views, *preds_2.shape[1:])
                pred_traj_2 = x_preds_2[i, 0].permute(0, 2, 3, 1).data.cpu().numpy()
                pred_traj_22 = x_preds_2[i, 1].permute(0, 2, 3, 1).data.cpu().numpy()
            else:
                pred_traj_2 = preds_2[i].permute(0, 2, 3, 1).data.cpu().numpy()
                pred_traj_22 = None
        else:
            pred_traj_2 = pred_traj_22 = None
        if accelerator is not None:
            if accelerator.is_main_process:
                animate_trajectories(gt_traj, pred_traj, pred_traj_2,
                                     path=os.path.join(fig_dir, f'{prefix}e{epoch}_traj_anim_{i}.gif'),
                                     duration=duration, rec_to_pred_t=cond_steps, t1='-D', t2='-S', title=lang_str_i,
                                     goal_img=x_goal_i,
                                     orig_trajectory2=gt_traj_12, pred_trajectory_12=pred_traj_12,
                                     pred_trajectory_22=pred_traj_22, goal_img2=x_goal_i2)
        else:
            animate_trajectories(gt_traj, pred_traj, pred_traj_2,
                                 path=os.path.join(fig_dir, f'{prefix}e{epoch}_traj_anim_{i}.gif'),
                                 duration=duration, rec_to_pred_t=cond_steps, t1='-D', t2='-S', title=lang_str_i,
                                 goal_img=x_goal_i, orig_trajectory2=gt_traj_12, pred_trajectory_12=pred_traj_12,
                                 pred_trajectory_22=pred_traj_22, goal_img2=x_goal_i2)
