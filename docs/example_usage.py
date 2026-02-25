"""
Example usage of DLPv3 and LPWM
"""
# imports
import os
import sys

sys.path.append(os.getcwd())
import argparse
# torch
import torch
# modules
from models import DLP

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Example Usage")
    parser.add_argument("--model_type", type=str, default='lpwm',
                        help="type of model to use for example: ['dlp', 'lpwm']")
    args = parser.parse_args()
    # parse input
    model_type = args.model_type

    if model_type == 'dlp':
        print("--- DLPv3 ---")
        # example hyper-parameters
        batch_size = 4
        beta_kl = 0.02
        beta_rec = 1.0
        beta_obj = 0.02
        kl_balance = 0.01  # balance between spatial attributes (x, y, scale, depth) and visual features
        n_kp_enc = 42
        n_kp_prior = 64
        patch_size = 8  # patch size for the prior to generate prior proposals
        anchor_s = 0.25  # effective patch size for the posterior: anchor_s * image_size
        image_size = 64
        ch = 3
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        timestep_horizon = 1
        deterministic = False
        warmup = False
        attn_norm_type = 'rms'

        learned_feature_dim = 4  # visual features
        learned_bg_feature_dim = 5  # visual features
        obj_res_from_fc = 4  # 8
        obj_ch_mult_prior = (1, 2)  # (1, 2)
        obj_ch_mult = (1, 2, 2)  # (1, 2)
        obj_base_ch = 32
        obj_final_cnn_ch = 32
        bg_res_from_fc = 8
        bg_ch_mult = (1, 1, 2, 4)
        # bg_ch_mult = (1, 1, 2)
        bg_base_ch = 32
        bg_final_cnn_ch = 32
        num_res_blocks = 1
        use_resblock = True

        model = DLP(cdim=ch,  # number of input image channels
                    image_size=image_size,
                    normalize_rgb=False,  # normalize to [-1, 1] or keep [0, 1]
                    n_kp_per_patch=1,
                    patch_size=patch_size,
                    anchor_s=anchor_s,
                    n_kp_enc=n_kp_enc,
                    n_kp_prior=n_kp_prior,
                    pad_mode='zeros',
                    features_dist='gauss',
                    learned_feature_dim=learned_feature_dim,
                    learned_bg_feature_dim=learned_bg_feature_dim,
                    n_fg_categories=8,
                    n_fg_classes=4,
                    n_bg_categories=4,
                    n_bg_classes=4,
                    scale_std=0.3,
                    offset_std=0.2,
                    obj_on_alpha=0.01,
                    obj_on_beta=0.01,
                    obj_on_min=1e-4,
                    obj_on_max=100,
                    obj_res_from_fc=obj_res_from_fc,
                    obj_ch_mult_prior=obj_ch_mult_prior,
                    obj_ch_mult=obj_ch_mult,
                    obj_base_ch=obj_base_ch,
                    obj_final_cnn_ch=obj_final_cnn_ch,
                    bg_res_from_fc=bg_res_from_fc,
                    bg_ch_mult=bg_ch_mult,
                    bg_base_ch=bg_base_ch,
                    bg_final_cnn_ch=bg_final_cnn_ch,
                    use_resblock=use_resblock,
                    num_res_blocks=num_res_blocks,
                    cnn_mid_blocks=False,
                    mlp_hidden_dim=256,
                    attn_norm_type=attn_norm_type,
                    pint_enc_layers=1,  # pint = particle interaction transformer
                    pint_enc_heads=1,
                    embed_init_std=0.2,
                    particle_positional_embed=True,  # add positional embeddings for particles in transformers
                    use_z_orig=True,  # for each particle, cat the patch center coordinates it originated from
                    particle_score=False,  # use the particle score as feature (i.e., the kp x-y sum of variances)
                    filtering_heuristic='none',  # how to filter prior keypoints, 'none' will keep all prior kp
                    # dynamics hyperparameters
                    timestep_horizon=timestep_horizon,
                    ).to(device)
        print(f'model.info():')
        print(model.info())
        print("----------------------------------")

        x_ts = (timestep_horizon + 1) if timestep_horizon > 1 else 1
        x = torch.rand(batch_size, x_ts, ch, image_size, image_size, device=device)

        model_output = model(x, deterministic, warmup, with_loss=True, beta_kl=beta_kl,
                             beta_rec=beta_rec, kl_balance=kl_balance,
                             recon_loss_type="mse", beta_obj=beta_obj)

        # let's see what's inside
        print(f'model(x) output:')
        for k in model_output.keys():
            if model_output[k] is not None and not isinstance(model_output[k], dict):
                print(f'{k}: {model_output[k].shape}')
        print("----------------------------------")
        """
        output: static
        model(x) output:
        kp_p: torch.Size([4, 64, 2])
        rec: torch.Size([4, 3, 64, 64])
        rec_rgb: torch.Size([4, 3, 64, 64])
        mu_anchor: torch.Size([4, 1, 42, 2])
        logvar_anchor: torch.Size([4, 1, 42, 2])
        z_base_var: torch.Size([4, 1, 42, 5])
        z_base: torch.Size([4, 1, 42, 2])
        z: torch.Size([4, 1, 42, 2])
        mu_offset: torch.Size([4, 1, 42, 2])
        logvar_offset: torch.Size([4, 1, 42, 2])
        z_offset: torch.Size([4, 1, 42, 2])
        mu_tot: torch.Size([4, 1, 42, 2])
        mu_features: torch.Size([4, 1, 42, 4])
        logvar_features: torch.Size([4, 1, 42, 4])
        z_features: torch.Size([4, 1, 42, 4])
        bg: torch.Size([4, 3, 64, 64])
        bg_rgb: torch.Size([4, 3, 64, 64])
        mu_bg_features: torch.Size([4, 1, 5])
        logvar_bg_features: torch.Size([4, 1, 5])
        z_bg_features: torch.Size([4, 1, 5])
        cropped_objects_original: torch.Size([168, 3, 16, 16])
        cropped_objects_original_rgb: torch.Size([168, 3, 16, 16])
        obj_on_a: torch.Size([4, 1, 42, 1])
        obj_on_b: torch.Size([4, 1, 42, 1])
        obj_on: torch.Size([4, 1, 42, 1])
        mu_obj_on: torch.Size([4, 1, 42, 1])
        dec_objects_original: torch.Size([4, 42, 4, 16, 16])
        dec_objects_original_rgb: torch.Size([4, 42, 4, 16, 16])
        dec_objects: torch.Size([4, 3, 64, 64])
        mu_depth: torch.Size([4, 1, 42, 1])
        logvar_depth: torch.Size([4, 1, 42, 1])
        z_depth: torch.Size([4, 1, 42, 1])
        mu_scale: torch.Size([4, 1, 42, 2])
        logvar_scale: torch.Size([4, 1, 42, 2])
        z_scale: torch.Size([4, 1, 42, 2])
        alpha_masks: torch.Size([4, 42, 1, 64, 64])
        mu_score: torch.Size([4, 1, 42, 1])
        logvar_score: torch.Size([4, 1, 42, 1])
        z_score: torch.Size([4, 1, 42, 1])
        ----------------------------------
        """

        # loss calculation
        all_losses = model.calc_elbo(x, model_output, beta_kl=beta_kl,
                                     beta_rec=beta_rec, kl_balance=kl_balance,
                                     recon_loss_type="mse", warmup=warmup, beta_obj=beta_obj)
        # let's see what's inside
        print(f'model.calc_elbo(): model losses:')
        for k in all_losses.keys():
            print(f'{k}: {all_losses[k]}')
        print("----------------------------------")
        """
        output: static
        model.calc_elbo(): model losses:
        loss: 104.4956283569336
        psnr: 10.748549461364746
        kl: 203.42578125
        kl_dyn: 0.0
        loss_rec: 1034.251708984375
        obj_on_l1: 17.903579711914062
        loss_kl_kp: 89.41305541992188
        loss_kl_feat: 491.08544921875
        loss_kl_obj_on: 58.43592834472656
        loss_kl_scale: 50.32316589355469
        loss_kl_depth: 0.3427771031856537
        loss_kl_context: 0.0
        loss_obj_reg: 331.80328369140625
        -------------------------------
        """
    elif model_type == 'lpwm':
        print("--- LPWM ---")
        # example hyper-parameters
        batch_size = 4
        beta_kl = 0.02
        beta_rec = 1.0
        beta_obj = 0.02
        kl_balance = 0.01  # balance between spatial attributes (x, y, scale, depth) and visual features
        n_kp_enc = 42
        n_kp_prior = 64
        patch_size = 8  # patch size for the prior to generate prior proposals
        anchor_s = 0.25  # effective patch size for the posterior: anchor_s * image_size
        image_size = 64
        ch = 3
        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        device = torch.device("cpu")
        # Context and dynamics transformer configuration
        pint_dyn_layers = 6  # Number of dynamics transformer layers
        pint_dyn_heads = 8  # Number of dynamics transformer heads
        pint_dim = 512  # Hidden dimension for PINT
        pint_ctx_layers = 4  # Number of context transformer layers
        pint_ctx_heads = 8  # Number of context transformer heads
        beta_dyn = 0.1  # beta-kl for the dynamics loss
        num_static_frames = 1  # "burn-in frames", number of initial frames with kl w.r.t. constant prior (as in DLPv2)
        context_dist = 'gauss'
        context_dim = 7
        timestep_horizon = 10
        deterministic = False
        warmup = False
        predict_delta = False
        attn_norm_type = 'rms'

        learned_feature_dim = 4  # visual features
        learned_bg_feature_dim = 5  # visual features
        obj_res_from_fc = 4  # 8
        obj_ch_mult_prior = (1, 2)  # (1, 2)
        obj_ch_mult = (1, 2, 2)  # (1, 2)
        obj_base_ch = 32
        obj_final_cnn_ch = 32
        bg_res_from_fc = 8
        bg_ch_mult = (1, 1, 2, 4)
        # bg_ch_mult = (1, 1, 2)
        bg_base_ch = 32
        bg_final_cnn_ch = 32
        num_res_blocks = 1
        use_resblock = True

        # actions
        action_cond = False
        action_d = 7
        null_actions = False
        action_in_ctx = True

        # random actions
        rand_action_cond = False
        rand_action_d = 6

        # language
        lang_cond = False
        lang_d = 128
        max_lang_len = 64

        # image goal condition
        img_goal_cond = False

        # number of views
        n_im_views = 1

        # episode done mask
        # ep_done_mask = None
        ep_dones = torch.randint(low=2, high=timestep_horizon + 2, size=(batch_size * n_im_views, 1),
                                 device=device)  # [bs, 1]
        ep_done_mask = torch.ones(batch_size * n_im_views, timestep_horizon + 1, dtype=torch.int, device=device)
        for i in range(ep_done_mask.shape[0]):
            if ep_dones[i] < ep_done_mask.shape[1]:
                ep_done_mask[i, ep_dones[i]:] = 0.0

        model = DLP(cdim=ch,  # number of input image channels
                    image_size=image_size,
                    normalize_rgb=False,  # normalize to [-1, 1] or keep [0, 1]
                    n_views=n_im_views,
                    n_kp_per_patch=1,
                    patch_size=patch_size,
                    anchor_s=anchor_s,
                    n_kp_enc=n_kp_enc,
                    n_kp_prior=n_kp_prior,
                    pad_mode='zeros',
                    dropout=0.1,
                    features_dist='gauss',
                    learned_feature_dim=learned_feature_dim,
                    learned_bg_feature_dim=learned_bg_feature_dim,
                    n_fg_categories=8,
                    n_fg_classes=4,
                    n_bg_categories=4,
                    n_bg_classes=4,
                    scale_std=0.3,
                    offset_std=0.2,
                    obj_on_alpha=0.01,
                    obj_on_beta=0.01,
                    obj_on_min=1e-4,
                    obj_on_max=100,
                    obj_res_from_fc=obj_res_from_fc,
                    obj_ch_mult_prior=obj_ch_mult_prior,
                    obj_ch_mult=obj_ch_mult,
                    obj_base_ch=obj_base_ch,
                    obj_final_cnn_ch=obj_final_cnn_ch,
                    bg_res_from_fc=bg_res_from_fc,
                    bg_ch_mult=bg_ch_mult,
                    bg_base_ch=bg_base_ch,
                    bg_final_cnn_ch=bg_final_cnn_ch,
                    use_resblock=use_resblock,
                    num_res_blocks=num_res_blocks,
                    cnn_mid_blocks=False,
                    mlp_hidden_dim=256,
                    attn_norm_type=attn_norm_type,
                    pint_enc_layers=1,  # pint = particle interaction transformer
                    pint_enc_heads=1,
                    embed_init_std=0.2,
                    particle_positional_embed=True,  # add positional embeddings for particles in transformers
                    use_z_orig=True,  # for each particle, cat the patch center coordinates it originated from
                    particle_score=False,  # use the particle score as feature (i.e., the kp x-y sum of variances)
                    filtering_heuristic='none',  # how to filter prior keypoints, 'none' will keep all prior kp
                    # dynamics hyperparameters
                    timestep_horizon=timestep_horizon,
                    n_static_frames=num_static_frames,
                    # how many initial frames should be optimized w.r.t static (constant) KL
                    predict_delta=predict_delta,
                    context_dim=context_dim,
                    ctx_dist=context_dist,
                    n_ctx_categories=8,
                    n_ctx_classes=4,
                    causal_ctx=True,  # model latent context with causal attention
                    ctx_pool_mode='none',  # how to pool the context latents, 'none'=a context latent for each particle
                    pint_dyn_layers=pint_dyn_layers,  # pint = particle interaction transformer
                    pint_dyn_heads=pint_dyn_heads,
                    pint_dim=pint_dim,
                    pint_ctx_layers=pint_ctx_layers,
                    pint_ctx_heads=pint_ctx_heads,
                    action_condition=action_cond,
                    action_dim=action_d,
                    random_action_condition=rand_action_cond,
                    random_action_dim=rand_action_d,
                    null_action_embed=null_actions,
                    action_in_ctx_module=action_in_ctx,
                    language_condition=lang_cond,
                    language_embed_dim=lang_d,
                    language_max_len=max_lang_len,
                    img_goal_condition=img_goal_cond
                    ).to(device)
        print(f'model.info():')
        print(model.info())
        print("----------------------------------")

        x_ts = (timestep_horizon + 1) if timestep_horizon > 1 else 1
        x = torch.rand(batch_size * n_im_views, x_ts, ch, image_size, image_size, device=device)
        if action_cond:
            actions_demo = torch.rand(batch_size * n_im_views, x_ts, action_d, device=device)
            if null_actions:
                actions_mask_demo = torch.rand(batch_size * n_im_views, x_ts, device=device) > 0.5
            else:
                actions_mask_demo = None
        else:
            actions_demo = None
            actions_mask_demo = None
        if lang_cond:
            lang_demo = torch.randn(batch_size, max_lang_len, lang_d, device=device)
        else:
            lang_demo = None
        if img_goal_cond:
            x_goal = torch.rand(batch_size * n_im_views, 1, ch, image_size, image_size, device=device)
        else:
            x_goal = None
        model_output = model(x, deterministic, warmup, with_loss=True, beta_kl=beta_kl,
                             beta_rec=beta_rec, kl_balance=kl_balance, beta_dyn=beta_dyn,
                             num_static=num_static_frames,
                             recon_loss_type="mse", actions=actions_demo, actions_mask=actions_mask_demo,
                             lang_embed=lang_demo, beta_obj=beta_obj, done_mask=ep_done_mask, x_goal=x_goal)
        # let's see what's inside
        print(f'model(x) output:')
        for k in model_output.keys():
            if model_output[k] is not None and not isinstance(model_output[k], dict):
                print(f'{k}: {model_output[k].shape}')
        print("----------------------------------")
        """
        output: dynamic
        model(x) output:
        kp_p: torch.Size([44, 64, 2])
        rec: torch.Size([44, 3, 64, 64])
        rec_rgb: torch.Size([44, 3, 64, 64])
        mu_anchor: torch.Size([4, 11, 64, 2])
        logvar_anchor: torch.Size([4, 11, 64, 2])
        z_base_var: torch.Size([4, 11, 64, 5])
        z_base: torch.Size([4, 11, 64, 2])
        z: torch.Size([4, 11, 64, 2])
        mu_offset: torch.Size([4, 11, 64, 2])
        logvar_offset: torch.Size([4, 11, 64, 2])
        z_offset: torch.Size([4, 11, 64, 2])
        mu_tot: torch.Size([4, 11, 64, 2])
        mu_features: torch.Size([4, 11, 64, 4])
        logvar_features: torch.Size([4, 11, 64, 4])
        z_features: torch.Size([4, 11, 64, 4])
        bg: torch.Size([44, 3, 64, 64])
        bg_rgb: torch.Size([44, 3, 64, 64])
        mu_bg_features: torch.Size([4, 11, 5])
        logvar_bg_features: torch.Size([4, 11, 5])
        z_bg_features: torch.Size([4, 11, 5])
        mu_context: torch.Size([4, 11, 65, 7])
        logvar_context: torch.Size([4, 11, 65, 7])
        z_context: torch.Size([4, 11, 65, 7])
        cropped_objects_original: torch.Size([1848, 3, 16, 16])
        cropped_objects_original_rgb: torch.Size([1848, 3, 16, 16])
        obj_on_a: torch.Size([4, 11, 64, 1])
        obj_on_b: torch.Size([4, 11, 64, 1])
        obj_on: torch.Size([4, 11, 64, 1])
        mu_obj_on: torch.Size([4, 11, 64, 1])
        dec_objects_original: torch.Size([44, 42, 4, 16, 16])
        dec_objects_original_rgb: torch.Size([44, 42, 4, 16, 16])
        dec_objects: torch.Size([44, 3, 64, 64])
        mu_depth: torch.Size([4, 11, 64, 1])
        logvar_depth: torch.Size([4, 11, 64, 1])
        z_depth: torch.Size([4, 11, 64, 1])
        mu_scale: torch.Size([4, 11, 64, 2])
        logvar_scale: torch.Size([4, 11, 64, 2])
        z_scale: torch.Size([4, 11, 64, 2])
        alpha_masks: torch.Size([44, 42, 1, 64, 64])
        mu_dyn: torch.Size([4, 10, 64, 2])
        logvar_dyn: torch.Size([4, 10, 64, 2])
        mu_features_dyn: torch.Size([4, 10, 64, 4])
        logvar_features_dyn: torch.Size([4, 10, 64, 4])
        obj_on_a_dyn: torch.Size([4, 10, 64])
        obj_on_b_dyn: torch.Size([4, 10, 64])
        mu_depth_dyn: torch.Size([4, 10, 64, 1])
        logvar_depth_dyn: torch.Size([4, 10, 64, 1])
        mu_scale_dyn: torch.Size([4, 10, 64, 2])
        logvar_scale_dyn: torch.Size([4, 10, 64, 2])
        mu_bg_dyn: torch.Size([4, 10, 5])
        logvar_bg_dyn: torch.Size([4, 10, 5])
        mu_context_dyn: torch.Size([4, 10, 65, 7])
        logvar_context_dyn: torch.Size([4, 10, 65, 7])
        mu_score: torch.Size([4, 11, 64, 1])
        logvar_score: torch.Size([4, 11, 64, 1])
        z_score: torch.Size([4, 11, 64, 1])
        """

        # loss calculation
        all_losses = model.calc_elbo(x, model_output, beta_kl=beta_kl,
                                     beta_rec=beta_rec, kl_balance=kl_balance, beta_dyn=beta_dyn,
                                     num_static=num_static_frames,
                                     recon_loss_type="mse", warmup=warmup, beta_obj=beta_obj, done_mask=ep_done_mask)
        # let's see what's inside
        print(f'model.calc_elbo(): model losses:')
        for k in all_losses.keys():
            print(f'{k}: {all_losses[k]}')
        print("----------------------------------")
        """
        output: dynamic
        model.calc_elbo(): model losses:
        loss: 113.88887023925781
        psnr: 10.76142692565918
        kl: 53.10963821411133
        kl_dyn: 969.53173828125
        loss_rec: 1031.409912109375
        obj_on_l1: 32.52882766723633
        loss_kl_kp: 24.626571655273438
        loss_kl_feat: 131.29530334472656
        loss_kl_obj_on: 13.286111831665039
        loss_kl_scale: 13.860746383666992
        loss_kl_depth: 0.023255351930856705
        loss_kl_context: 61.956878662109375
        loss_kl_score_dyn: 0.0
        loss_kl_kp_dyn: 1196.411376953125
        loss_kl_feat_dyn: 8304.888671875
        loss_kl_obj_on_dyn: 13.850823402404785
        loss_kl_scale_dyn: 1304.9859619140625
        loss_kl_depth_dyn: 2.530747175216675
        loss_obj_reg: 163.38864135742188
        """

        # sampling
        if timestep_horizon > 1:
            num_steps = 15
            cond_steps = 5
            x = torch.rand(1 * n_im_views, num_steps + cond_steps, ch, image_size, image_size, device=device)
            if action_cond:
                actions_demo = torch.rand(1 * n_im_views, num_steps + cond_steps, action_d, device=device)
                if null_actions:
                    actions_mask_demo = torch.rand(1 * n_im_views, num_steps + cond_steps, device=device) > 0.5
                else:
                    actions_mask_demo = None
            else:
                actions_demo = None
                actions_mask_demo = None
            if lang_cond:
                lang_demo = torch.randn(1, max_lang_len, lang_d, device=device)
            else:
                lang_demo = None
            if img_goal_cond:
                x_goal = torch.rand(1 * n_im_views, 1, ch, image_size, image_size, device=device)
            else:
                x_goal = None
            sample_out, sample_z_out = model.sample_from_x(x, cond_steps=cond_steps, num_steps=num_steps,
                                                           deterministic=False,
                                                           return_z=True, actions=actions_demo,
                                                           actions_mask=actions_mask_demo, lang_embed=lang_demo,
                                                           x_goal=x_goal)
            # let's see what's inside
            print(f'model.sample_from_x(): model dynamics unrolling:')
            print(f'sample_out: {sample_out.shape}')
            print(f'sample_z_out:')
            for k in sample_z_out.keys():
                if sample_z_out[k] is not None:
                    print(f'{k}: {sample_z_out[k].shape}')
            print("----------------------------------")
            """
            output:
            model.sample_from_x(): model dynamics unrolling:
            sample_out: torch.Size([1, 20, 3, 64, 64])
            sample_z_out:
            z_pos: torch.Size([1, 20, 64, 2])
            z_scale: torch.Size([1, 20, 64, 2])
            z_obj_on: torch.Size([1, 20, 64, 1])
            z_depth: torch.Size([1, 20, 64, 1])
            z_features: torch.Size([1, 20, 64, 4])
            z_context: torch.Size([1, 19, 65, 7])
            z_bg_features: torch.Size([1, 20, 5])
            z_ids: torch.Size([1, 20, 64])
            z_score: torch.Size([1, 20, 64, 1])
            z_context_posterior: torch.Size([1, 4, 65, 7])
            mu_context_posterior: torch.Size([1, 4, 65, 7])
            """
            z = sample_z_out['z_pos']
            z_scale = sample_z_out['z_scale']
            z_obj_on = sample_z_out['z_obj_on']
            z_depth = sample_z_out['z_depth']
            z_features = sample_z_out['z_features']
            z_bg_features = sample_z_out['z_bg_features']
            z_context = sample_z_out['z_context']
            z_score = sample_z_out['z_score']
            z_goal_proj = sample_z_out['z_goal_proj']
            if action_cond:
                actions_demo = torch.rand(1 * n_im_views, num_steps + cond_steps + num_steps, action_d, device=device)
                if null_actions:
                    actions_mask_demo = torch.rand(1 * n_im_views, num_steps + cond_steps + num_steps,
                                                   device=device) > 0.5
                else:
                    actions_mask_demo = None
            else:
                actions_demo = None
                actions_mask_demo = None
            z_out, rec_dyn = model.sample_from_z(z, z_scale, z_obj_on, z_depth, z_features, z_bg_features, z_context,
                                                 z_score, num_steps=num_steps, deterministic=True,
                                                 decode=True, actions=actions_demo, actions_mask=actions_mask_demo,
                                                 lang_embed=lang_demo, z_goal=z_goal_proj)
            # let's see what's inside
            print(f'model.sample_from_z(): model dynamics unrolling:')
            print(f'rec_dyn: {rec_dyn.shape}')
            print(f'z_out:')
            for k in z_out.keys():
                if z_out[k] is not None:
                    print(f'{k}: {z_out[k].shape}')
            print("----------------------------------")

            """
            output:
            model.sample_from_z(): model dynamics unrolling:
            rec_dyn: torch.Size([1, 35, 3, 64, 64])
            z_out:
            z_pos: torch.Size([1, 35, 64, 2])
            z_scale: torch.Size([1, 35, 64, 2])
            z_obj_on: torch.Size([1, 35, 64, 1])
            z_depth: torch.Size([1, 35, 64, 1])
            z_features: torch.Size([1, 35, 64, 4])
            z_context: torch.Size([1, 34, 65, 7])
            z_bg_features: torch.Size([1, 35, 5])
            z_ids: torch.Size([1, 35, 64])
            z_score: torch.Size([1, 35, 64, 1])
            """

            # context_conditioned
            num_steps = 23
            cond_steps = 5
            x = torch.rand(1 * n_im_views, num_steps + cond_steps, ch, image_size, image_size, device=device)
            if action_cond:
                actions_demo = torch.rand(1 * n_im_views, num_steps + cond_steps, action_d, device=device)
                if null_actions:
                    actions_mask_demo = torch.rand(1 * n_im_views, num_steps + cond_steps, device=device) > 0.5
                else:
                    actions_mask_demo = None
            else:
                actions_demo = None
                actions_mask_demo = None
            if lang_cond:
                lang_demo = torch.randn(1, max_lang_len, lang_d, device=device)
            else:
                lang_demo = None
            if img_goal_cond:
                x_goal = torch.rand(1 * n_im_views, 1, ch, image_size, image_size, device=device)
            else:
                x_goal = None
            sample_out, sample_z_out = model.sample_from_x(x, cond_steps=cond_steps, num_steps=num_steps,
                                                           deterministic=False,
                                                           return_z=True, use_all_ctx=True, actions=actions_demo,
                                                           actions_mask=actions_mask_demo, lang_embed=lang_demo,
                                                           x_goal=x_goal)
            # let's see what's inside
            print(f'model.sample_from_x(use_all_ctx=True): model dynamics unrolling:')
            print(f'sample_out: {sample_out.shape}')
            print(f'sample_z_out:')
            for k in sample_z_out.keys():
                if sample_z_out[k] is not None:
                    print(f'{k}: {sample_z_out[k].shape}')

            print("----------------------------------")
            """
            output:
            model.sample_from_x(use_all_ctx=True): model dynamics unrolling:
            sample_out: torch.Size([1, 28, 3, 64, 64])
            sample_z_out:
            z_pos: torch.Size([1, 28, 64, 2])
            z_scale: torch.Size([1, 28, 64, 2])
            z_obj_on: torch.Size([1, 28, 64, 1])
            z_depth: torch.Size([1, 28, 64, 1])
            z_features: torch.Size([1, 28, 64, 4])
            z_context: torch.Size([1, 27, 65, 7])
            z_bg_features: torch.Size([1, 28, 5])
            z_ids: torch.Size([1, 28, 64])
            z_score: torch.Size([1, 28, 64, 1])
            z_context_posterior: torch.Size([1, 27, 65, 7])
            mu_context_posterior: torch.Size([1, 27, 65, 7])
            """
    else:
        raise ValueError("Invalid model type: choose between ['dlp', 'lpwm']")

