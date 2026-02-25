"""
Main DLP model for single-image and dynamics.
"""

# imports
import numpy as np
# torch
import torch
import torch.nn.functional as F
import torch.nn as nn

from modules.modules import DLPEncoder, DLPDecoder, DLPContext
from modules.modules import DLPDynamics
# util functions
from utils.util_func import calc_model_size, generate_dlp_logo
from utils.loss_functions import calc_reconstruction_loss, calc_kl_beta_dist, calc_kl, LossLPIPS, calc_kl_categorical, \
    ChamferLossKL
from modules.vision_modules import rgb_to_minusoneone, minusoneone_to_rgb


class DLP(nn.Module):
    def __init__(self,
                 # Input configuration
                 cdim=3,  # Number of input image channels
                 image_size=64,  # Input image size (assumed square)
                 normalize_rgb=False,  # If True, normalize RGB to [-1, 1], else keep [0, 1]
                 n_views=1,  # number of input views (e.g., multiple cameras)

                 # Keypoint and patch configuration
                 n_kp_per_patch=1,  # Number of proposal/prior keypoints to extract per patch
                 patch_size=16,  # Size of patches for keypoint proposal network
                 anchor_s=0.25,  # Glimpse size ratio relative to image size
                 n_kp_enc=20,  # Number of posterior keypoints to learn
                 n_kp_prior=64,  # Number of keypoints to filter from prior proposals
                 warmup_n_kp_ratio=1.0,
                 mask_bg_in_enc=True,  # before encoding the bg, mask with the particles' obj_on

                 # Network configuration
                 pad_mode='zeros',  # Padding mode for CNNs ('zeros' or 'replicate')
                 dropout=0.1,  # Dropout rate for transformers

                 # Feature representation
                 features_dist='gauss',  # Distribution type for features ('gauss' or 'categorical')
                 learned_feature_dim=16,  # Dimension of learned visual features
                 learned_bg_feature_dim=None,  # Background feature dimension (if None, equals learned_feature_dim)
                 n_fg_categories=8,  # Number of foreground feature categories (if categorical)
                 n_fg_classes=4,  # Number of foreground feature classes per category
                 n_bg_categories=4,  # Number of background feature categories
                 n_bg_classes=4,  # Number of background feature classes per category

                 # Prior distributions parameters
                 scale_std=0.3,  # Prior standard deviation for scale
                 offset_std=0.2,  # Prior standard deviation for offset
                 # z_t (transparency)
                 obj_on_alpha=0.01,  # Alpha parameter for transparency Beta distribution
                 obj_on_beta=0.01,  # Beta parameter for transparency Beta distribution
                 obj_on_min=1e-4,  # Minimum concentration in Beta dist transparency value
                 obj_on_max=100,  # Maximum  concentration in Beta dist for transparency value

                 # Object decoder architecture
                 obj_res_from_fc=8,  # Initial resolution for object encoder-decoder
                 obj_ch_mult_prior=(1, 2, 3),  # Channel multipliers for prior patch encoder (kp proposal)
                 obj_ch_mult=(1, 2, 3),  # Channel multipliers for object encoder-decoder
                 obj_base_ch=32,  # Base channels for object encoder-decoder
                 obj_final_cnn_ch=32,  # Final CNN channels for object encoder-decoder

                 # Background decoder architecture
                 bg_res_from_fc=8,  # Initial resolution for background encoder-decoder
                 bg_ch_mult=(1, 2, 3),  # Channel multipliers for background encoder-decoder
                 bg_base_ch=32,  # Base channels for background decoder
                 bg_final_cnn_ch=32,  # Final CNN channels for background encoder-decoder

                 # Network architecture options
                 use_resblock=True,  # Use residual blocks in decoders
                 num_res_blocks=2,  # Number of residual blocks per resolution
                 cnn_mid_blocks=False,  # Use middle blocks in CNN
                 mlp_hidden_dim=256,  # Hidden dimension for MLPs
                 attn_norm_type='rms',  # Normalization type for attention ('rms' or 'ln')

                 # Particle interaction transformer (PINT) configuration
                 pint_enc_layers=1,  # Number of PINT encoder layers
                 pint_enc_heads=1,  # Number of PINT encoder attention heads
                 embed_init_std=0.02,  # Standard deviation for embedding initialization
                 particle_positional_embed=True,  # Use positional embeddings for particles
                 use_z_orig=True,  # Include patch center coordinates in particle features
                 particle_score=False,  # Use particle confidence score as feature
                 filtering_heuristic='none',  # Method to filter prior keypoints ('none','distance','variance','random')

                 # Dynamics configuration
                 timestep_horizon=10,  # Number of timesteps to predict ahead
                 n_static_frames=1,  # Number of initial frames for static KL optimization
                 predict_delta=False,  # Predict position deltas instead of absolute positions
                 context_dim=None,  # Context latent dimension (if None, equals learned_feature_dim)
                 ctx_dist='gauss',  # Context distribution type ('gauss' or 'categorical')
                 n_ctx_categories=8,  # Number of context categories (if categorical)
                 n_ctx_classes=4,  # Number of context classes per category
                 causal_ctx=True,  # Use causal attention for context modeling
                 ctx_pool_mode='none',  # Context pooling mode ('none' = per-particle context)
                 global_ctx_pool=False,  # learn global latent context in addition to per-particle context
                 # global_ctx_pool: EXPERIMENTAL, NOT USED IN THE PAPER
                 # EXPERIMENTAL, NOT USED IN THE PAPER:
                 pool_ctx_dim=7,  # pool dimension for the global ctx latent
                 n_pool_ctx_categories=8,  # Number of global context categories (if categorical)
                 n_pool_ctx_classes=4,  # Number of global context classes per category
                 global_local_fuse_mode='none',  # concatenate/add global and local z_ctx to condition the dynamics
                 condition_local_on_global=True,  # condition z_context on z_context_global
                 # END EXPERIMETNAL

                 # Context and dynamics transformer configuration
                 pint_dyn_layers=6,  # Number of dynamics transformer layers
                 pint_dyn_heads=8,  # Number of dynamics transformer heads
                 pint_dim=512,  # Hidden dimension for PINT
                 pint_ctx_layers=4,  # Number of context transformer layers
                 pint_ctx_heads=8,  # Number of context transformer heads

                 # external conditioning
                 action_condition=False,  # condition on actions
                 action_dim=0,  # dimension of input actions
                 null_action_embed=False,  # learn a "no-input-action" embedding, to learn on action-free videos as well
                 random_action_condition=False,  # condition on random actions
                 random_action_dim=0,  # dimension of sampled random actions
                 action_in_ctx_module=True,  # use action to condition context generation
                 language_condition=False,  # condition on language embedding
                 language_embed_dim=0,  # embedding dimension for each token
                 language_max_len=64,  # maximum tokens per prompt
                 img_goal_condition=False,  # image as goal conditioning for dynamics

                 # initialization
                 init_zero_bias=True,  # zero bias for conv and linear layers
                 init_ssm_last_layer=True,  # spatial softmax initialization
                 init_conv_layers=True,  # initialize conv layers with normal dist
                 init_conv_fg_std=0.02,  # std for conv fg normal dist
                 init_conv_bg_std=0.005,  # std for conv bg normal dist (<fg -> prioritize fg in learning)
                 ):
        super(DLP, self).__init__()
        """
        Args:
        cdim (int): Number of input image channels. Defaults to 3.
        image_size (int): Size of input images (assumed square). Defaults to 64.
        normalize_rgb (bool): Normalize RGB values to [-1, 1] instead of [0, 1]. Defaults to False.
        n_kp_per_patch (int): Number of keypoints to extract per patch. Defaults to 1.
        patch_size (int): Size of patches for keypoint proposal network. Defaults to 16.
        anchor_s (float): Glimpse size as ratio of image_size (e.g., 0.25 for 32px glimpse on 128px image). Defaults to 0.25.
        n_kp_enc (int): Number of posterior keypoints to learn. Defaults to 20.
        n_kp_prior (int): Number of keypoints to filter from prior proposals. Defaults to 64.
        pad_mode (str): Padding mode for CNNs ('zeros' or 'replicate'). Defaults to 'zeros'.
        dropout (float): Dropout rate for transformers. Defaults to 0.1.
        features_dist (str): Distribution type for features ('gauss' or 'categorical'). Defaults to 'gauss'.
        learned_feature_dim (int): Dimension of learned visual features. Defaults to 16.
        learned_bg_feature_dim (Optional[int]): Background feature dimension. If None, equals learned_feature_dim. Defaults to None.
        n_fg_categories (int): Number of foreground feature categories if categorical. Defaults to 8.
        n_fg_classes (int): Number of foreground feature classes per category. Defaults to 4.
        n_bg_categories (int): Number of background feature categories. Defaults to 4.
        n_bg_classes (int): Number of background feature classes per category. Defaults to 4.
        scale_std (float): Prior standard deviation for scale. Defaults to 0.3.
        offset_std (float): Prior standard deviation for offset. Defaults to 0.2.
        obj_on_alpha (float): Alpha parameter for transparency Beta distribution. Defaults to 0.01.
        obj_on_beta (float): Beta parameter for transparency Beta distribution. Defaults to 0.01.
        obj_on_min (float): Minimum concentration value in Beta dist for transparency value. Defaults to 1e-4.
        obj_on_max (float): Maximum concentration value in Beta dist transparency value. Defaults to 100.
        obj_res_from_fc (int): Initial resolution for object encoder-decoder. Defaults to 8.
        obj_ch_mult_prior (tuple): Channel multipliers for prior patch encoder (kp proposals). Defaults to (1, 2, 3).
        obj_ch_mult (tuple): Channel multipliers for object encoder-decoder. Defaults to (1, 2, 3).
        obj_base_ch (int): Base channels for object encoder-decoder. Defaults to 32.
        obj_final_cnn_ch (int): Final CNN channels for object encoder-decoder. Defaults to 32.
        bg_res_from_fc (int): Initial resolution for background encoder-decoder. Defaults to 8.
        bg_ch_mult (tuple): Channel multipliers for background encoder-decoder. Defaults to (1, 2, 3).
        bg_base_ch (int): Base channels for background encoder-decoder. Defaults to 32.
        bg_final_cnn_ch (int): Final CNN channels for background encoder-decoder. Defaults to 32.
        use_resblock (bool): Use residual blocks in encoders-decoders. Defaults to True.
        num_res_blocks (int): Number of residual blocks per resolution. Defaults to 2.
        cnn_mid_blocks (bool): Use middle blocks in CNN. Defaults to False.
        mlp_hidden_dim (int): Hidden dimension for MLPs. Defaults to 256.
        attn_norm_type (str): Normalization type for attention ('rms' or 'layer'). Defaults to 'rms'.
        pint_enc_layers (int): Number of PINT encoder layers. Defaults to 1.
        pint_enc_heads (int): Number of PINT encoder attention heads. Defaults to 1.
        embed_init_std (float): Standard deviation for embedding initialization. Defaults to 0.2.
        particle_positional_embed (bool): Use positional embeddings for particles. Defaults to True.
        use_z_orig (bool): Include patch center coordinates in particle features. Defaults to True.
        particle_score (bool): Use particle confidence score as feature. Defaults to False.
        filtering_heuristic (str): Method to filter prior keypoints ('none','distance','variance','random'). Defaults to 'none'.
        timestep_horizon (int): Number of timesteps to predict ahead. Defaults to 10.
        n_static_frames (int): Number of initial frames for static KL optimization. Defaults to 1.
        predict_delta (bool): Predict position deltas instead of absolute positions. Defaults to False.
        context_dim (Optional[int]): Context latent dimension. If None, equals learned_feature_dim. Defaults to None.
        ctx_dist (str): Context distribution type ('gauss' or 'categorical'). Defaults to 'gauss'.
        n_ctx_categories (int): Number of context categories if categorical. Defaults to 8.
        n_ctx_classes (int): Number of context classes per category. Defaults to 4.
        causal_ctx (bool): Use causal attention for context modeling. Defaults to True.
        ctx_pool_mode (str): Context pooling mode ('none' = per-particle context). Defaults to 'none'.
        pint_dyn_layers (int): Number of dynamics transformer layers. Defaults to 6.
        pint_dyn_heads (int): Number of dynamics transformer heads. Defaults to 8.
        pint_dim (int): Hidden dimension for PINT. Defaults to 512.
        pint_ctx_layers (int): Number of context transformer layers. Defaults to 4.
        pint_ctx_heads (int): Number of context transformer heads. Defaults to 8.
        
        Example: see in models.py after model definition
        ----
        Deep Latent Particles (DLP) Model

        DLP is an unsupervised/self-supervised object-centric model that decomposes input images into a set
        of latent particles. Each particle represents a local region in the image and is characterized by:
        - Position (x,y): 2D coordinate-keypoint (Gaussian distributed)
        - Scale: 2D bounding box dimensions (Gaussian distributed)
        - Depth: Local depth ordering parameter (Gaussian distributed)
        - Transparency: Visibility parameter in [0,1] (Beta distributed)
        - Features: Visual features within the bounding box (Gaussian or Categorical distributed)

        The background is modeled as a single particle with its own feature dimension.

        For dynamic scenes (LPWM), the model includes latent context variables that capture transitions
        between particles in consecutive timesteps, similar to latent actions.

        Pipeline:
        1. Prior Network: Proposes n_kp_prior keypoints by:
           - Processing image patches through a CNN
           - Using spatial-softmax to locate highest activations
           - Generating keypoint proposals per activation map

        2. Posterior Network: Filters n_kp_enc keypoints by:
           - Processing each proposal to extract particle attributes
           - Optionally filtering based on keypoint variance (confidence)
           - Modeling positions as offsets from prior keypoints

        3. Background Processing:
           - Creates background mask using particle transparency
           - Masks out regions modeled by active particles
           - Encodes masked background separately

        4. Decoding:
           - Decodes particles and background separately
           - Stitches complete image using differentiable spatial transformer network (STN)

        5. Dynamic Modeling (LPWM):
           - Context encoder with shared causal transformer backbone:
             * Posterior (inverse model): p(c_t|z_t+1, z_t)
             * Prior (policy): p(c_t|z_t)
           - Dynamics module: p(z_t+1|z_t, c_t)
           - Uses AdaLN conditioning for particle transitions
           - Optimizes KL divergence between posterior and prior

        Note: Patch extraction and stitching use differentiable spatial transformer networks (STN).
        """
        self.cdim = cdim  # number of input image channels
        self.image_size = image_size
        self.normalize_rgb = normalize_rgb  # normalize to [-1, 1] or keep [0, 1]
        self.n_views = n_views  # number of input views (e.g., multiple cameras)
        self.dropout = dropout
        self.num_patches = int((image_size // patch_size) ** 2)
        self.filter_particles_in_decoder = (timestep_horizon > 1)
        self.n_kp_per_patch = n_kp_per_patch
        self.n_kp_total = self.n_kp_per_patch * self.num_patches
        self.n_kp_prior = min(self.n_kp_total, n_kp_prior)
        self.n_kp_enc = self.n_kp_prior if self.filter_particles_in_decoder else n_kp_enc
        self.n_kp_dec = n_kp_enc
        self.warmup_n_kp_ratio = warmup_n_kp_ratio
        self.kp_range = (-1, 1)
        self.kp_activation = 'tanh'  # since keypoints are in [-1, 1], we use tanh activation for kp heads
        self.anchor_s = anchor_s  # posterior patch ratio, i.e., anchor size, glimpse-size = anchor_s * image_size
        self.patch_size = patch_size  # prior patch size, to propose prior keypoints
        self.obj_patch_size = np.round(self.anchor_s * (image_size - 1)).astype(int)
        self.mask_bg_in_enc = mask_bg_in_enc  # before encoding the bg, mask with the particles' obj_on

        self.features_dist = features_dist
        self.n_fg_categories = n_fg_categories
        self.n_fg_classes = n_fg_classes
        self.n_bg_categories = n_bg_categories
        self.n_bg_classes = n_bg_classes
        if self.features_dist == 'categorical':
            self.learned_feature_dim = int(self.n_fg_categories * self.n_fg_classes)
            self.learned_bg_feature_dim = int(self.n_bg_categories * self.n_bg_classes)
        else:
            self.learned_feature_dim = learned_feature_dim
            self.learned_bg_feature_dim = learned_feature_dim if learned_bg_feature_dim is None else learned_bg_feature_dim
        assert learned_feature_dim > 0, "learned_feature_dim must be greater than 0"
        assert self.learned_bg_feature_dim > 0, "bg_learned_feature_dim must be greater than 0"

        self.obj_on_min = np.log(obj_on_min)
        self.obj_on_max = np.log(obj_on_max)

        assert filtering_heuristic in ['distance', 'variance',
                                       'random', 'none'], f'unknown filtering heuristic: {filtering_heuristic}'
        self.filtering_heuristic = filtering_heuristic

        self.particle_score = particle_score
        self.use_z_orig = use_z_orig if self.n_kp_enc == self.n_kp_prior else False

        # attention hyper-parameters
        self.attn_norm_type = attn_norm_type
        self.pint_enc_layers = pint_enc_layers
        self.pint_enc_heads = pint_enc_heads
        self.particle_positional_embed = particle_positional_embed if self.n_kp_enc == self.n_kp_prior else False
        self.embed_init_std = embed_init_std

        # cnn hyper-parameters
        self.use_resblock = use_resblock
        self.num_res_blocks = num_res_blocks
        self.cnn_mid_blocks = cnn_mid_blocks
        self.mlp_hidden_dim = mlp_hidden_dim
        self.pad_mode = pad_mode
        self.obj_res_from_fc = obj_res_from_fc
        self.obj_ch_mult_prior = obj_ch_mult_prior
        self.obj_ch_mult = obj_ch_mult
        self.obj_base_ch = obj_base_ch
        self.obj_final_cnn_ch = obj_final_cnn_ch
        self.bg_res_from_fc = bg_res_from_fc
        self.bg_ch_mult = bg_ch_mult
        self.bg_base_ch = bg_base_ch
        self.bg_final_cnn_ch = bg_final_cnn_ch

        # priors
        self.register_buffer('logvar_kp', torch.log(torch.tensor(1.0 ** 2)))
        self.register_buffer('mu_scale_prior',
                             torch.tensor(np.log(0.75 * self.anchor_s / (1 - 0.75 * self.anchor_s + 1e-5))))
        self.register_buffer('logvar_scale_p', torch.log(torch.tensor(scale_std ** 2)))
        self.register_buffer('logvar_offset_p', torch.log(torch.tensor(offset_std ** 2)))
        self.register_buffer('obj_on_a_p', torch.tensor(obj_on_alpha))
        self.register_buffer('obj_on_b_p', torch.tensor(obj_on_beta))

        # dynamics
        self.timestep_horizon = timestep_horizon
        self.is_dynamics_model = (self.timestep_horizon > 1)
        self.n_static_frames = n_static_frames
        self.predict_delta = predict_delta

        self.context_dist = ctx_dist
        assert self.context_dist in ["gauss", "beta", "categorical"], f'ctx distribution {ctx_dist} unrecognized'
        self.ctx_pool_mode = ctx_pool_mode
        assert self.ctx_pool_mode in ["none", "token", "mlp", "mean", "last"], \
            f'ctx pooling {ctx_pool_mode} unrecognized'
        self.n_ctx_categories = n_ctx_categories
        self.n_ctx_classes = n_ctx_classes
        if self.is_dynamics_model:
            if self.context_dist == 'categorical':
                self.context_dim = int(self.n_ctx_categories * self.n_ctx_classes) if self.is_dynamics_model else None
            else:
                if context_dim is None:
                    self.context_dim = learned_feature_dim
                else:
                    self.context_dim = context_dim
        else:
            self.context_dim = 0
        self.causal_ctx = causal_ctx
        # global latent context: EXPERIMENTAL, NOT USED IN THE PAPER
        self.global_ctx_pool = global_ctx_pool
        self.pool_ctx_dim = pool_ctx_dim
        self.n_pool_ctx_categories = n_pool_ctx_categories
        self.n_pool_ctx_classes = n_pool_ctx_classes
        if self.is_dynamics_model and self.context_dist == 'categorical':
            self.pool_ctx_dim = int(self.n_pool_ctx_categories * self.n_pool_ctx_classes)
        self.global_local_fuse_mode = global_local_fuse_mode
        self.condition_local_on_global = condition_local_on_global

        self.pint_dyn_layers = pint_dyn_layers
        self.pint_dyn_heads = pint_dyn_heads
        self.pint_ctx_layers = pint_ctx_layers
        self.pint_ctx_heads = pint_ctx_heads
        self.pint_dim = pint_dim
        pint_inner_dim = self.pint_dim
        pte_dropout = dropout
        max_particles = n_kp_enc + 2  # particle positional bias, +1 for the bg particle, +1 for context

        # dynamics conditioning
        # actions
        self.action_condition = action_condition
        self.action_dim = action_dim
        self.random_action_condition = random_action_condition
        self.random_action_dim = random_action_dim
        self.learn_null_action_embed = null_action_embed
        self.action_in_ctx_module = action_in_ctx_module
        # language
        self.language_condition = language_condition
        self.language_embed_dim = language_embed_dim
        self.language_max_len = language_max_len
        # image
        self.img_goal_condition = img_goal_condition

        # initialization
        self.init_zero_bias = init_zero_bias  # zero bias for conv and linear layers
        self.init_ssm_last_layer = init_ssm_last_layer  # spatial softmax initialization
        self.init_conv_layers = init_conv_layers  # initialize conv layers with normal dist
        self.init_conv_fg_std = init_conv_fg_std  # std for conv fg normal dist
        self.init_conv_bg_std = init_conv_bg_std  # std for conv bg normal dist

        # encoder
        self.encoder_module = DLPEncoder(cdim=self.cdim,
                                         image_size=self.image_size,
                                         n_views=self.n_views,
                                         patch_size=self.patch_size,
                                         n_kp_per_patch=self.n_kp_per_patch,
                                         n_kp_enc=self.n_kp_enc,
                                         n_kp_prior=self.n_kp_prior,
                                         n_kp_dec=self.n_kp_dec,
                                         warmup_n_kp_ratio=self.warmup_n_kp_ratio,
                                         kp_range=self.kp_range,
                                         kp_activation=self.kp_activation,
                                         anchor_s=self.anchor_s,
                                         mask_bg_in_enc=self.mask_bg_in_enc,
                                         features_dist=self.features_dist,
                                         n_fg_categories=n_fg_categories,
                                         n_fg_classes=n_fg_classes,
                                         n_bg_categories=n_bg_categories,
                                         n_bg_classes=n_bg_classes,
                                         obj_on_min=self.obj_on_min,
                                         obj_on_max=self.obj_on_max,
                                         use_z_orig=self.use_z_orig,
                                         learned_feature_dim=self.learned_feature_dim,
                                         learned_bg_feature_dim=self.learned_bg_feature_dim,
                                         pad_mode=self.pad_mode,
                                         obj_ch_mult_prior=self.obj_ch_mult_prior,
                                         obj_ch_mult=self.obj_ch_mult,
                                         obj_base_ch=self.obj_base_ch,
                                         obj_final_cnn_ch=self.obj_final_cnn_ch,
                                         bg_ch_mult=self.bg_ch_mult,
                                         bg_base_ch=self.bg_base_ch,
                                         bg_final_cnn_ch=self.bg_final_cnn_ch,
                                         use_resblock=self.use_resblock,
                                         num_res_blocks=self.num_res_blocks,
                                         cnn_mid_blocks=self.cnn_mid_blocks,
                                         mlp_hidden_dim=self.mlp_hidden_dim,
                                         particle_score=self.particle_score,
                                         embed_init_std=self.embed_init_std,
                                         attn_norm_type=self.attn_norm_type,
                                         pte_layers=self.pint_enc_layers,
                                         pte_heads=self.pint_enc_heads,
                                         dropout=pte_dropout,
                                         particle_positional_embed=self.particle_positional_embed,
                                         projection_dim=self.mlp_hidden_dim,
                                         interaction_obj_on=False,  # use attention for transparency
                                         interaction_depth=True,  # use attention for depth
                                         interaction_features=True,  # use attention for visual features
                                         timestep_horizon=self.timestep_horizon,
                                         add_particle_temp_embed=False,
                                         context_dim=self.context_dim,
                                         init_zero_bias=init_zero_bias,  # zero bias for conv and linear layers
                                         init_ssm_last_layer=init_ssm_last_layer,  # spatial softmax initialization
                                         init_conv_layers=init_conv_layers,  # initialize conv layers with normal dist
                                         init_conv_fg_std=init_conv_fg_std,  # std for conv fg normal dist
                                         init_conv_bg_std=init_conv_bg_std  # std for conv bg normal dist
                                         )

        # prior
        self.prior_module = self.encoder_module.prior_encoder
        particle_anchors = self.encoder_module.patch_centers[:, :-1]  # [1, n_patches, 2], no need for (0,0)-the bg
        particle_anchors = particle_anchors.unsqueeze(-2).repeat(1, 1, self.n_kp_per_patch, 1).view(1, -1, 2)
        # [1, n_patches * n_kp_per_patch, 2]

        # decoder
        self.decoder_module = DLPDecoder(cdim=cdim, image_size=image_size,
                                         learned_feature_dim=self.learned_feature_dim,
                                         learned_bg_feature_dim=self.learned_bg_feature_dim,
                                         anchor_s=anchor_s, n_kp_enc=self.n_kp_dec, pad_mode=pad_mode,
                                         context_dim=self.context_dim,
                                         obj_res_from_fc=obj_res_from_fc, obj_ch_mult=obj_ch_mult,
                                         obj_base_ch=obj_base_ch, obj_final_cnn_ch=obj_final_cnn_ch,
                                         bg_res_from_fc=bg_res_from_fc, bg_ch_mult=bg_ch_mult, bg_base_ch=bg_base_ch,
                                         bg_final_cnn_ch=bg_final_cnn_ch,
                                         num_res_blocks=num_res_blocks, decode_with_ctx=False,
                                         timestep_horizon=timestep_horizon, use_resblock=use_resblock,
                                         normalize_rgb=normalize_rgb, cnn_mid_blocks=cnn_mid_blocks,
                                         mlp_hidden_dim=mlp_hidden_dim,
                                         init_zero_bias=init_zero_bias,  # zero bias for conv and linear layers
                                         init_conv_layers=init_conv_layers,  # initialize conv layers with normal dist
                                         init_conv_fg_std=init_conv_fg_std,  # std for conv fg normal dist
                                         init_conv_bg_std=init_conv_bg_std  # std for conv bg normal dist
                                         )

        # context (latent actions)
        if self.context_dim > 0:
            self.ctx_module = DLPContext(n_kp_enc=self.n_kp_enc, dropout=pte_dropout,
                                         learned_feature_dim=self.learned_feature_dim,
                                         learned_bg_feature_dim=self.learned_bg_feature_dim,
                                         embed_init_std=embed_init_std, projection_dim=pint_inner_dim,
                                         timestep_horizon=timestep_horizon, pte_layers=pint_ctx_layers,
                                         pte_heads=pint_ctx_heads,
                                         attn_norm_type=attn_norm_type,
                                         context_dim=self.context_dim,
                                         hidden_dim=pint_inner_dim,
                                         ctx_pool_mode=self.ctx_pool_mode,
                                         bg=True, n_views=self.n_views,
                                         particle_positional_embed=particle_positional_embed,
                                         particle_score=self.particle_score,
                                         causal=self.causal_ctx, norm_layer=True,
                                         shared_logvar=False, ctx_dist=ctx_dist,
                                         n_ctx_categories=n_ctx_categories, n_ctx_classes=n_ctx_classes,
                                         particle_anchors=particle_anchors, use_z_orig=self.use_z_orig,
                                         global_ctx_pool=self.global_ctx_pool,
                                         ctx_pool_dim=self.pool_ctx_dim,
                                         n_pool_ctx_categories=self.n_pool_ctx_categories,
                                         n_pool_ctx_classes=self.n_pool_ctx_classes,
                                         global_local_fuse_mode=global_local_fuse_mode,
                                         condition_local_on_global=condition_local_on_global,
                                         # external conditioning
                                         action_condition=self.action_condition,
                                         # condition on actions
                                         action_dim=self.action_dim,  # dimension of input actions
                                         random_action_condition=self.random_action_condition,
                                         random_action_dim=self.random_action_dim,
                                         null_action_embed=self.learn_null_action_embed,
                                         # learn a "no-input-action" embedding, to learn on action-free videos as well
                                         action_as_particle=self.action_condition and not self.action_in_ctx_module,
                                         language_condition=self.language_condition,  # condition on language embedding
                                         language_embed_dim=self.language_embed_dim,
                                         # embedding dimension for each token
                                         language_max_len=self.language_max_len,  # maximum tokens per prompt
                                         img_goal_condition=self.img_goal_condition
                                         )
            self.encoder_module.ctx_enc = self.ctx_module
        else:
            self.ctx_module = None

        # dynamics
        if self.is_dynamics_model:
            dyn_activ = self.kp_activation
            ctx_cond_mode = 'adaln'
            context_decoder_dyn = self.ctx_module
            dyn_particle_anchors = particle_anchors if (self.n_kp_enc == self.n_kp_prior) else None
            if self.global_ctx_pool and self.global_local_fuse_mode == 'concat':
                dyn_ctx_dim = self.pool_ctx_dim + self.context_dim
            else:
                dyn_ctx_dim = self.context_dim
            self.dyn_module = DLPDynamics(self.learned_feature_dim,
                                          self.learned_bg_feature_dim,
                                          pint_inner_dim,
                                          pint_inner_dim,
                                          n_head=pint_dyn_heads,
                                          n_layer=pint_dyn_layers,
                                          block_size=timestep_horizon,
                                          dropout=dropout,
                                          kp_activation=dyn_activ,
                                          predict_delta=predict_delta,
                                          max_delta=1.0,
                                          positional_bias=False,
                                          max_particles=max_particles,
                                          context_dim=dyn_ctx_dim,
                                          attn_norm_type=attn_norm_type,
                                          n_fg_particles=self.n_kp_enc,
                                          ctx_pool_mode=ctx_pool_mode,
                                          particle_positional_embed=particle_positional_embed,
                                          particle_anchors=dyn_particle_anchors,
                                          particle_score=self.particle_score,
                                          init_std=self.embed_init_std,
                                          ctx_mode=ctx_cond_mode,
                                          pint_ctx_layers=pint_ctx_layers,
                                          pint_ctx_heads=pint_ctx_heads,
                                          ctx_dist=ctx_dist,
                                          n_ctx_categories=n_ctx_categories,
                                          n_ctx_classes=n_ctx_classes,
                                          context_decoder=context_decoder_dyn,
                                          features_dist=self.features_dist,
                                          n_fg_categories=n_fg_categories,
                                          n_fg_classes=n_fg_classes,
                                          n_bg_categories=n_bg_categories,
                                          n_bg_classes=n_bg_classes,
                                          scale_init=self.anchor_s,
                                          obj_on_min=self.obj_on_min,
                                          obj_on_max=self.obj_on_max,
                                          use_z_orig=self.use_z_orig,
                                          n_views=self.n_views,
                                          # external conditioning
                                          action_condition=(self.action_condition and not self.action_in_ctx_module),
                                          # condition on actions
                                          action_dim=self.action_dim,  # dimension of input actions
                                          random_action_condition=(
                                                  self.random_action_condition and not self.action_in_ctx_module),
                                          random_action_dim=self.random_action_dim,
                                          null_action_embed=(
                                                  self.learn_null_action_embed and not self.action_in_ctx_module),
                                          # learn a "no-input-action" embedding, to learn on action-free videos as well
                                          )
        else:
            self.dyn_module = nn.Identity()
        self.init_weights()

    def init_weights(self):
        if self.init_zero_bias:
            # all conv, linear layers are specific to modules
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
        self.prior_module.init_weights()
        self.encoder_module.init_weights()
        self.decoder_module.init_weights()
        if isinstance(self.ctx_module, DLPContext):
            self.ctx_module.init_weights()
        if isinstance(self.dyn_module, DLPDynamics):
            self.dyn_module.init_weights()

    def info(self):
        # Create sections for different parts of the model
        def create_section_header(title):
            return f"\n{'=' * 80}\n{title}\n{'=' * 80}\n"

        def format_row(label, value):
            return f"{label:<45} | {value}"

        sections = []

        # DLP Logo
        sections.append(generate_dlp_logo())

        # [Previous sections remain the same until Latent Information]
        basic_config = [
            ("Prior Keypoint Filtering", f"{self.n_kp_total} -> {self.n_kp_prior}"),
            ("Filtering Heuristic", self.filtering_heuristic),
            ("Prior Patch Size", self.patch_size),
            ("Posterior Particles (Encoder)", self.n_kp_enc),
            ("Posterior Particles (Decoder)", self.n_kp_dec),
            ("Filter Particles in Decoder", self.filter_particles_in_decoder),
            ("Include Origin Patch Center", self.use_z_orig),
            ("Posterior Object Patch Size", self.obj_patch_size),
            ("Attention Layer Normalization", self.attn_norm_type),
            ("Number of Input Views (Cameras)", self.n_views),
        ]

        sections.append(create_section_header("Basic Configuration"))
        sections.extend(format_row(label, value) for label, value in basic_config)

        # Feature Distribution Information
        sections.append(create_section_header("Feature Distribution"))
        if self.features_dist == 'categorical':
            feature_info = [
                ("Distribution Type", self.features_dist),
                ("Foreground Dimension", self.learned_feature_dim),
                ("Background Dimension", self.learned_bg_feature_dim),
                ("Foreground Categories/Classes", f"{self.n_fg_categories}/{self.n_fg_classes}"),
                ("Background Categories/Classes", f"{self.n_bg_categories}/{self.n_bg_classes}")
            ]
        else:
            feature_info = [
                ("Distribution Type", self.features_dist),
                ("Particle Visual Feature Dimension", self.learned_feature_dim),
                ("Background Visual Feature Dimension", self.learned_bg_feature_dim)
            ]
        sections.extend(format_row(label, value) for label, value in feature_info)

        # Context Distribution
        sections.append(create_section_header("Context Information"))
        if self.context_dist == 'categorical':
            context_info = [
                ("Distribution Type", self.context_dist),
                ("Dimension", self.context_dim),
                ("Categories/Classes", f"{self.n_ctx_categories}/{self.n_ctx_classes}")
            ]
        else:
            context_info = [
                ("Distribution Type", self.context_dist),
                ("Dimension", self.context_dim)
            ]

        if self.ctx_module is not None:
            ctx_size_dict = calc_model_size(self.ctx_module)
            ctx_n_params = ctx_size_dict['n_params']
            context_info.append(("CTX Module Parameters", f"{ctx_n_params} ({ctx_n_params / 1e6:.4f}M)"))

        sections.extend(format_row(label, value) for label, value in context_info)

        # random actions
        sections.append(create_section_header("Random Action Conditioning via AdaLN Information"))
        if self.random_action_condition:
            rand_action_info = [
                ("Random Action Conditioning", self.random_action_condition),
                ("Dimension", self.random_action_dim),
                ("Condition in CTX Module", self.action_in_ctx_module),
            ]
        else:
            rand_action_info = [
                ("Random Action Conditioning", self.random_action_condition),
            ]
        sections.extend(format_row(label, value) for label, value in rand_action_info)

        # actions
        sections.append(create_section_header("Action Conditioning via AdaLN Information"))
        if self.action_condition:
            action_info = [
                ("Action Conditioning", self.action_condition),
                ("Dimension", self.action_dim),
                ("Condition in CTX Module", self.action_in_ctx_module),
                ("Learn Null Embedding for Actions", self.learn_null_action_embed),
            ]
        else:
            action_info = [
                ("Action Conditioning", self.action_condition),
            ]
        sections.extend(format_row(label, value) for label, value in action_info)

        # language
        sections.append(create_section_header("Language Conditioning"))
        if self.language_condition:
            lang_info = [
                ("Language Conditioning", self.language_condition),
                ("Dimension", self.language_embed_dim),
                ("Maximum Language Tokens", self.language_max_len),
            ]
        else:
            lang_info = [
                ("Language Conditioning", self.language_condition),
            ]
        sections.extend(format_row(label, value) for label, value in lang_info)

        # image goal conditioning
        sections.append(create_section_header("Image Goal Conditioning"))
        lang_info = [
            ("Image Goal Conditioning", self.img_goal_condition),
        ]
        sections.extend(format_row(label, value) for label, value in lang_info)

        # CNN Architecture
        sections.append(create_section_header("CNN Architecture"))
        cnn_info = [
            ("Prior CNN Pre-pool Output Size", self.prior_module.enc.conv_output_size),
            ("Object CNN Output Shape", self.encoder_module.particle_enc.particle_features_enc.cnn_out_shape),
            ("Background CNN Output Shape", self.encoder_module.bg_encoder.cnn_out_shape),
            ("Decoder Background Upsamples", self.decoder_module.num_bg_upsample),
            ("Decoder Object Upsamples", self.decoder_module.num_obj_upsample)
        ]
        sections.extend(format_row(label, value) for label, value in cnn_info)

        # Latent Information
        sections.append(create_section_header("Latent Space Information"))
        context_coeff = 1 if self.ctx_pool_mode != 'none' else (self.n_kp_enc + 1)
        latent_dim = ((6 + self.learned_feature_dim) * self.n_kp_enc
                      + self.learned_bg_feature_dim
                      + context_coeff * self.context_dim)

        sections.append(
            format_row("Encoder Particle Features", self.encoder_module.particle_enc.particle_features_enc.info))
        sections.append(format_row("Background Encoder", self.encoder_module.bg_encoder.info))
        if self.encoder_module.particle_inter_enc is not None:
            sections.append(format_row("Particle Intermediate Encoder", self.encoder_module.particle_inter_enc.info))
        sections.append(format_row("Particle Decoder", self.decoder_module.particle_dec.info))
        sections.append(format_row("Background Decoder", self.decoder_module.bg_dec.info))

        # Add latent dimension with formula
        latent_formula = (f"(6 + {self.learned_feature_dim}) * {self.n_kp_enc} + "
                          f"{self.learned_bg_feature_dim} + "
                          f"{context_coeff} * {self.context_dim}")
        sections.append(format_row("Latent Dimension Formula", latent_formula))
        sections.append(format_row("Total Latent Dimension", f"{latent_formula} = {latent_dim}"))

        # Dynamic Module Information (if applicable)
        if self.is_dynamics_model:
            sections.append(create_section_header("Dynamics Module Information"))
            pint_size_dict = calc_model_size(self.dyn_module)
            pint_n_params = pint_size_dict['n_params']

            if self.context_dim > 0:
                ctx_size_dict = calc_model_size(self.dyn_module.context_decoder)
                pint_n_params = pint_n_params - ctx_size_dict['n_params']

            dynamics_info = [
                ("Dropout (PINT)", self.dropout),
                ("Burn-in Frames", self.n_static_frames),
                ("Prior Predicts Delta", self.predict_delta),
                ("PINT Relative Positional Bias", self.dyn_module.particle_transformer.positional_bias),
                ("PINT Parameters", f"{pint_n_params} ({pint_n_params / 1e6:.4f}M)")
            ]
            sections.extend(format_row(label, value) for label, value in dynamics_info)

        # Model Size Information
        sections.append(create_section_header("Model Size Information"))
        prior_size_dict = calc_model_size(self.prior_module)
        enc_size_dict = calc_model_size(self.encoder_module)
        enc_n_params = enc_size_dict['n_params']
        if self.ctx_module is not None:
            ctx_size_dict = calc_model_size(self.encoder_module.ctx_enc)
            enc_n_params = enc_n_params - ctx_size_dict['n_params']
        dec_size_dict = calc_model_size(self.decoder_module)
        size_dict = calc_model_size(self)

        model_size_info = [
            ("Prior Parameters", f"{prior_size_dict['n_params']} ({prior_size_dict['n_params'] / 1e6:.4f}M)"),
            ("Encoder Parameters", f"{enc_n_params} ({enc_n_params / 1e6:.4f}M)"),
            ("Decoder Parameters", f"{dec_size_dict['n_params']} ({dec_size_dict['n_params'] / 1e6:.4f}M)"),
            ("Total Parameters", f"{size_dict['n_params']} ({size_dict['n_params'] / 1e6:.4f}M)"),
            ("Estimated Size on Disk", f"{size_dict['size_mb']:.3f}MB")
        ]
        sections.extend(format_row(label, value) for label, value in model_size_info)

        return "\n".join(sections)

    def encode_prior(self, x):
        return self.prior_module(x)

    def encode_all(self, x, deterministic=False, warmup=False, actions=None, actions_mask=None, lang_embed=None,
                   x_goal=None):
        """
        encode posterior particles
        """
        # x: [bs, timestep_horizon, ch, h, w
        # # make sure x is [bs, T, ch, h, w]
        # x_goal: [bs, 1, ch, h, w]
        if len(x.shape) == 4:
            # that means x: [bs, ch, h, w]
            x = x.unsqueeze(1)  # -> [bs, T=1, ch, h, w]]
        enc_dict = self.encoder_module(x, deterministic, warmup, actions=actions, actions_mask=actions_mask,
                                       lang_embed=lang_embed, x_goal=x_goal)
        cropped_objects = enc_dict['cropped_objects']
        if self.normalize_rgb:
            cropped_objects_rgb = minusoneone_to_rgb(cropped_objects)
        else:
            cropped_objects_rgb = cropped_objects
        enc_dict['cropped_objects_rgb'] = cropped_objects_rgb
        return enc_dict

    def decode_all(self, z, z_scale, z_features, obj_on_sample, z_depth, z_bg_features, z_ctx,
                   warmup=False, filter_key=None):
        if filter_key is not None:
            orig_shape = z.shape
            # filter_key: [batch_size, n_kp]
            if len(filter_key.shape) == 3:
                # [bs, T, n_kp]
                filter_key = filter_key.view(-1, filter_key.shape[-1])
            if len(orig_shape) == 4:
                # [bs, T, n_kp, ...] -> [bs * T, n_kp, ...]
                z = z.view(-1, *z.shape[2:])
                z_scale = z_scale.view(-1, *z_scale.shape[2:])
                z_depth = z_depth.view(-1, *z_depth.shape[2:])
                z_features = z_features.view(-1, *z_features.shape[2:])
                obj_on_sample = obj_on_sample.view(-1, *obj_on_sample.shape[2:])
            # k = self.n_kp_dec
            # discourage "lazy" particles that don't move by choking the model to use less particles for reconstruction
            k = self.n_kp_dec if not warmup else min(self.n_kp_dec, int(self.warmup_n_kp_ratio * self.n_kp_enc))
            _, embed_ind = torch.topk(filter_key, k=k, dim=-1, largest=False)
            # make selection
            batch_ind = torch.arange(z.shape[0], device=z.device)[:, None]

            z = z[batch_ind, embed_ind]  # [bs * T, n_kp_dec, 2]
            z_scale = z_scale[batch_ind, embed_ind]  # [bs * T, n_kp_dec, 2]
            obj_on_sample = obj_on_sample[batch_ind, embed_ind]  # [bs * T, n_kp_dec, 1]
            z_depth = z_depth[batch_ind, embed_ind]  # [bs * T, n_kp_dec, 1]
            z_features = z_features[batch_ind, embed_ind]  # [bs * T, n_kp_dec, features_dim]

            if len(orig_shape) == 4:
                # [bs * T, n_kp, ...] -> [bs, T, n_kp, ...]
                z = z.reshape(orig_shape[0], orig_shape[1], *z.shape[1:])
                z_scale = z_scale.reshape(orig_shape[0], orig_shape[1], *z_scale.shape[1:])
                z_depth = z_depth.reshape(orig_shape[0], orig_shape[1], *z_depth.shape[1:])
                z_features = z_features.reshape(orig_shape[0], orig_shape[1], *z_features.shape[1:])
                obj_on_sample = obj_on_sample.reshape(orig_shape[0], orig_shape[1], *obj_on_sample.shape[1:])

        dec_dict = self.decoder_module(z, z_scale, z_features, obj_on_sample, z_depth, z_bg_features, z_ctx, warmup)

        dec_objects = dec_dict['dec_objects']
        dec_objects_trans = dec_dict['dec_objects_trans']
        rec = dec_dict['rec']
        bg_rec = dec_dict['bg_rec']
        if self.normalize_rgb:
            rec_rgb = minusoneone_to_rgb(rec)
            bg_rec_rgb = minusoneone_to_rgb(bg_rec)
            dec_objects_trans = minusoneone_to_rgb(dec_objects_trans)
            dec_objects_rgb = minusoneone_to_rgb(dec_objects)
        else:
            rec_rgb = rec
            bg_rec_rgb = bg_rec
            dec_objects_rgb = dec_objects

        dec_dict['rec_rgb'] = rec_rgb
        dec_dict['bg_rgb'] = bg_rec_rgb
        dec_dict['dec_objects_trans'] = dec_objects_trans
        dec_dict['dec_objects_original_rgb'] = dec_objects_rgb
        return dec_dict

    def sample_from_x(self, x, num_steps=10, deterministic=True, cond_steps=None, return_z=False, use_all_ctx=False,
                      actions=None, actions_mask=None, lang_embed=None, x_goal=None, decode=True, n_pred_eq_gt=True,
                      return_context_posterior=False):
        """
        (Conditional) Sampling from LPWM: x is the conditional frames, encoded to latent particles
         which are unrolled to the future with PINT, and the predicted particles are then decoded to a sequence
         of RGB images.
        """
        # use_all_ctx: if True, will encode context from the entire trajectory to condition the prediction
        # this is meant to see if the model is able to follow conditions and reconstruct stochastic trajectories
        # that involve latent stochastic actions
        # x: [bs, T, ...]
        assert self.is_dynamics_model, f"model timesteps: {self.timestep_horizon} -> non-dynamics model"
        # encode-decode
        batch_size, timestep_horizon_all = x.size(0), x.size(1)
        timestep_horizon = self.timestep_horizon if cond_steps is None else cond_steps

        if self.normalize_rgb:
            x = rgb_to_minusoneone(x)

        if self.action_condition and actions is not None:
            actions_enc = actions[:, :timestep_horizon].contiguous()
        else:
            actions_enc = None
        if self.action_condition and actions_mask is not None:
            actions_mask_enc = actions_mask[:, :timestep_horizon].contiguous()
        else:
            actions_mask_enc = None

        # encode particles
        enc_dict = self.encode_all(x[:, :timestep_horizon].contiguous(), deterministic=True, actions=actions_enc,
                                   actions_mask=actions_mask_enc, lang_embed=lang_embed, x_goal=x_goal)

        x_in = x[:, :timestep_horizon].reshape(-1, *x.shape[2:])  # [bs * T, ...]
        # encoder
        z = enc_dict['z']
        z_features = enc_dict['z_features']
        z_bg_features = enc_dict['z_bg_features']
        z_obj_on = enc_dict['obj_on']
        z_depth = enc_dict['z_depth']
        z_scale = enc_dict['z_scale']
        z_context = enc_dict['z_context']
        z_score = enc_dict['z_score']
        z_goal_proj = enc_dict['z_goal_proj']  # [bs, 1, N, proj_dim] if img_goal_cond else None
        filter_key = enc_dict['z_base_var'].sum(-1) if self.filter_particles_in_decoder else None
        # "latent actions", [bs, T-1, ctx_dim], ctx models every pair of consecutive steps
        if timestep_horizon_all == 1:
            z_context = None
        else:
            z_context = z_context[:, 1:].contiguous() if z_context is not None else None

        if timestep_horizon_all > 1 and use_all_ctx and z_context is not None:
            # encode context from the entire trajectory
            while z_context.shape[1] < timestep_horizon_all - 1:
                causal = self.causal_ctx
                if causal:
                    end_step = z_context.shape[1] + 1
                    start_step = max(end_step - self.timestep_horizon, 0)
                    if self.action_condition and actions is not None:
                        actions_enc = actions[:, start_step:end_step + 1].contiguous()
                    else:
                        actions_enc = None
                    if self.action_condition and actions_mask is not None:
                        actions_mask_enc = actions_mask[:, start_step:end_step + 1].contiguous()
                    else:
                        actions_mask_enc = None
                    enc_dict = self.encode_all(x[:, start_step:end_step + 1].contiguous(), deterministic,
                                               actions=actions_enc, actions_mask=actions_mask_enc, x_goal=x_goal,
                                               lang_embed=lang_embed)
                    z_context_t = enc_dict['z_context'][:, -1:].contiguous()
                    z_context = torch.cat([z_context, z_context_t], dim=1)
                else:
                    start_step = z_context.shape[1]
                    end_step = start_step + self.timestep_horizon + 1
                    enc_dict = self.encode_all(x[:, start_step:end_step].contiguous(), deterministic)
                    z_context_t = enc_dict['z_context'][:, 1:].contiguous()
                    z_context = torch.cat([z_context, z_context_t], dim=1)

        # decoder
        if decode:
            if z_context is not None:
                z_ctx = z_context[:, :timestep_horizon - 1].contiguous()
            else:
                z_ctx = None
            dec_dict = self.decode_all(z, z_scale, z_features, z_obj_on, z_depth, z_bg_features,
                                       z_ctx=z_ctx, filter_key=filter_key)
            rec = dec_dict['rec_rgb']

            rec = rec.view(batch_size, -1, *rec.shape[1:])
        else:
            rec = None

        # dynamics
        if self.action_condition and actions is not None:
            actions_dyn = actions[:, :timestep_horizon + num_steps].contiguous()
        else:
            actions_dyn = None
        if self.action_condition and actions_mask is not None:
            actions_mask_dyn = actions_mask[:, :timestep_horizon + num_steps].contiguous()
        else:
            actions_mask_dyn = None
        dyn_out = self.dyn_module.sample(z, z_scale, z_obj_on, z_depth, z_features, z_bg_features, z_context,
                                         z_score, steps=num_steps, deterministic=deterministic, actions=actions_dyn,
                                         actions_mask=actions_mask_dyn, lang_embed=lang_embed, z_goal=z_goal_proj,
                                         return_context_posterior=return_context_posterior)
        z_dyn = dyn_out['z']
        z_scale_dyn = dyn_out['z_scale']
        z_obj_on_dyn = dyn_out['z_obj_on']
        z_depth_dyn = dyn_out['z_depth']
        z_features_dyn = dyn_out['z_features']
        z_bg_features_dyn = dyn_out['z_bg_features']
        z_context_dyn = dyn_out['z_context']
        z_score_dyn = dyn_out['z_score']
        z_context_posterior = dyn_out['z_context_posterior']
        mu_context_posterior = dyn_out['mu_context_posterior']
        if return_z:
            z_ids = 1 + torch.arange(z_dyn.shape[2], device=z_dyn.device)  # num_particles, ids start from 1
            z_ids = z_ids[None, None, :].repeat(z_dyn.shape[0], z_dyn.shape[1], 1)  # [bs, T, n_particles]
            z_out = {'z_pos': z_dyn.detach(), 'z_scale': z_scale_dyn.detach(), 'z_obj_on': z_obj_on_dyn.detach(),
                     'z_depth': z_depth_dyn.detach(), 'z_features': z_features_dyn.detach(),
                     'z_context': z_context_dyn.detach(), 'z_bg_features': z_bg_features_dyn.detach(), 'z_ids': z_ids,
                     'z_score': z_score_dyn, 'z_goal_proj': z_goal_proj,
                     'z_context_posterior': z_context_posterior, 'mu_context_posterior': mu_context_posterior}
        else:
            z_out = None

        z_dyn = z_dyn[:, -num_steps:].contiguous()
        z_features_dyn = z_features_dyn[:, -num_steps:].contiguous()
        z_bg_features_dyn = z_bg_features_dyn[:, -num_steps:].contiguous()
        z_obj_on_dyn = z_obj_on_dyn[:, -num_steps:].contiguous()
        z_depth_dyn = z_depth_dyn[:, -num_steps:].contiguous()
        z_scale_dyn = z_scale_dyn[:, -num_steps:].contiguous()
        z_context_dyn = z_context_dyn[:, -num_steps:].contiguous()
        z_score_dyn = z_score_dyn[:, -num_steps:].contiguous()

        if self.filter_particles_in_decoder:
            if self.particle_score:
                filter_key = z_score_dyn.sum(-1) if len(z_score_dyn.shape) == 4 else z_score_dyn
            else:
                filter_key = (1 - z_obj_on_dyn).sum(-1) if len(z_obj_on_dyn.shape) == 4 else (1 - z_obj_on_dyn)

        else:
            filter_key = None

        if decode:
            dec_out = self.decode_all(z_dyn, z_scale_dyn, z_features_dyn, z_obj_on_dyn, z_depth_dyn, z_bg_features_dyn,
                                      z_context_dyn, filter_key=filter_key)
            rec_dyn = dec_out['rec_rgb']
            rec_dyn = rec_dyn.reshape(batch_size, -1, *rec_dyn.shape[1:])
            rec = torch.cat([rec, rec_dyn], dim=1)
            if n_pred_eq_gt:
                assert timestep_horizon_all == rec.shape[1],\
                    f"prediction {rec.shape[1]} and gt {timestep_horizon_all} frames shape don't match"
        else:
            rec = None
        if return_z:
            return rec, z_out
        return rec

    def sample_from_z(self, z, z_scale, z_obj_on, z_depth, z_features, z_bg_features, z_context,
                      z_score=None, num_steps=10, deterministic=True, decode=False, actions=None, actions_mask=None,
                      lang_embed=None, z_goal=None):
        assert self.is_dynamics_model, f"model timesteps: {self.timestep_horizon} -> non-dynamics model"
        batch_size = z.shape[0]
        # dynamics
        dyn_out = self.dyn_module.sample(z, z_scale, z_obj_on, z_depth, z_features, z_bg_features, z_context,
                                         z_score, steps=num_steps, deterministic=deterministic, actions=actions,
                                         actions_mask=actions_mask, lang_embed=lang_embed, z_goal=z_goal)
        z_dyn = dyn_out['z']
        z_scale_dyn = dyn_out['z_scale']
        z_obj_on_dyn = dyn_out['z_obj_on']
        z_depth_dyn = dyn_out['z_depth']
        z_features_dyn = dyn_out['z_features']
        z_bg_features_dyn = dyn_out['z_bg_features']
        z_context_dyn = dyn_out['z_context']
        z_score_dyn = dyn_out['z_score']
        z_ids = 1 + torch.arange(z_dyn.shape[2], device=z_dyn.device)  # num_particles, ids start from 1
        z_ids = z_ids[None, None, :].repeat(z_dyn.shape[0], z_dyn.shape[1], 1)  # [bs, T, n_particles]
        z_out = {'z_pos': z_dyn.detach(), 'z_scale': z_scale_dyn.detach(), 'z_obj_on': z_obj_on_dyn.detach(),
                 'z_depth': z_depth_dyn.detach(), 'z_features': z_features_dyn.detach(),
                 'z_context': z_context_dyn.detach(), 'z_bg_features': z_bg_features_dyn.detach(), 'z_ids': z_ids,
                 'z_score': z_score_dyn}
        if decode:
            # decode
            if self.filter_particles_in_decoder:
                if self.particle_score:
                    filter_key = z_score_dyn.sum(-1) if len(z_score_dyn.shape) == 4 else z_score_dyn
                else:
                    filter_key = (1 - z_obj_on_dyn).sum(-1) if len(z_obj_on_dyn.shape) == 4 else (1 - z_obj_on_dyn)
            else:
                filter_key = None
            dec_out = self.decode_all(z_dyn, z_scale_dyn, z_features_dyn, z_obj_on_dyn, z_depth_dyn, z_bg_features_dyn,
                                      z_context_dyn, filter_key=filter_key)
            rec_dyn = dec_out['rec_rgb']
            rec_dyn = rec_dyn.reshape(batch_size, -1, *rec_dyn.shape[1:])
        else:
            rec_dyn = None
        return z_out, rec_dyn

    def forward(self, x, deterministic=False, warmup=False, with_loss=False, beta_kl=0.1, beta_dyn=0.1,
                beta_rec=1.0, kl_balance=0.001, dynamic_discount=None, recon_loss_type="mse", recon_loss_func=None,
                balance=0.5, beta_dyn_rec=1.0, num_static=None, actions=None, actions_mask=None, lang_embed=None,
                beta_obj=0.0, done_mask=None, x_goal=None):
        if len(x.shape) == 4:
            # x: [bs, ch, h, w]
            batch_size = x.size(0)
            timestep_horizon = 1
            x = x.unsqueeze(1)
        else:
            # x: [bs, T + 1, ch, h, w]
            batch_size, timestep_horizon = x.size(0), x.size(1)

        if self.normalize_rgb:
            x = rgb_to_minusoneone(x)
            if x_goal is not None:
                x_goal = rgb_to_minusoneone(x_goal)

        # encode particles
        enc_dict = self.encode_all(x, deterministic, warmup=warmup, actions=actions, actions_mask=actions_mask,
                                   lang_embed=lang_embed, x_goal=x_goal)

        # unpack encoder output: [bs, T, ...]
        kp_p = enc_dict['kp_p']
        mu_anchor = enc_dict['mu_anchor']  # mu_anchor = z_base = top-k(kp_p)
        logvar_anchor = enc_dict['logvar_anchor']
        z_base = enc_dict['z_base']
        z_base_var = enc_dict['z_base_var']
        z = enc_dict['z']
        mu_offset = enc_dict['mu_offset']
        logvar_offset = enc_dict['logvar_offset']
        z_offset = enc_dict['z_offset']
        mu_tot = enc_dict['mu_tot']
        mu_features = enc_dict['mu_features']
        logvar_features = enc_dict['logvar_features']
        z_features = enc_dict['z_features']
        # cropped_objects = enc_dict['cropped_objects_original']
        obj_on_a = enc_dict['obj_on_a']
        obj_on_b = enc_dict['obj_on_b']
        z_obj_on = enc_dict['obj_on']
        mu_obj_on = enc_dict['mu_obj_on']
        mu_depth = enc_dict['mu_depth']
        logvar_depth = enc_dict['logvar_depth']
        z_depth = enc_dict['z_depth']
        mu_scale = enc_dict['mu_scale']
        logvar_scale = enc_dict['logvar_scale']
        z_scale = enc_dict['z_scale']
        mu_bg_features = enc_dict['mu_bg_features']
        logvar_bg_features = enc_dict['logvar_bg_features']
        z_bg_features = enc_dict['z_bg_features']
        mu_context_global = enc_dict['mu_context_global']
        logvar_context_global = enc_dict['logvar_context_global']
        z_context_global = enc_dict['z_context_global']
        mu_context = enc_dict['mu_context']
        logvar_context = enc_dict['logvar_context']
        z_context = enc_dict['z_context']
        cropped_objects = enc_dict['cropped_objects']
        cropped_objects_rgb = enc_dict['cropped_objects_rgb']

        mu_score = enc_dict['mu_score']
        logvar_score = enc_dict['logvar_score']
        z_score = enc_dict['z_score']

        if self.context_dim > 0:
            mu_context_dyn = enc_dict['mu_context_dyn'][:, :-1]
            if self.context_dist != 'categorical':
                logvar_context_dyn = enc_dict['logvar_context_dyn'][:, :-1]
            else:
                logvar_context_dyn = mu_context_dyn
            z_context_dyn = enc_dict['z_context_dyn'][:, :-1]

            if self.global_ctx_pool:
                mu_context_global_dyn = enc_dict['mu_context_global_dyn'][:, :-1]
                if self.context_dist != 'categorical':
                    logvar_context_global_dyn = enc_dict['logvar_context_global_dyn'][:, :-1]
                else:
                    logvar_context_global_dyn = mu_context_global_dyn
                z_context_global_dyn = enc_dict['z_context_global_dyn'][:, :-1]
            else:
                mu_context_global_dyn = logvar_context_global_dyn = z_context_global_dyn = None
        else:
            mu_context_dyn = logvar_context_dyn = z_context_dyn = None
            mu_context_global_dyn = logvar_context_global_dyn = z_context_global_dyn = None

        filter_key = z_base_var.sum(-1) if (
                self.filter_particles_in_decoder and self.n_kp_enc != self.n_kp_dec) else None
        dec_dict = self.decode_all(z, z_scale, z_features, z_obj_on, z_depth, z_bg_features, z_context,
                                   warmup, filter_key=filter_key)

        bg_mask = dec_dict['bg_mask']
        dec_objects = dec_dict['dec_objects']
        dec_objects_trans = dec_dict['dec_objects_trans']
        alpha_masks = dec_dict['alpha_masks']
        rec = dec_dict['rec']
        bg_rec = dec_dict['bg_rec']

        rec_rgb = dec_dict['rec_rgb']
        bg_rec_rgb = dec_dict['bg_rgb']
        dec_objects_rgb = dec_dict['dec_objects_original_rgb']

        # dynamics - all but the last timestep
        if self.is_dynamics_model:
            detach_dyn_inputs = False
            # forward PINT
            # [bs, T-1, n_kp, attribute/feature_dim]

            z_dyn = z_base + z_offset  # = z, but can now detach z_base if more stable
            z_v = z_dyn[:, :-1].detach() if detach_dyn_inputs else z_dyn[:, :-1]
            z_scale_v = z_scale[:, :-1].detach() if detach_dyn_inputs else z_scale[:, :-1]
            z_obj_on_v = z_obj_on[:, :-1].detach() if detach_dyn_inputs else z_obj_on[:, :-1]
            z_depth_v = z_depth[:, :-1].detach() if detach_dyn_inputs else z_depth[:, :-1]
            z_features_v = z_features[:, :-1].detach() if detach_dyn_inputs else z_features[:, :-1]
            z_bg_features_v = z_bg_features[:, :-1].detach() if detach_dyn_inputs else z_bg_features[:, :-1]
            if z_context is not None:
                z_context_v = z_context[:, 1:]
            else:
                z_context_v = None
            # [bs, T-1, context_dim]
            if z_score is not None:
                z_score_v = z_score[:, :-1]
            else:
                z_score_v = None
            if actions is not None:
                actions_v = actions[:, :-1]
            else:
                actions_v = None
            if actions_mask is not None:
                actions_mask_v = actions_mask[:, :-1]
            else:
                actions_mask_v = None
            dyn_out = self.dyn_module(z_v,
                                      z_scale_v,
                                      z_obj_on_v,
                                      z_depth_v,
                                      z_features_v,
                                      z_bg_features_v,
                                      z_context_v,
                                      z_score_v,
                                      actions=actions_v,
                                      actions_mask=actions_mask_v)

            mu_dyn = dyn_out['mu']
            logvar_dyn = dyn_out['logvar']

            mu_features_dyn = dyn_out['mu_features']
            logvar_features_dyn = dyn_out['logvar_features']

            obj_on_a_dyn = dyn_out['obj_on_a']
            obj_on_b_dyn = dyn_out['obj_on_b']

            mu_depth_dyn = dyn_out['mu_depth']
            logvar_depth_dyn = dyn_out['logvar_depth']

            mu_scale_dyn = dyn_out['mu_scale']
            logvar_scale_dyn = dyn_out['logvar_scale']

            mu_bg_features_dyn = dyn_out['mu_bg_features']
            logvar_bg_features_dyn = dyn_out['logvar_bg_features']

            mu_score_dyn = dyn_out['mu_score']
            logvar_score_dyn = dyn_out['logvar_score']

        else:
            mu_dyn = None
            logvar_dyn = None

            mu_features_dyn = None
            logvar_features_dyn = None

            obj_on_a_dyn = None
            obj_on_b_dyn = None

            mu_depth_dyn = None
            logvar_depth_dyn = None

            mu_scale_dyn = None
            logvar_scale_dyn = None

            mu_bg_features_dyn = None
            logvar_bg_features_dyn = None

            mu_context_dyn = None
            logvar_context_dyn = None

            mu_context_global_dyn = None
            logvar_context_global_dyn = None

            mu_score_dyn = None
            logvar_score_dyn = None

        output_dict = {'kp_p': kp_p, 'rec': rec, 'rec_rgb': rec_rgb, 'mu_anchor': mu_anchor,
                       'logvar_anchor': logvar_anchor, 'z_base_var': z_base_var,
                       'z_base': z_base, 'z': z,
                       'mu_offset': mu_offset, 'logvar_offset': logvar_offset, 'z_offset': z_offset,
                       'mu_tot': mu_tot, 'mu_features': mu_features, 'logvar_features': logvar_features,
                       'z_features': z_features, 'bg': bg_rec, 'bg_rgb': bg_rec_rgb, 'mu_bg_features': mu_bg_features,
                       'logvar_bg_features': logvar_bg_features, 'z_bg_features': z_bg_features,
                       'mu_context': mu_context, 'logvar_context': logvar_context, 'z_context': z_context,
                       'cropped_objects_original': cropped_objects, 'cropped_objects_original_rgb': cropped_objects_rgb,
                       'obj_on_a': obj_on_a, 'obj_on_b': obj_on_b,
                       'obj_on': z_obj_on, 'mu_obj_on': mu_obj_on, 'dec_objects_original': dec_objects,
                       'dec_objects_original_rgb': dec_objects_rgb, 'dec_objects': dec_objects_trans,
                       'mu_depth': mu_depth, 'logvar_depth': logvar_depth, 'z_depth': z_depth, 'mu_scale': mu_scale,
                       'logvar_scale': logvar_scale, 'z_scale': z_scale,
                       'alpha_masks': alpha_masks, 'mu_dyn': mu_dyn,
                       'logvar_dyn': logvar_dyn, 'mu_features_dyn': mu_features_dyn,
                       'logvar_features_dyn': logvar_features_dyn, 'obj_on_a_dyn': obj_on_a_dyn,
                       'obj_on_b_dyn': obj_on_b_dyn, 'mu_depth_dyn': mu_depth_dyn, 'logvar_depth_dyn': logvar_depth_dyn,
                       'mu_scale_dyn': mu_scale_dyn, 'logvar_scale_dyn': logvar_scale_dyn,
                       'mu_bg_dyn': mu_bg_features_dyn, 'logvar_bg_dyn': logvar_bg_features_dyn,
                       'mu_context_dyn': mu_context_dyn, 'logvar_context_dyn': logvar_context_dyn,
                       'mu_score': mu_score, 'logvar_score': logvar_score, 'z_score': z_score,
                       'mu_score_dyn': mu_score_dyn, 'logvar_score_dyn': logvar_score_dyn,
                       'mu_context_global': mu_context_global, 'logvar_context_global': logvar_context_global,
                       'z_context_global': z_context_global,
                       'mu_context_global_dyn': mu_context_global_dyn,
                       'logvar_context_global_dyn': logvar_context_global_dyn,
                       'z_context_global_dyn': z_context_global_dyn,
                       }

        if with_loss:
            if num_static is None:
                num_static = self.n_static_frames
            loss_dict = self.calc_elbo(x, output_dict, warmup=warmup, beta_kl=beta_kl,
                                       beta_dyn=beta_dyn, beta_rec=beta_rec, kl_balance=kl_balance,
                                       dynamic_discount=dynamic_discount, recon_loss_type=recon_loss_type,
                                       recon_loss_func=recon_loss_func, beta_dyn_rec=beta_dyn_rec,
                                       num_static=num_static, beta_obj=beta_obj, done_mask=done_mask)
            output_dict['loss_dict'] = loss_dict
        else:
            output_dict['loss_dict'] = None

        return output_dict

    def calc_elbo(self, x, model_output, warmup=False, beta_kl=0.1, beta_dyn=0.1, beta_rec=1.0,
                  kl_balance=0.001, dynamic_discount=None, recon_loss_type="mse", recon_loss_func=None, balance=0.5,
                  beta_dyn_rec=1.0, num_static=1, use_kl_mask=True, apply_mask_on_obj_on=False, beta_obj=0.0,
                  done_mask=None):
        # beta_obj = beta_reg in the paper
        if self.is_dynamics_model:
            return self.calc_dyn_elbo(x, model_output, warmup, beta_kl, beta_dyn, beta_rec,
                                      kl_balance, dynamic_discount, recon_loss_type,
                                      recon_loss_func,
                                      balance, beta_dyn_rec, num_static, use_kl_mask=use_kl_mask,
                                      apply_mask_on_obj_on=apply_mask_on_obj_on, beta_obj=beta_obj, done_mask=done_mask)
        else:
            return self.calc_static_elbo(x, model_output, warmup, beta_kl, beta_dyn, beta_rec,
                                         kl_balance, dynamic_discount, recon_loss_type, recon_loss_func,
                                         balance, use_kl_mask=use_kl_mask, apply_mask_on_obj_on=apply_mask_on_obj_on,
                                         beta_obj=beta_obj)

    def calc_dyn_elbo(self, x, model_output, warmup=False, beta_kl=0.1, beta_dyn=0.1, beta_rec=1.0,
                      kl_balance=0.001, dynamic_discount=None, recon_loss_type="mse", recon_loss_func=None,
                      balance=0.5, beta_dyn_rec=1.0, num_static=1, use_kl_mask=True, apply_mask_on_obj_on=False,
                      beta_obj=0.0, done_mask=None):
        # x: [batch_size, timestep_horizon, ch, h, w]
        # num_static: "burn-in frames" number of timesteps to consider as posterior for the kl with constant priors
        # done_mask: [bs, T+1], 1 for t <= T_end_of_ep else 0
        # beta_obj = beta_reg in the paper
        # define losses
        kl_loss_func = ChamferLossKL(use_reverse_kl=False)
        if recon_loss_type == "vgg":
            if recon_loss_func is None:
                recon_loss_func = LossLPIPS(normalized_rgb=self.normalize_rgb).to(x.device)
        else:
            recon_loss_func = calc_reconstruction_loss

        # unpack output
        mu_p = model_output['kp_p']
        mu_anchor = model_output['mu_anchor']
        logvar_anchor = model_output['logvar_anchor']
        z = model_output['z']
        z_base = model_output['z_base']
        z_base_var = model_output['z_base_var']
        mu_offset = model_output['mu_offset']
        logvar_offset = model_output['logvar_offset']
        rec_x = model_output['rec']
        mu_features = model_output['mu_features']
        logvar_features = model_output['logvar_features']
        z_features = model_output['z_features']
        mu_bg = model_output['mu_bg_features']
        logvar_bg = model_output['logvar_bg_features']
        z_bg = model_output['z_bg_features']
        mu_scale = model_output['mu_scale']
        logvar_scale = model_output['logvar_scale']
        z_scale = model_output['z_scale']
        mu_depth = model_output['mu_depth']
        logvar_depth = model_output['logvar_depth']
        z_depth = model_output['z_depth']
        mu_context = model_output['mu_context']
        logvar_context = model_output['logvar_context']
        mu_context_global = model_output['mu_context_global']
        logvar_context_global = model_output['logvar_context_global']
        # object stuff
        dec_objects_original = model_output['dec_objects_original']
        cropped_objects_original = model_output['cropped_objects_original']
        obj_on = model_output['obj_on']  # [batch_size, n_kp]
        obj_on_a = model_output['obj_on_a']  # [batch_size, n_kp]
        obj_on_b = model_output['obj_on_b']  # [batch_size, n_kp]
        alpha_masks = model_output['alpha_masks']  # [batch_size, n_kp, 1, h, w]

        mu_score = model_output['mu_score']
        logvar_score = model_output['logvar_score']
        z_score = model_output['z_score']

        # dynamics stuff
        mu_dyn = model_output['mu_dyn']
        logvar_dyn = model_output['logvar_dyn']
        mu_features_dyn = model_output['mu_features_dyn']
        logvar_features_dyn = model_output['logvar_features_dyn']
        obj_on_a_dyn = model_output['obj_on_a_dyn']
        obj_on_b_dyn = model_output['obj_on_b_dyn']
        mu_depth_dyn = model_output['mu_depth_dyn']
        logvar_depth_dyn = model_output['logvar_depth_dyn']
        mu_scale_dyn = model_output['mu_scale_dyn']
        logvar_scale_dyn = model_output['logvar_scale_dyn']
        mu_bg_features_dyn = model_output['mu_bg_dyn']
        logvar_bg_features_dyn = model_output['logvar_bg_dyn']
        mu_context_dyn = model_output['mu_context_dyn']
        logvar_context_dyn = model_output['logvar_context_dyn']
        mu_context_global_dyn = model_output['mu_context_global_dyn']
        logvar_context_global_dyn = model_output['logvar_context_global_dyn']

        mu_score_dyn = model_output['mu_score_dyn']
        logvar_score_dyn = model_output['logvar_score_dyn']

        batch_size = x.shape[0]
        timestep_horizon = self.timestep_horizon
        x = x.reshape(-1, *x.shape[2:])
        if warmup:
            num_static = timestep_horizon - 1  # optimize only the last step for dynamics during warmup
        # discount for future steps
        if dynamic_discount is None:
            discount = torch.ones(size=(timestep_horizon - num_static + 1,), device=x.device)
        else:
            discount = dynamic_discount[:timestep_horizon - num_static + 1]

        if done_mask is None or warmup:
            done_mask_0 = done_mask_dyn = done_norm = 1.0
        else:
            done_norm = 1 / done_mask.sum(-1, keepdim=True)
            done_mask_0 = done_mask[:, :num_static] * done_norm
            done_mask_dyn = done_mask[:, num_static:] * done_norm

        # --- reconstruction error --- #
        if recon_loss_type == "vgg":
            loss_rec = recon_loss_func(x, rec_x, reduction="none")
            loss_rec = (x.shape[1] * x.shape[2] * x.shape[3]) * loss_rec  # [h * w * c]
        else:
            loss_rec = calc_reconstruction_loss(x, rec_x, loss_type='mse', reduction='none')

        loss_rec = loss_rec.view(batch_size, timestep_horizon + 1, -1)
        # consider discount
        loss_rec_0, loss_rec_future = loss_rec.split([num_static, timestep_horizon - num_static + 1], dim=1)
        loss_rec_future = beta_dyn_rec * discount[None, :, None] * loss_rec_future
        # loss_rec = loss_rec_0.sum(dim=(-2, -1)).mean() + loss_rec_future.sum(dim=(-2, -1)).mean()
        loss_rec_0 = (loss_rec_0.sum(-1) * done_mask_0).sum(-1).mean()
        loss_rec_future = (loss_rec_future.sum(-1) * done_mask_dyn).sum(-1).mean()
        loss_rec = loss_rec_0 + loss_rec_future

        with torch.no_grad():
            psnr = -10 * torch.log10(F.mse_loss(rec_x, x))
        # --- end reconstruction error --- #

        # --- isolate the first timestep for kl with constant priors --- #
        # let num_static = t, timestep_horizon = T
        mu_0 = mu_anchor[:, :num_static]  # [bs, t n_kp, 2]
        logvar_0 = logvar_anchor[:, :num_static]  # [bs, t, n_kp, 2]
        mu_p_0 = mu_p.reshape(batch_size, timestep_horizon + 1, *mu_p.shape[1:])[:,
                 :num_static]  # [bs, t, n_kp_prior, 2]
        mu_offset_0 = mu_offset[:, :num_static]  # [bs, t, n_kp, 2]
        logvar_offset_0 = logvar_offset[:, :num_static]
        mu_depth_0 = mu_depth[:, :num_static]  # [bs, n_kp, 1]
        logvar_depth_0 = logvar_depth[:, :num_static]
        mu_scale_0 = mu_scale[:, :num_static]  # [bs, t, n_kp, 2]
        logvar_scale_0 = logvar_scale[:, :num_static]
        obj_on_a_0 = obj_on_a[:, :num_static]  # [bs, t, n_kp]
        obj_on_b_0 = obj_on_b[:, :num_static]  # [bs, t, n_kp]
        mu_features_0 = mu_features[:, :num_static]
        if logvar_features is None:
            logvar_features_0 = mu_features_0
        else:
            logvar_features_0 = logvar_features[:, :num_static]
        # mu/logvar_features_0: [bs, n_kp, feat_dim]
        mu_bg_0 = mu_bg[:, :num_static]  # [bs, t, feat_dim]
        if logvar_bg is None:
            logvar_bg_0 = mu_bg_0
        else:
            logvar_bg_0 = logvar_bg[:, :num_static]
        # note: bg latent dim = a single particle's latent dim

        mu_score_0 = mu_score[:, :num_static]  # [bs, n_kp, 1]
        logvar_score_0 = logvar_score[:, :num_static]

        if use_kl_mask:
            # mask_c = 2.0 if warmup else 1.0
            # kl_mask_0 = obj_on[:, :num_static].reshape(obj_on.shape[0], num_static, obj_on.shape[2]) * mask_c
            kl_mask_0 = obj_on[:, :num_static].reshape(obj_on.shape[0], num_static, obj_on.shape[2])
            # adaptive_beta_kl_0 = kl_mask_0.sum(-1) + 1 # [bs, n_static]
            adaptive_beta_kl_0 = 1.0
            # if warmup:
            #     kl_mask_0 = 1.0
        else:
            kl_mask_0 = 1.0
            adaptive_beta_kl_0 = 1.0

        # --- end isolate the first timestep for kl with constant priors --- #

        # --- define priors --- #
        logvar_kp = self.logvar_kp.expand_as(mu_p_0)
        logvar_offset_p = self.logvar_offset_p
        logvar_scale_p = self.logvar_scale_p
        # optional, smoother
        # obj_on_a_prior = self.obj_on_a_p * 10 if warmup else self.obj_on_a_p
        # obj_on_b_prior = self.obj_on_b_p * 10 if warmup else self.obj_on_b_p
        obj_on_a_prior = self.obj_on_a_p
        obj_on_b_prior = self.obj_on_b_p
        mu_scale_prior = self.mu_scale_prior
        mu_feat_prior = logvar_feat_prior = torch.tensor(0.0, device=mu_features_0.device)
        mu_bg_feat_prior = logvar_bg_feat_prior = torch.tensor(0.0, device=mu_bg_0.device)

        mu_tot = z_base + mu_offset
        logvar_tot = logvar_offset
        # --- end priors --- #

        # --- kl-divergence for t <= tau --- #
        # kl-divergence and priors
        mu_prior = mu_p_0.reshape(-1, *mu_p_0.shape[2:])  # [bs * t, n_kp_prior, 2]
        logvar_prior = logvar_kp.reshape(-1, *logvar_kp.shape[2:])  # [bs * t, n_kp_prior, 2]
        # for t < tau, we separate for base and offset
        mu_post = mu_0.reshape(-1, *mu_0.shape[2:])  # [bs * t, n_kp, 2]
        # deterministic chamfer (separable chamfer-kl)
        logvar_post = torch.zeros_like(mu_post)
        # note: mu_p_0 is a duplication of the prior keypoints for the first timestep
        loss_kl_kp_base = kl_loss_func(mu_preds=mu_post, logvar_preds=logvar_post, mu_gts=mu_prior,
                                       logvar_gts=logvar_prior)  # [batch_size, ]
        # this ensures coverage of the keypoints:
        loss_kl_kp_base = (loss_kl_kp_base.view(batch_size, -1) * adaptive_beta_kl_0 * done_mask_0).sum(-1).mean()

        loss_kl_kp_offset = calc_kl(logvar_offset_0.reshape(-1, logvar_offset_0.shape[-1]),
                                    mu_offset_0.reshape(-1, mu_offset_0.shape[-1]), logvar_o=logvar_offset_p,
                                    reduce='none')
        # loss_kl_kp_offset = (loss_kl_kp_offset.view(batch_size, -1, mu_offset.shape[2]) * kl_mask_0).sum(
        #     dim=(-2, -1)).mean()
        loss_kl_kp_offset = (loss_kl_kp_offset.view(batch_size, -1, mu_offset.shape[2]) * kl_mask_0).sum(-1)
        loss_kl_kp_offset = (loss_kl_kp_offset * adaptive_beta_kl_0 * done_mask_0).sum(-1).mean()
        loss_kl_kp = 0.5 * kl_balance * loss_kl_kp_base + loss_kl_kp_offset

        # depth
        loss_kl_depth = calc_kl(logvar_depth_0.reshape(-1, logvar_depth_0.shape[-1]),
                                mu_depth_0.reshape(-1, mu_depth_0.shape[-1]), reduce='none')
        # loss_kl_depth = (loss_kl_depth.view(batch_size, -1, mu_depth.shape[2]) * kl_mask_0).sum(dim=(-2, -1)).mean()
        loss_kl_depth = (loss_kl_depth.view(batch_size, -1, mu_depth.shape[2]) * kl_mask_0).sum(-1)
        loss_kl_depth = (loss_kl_depth * adaptive_beta_kl_0 * done_mask_0).sum(-1).mean()

        # scale
        # assume sigmoid activation on z_scale
        loss_kl_scale = calc_kl(logvar_scale_0.reshape(-1, logvar_scale_0.shape[-1]),
                                mu_scale_0.reshape(-1, mu_scale_0.shape[-1]),
                                mu_o=mu_scale_prior, logvar_o=logvar_scale_p,
                                reduce='none')  # [bs * n_static, n_particles]
        # loss_kl_scale = (loss_kl_scale.view(batch_size, -1, mu_scale.shape[2]) * kl_mask_0).sum(dim=(-2, -1)).mean()
        loss_kl_scale = (loss_kl_scale.view(batch_size, -1, mu_scale.shape[2]) * kl_mask_0).sum(-1)  # [bs, n_static]
        loss_kl_scale = (loss_kl_scale * adaptive_beta_kl_0 * done_mask_0).sum(-1).mean()

        # obj_on (z_t, transparency)
        loss_kl_obj_on = calc_kl_beta_dist(obj_on_a_0, obj_on_b_0,
                                           obj_on_a_prior,
                                           obj_on_b_prior)
        if apply_mask_on_obj_on:
            loss_kl_obj_on = (loss_kl_obj_on * kl_mask_0).sum(dim=-1)  # [bs, T]
        else:
            loss_kl_obj_on = (loss_kl_obj_on).sum(dim=-1)  # [bs, T]
        loss_kl_obj_on = (loss_kl_obj_on * adaptive_beta_kl_0 * done_mask_0).sum(-1).mean()
        # the following is not used as part of the loss, it shows the average number of turned-on particles
        obj_on_l1 = torch.abs(obj_on.squeeze(-1)).sum(-1).mean()  # just to get an idea how many particles are turned on

        # features
        if self.features_dist == 'categorical':
            logits_feat_post = mu_features_0.reshape(-1, mu_features_0.shape[-1])
            logits_feat_prior = (1 / self.n_fg_classes) * torch.ones_like(logits_feat_post)
            logits_feat_prior = torch.log(logits_feat_prior)
            loss_kl_feat_obj = calc_kl_categorical(logits_feat_post, logits_feat_prior,
                                                   num_classes=self.n_fg_classes, reduce='none')
            loss_kl_feat_obj = loss_kl_feat_obj.view(batch_size, -1, mu_features.shape[2]) * kl_mask_0
            loss_kl_feat_obj = (loss_kl_feat_obj.sum(-1) * adaptive_beta_kl_0 * done_mask_0).sum(-1).mean()

            logits_feat_bg_post = mu_bg_0.reshape(-1, mu_bg_0.shape[-1])
            logits_feat_bg_prior = (1 / self.n_bg_classes) * torch.ones_like(logits_feat_bg_post)
            logits_feat_bg_prior = torch.log(logits_feat_bg_prior)
            loss_kl_feat_bg = calc_kl_categorical(logits_feat_bg_post, logits_feat_bg_prior,
                                                  num_classes=self.n_bg_classes, reduce='none')
            loss_kl_feat_bg = (loss_kl_feat_bg.view(batch_size, -1) * adaptive_beta_kl_0 * done_mask_0).sum(-1).mean()

            loss_kl_feat = loss_kl_feat_obj + loss_kl_feat_bg
        else:
            loss_kl_feat = calc_kl(logvar_features_0.reshape(-1, logvar_features_0.shape[-1]),
                                   mu_features_0.reshape(-1, mu_features_0.shape[-1]),
                                   mu_o=mu_feat_prior, logvar_o=logvar_feat_prior,
                                   reduce='none')
            loss_kl_feat_obj = loss_kl_feat.view(batch_size, -1, mu_features.shape[2]) * kl_mask_0
            loss_kl_feat_obj = ((loss_kl_feat_obj.sum(-1) * adaptive_beta_kl_0 * done_mask_0).sum(-1)).mean()

            loss_kl_feat_bg = calc_kl(logvar_bg_0.reshape(-1, logvar_bg_0.shape[-1]),
                                      mu_bg_0.reshape(-1, mu_bg_0.shape[-1]),
                                      mu_o=mu_bg_feat_prior,
                                      logvar_o=logvar_bg_feat_prior,
                                      reduce='none')
            loss_kl_feat_bg = (loss_kl_feat_bg.view(batch_size, -1) * adaptive_beta_kl_0 * done_mask_0).sum(-1).mean()

            loss_kl_feat = loss_kl_feat_obj + loss_kl_feat_bg

        # --- end kl-divergence for t < tau --- #

        # --- kl-divergence for t >= tau --- #
        # dynamics
        if use_kl_mask:
            kl_mask = obj_on[:, num_static:].reshape(-1, obj_on.shape[2])  # [bs * t, n_kp]
            adaptive_beta_kl_dyn = torch.tensor(1.0, device=obj_on_a.device)
        else:
            kl_mask = torch.tensor(1.0, device=obj_on_a.device)
            adaptive_beta_kl_dyn = torch.tensor(1.0, device=obj_on_a.device)

        # transparency
        obj_on_a_post = obj_on_a[:, num_static:]
        obj_on_b_post = obj_on_b[:, num_static:]

        obj_on_a_post = obj_on_a_post.reshape(-1, *obj_on_a_post.shape[2:])
        obj_on_b_post = obj_on_b_post.reshape(-1, *obj_on_b_post.shape[2:])
        obj_on_a_dyn = obj_on_a_dyn[:, num_static - 1:].reshape(-1, *obj_on_a_dyn.shape[2:])
        obj_on_b_dyn = obj_on_b_dyn[:, num_static - 1:].reshape(-1, *obj_on_b_dyn.shape[2:])

        # position
        mu_dyn_post_offset = mu_tot[:, num_static:].reshape(-1, *mu_tot.shape[2:])
        logvar_dyn_post_offset = logvar_tot[:, num_static:].reshape(-1, *logvar_tot.shape[2:])
        mu_dyn_prior = mu_dyn[:, num_static - 1:].reshape(-1, *mu_dyn.shape[2:])
        logvar_dyn_prior = logvar_dyn[:, num_static - 1:].reshape(-1, *logvar_dyn.shape[2:])

        # depth
        mu_depth_dyn_post = mu_depth[:, num_static:].reshape(-1, *mu_depth.shape[2:])
        logvar_depth_dyn_post = logvar_depth[:, num_static:].reshape(-1, *logvar_depth.shape[2:])
        mu_depth_dyn_prior = mu_depth_dyn[:, num_static - 1:].reshape(-1, *mu_depth_dyn.shape[2:])
        logvar_depth_dyn_prior = logvar_depth_dyn[:, num_static - 1:].reshape(-1,
                                                                              *logvar_depth_dyn.shape[2:])

        # scale
        mu_scale_dyn_post = mu_scale[:, num_static:].reshape(-1, *mu_scale.shape[2:])
        logvar_scale_dyn_post = logvar_scale[:, num_static:].reshape(-1, *logvar_scale.shape[2:])
        mu_scale_dyn_prior = mu_scale_dyn[:, num_static - 1:].reshape(-1, *mu_scale_dyn.shape[2:])
        logvar_scale_dyn_prior = logvar_scale_dyn[:, num_static - 1:].reshape(-1,
                                                                              *logvar_scale_dyn.shape[2:])

        # object features
        mu_features_dyn_post = mu_features[:, num_static:].reshape(-1, *mu_features.shape[2:])
        logvar_features_dyn_post = logvar_features[:, num_static:].reshape(-1, *logvar_features.shape[2:])
        mu_features_dyn_prior = mu_features_dyn[:, num_static - 1:].reshape(-1, *mu_features_dyn.shape[2:])
        logvar_features_dyn_prior = logvar_features_dyn[:, num_static - 1:].reshape(-1, *logvar_features_dyn.shape[2:])

        # score
        if mu_score_dyn is not None:
            mu_score_dyn_post = mu_score[:, num_static:].reshape(-1, *mu_score.shape[2:])
            logvar_score_dyn_post = logvar_score[:, num_static:].reshape(-1, *logvar_score.shape[2:])
            mu_score_dyn_prior = mu_score_dyn[:, num_static - 1:].reshape(-1, *mu_score_dyn.shape[2:])
            logvar_score_dyn_prior = logvar_score_dyn[:, num_static - 1:].reshape(-1, *logvar_score_dyn.shape[2:])

            loss_kl_score_dyn = calc_kl_jit(mu=mu_score_dyn_post.view(-1, mu_score_dyn_post.shape[-1]),
                                            logvar=logvar_score_dyn_post.view(-1, logvar_score_dyn_post.shape[-1]),
                                            mu_o=mu_score_dyn_prior.view(-1, mu_score_dyn_prior.shape[-1]),
                                            logvar_o=logvar_score_dyn_prior.view(-1, logvar_score_dyn_prior.shape[-1]),
                                            reduce='none')
            loss_kl_score_dyn = (loss_kl_score_dyn.view(mu_score_dyn_post.shape[0],
                                                        mu_score_dyn_post.shape[1])).sum(-1)
            loss_kl_score_dyn = loss_kl_score_dyn.reshape(batch_size, -1)
            loss_kl_score_dyn = ((loss_kl_score_dyn * discount[None, :]) * adaptive_beta_kl_dyn * done_mask_dyn).sum(
                -1).mean()
        else:
            loss_kl_score_dyn = torch.tensor(0.0, device=x.device)

        # bg features
        mu_bg_dyn_post = mu_bg[:, num_static:].reshape(-1, *mu_bg.shape[2:])
        logvar_bg_dyn_post = logvar_bg[:, num_static:].reshape(-1, *logvar_bg.shape[2:])
        mu_bg_dyn_prior = mu_bg_features_dyn[:, num_static - 1:].reshape(-1, *mu_bg_features_dyn.shape[2:])
        logvar_bg_dyn_prior = logvar_bg_features_dyn[:, num_static - 1:].reshape(-1, *logvar_bg_features_dyn.shape[2:])

        dyn_kl_balance = balance
        features_weight = 1.0
        loss_kl_dyn_dict = calc_dynamic_kl(mu_post=mu_dyn_post_offset, logvar_post=logvar_dyn_post_offset,
                                           mu_prior=mu_dyn_prior, logvar_prior=logvar_dyn_prior,
                                           mu_depth_post=mu_depth_dyn_post, logvar_depth_post=logvar_depth_dyn_post,
                                           mu_depth_prior=mu_depth_dyn_prior, logvar_depth_prior=logvar_depth_dyn_prior,
                                           mu_scale_post=mu_scale_dyn_post, logvar_scale_post=logvar_scale_dyn_post,
                                           mu_scale_prior=mu_scale_dyn_prior, logvar_scale_prior=logvar_scale_dyn_prior,
                                           mu_features_post=mu_features_dyn_post,
                                           logvar_features_post=logvar_features_dyn_post,
                                           mu_features_prior=mu_features_dyn_prior,
                                           logvar_features_prior=logvar_features_dyn_prior,
                                           mu_bg_post=mu_bg_dyn_post, logvar_bg_post=logvar_bg_dyn_post,
                                           mu_bg_prior=mu_bg_dyn_prior, logvar_bg_prior=logvar_bg_dyn_prior,
                                           obj_on_a_post=obj_on_a_post, obj_on_b_post=obj_on_b_post,
                                           obj_on_a_prior=obj_on_a_dyn, obj_on_b_prior=obj_on_b_dyn, kl_mask=kl_mask,
                                           balance=dyn_kl_balance, reduce='none',
                                           features_weight=features_weight, features_dist=self.features_dist,
                                           n_fg_classes=self.n_fg_classes, n_bg_classes=self.n_bg_classes,
                                           apply_mask_on_obj_on=apply_mask_on_obj_on)
        loss_kl_dyn = loss_kl_dyn_dict['loss_kl']
        loss_kl_kp_dyn = loss_kl_dyn_dict['loss_kl_kp'] / batch_size
        loss_kl_scale_dyn = loss_kl_dyn_dict['loss_kl_scale'] / batch_size
        loss_kl_depth_dyn = loss_kl_dyn_dict['loss_kl_depth'] / batch_size
        loss_kl_obj_on_dyn = loss_kl_dyn_dict['loss_kl_obj_on'] / batch_size
        loss_kl_feat_dyn = loss_kl_dyn_dict['loss_kl_features'] / batch_size
        loss_kl_dyn = loss_kl_dyn.reshape(batch_size, -1)
        loss_kl_dyn = ((loss_kl_dyn * discount[None, :]) * adaptive_beta_kl_dyn * done_mask_dyn).sum(-1).mean()

        loss_kl_dyn = (loss_kl_dyn + loss_kl_score_dyn)

        if self.context_dist == 'beta':
            mu_context_dyn_post = mu_context[:, num_static:]
            logvar_context_dyn_post = logvar_context[:, num_static:]
            mu_context_dyn_prior = mu_context_dyn[:, num_static - 1:]
            logvar_context_dyn_prior = logvar_context_dyn[:, num_static - 1:]

            loss_kl_context_dyn = calc_kl_beta_dist(mu_context_dyn_post.reshape(-1, mu_context_dyn_post.shape[-1]),
                                                    logvar_context_dyn_post.reshape(-1,
                                                                                    logvar_context_dyn_post.shape[-1]),
                                                    mu_context_dyn_prior.reshape(-1, mu_context_dyn_prior.shape[-1]),
                                                    logvar_context_dyn_prior.reshape(-1,
                                                                                     logvar_context_dyn_prior.shape[
                                                                                         -1]),
                                                    reduce='none')
        elif self.context_dist == 'categorical':
            logits_context_dyn_post = mu_context_dyn_post = mu_context[:, num_static:]
            logits_context_dyn_prior = mu_context_dyn[:, num_static - 1:]
            loss_kl_context_dyn = calc_kl_categorical(
                logits_context_dyn_post.reshape(-1, logits_context_dyn_post.shape[-1]),
                logits_context_dyn_prior.reshape(-1, logits_context_dyn_prior.shape[-1]),
                num_classes=self.n_ctx_classes, reduce='none', balance=balance)
        else:
            mu_context_dyn_post = mu_context[:, num_static:]
            logvar_context_dyn_post = logvar_context[:, num_static:]
            mu_context_dyn_prior = mu_context_dyn[:, num_static - 1:]
            logvar_context_dyn_prior = logvar_context_dyn[:, num_static - 1:]

            loss_kl_context_dyn = calc_kl(logvar_context_dyn_post.reshape(-1, logvar_context_dyn_post.shape[-1]),
                                          mu_context_dyn_post.reshape(-1, mu_context_dyn_post.shape[-1]),
                                          mu_o=mu_context_dyn_prior.reshape(-1, mu_context_dyn.shape[-1]),
                                          logvar_o=logvar_context_dyn_prior.reshape(-1, logvar_context_dyn.shape[-1]),
                                          reduce='none', balance=balance)
        loss_kl_context_dyn = loss_kl_context_dyn.view(batch_size, mu_context_dyn_post.shape[1], -1).sum(-1)
        loss_kl_context_dyn = (loss_kl_context_dyn * adaptive_beta_kl_dyn * done_mask_dyn).sum(-1).mean()

        if self.global_ctx_pool and mu_context_global is not None:
            if self.context_dist == 'beta':
                mu_context_global_dyn_post = mu_context_global[:, num_static:]
                logvar_context_global_dyn_post = logvar_context_global[:, num_static::]
                mu_context_global_dyn_prior = mu_context_global_dyn[:, num_static - 1:]
                logvar_context_global_dyn_prior = logvar_context_global_dyn[:, num_static - 1:]

                loss_kl_context_global_dyn = calc_kl_beta_dist(
                    mu_context_global_dyn_post.reshape(-1, mu_context_global_dyn_post.shape[-1]),
                    logvar_context_global_dyn_post.reshape(-1,
                                                           logvar_context_global_dyn_post.shape[
                                                               -1]),
                    mu_context_global_dyn_prior.reshape(-1,
                                                        mu_context_global_dyn_prior.shape[-1]),
                    logvar_context_global_dyn_prior.reshape(-1,
                                                            logvar_context_global_dyn_prior.shape[
                                                                -1]),
                    reduce='none')
            elif self.context_dist == 'categorical':
                logits_context_global_dyn_post = mu_context_global_dyn_post = mu_context_global[:, num_static:]
                logits_context_global_dyn_prior = mu_context_global_dyn[:, num_static - 1:]
                loss_kl_context_global_dyn = calc_kl_categorical(
                    logits_context_global_dyn_post.reshape(-1, logits_context_global_dyn_post.shape[-1]),
                    logits_context_global_dyn_prior.reshape(-1, logits_context_global_dyn_prior.shape[-1]),
                    num_classes=self.n_pool_ctx_classes, reduce='none', balance=balance)
            else:
                mu_context_global_dyn_post = mu_context_global[:, num_static:]
                logvar_context_global_dyn_post = logvar_context_global[:, num_static:]
                mu_context_global_dyn_prior = mu_context_global_dyn[:, num_static - 1:]
                logvar_context_global_dyn_prior = logvar_context_global_dyn[:, num_static - 1:]

                loss_kl_context_global_dyn = calc_kl(
                    logvar_context_global_dyn_post.reshape(-1, logvar_context_global_dyn_post.shape[-1]),
                    mu_context_global_dyn_post.reshape(-1, mu_context_global_dyn_post.shape[-1]),
                    mu_o=mu_context_global_dyn_prior.reshape(-1, mu_context_global_dyn.shape[-1]),
                    logvar_o=logvar_context_global_dyn_prior.reshape(-1,
                                                                     logvar_context_global_dyn.shape[-1]),
                    reduce='none', balance=balance)
            loss_kl_context_global_dyn = loss_kl_context_global_dyn.view(batch_size,
                                                                         mu_context_global_dyn_post.shape[1], -1).sum(
                -1)
            loss_kl_context_global_dyn = (loss_kl_context_global_dyn * adaptive_beta_kl_dyn * done_mask_dyn).sum(
                -1).mean()

            loss_kl_context_dyn = loss_kl_context_dyn + loss_kl_context_global_dyn

        # --- end kl-divergence for t >= tau --- #

        # normalization coefficients
        # normalize by number of particles
        n_particles = self.n_kp_prior + 1  # K fg + 1 bg
        # normalize by number of timesteps
        if done_mask is None or warmup:
            norm_f = 1 / (timestep_horizon + 1)
        else:
            norm_f = 1

        # obj regularization
        loss_obj_reg = (((kl_mask_0.sum(-1) ** 2) * done_mask_0).sum(-1)).mean()

        # total losses
        loss_kl = loss_kl_kp + loss_kl_scale + loss_kl_obj_on + loss_kl_depth + kl_balance * loss_kl_feat
        loss_kl_static = loss_kl

        # total dynamics kl
        loss_kl_context = loss_kl_context_dyn
        losses = [beta_rec * loss_rec,
                  beta_kl * loss_kl,
                  beta_dyn * loss_kl_dyn,
                  beta_dyn * loss_kl_context_dyn,
                  beta_obj * loss_obj_reg]  # beta_obj = beta_reg in the paper
        loss = norm_f * sum(losses)

        loss_scale = 0.1 if recon_loss_type == 'mse' else 0.01
        loss = loss_scale * loss
        loss_dict = {'loss': loss, 'psnr': psnr.detach(), 'kl': loss_kl_static, 'kl_dyn': loss_kl_dyn,
                     'loss_rec': loss_rec,
                     'obj_on_l1': obj_on_l1, 'loss_kl_kp': loss_kl_kp, 'loss_kl_feat': loss_kl_feat,
                     'loss_kl_obj_on': loss_kl_obj_on, 'loss_kl_scale': loss_kl_scale, 'loss_kl_depth': loss_kl_depth,
                     'loss_kl_context': loss_kl_context,
                     'loss_kl_score_dyn': loss_kl_score_dyn, 'loss_kl_kp_dyn': loss_kl_kp_dyn,
                     'loss_kl_feat_dyn': loss_kl_feat_dyn,
                     'loss_kl_obj_on_dyn': loss_kl_obj_on_dyn, 'loss_kl_scale_dyn': loss_kl_scale_dyn,
                     'loss_kl_depth_dyn': loss_kl_depth_dyn, 'loss_obj_reg': loss_obj_reg}
        return loss_dict

    def calc_static_elbo(self, x, model_output, warmup=False, beta_kl=0.05, beta_dyn=1.0, beta_rec=1.0,
                         kl_balance=0.001, dynamic_discount=None, recon_loss_type="mse", recon_loss_func=None,
                         balance=0.5, use_kl_mask=True, apply_mask_on_obj_on=False, beta_obj=0.0):
        # x: [batch_size, timestep_horizon, ch, h, w]
        # constant prior for all timesteps (single image DLP)
        # balance: kl balance for dynamics kl posterior and prior
        # beta_obj = beta_reg in the paper
        # define losses
        kl_loss_func = ChamferLossKL(use_reverse_kl=False)
        if recon_loss_type == "vgg":
            if recon_loss_func is None:
                recon_loss_func = LossLPIPS(normalized_rgb=self.normalize_rgb).to(x.device)
        else:
            recon_loss_func = calc_reconstruction_loss

        # unpack output
        mu_p = model_output['kp_p']
        mu_anchor = model_output['mu_anchor']
        logvar_anchor = model_output['logvar_anchor']
        z = model_output['z']
        z_base = model_output['z_base']
        mu_offset = model_output['mu_offset']
        logvar_offset = model_output['logvar_offset']
        rec_x = model_output['rec']
        mu_features = model_output['mu_features']
        logvar_features = model_output['logvar_features']
        z_features = model_output['z_features']
        mu_bg = model_output['mu_bg_features']
        logvar_bg = model_output['logvar_bg_features']
        z_bg = model_output['z_bg_features']
        mu_scale = model_output['mu_scale']
        logvar_scale = model_output['logvar_scale']
        z_scale = model_output['z_scale']
        mu_depth = model_output['mu_depth']
        logvar_depth = model_output['logvar_depth']
        z_depth = model_output['z_depth']
        mu_context = model_output['mu_context']
        logvar_context = model_output['logvar_context']
        # object stuff
        dec_objects_original = model_output['dec_objects_original']
        cropped_objects_original = model_output['cropped_objects_original']
        obj_on = model_output['obj_on']  # [batch_size, n_kp]
        obj_on_a = model_output['obj_on_a']  # [batch_size, n_kp]
        obj_on_b = model_output['obj_on_b']  # [batch_size, n_kp]
        alpha_masks = model_output['alpha_masks']  # [batch_size, n_kp, 1, h, w]

        batch_size = x.shape[0]
        timestep_horizon = self.timestep_horizon
        x = x.view(-1, *x.shape[2:])

        if recon_loss_type == "vgg":
            loss_rec = recon_loss_func(x, rec_x, reduction="mean")
            loss_rec = (x.shape[1] * x.shape[2] * x.shape[3]) * loss_rec
        else:
            loss_rec = calc_reconstruction_loss(x, rec_x, loss_type='mse', reduction='none')
            loss_rec = loss_rec.view(batch_size, timestep_horizon, -1)
            loss_rec = loss_rec.sum(-1).mean()

        with torch.no_grad():
            psnr = -10 * torch.log10(F.mse_loss(rec_x, x))

        # --- end reconstruction error --- #

        if use_kl_mask:
            kl_mask = obj_on.reshape(obj_on.shape[0], obj_on.shape[2])
            adaptive_beta_kl = 1.0
        else:
            kl_mask = 1.0
            adaptive_beta_kl = 1.0

        # --- define priors --- #
        logvar_kp = self.logvar_kp.expand_as(mu_p)
        logvar_offset_p = self.logvar_offset_p
        logvar_scale_p = self.logvar_scale_p
        # encourage objects to be 'on' during warmup
        obj_on_a_prior = self.obj_on_a_p
        obj_on_b_prior = self.obj_on_b_p
        mu_scale_prior = self.mu_scale_prior
        # --- end priors --- #

        # --- kl-divergence for t = 0 --- #
        # kl-divergence and priors
        mu_prior = mu_p
        logvar_prior = logvar_kp
        mu_post = mu_anchor.squeeze(1)
        logvar_post = torch.zeros_like(mu_post)
        loss_kl_kp_base = kl_loss_func(mu_preds=mu_post, logvar_preds=logvar_post, mu_gts=mu_prior,
                                       logvar_gts=logvar_prior)  # [batch_size, ]
        loss_kl_kp_base = (loss_kl_kp_base * adaptive_beta_kl).mean()

        loss_kl_kp_offset = calc_kl(logvar_offset.reshape(-1, logvar_offset.shape[-1]),
                                    mu_offset.reshape(-1, mu_offset.shape[-1]), logvar_o=logvar_offset_p,
                                    reduce='none')
        loss_kl_kp_offset = (loss_kl_kp_offset.view(-1, mu_offset.shape[2]) * kl_mask).sum(-1)
        loss_kl_kp_offset = (loss_kl_kp_offset * adaptive_beta_kl).mean()
        loss_kl_kp = 0.5 * kl_balance * loss_kl_kp_base + loss_kl_kp_offset

        # depth
        loss_kl_depth = calc_kl(logvar_depth.reshape(-1, logvar_depth.shape[-1]),
                                mu_depth.reshape(-1, mu_depth.shape[-1]), reduce='none')
        loss_kl_depth = ((loss_kl_depth.view(-1, mu_depth.shape[2]) * kl_mask).sum(-1) * adaptive_beta_kl).mean()

        # scale
        # assume sigmoid activation on z_scale
        loss_kl_scale = calc_kl(logvar_scale.reshape(-1, logvar_scale.shape[-1]),
                                mu_scale.reshape(-1, mu_scale.shape[-1]),
                                mu_o=mu_scale_prior, logvar_o=logvar_scale_p,
                                reduce='none')
        loss_kl_scale = ((loss_kl_scale.view(-1, mu_scale.shape[2]) * kl_mask).sum(-1) * adaptive_beta_kl).mean()

        # obj_on
        loss_kl_obj_on = calc_kl_beta_dist(obj_on_a, obj_on_b,
                                           obj_on_a_prior,
                                           obj_on_b_prior)
        if apply_mask_on_obj_on:
            loss_kl_obj_on = (loss_kl_obj_on * kl_mask).sum(-1)
        else:
            loss_kl_obj_on = loss_kl_obj_on.sum(-1)
        loss_kl_obj_on = (loss_kl_obj_on * adaptive_beta_kl).mean()

        # the following is not used as part of the loss, it shows the average number of turned-on particles
        obj_on_l1 = torch.abs(obj_on.squeeze(-1)).sum(-1).mean()  # just to get an idea how many particles are turned on

        # features
        if self.features_dist == 'categorical':
            logits_feat_post = mu_features.reshape(-1, mu_features.shape[-1])
            logits_feat_prior = (1 / self.n_fg_classes) * torch.ones_like(logits_feat_post)
            logits_feat_prior = torch.log(logits_feat_prior)
            loss_kl_feat_obj = calc_kl_categorical(logits_feat_post, logits_feat_prior,
                                                   num_classes=self.n_fg_classes, reduce='none', balance=balance)
            loss_kl_feat_obj = loss_kl_feat_obj.view(-1, mu_features.shape[2]) * kl_mask
            loss_kl_feat_obj = (loss_kl_feat_obj.sum(-1) * adaptive_beta_kl).mean()

            logits_feat_bg_post = mu_bg.reshape(-1, mu_bg.shape[-1])
            logits_feat_bg_prior = (1 / self.n_bg_classes) * torch.ones_like(logits_feat_bg_post)
            logits_feat_bg_prior = torch.log(logits_feat_bg_prior)
            loss_kl_feat_bg = calc_kl_categorical(logits_feat_bg_post, logits_feat_bg_prior,
                                                  num_classes=self.n_bg_classes, reduce='none', balance=balance)
            loss_kl_feat_bg = (loss_kl_feat_bg * adaptive_beta_kl).mean()

            loss_kl_feat = loss_kl_feat_obj + loss_kl_feat_bg
        else:
            loss_kl_feat = calc_kl(logvar_features.reshape(-1, logvar_features.shape[-1]),
                                   mu_features.reshape(-1, mu_features.shape[-1]), reduce='none')
            loss_kl_feat_obj = loss_kl_feat.view(-1, mu_features.shape[2]) * kl_mask
            loss_kl_feat_obj = (loss_kl_feat_obj.sum(-1) * adaptive_beta_kl).mean()

            loss_kl_feat_bg = calc_kl(logvar_bg, mu_bg, reduce='none')
            loss_kl_feat_bg = (loss_kl_feat_bg * adaptive_beta_kl).mean()
            loss_kl_feat = loss_kl_feat_obj + loss_kl_feat_bg

        loss_kl_context = torch.tensor(0.0, device=x.device)
        # total losses
        loss_kl = loss_kl_kp + loss_kl_scale + loss_kl_obj_on + loss_kl_depth + kl_balance * loss_kl_feat
        loss_kl_static = loss_kl
        n_particles = self.n_kp_prior + 1  # K fg + 1 bg

        loss_obj_reg = (kl_mask.sum(-1) ** 2).mean()
        loss = beta_rec * loss_rec + beta_kl * loss_kl + beta_obj * loss_obj_reg
        loss_kl_dyn = torch.tensor(0.0, device=x.device)
        loss_scale = 0.1 if recon_loss_type == 'mse' else 0.01
        loss = loss_scale * loss
        loss_dict = {'loss': loss, 'psnr': psnr.detach(), 'kl': loss_kl_static, 'kl_dyn': loss_kl_dyn,
                     'loss_rec': loss_rec,
                     'obj_on_l1': obj_on_l1, 'loss_kl_kp': loss_kl_kp, 'loss_kl_feat': loss_kl_feat,
                     'loss_kl_obj_on': loss_kl_obj_on, 'loss_kl_scale': loss_kl_scale, 'loss_kl_depth': loss_kl_depth,
                     'loss_kl_context': loss_kl_context, 'loss_obj_reg': loss_obj_reg}
        return loss_dict

    def lerp(self, other, betta):
        # weight interpolation for ema - not used in the paper
        if hasattr(other, 'module'):
            other = other.module
        with torch.no_grad():
            params = self.parameters()
            other_param = other.parameters()
            for p, p_other in zip(params, other_param):
                p.data.lerp_(p_other.data, 1.0 - betta)


"""
JIT scripts
"""


@torch.jit.script
def reparam(mu, logvar):
    return mu + torch.randn_like(mu, device=mu.device) * torch.exp(0.5 * logvar)


@torch.jit.script
def calc_kl_jit(logvar, mu, mu_o: torch.Tensor, logvar_o: torch.Tensor, reduce: str = 'none', balance: float = 0.5):
    """
    Calculate kl-divergence
    :param logvar: log-variance from the encoder
    :param mu: mean from the encoder
    :param mu_o: negative mean for outliers (hyper-parameter)
    :param logvar_o: negative log-variance for outliers (hyper-parameter)
    :param reduce: type of reduce: 'sum', 'none'
    :param balance: balancing coefficient between posterior and prior
    :return: kld
    """
    if balance == 0.5:
        kl = -0.5 * (1 + logvar - logvar_o - torch.exp(logvar - logvar_o) - (mu - mu_o).pow(2) * torch.exp(
            -logvar_o)).sum(-1)
    else:
        # detach post
        mu_post = mu.detach()
        logvar_post = logvar.detach()
        mu_prior = mu_o
        logvar_prior = logvar_o
        kl_a = -0.5 * (1 + logvar_post - logvar_prior - torch.exp(logvar_post - logvar_prior) - (
                mu_post - mu_prior).pow(2) * torch.exp(-logvar_prior)).sum(-1)
        # detach prior
        mu_post = mu
        logvar_post = logvar
        mu_prior = mu_o.detach()
        logvar_prior = logvar_o.detach()
        kl_b = -0.5 * (1 + logvar_post - logvar_prior - torch.exp(logvar_post - logvar_prior) - (
                mu_post - mu_prior).pow(2) * torch.exp(-logvar_prior)).sum(-1)
        kl = (1 - balance) * kl_a + balance * kl_b
    if reduce == 'sum':
        kl = torch.sum(kl)
    elif reduce == 'mean':
        kl = torch.mean(kl)
    return kl


@torch.jit.script
def calc_dynamic_kl(mu_post, logvar_post, mu_prior, logvar_prior,
                    mu_depth_post, logvar_depth_post, mu_depth_prior, logvar_depth_prior,
                    mu_scale_post, logvar_scale_post, mu_scale_prior, logvar_scale_prior,
                    mu_features_post, logvar_features_post, mu_features_prior, logvar_features_prior,
                    mu_bg_post, logvar_bg_post, mu_bg_prior, logvar_bg_prior,
                    obj_on_a_post, obj_on_b_post, obj_on_a_prior, obj_on_b_prior, kl_mask,
                    reduce: str = 'none', features_weight: float = 1.0, balance: float = 0.5,
                    features_dist: str = 'gauss', n_fg_classes: int = 4,
                    n_bg_classes: int = 4, apply_mask_on_obj_on: bool = False):
    # mu_post, mu_depth_post, mu_scale_post, mu_features_post: [bs, n_kp_a, dim]
    # logvar_post, logvar_depth_post, logvar_scale_post, logvar_features_post: [bs, n_kp_a, dim]
    # mu_bg_post, logvar_bg_post: [bs, dim]
    # obj_on_a_post, obj_on_b_post: [bs, n_kp_a, 1]
    # prior: similar, but n_kp_b instead of n_kp_a
    bs = mu_post.shape[0]
    n_particles = mu_post.shape[1]
    # n_prior = mu_prior.shape[1]

    # calc batch pairwise kls
    loss_kl_kp = calc_kl_jit(mu=mu_post.view(-1, mu_post.shape[-1]),
                             logvar=logvar_post.view(-1, logvar_post.shape[-1]),
                             mu_o=mu_prior.view(-1, mu_prior.shape[-1]),
                             logvar_o=logvar_prior.view(-1, logvar_prior.shape[-1]),
                             reduce='none', balance=balance)
    loss_kl_kp = (loss_kl_kp.view(bs, n_particles) * kl_mask).sum(-1)

    loss_kl_depth = calc_kl_jit(mu=mu_depth_post.view(-1, mu_depth_post.shape[-1]),
                                logvar=logvar_depth_post.view(-1, logvar_depth_post.shape[-1]),
                                mu_o=mu_depth_prior.view(-1, mu_depth_prior.shape[-1]),
                                logvar_o=logvar_depth_prior.view(-1, logvar_depth_prior.shape[-1]),
                                reduce='none', balance=balance)
    loss_kl_depth = (loss_kl_depth.view(bs, n_particles) * kl_mask).sum(-1)

    loss_kl_scale = calc_kl_jit(mu=mu_scale_post.view(-1, mu_scale_post.shape[-1]),
                                logvar=logvar_scale_post.view(-1, logvar_scale_post.shape[-1]),
                                mu_o=mu_scale_prior.view(-1, mu_scale_prior.shape[-1]),
                                logvar_o=logvar_scale_prior.view(-1, logvar_scale_prior.shape[-1]),
                                reduce='none', balance=balance)
    loss_kl_scale = (loss_kl_scale.view(bs, n_particles) * kl_mask).sum(-1)

    if features_dist == 'categorical':
        logits_fg_dyn_post = mu_features_post
        logits_fg_dyn_prior = mu_features_prior
        loss_kl_features = calc_kl_categorical(
            logits_fg_dyn_post.reshape(-1, logits_fg_dyn_post.shape[-1]),
            logits_fg_dyn_prior.reshape(-1, logits_fg_dyn_prior.shape[-1]),
            num_classes=n_fg_classes, reduce='none', balance=balance)
        loss_kl_features = (loss_kl_features.view(bs, n_particles) * kl_mask).sum(-1)

        logits_bg_dyn_post = mu_bg_post
        logits_bg_dyn_prior = mu_bg_prior
        loss_kl_bg = calc_kl_categorical(logits_bg_dyn_post, logits_bg_dyn_prior,
                                         num_classes=n_bg_classes, reduce='none', balance=balance)
    else:
        loss_kl_features = calc_kl_jit(mu=mu_features_post.view(-1, mu_features_post.shape[-1]),
                                       logvar=logvar_features_post.view(-1, logvar_features_post.shape[-1]),
                                       mu_o=mu_features_prior.view(-1, mu_features_prior.shape[-1]),
                                       logvar_o=logvar_features_prior.view(-1, logvar_features_prior.shape[-1]),
                                       reduce='none', balance=balance)
        loss_kl_features = (loss_kl_features.view(bs, n_particles) * kl_mask).sum(-1)
        loss_kl_bg = calc_kl_jit(mu=mu_bg_post, logvar=logvar_bg_post, mu_o=mu_bg_prior, logvar_o=logvar_bg_prior,
                                 reduce='none', balance=balance)  # [bs, ]
    # [bs, ]
    if len(obj_on_a_post.shape) == 2:
        obj_on_a_post = obj_on_a_post.unsqueeze(-1)
        obj_on_b_post = obj_on_b_post.unsqueeze(-1)
    if len(obj_on_a_prior.shape) == 2:
        obj_on_a_prior = obj_on_a_prior.unsqueeze(-1)
        obj_on_b_prior = obj_on_b_prior.unsqueeze(-1)

    loss_kl_obj_on = calc_kl_beta_dist(obj_on_a_post.view(-1, obj_on_a_post.shape[-1]),
                                       obj_on_b_post.view(-1, obj_on_b_post.shape[-1]),
                                       obj_on_a_prior.view(-1, obj_on_a_prior.shape[-1]),
                                       obj_on_b_prior.view(-1, obj_on_b_prior.shape[-1]),
                                       reduce='none', balance=balance)
    if apply_mask_on_obj_on:
        loss_kl_obj_on = (loss_kl_obj_on.view(bs, n_particles) * kl_mask).sum(-1)
    else:
        loss_kl_obj_on = loss_kl_obj_on.view(bs, n_particles).sum(-1)
    # [bs, n_kp_a, n_kp_b]
    loss_kl_all = loss_kl_kp + loss_kl_depth + loss_kl_scale + loss_kl_obj_on + features_weight * loss_kl_features

    loss_kl = loss_kl_all + features_weight * loss_kl_bg  # [bs, ]
    if reduce == 'mean':
        loss_kl = loss_kl.mean()
    elif reduce == 'sum':
        loss_kl = loss_kl.sum()

    loss_dict = {'loss_kl': loss_kl, 'loss_kl_kp': loss_kl_kp.sum(),
                 'loss_kl_depth': loss_kl_depth.sum(),
                 'loss_kl_scale': loss_kl_scale.sum(), 'loss_kl_obj_on': loss_kl_obj_on.sum(),
                 'loss_kl_features': loss_kl_features.sum()}
    return loss_dict


if __name__ == '__main__':
    # example hyper-parameters
    batch_size = 4
    # batch_size = 1
    beta_kl = 0.1
    beta_rec = 1.0
    beta_obj = 0.125
    kl_balance = 0.001  # balance between spatial attributes (x, y, scale, depth) and visual features
    n_kp_enc = 42
    n_kp_prior = 64
    patch_size = 4  # patch size for the prior to generate prior proposals
    # patch_size = 64  # patch size for the prior to generate prior proposals
    anchor_s = 0.25  # effective patch size for the posterior: anchor_s * image_size
    image_size = 32
    # image_size = 256
    ch = 3
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    pint_layers = 6  # transformer-based dynamics module number of layers
    pint_heads = 8  # transformer-based dynamics module attention heads
    pint_dim = 256  # transformer-based dynamics module inner dimension (+projection dim)
    beta_dyn = 0.1  # beta-kl for the dynamics loss
    num_static_frames = 1  # "burn-in frames", number of initial frames with kl w.r.t. constant prior (as in DLPv2)
    context_dist = 'gauss'
    # context_dist = 'beta'
    # context_dist = 'categorical'
    context_dim = 7
    timestep_horizon = 10
    deterministic = False
    warmup = False
    predict_delta = False
    # attn_norm_type = 'ln'
    # attn_norm_type = 'pn'
    attn_norm_type = 'rms'

    # new stuff
    learned_feature_dim = 4  # visual features
    learned_bg_feature_dim = 64  # visual features
    obj_res_from_fc = 4  # 8
    obj_ch_mult_prior = (1, 2)  # (1, 2)
    obj_ch_mult = (1, 2, 2)  # (1, 2)
    obj_base_ch = 32
    obj_final_cnn_ch = 32
    bg_res_from_fc = 8
    # bg_ch_mult = (1, 1, 2, 4)
    bg_ch_mult = (1, 1, 2)
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
    ep_dones = torch.randint(low=2, high=timestep_horizon + 2, size=(batch_size * n_im_views, 1), device=device)  # [bs, 1]
    ep_done_mask = torch.ones(batch_size * n_im_views, timestep_horizon + 1, dtype=torch.int, device=device)
    for i in range(ep_done_mask.shape[0]):
        if ep_dones[i] < ep_done_mask.shape[1]:
            ep_done_mask[i, ep_dones[i]:] = 0.0

    print("--- DLP ---")
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
                global_ctx_pool=False,
                pool_ctx_dim=7,
                global_local_fuse_mode='add',
                condition_local_on_global=True,
                pint_dyn_layers=pint_layers,  # pint = particle interaction transformer
                pint_dyn_heads=pint_heads,
                pint_dim=pint_dim,
                pint_ctx_layers=4,
                pint_ctx_heads=8,
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
    output: static
    model(x) output:
    kp_p: torch.Size([4, 64, 2])
    rec: torch.Size([4, 3, 32, 32])
    rec_rgb: torch.Size([4, 3, 32, 32])
    mu_anchor: torch.Size([4, 1, 20, 2])
    logvar_anchor: torch.Size([4, 1, 20, 2])
    z_base_var: torch.Size([4, 1, 20, 5])
    z_base: torch.Size([4, 1, 20, 2])
    z: torch.Size([4, 1, 20, 2])
    mu_offset: torch.Size([4, 1, 20, 2])
    logvar_offset: torch.Size([4, 1, 20, 2])
    z_offset: torch.Size([4, 1, 20, 2])
    mu_tot: torch.Size([4, 1, 20, 2])
    mu_features: torch.Size([4, 1, 20, 16])
    logvar_features: torch.Size([4, 1, 20, 16])
    z_features: torch.Size([4, 1, 20, 16])
    bg: torch.Size([4, 3, 32, 32])
    bg_rgb: torch.Size([4, 3, 32, 32])
    mu_bg_features: torch.Size([4, 1, 64])
    logvar_bg_features: torch.Size([4, 1, 64])
    z_bg_features: torch.Size([4, 1, 64])
    cropped_objects_original: torch.Size([80, 3, 8, 8])
    cropped_objects_original_rgb: torch.Size([80, 3, 8, 8])
    obj_on_a: torch.Size([4, 1, 20, 1])
    obj_on_b: torch.Size([4, 1, 20, 1])
    obj_on: torch.Size([4, 1, 20, 1])
    mu_obj_on: torch.Size([4, 1, 20, 1])
    dec_objects_original: torch.Size([4, 20, 4, 8, 8])
    dec_objects_original_rgb: torch.Size([4, 20, 4, 8, 8])
    dec_objects: torch.Size([4, 3, 32, 32])
    mu_depth: torch.Size([4, 1, 20, 1])
    logvar_depth: torch.Size([4, 1, 20, 1])
    z_depth: torch.Size([4, 1, 20, 1])
    mu_scale: torch.Size([4, 1, 20, 2])
    logvar_scale: torch.Size([4, 1, 20, 2])
    z_scale: torch.Size([4, 1, 20, 2])
    alpha_masks: torch.Size([4, 20, 1, 32, 32])
    mu_score: torch.Size([4, 1, 20, 1])
    logvar_score: torch.Size([4, 1, 20, 1])
    z_score: torch.Size([4, 1, 20, 1])
    -----------------------------------
    output: dynamic
    kp_p: torch.Size([44, 64, 2])
    rec: torch.Size([44, 3, 32, 32])
    rec_rgb: torch.Size([44, 3, 32, 32])
    mu_anchor: torch.Size([4, 11, 64, 2])
    logvar_anchor: torch.Size([4, 11, 64, 2])
    z_base_var: torch.Size([4, 11, 64, 5])
    z_base: torch.Size([4, 11, 64, 2])
    z: torch.Size([4, 11, 64, 2])
    mu_offset: torch.Size([4, 11, 64, 2])
    logvar_offset: torch.Size([4, 11, 64, 2])
    z_offset: torch.Size([4, 11, 64, 2])
    mu_tot: torch.Size([4, 11, 64, 2])
    mu_features: torch.Size([4, 11, 64, 16])
    logvar_features: torch.Size([4, 11, 64, 16])
    z_features: torch.Size([4, 11, 64, 16])
    bg: torch.Size([44, 3, 32, 32])
    bg_rgb: torch.Size([44, 3, 32, 32])
    mu_bg_features: torch.Size([4, 11, 64])
    logvar_bg_features: torch.Size([4, 11, 64])
    z_bg_features: torch.Size([4, 11, 64])
    mu_context: torch.Size([4, 11, 65, 7])
    logvar_context: torch.Size([4, 11, 65, 7])
    z_context: torch.Size([4, 11, 65, 7])
    cropped_objects_original: torch.Size([880, 3, 8, 8])
    cropped_objects_original_rgb: torch.Size([880, 3, 8, 8])
    obj_on_a: torch.Size([4, 11, 64, 1])
    obj_on_b: torch.Size([4, 11, 64, 1])
    obj_on: torch.Size([4, 11, 64, 1])
    mu_obj_on: torch.Size([4, 11, 64, 1])
    dec_objects_original: torch.Size([44, 20, 4, 8, 8])
    dec_objects_original_rgb: torch.Size([44, 20, 4, 8, 8])
    dec_objects: torch.Size([44, 3, 32, 32])
    mu_depth: torch.Size([4, 11, 64, 1])
    logvar_depth: torch.Size([4, 11, 64, 1])
    z_depth: torch.Size([4, 11, 64, 1])
    mu_scale: torch.Size([4, 11, 64, 2])
    logvar_scale: torch.Size([4, 11, 64, 2])
    z_scale: torch.Size([4, 11, 64, 2])
    alpha_masks: torch.Size([44, 20, 1, 32, 32])
    mu_dyn: torch.Size([4, 10, 64, 2])
    logvar_dyn: torch.Size([4, 10, 64, 2])
    mu_features_dyn: torch.Size([4, 10, 64, 16])
    logvar_features_dyn: torch.Size([4, 10, 64, 16])
    obj_on_a_dyn: torch.Size([4, 10, 64])
    obj_on_b_dyn: torch.Size([4, 10, 64])
    mu_depth_dyn: torch.Size([4, 10, 64, 1])
    logvar_depth_dyn: torch.Size([4, 10, 64, 1])
    mu_scale_dyn: torch.Size([4, 10, 64, 2])
    logvar_scale_dyn: torch.Size([4, 10, 64, 2])
    mu_bg_dyn: torch.Size([4, 10, 64])
    logvar_bg_dyn: torch.Size([4, 10, 64])
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
    output: static
    model.calc_elbo(): model losses:
    loss: 29.029062271118164
    psnr: 10.500645637512207
    kl: 165.39024353027344
    kl_dyn: 0.0
    loss_rec: 273.7515869140625
    obj_on_l1: 10.990203857421875
    loss_kl_kp: 56.14067077636719
    loss_kl_feat: 1539.14306640625
    loss_kl_obj_on: 27.826616287231445
    loss_kl_scale: 79.79740142822266
    loss_kl_depth: 0.08641384541988373
    loss_kl_context: 0.0
    -------------------------------
    output: dynamic
    model.calc_elbo(): model losses:
    loss: 66.55062103271484
    psnr: 10.32331657409668
    kl: 504.9709777832031
    kl_dyn: 41333.03515625
    loss_rec: 3136.766357421875
    obj_on_l1: 31.788606643676758
    loss_kl_kp: 173.0999298095703
    loss_kl_feat: 3799.663330078125
    loss_kl_obj_on: 89.04517364501953
    loss_kl_scale: 238.95504760742188
    loss_kl_depth: 0.07118013501167297
    loss_kl_context: 550.4608764648438
    loss_kl_score_dyn: 0.0
    loss_kl_kp_dyn: 1333.2117919921875
    loss_kl_feat_dyn: 32615.171875
    loss_kl_obj_on_dyn: 101.78546905517578
    loss_kl_scale_dyn: 2614.52734375
    loss_kl_depth_dyn: 5.365377426147461
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
        sample_out: torch.Size([1, 20, 3, 32, 32])
        sample_z_out:
        z_pos: torch.Size([1, 20, 64, 2])
        z_scale: torch.Size([1, 20, 64, 2])
        z_obj_on: torch.Size([1, 20, 64, 1])
        z_depth: torch.Size([1, 20, 64, 1])
        z_features: torch.Size([1, 20, 64, 16])
        z_context: torch.Size([1, 19, 65, 7])
        z_bg_features: torch.Size([1, 20, 64])
        z_ids: torch.Size([1, 20, 64])
        z_score: torch.Size([1, 20, 64, 1])
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
                actions_mask_demo = torch.rand(1 * n_im_views, num_steps + cond_steps + num_steps, device=device) > 0.5
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
        rec_dyn: torch.Size([1, 35, 3, 32, 32])
        z_out:
        z_pos: torch.Size([1, 35, 64, 2])
        z_scale: torch.Size([1, 35, 64, 2])
        z_obj_on: torch.Size([1, 35, 64, 1])
        z_depth: torch.Size([1, 35, 64, 1])
        z_features: torch.Size([1, 35, 64, 16])
        z_context: torch.Size([1, 34, 65, 7])
        z_bg_features: torch.Size([1, 35, 64])
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
        sample_out: torch.Size([1, 28, 3, 32, 32])
        sample_z_out:
        z_pos: torch.Size([1, 28, 64, 2])
        z_scale: torch.Size([1, 28, 64, 2])
        z_obj_on: torch.Size([1, 28, 64, 1])
        z_depth: torch.Size([1, 28, 64, 1])
        z_features: torch.Size([1, 28, 64, 16])
        z_context: torch.Size([1, 27, 65, 7])
        z_bg_features: torch.Size([1, 28, 64])
        z_ids: torch.Size([1, 28, 64])
        z_score: torch.Size([1, 28, 64, 1])
        """
