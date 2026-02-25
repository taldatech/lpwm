"""
Modules for DLP
"""
# imports
import math
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.distributions import Beta
from utils.util_func import reparameterize, spatial_transform, create_masks_fast, create_masks_with_scale, \
    modulate
# modules
from modules.vision_modules import Encoder, Decoder

"""
Basic Modules
"""


class AlternativeSpatialSoftmaxKP(torch.nn.Module):
    """
    This module performs spatial-softmax (ssm) by performing marginalization over heatmaps.
    """

    def __init__(self, kp_range=(-1, 1)):
        super().__init__()
        self.kp_range = kp_range

    def forward(self, heatmap, probs=False, variance=False):
        batch_size, n_kp, height, width = heatmap.shape
        # p(x) = \int p(x,y)dy
        logits = heatmap.view(batch_size, n_kp, -1)  # [batch_size, n_kp, h * w]
        scores = torch.softmax(logits, dim=-1)  # [batch_size, n_kp, h * w]
        scores = scores.view(batch_size, n_kp, height, width)  # [batch_size, n_kp, h, w]
        y_axis = torch.linspace(self.kp_range[0], self.kp_range[1], height,
                                device=scores.device).type_as(scores).expand(1, 1, -1)  # [1, 1, features_dim_height]
        x_axis = torch.linspace(self.kp_range[0], self.kp_range[1], width,
                                device=scores.device).type_as(scores).expand(1, 1, -1)  # [1, 1, features_dim_width]

        # marginalize over x (width) and y (height)
        sm_h = scores.sum(dim=-1)  # [batch_size, n_kp, h]
        sm_w = scores.sum(dim=-2)  # [batch_size, n_kp, w]

        # # expected value: probability per coordinate * coordinate
        kp_h = torch.sum(sm_h * y_axis, dim=-1, keepdim=True)  # [batch_size, n_kp, 1]
        kp_h = kp_h.squeeze(-1)  # [batch_size, n_kp], y coordinate of each kp

        kp_w = torch.sum(sm_w * x_axis, dim=-1, keepdim=True)  # [batch_size, n_kp, 1]
        kp_w = kp_w.squeeze(-1)  # [batch_size, n_kp], x coordinate of each kp

        # stack keypoints
        kp = torch.stack([kp_h, kp_w], dim=-1)  # [batch_size, n_kp, 2], x, y coordinates of each kp

        if variance:
            # sigma^2 = E[x^2] - (E[x])^2
            y_sq = (scores * (y_axis.unsqueeze(-1) ** 2)).sum(dim=(-2, -1))  # [batch_size, n_kp]
            v_h = (y_sq - kp_h ** 2).clamp_min(1e-6)  # [batch_size, n_kp]
            x_sq = (scores * (x_axis.unsqueeze(-2) ** 2)).sum(dim=(-2, -1))  # [batch_size, n_kp]
            v_w = (x_sq - kp_w ** 2).clamp_min(1e-6)  # [batch_size, n_kp]

            # covariance: E[xy] - E[x]E[y]
            xy_sq = (scores * (y_axis.unsqueeze(-1) * x_axis.unsqueeze(-2))).sum(dim=(-2, -1))  # [batch_size, n_kp]
            cov = xy_sq - kp_h * kp_w

            var = torch.stack([v_h, v_w, cov], dim=-1)
            return kp, var
        if probs:
            return kp, sm_h, sm_w
        else:
            return kp


class ImagePatcher(nn.Module):
    """
    Author: Tal Daniel
    This module take an image of size B x cdim x H x W and return a patchified tesnor
    B x cdim x num_patches x patch_size x patch_size. It also gives you the global location of the patch
    w.r.t the original image. We use this module to extract prior KP from patches, and we need to know their
    global coordinates for the Chamfer-KL.
    """

    def __init__(self, cdim=3, image_size=64, patch_size=16):
        super(ImagePatcher, self).__init__()
        self.cdim = cdim
        self.image_size = image_size
        self.patch_size = patch_size
        self.kh, self.kw = self.patch_size, self.patch_size  # kernel size
        self.dh, self.dw = self.patch_size, patch_size  # stride
        self.unfold_shape = self.get_unfold_shape()
        self.patch_location_idx = self.get_patch_location_idx()
        # print(f'unfold shape: {self.unfold_shape}')
        # print(f'patch locations: {self.patch_location_idx}')

    def get_patch_location_idx(self):
        h = np.arange(0, self.image_size)[::self.patch_size]
        w = np.arange(0, self.image_size)[::self.patch_size]
        ww, hh = np.meshgrid(h, w)
        hw = np.stack((hh, ww), axis=-1)
        hw = hw.reshape(-1, 2)
        # return torch.from_numpy(hw).int()
        return torch.tensor(hw, dtype=torch.int)

    def get_patch_centers(self):
        mid = self.patch_size // 2
        patch_locations_idx = self.get_patch_location_idx()
        patch_locations_idx += mid
        return patch_locations_idx

    def get_unfold_shape(self):
        dummy_input = torch.zeros(1, self.cdim, self.image_size, self.image_size)
        patches = dummy_input.unfold(2, self.kh, self.dh).unfold(3, self.kw, self.dw)
        unfold_shape = patches.shape[1:]
        return unfold_shape

    def img_to_patches(self, x):
        patches = x.unfold(2, self.kh, self.dh).unfold(3, self.kw, self.dw)
        patches = patches.contiguous().view(patches.shape[0], patches.shape[1], -1, self.kh, self.kw)
        return patches

    def patches_to_img(self, x):
        patches_orig = x.view(x.shape[0], *self.unfold_shape)
        output_h = self.unfold_shape[1] * self.unfold_shape[3]
        output_w = self.unfold_shape[2] * self.unfold_shape[4]
        patches_orig = patches_orig.permute(0, 1, 2, 4, 3, 5).contiguous()
        patches_orig = patches_orig.view(-1, self.cdim, output_h, output_w)
        return patches_orig

    def forward(self, x, patches=True):
        # x [batch_size, 3, image_size, image_size] or [batch_size, 3, num_patches, image_size, image_size]
        if patches:
            return self.img_to_patches(x)
        else:
            return self.patches_to_img(x)


"""
Normalization
"""


class ParticleNorm(nn.Module):
    """
    experimental particle normalization module, not used in the code but left here for research
    """

    def __init__(self, particle_dim, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.particle_dim = particle_dim
        self.a = nn.Parameter(torch.ones(1, 1, 1, self.particle_dim))
        self.g = nn.Parameter(torch.ones(1, 1, 1, self.particle_dim))
        self.s = nn.Parameter(torch.zeros(1, 1, 1, self.particle_dim))

    def forward(self, x):
        # [bs, n_particles, T, dim]
        dims = (1,)
        mean = x.mean(dim=dims, keepdim=True)
        var = x.var(dim=dims, unbiased=False, keepdim=True)
        if len(x.shape) == 3:
            d_n = (x - self.a.squeeze(2) * mean) / (var + self.eps).sqrt()
            out = d_n * self.g.squeeze(2) + self.s.squeeze(2)
        else:
            d_n = (x - self.a * mean) / (var + self.eps).sqrt()
            out = d_n * self.g + self.s
        return out


class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.g = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # F.normalize: x = x / (x ** 2).sum(-1, keepdim=True).sqrt()
        return F.normalize(x, dim=-1) * self.scale * self.g


"""
Attention-based modules
"""


class SimpleRelativePositionalBias(nn.Module):
    # adapted from https://github.com/facebookresearch/mega
    def __init__(self, max_positions, num_heads=1, max_particles=None, layer_norm=False):
        super().__init__()
        self.max_positions = max_positions
        self.num_heads = num_heads
        self.max_particles = max_particles
        self.rel_pos_bias = nn.Parameter(torch.Tensor(2 * max_positions - 1, self.num_heads))
        self.ln_t = nn.LayerNorm([2 * max_positions - 1, self.num_heads]) if layer_norm else nn.Identity()

        if self.max_particles is not None:
            self.particle_rel_pos_bias = nn.Parameter(torch.Tensor(2 * max_particles - 1, self.num_heads))
            self.ln_p = nn.LayerNorm([2 * max_particles - 1, self.num_heads]) if layer_norm else nn.Identity()
        self.reset_parameters()

    def reset_parameters(self):
        std = 0.02
        nn.init.normal_(self.rel_pos_bias, mean=0.0, std=std)
        if self.max_particles is not None:
            nn.init.normal_(self.particle_rel_pos_bias, mean=0.0, std=std)

    def get_particle_rel_position(self, num_particles):
        if self.max_particles is None:
            return 0.0
        if num_particles > self.max_particles:
            raise ValueError('Num particles {} going beyond max particles {}'.format(num_particles, self.max_particles))

        # seq_len * 2 -1
        in_ln = self.ln_p(self.particle_rel_pos_bias)
        b = in_ln[(self.max_particles - num_particles):(self.max_particles + num_particles - 1)]
        # seq_len * 3 - 1
        t = F.pad(b, (0, 0, 0, num_particles))
        # (seq_len * 3 - 1) * seq_len
        t = torch.tile(t, (num_particles, 1))
        t = t[:-num_particles]
        # seq_len x (3 * seq_len - 2)
        t = t.view(num_particles, 3 * num_particles - 2, b.shape[-1])
        r = (2 * num_particles - 1) // 2
        start = r
        end = t.size(1) - r
        t = t[:, start:end]  # [seq_len, seq_len, n_heads]
        t = t.permute(2, 0, 1).unsqueeze(0)  # [1, n_heads, seq_len, seq_len]
        return t

    def forward(self, seq_len, num_particles=None):
        if seq_len > self.max_positions:
            raise ValueError('Sequence length {} going beyond max length {}'.format(seq_len, self.max_positions))

        # seq_len * 2 -1
        in_ln = self.ln_t(self.rel_pos_bias)
        b = in_ln[(self.max_positions - seq_len):(self.max_positions + seq_len - 1)]
        # seq_len * 3 - 1
        t = F.pad(b, (0, 0, 0, seq_len))
        # (seq_len * 3 - 1) * seq_len
        t = torch.tile(t, (seq_len, 1))
        t = t[:-seq_len]
        # seq_len x (3 * seq_len - 2)
        t = t.view(seq_len, 3 * seq_len - 2, b.shape[-1])
        r = (2 * seq_len - 1) // 2
        start = r
        end = t.size(1) - r
        t = t[:, start:end]  # [seq_len, seq_len, n_heads]
        t = t.permute(2, 0, 1).unsqueeze(0)  # [1, n_heads, seq_len, seq_len]
        p = None
        if num_particles is not None and self.max_particles is not None:
            p = self.get_particle_rel_position(num_particles)  # [1, n_heads, n_part, n_part]
            t = t[:, :, None, :, None, :]
            p = p[:, :, :, None, :, None]
        return t, p


class CausalParticleSelfAttention(nn.Module):
    """
    A particle-based multi-head masked self-attention layer with a projection at the end.
    """

    def __init__(self, n_embed, n_head, block_size, attn_pdrop=0.1, resid_pdrop=0.1,
                 positional_bias=False, max_particles=None, linear_bias=False, torch_attn=False):
        super().__init__()
        assert n_embed % n_head == 0
        self.attn_pdrop = attn_pdrop
        self.resid_pdrop = resid_pdrop
        self.torch_attn = torch_attn
        # key, query, value projections for all heads
        self.key = nn.Linear(n_embed, n_embed, bias=linear_bias)
        self.query = nn.Linear(n_embed, n_embed, bias=linear_bias)
        self.value = nn.Linear(n_embed, n_embed, bias=linear_bias)
        # regularization
        self.attn_drop = nn.Dropout(attn_pdrop) if not self.torch_attn else nn.Identity()
        # output projection
        self.proj = nn.Linear(n_embed, n_embed, bias=linear_bias)

        self.resid_drop = nn.Dropout(resid_pdrop)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size))
                             .view(1, 1, 1, block_size, 1, block_size))
        self.n_head = n_head
        self.positional_bias = positional_bias
        self.max_particles = max_particles
        if self.positional_bias:
            self.rel_pos_bias = SimpleRelativePositionalBias(block_size, n_head, max_particles=max_particles)
        else:
            self.rel_pos_bias = nn.Identity()

    def forward(self, x):
        B, N, T, C = x.size()  # batch size, n_particles, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, N * T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, N * T, hs)
        q = self.query(x).view(B, N * T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, N * T, hs)
        v = self.value(x).view(B, N * T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, N * T, hs)

        if self.torch_attn:
            y = F.scaled_dot_product_attention(query=q, key=k, value=v, is_causal=True,
                                               dropout_p=self.attn_pdrop if self.training else 0.0)

        else:
            # causal self-attention; Self-attend: (B, nh, N * T, hs) x (B, nh, hs, N  *T) -> (B, nh, N * T, N *T )
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))  # (B, nh, N * T, N * T)
            att = att.view(B, -1, N, T, N, T)  # (B, nh, N, T, N, T)
            if self.positional_bias:
                if self.max_particles is not None:
                    bias_t, bias_p = self.rel_pos_bias(T, num_particles=N)
                    bias_t = bias_t.view(1, bias_t.shape[1], 1, T, 1, T)
                    bias_p = bias_p.view(1, bias_p.shape[1], N, 1, N, 1)
                    att = att + bias_t + bias_p
                else:
                    bias_t, _ = self.rel_pos_bias(T)
                    bias_t = bias_t.view(1, bias_t.shape[1], 1, T, 1, T)
                    att = att + bias_t
            att = att.masked_fill(self.mask[:, :, :, :T, :, :T] == 0, float('-inf'))
            att = att.view(B, -1, N * T, N * T)  # (B, nh, N * T, N * T)
            att = F.softmax(att, dim=-1)
            att = self.attn_drop(att)
            y = att @ v  # (B, nh, N*T, N*T) x (B, nh, N*T, hs) -> (B, nh, N*T, hs)

        y = y.transpose(1, 2).contiguous().view(B, N * T, C)  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        y = y.view(B, N, T, -1)
        return y


class ParticleSelfAttention(nn.Module):
    """
    A particle-based multi-head masked self-attention layer with a projection at the end.
    """

    def __init__(self, n_embed, n_head, block_size, attn_pdrop=0.1, resid_pdrop=0.1,
                 positional_bias=False, max_particles=None, linear_bias=False, torch_attn=False):
        super().__init__()
        assert n_embed % n_head == 0
        self.attn_pdrop = attn_pdrop
        self.resid_pdrop = resid_pdrop
        self.torch_attn = torch_attn
        # key, query, value projections for all heads
        self.key = nn.Linear(n_embed, n_embed, bias=linear_bias)
        self.query = nn.Linear(n_embed, n_embed, bias=linear_bias)
        self.value = nn.Linear(n_embed, n_embed, bias=linear_bias)
        # regularization
        self.attn_drop = nn.Dropout(attn_pdrop) if not self.torch_attn else nn.Identity()
        # output projection
        self.proj = nn.Linear(n_embed, n_embed, bias=linear_bias)

        self.resid_drop = nn.Dropout(resid_pdrop)
        self.n_head = n_head
        self.positional_bias = positional_bias
        self.max_particles = max_particles
        if self.positional_bias:
            self.rel_pos_bias = SimpleRelativePositionalBias(block_size, n_head, max_particles=max_particles)
        else:
            self.rel_pos_bias = nn.Identity()

    def forward(self, x):
        B, N, T, C = x.size()  # batch size, n_particles, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, N * T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, N * T, hs)
        q = self.query(x).view(B, N * T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, N * T, hs)
        v = self.value(x).view(B, N * T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, N * T, hs)

        if self.torch_attn:
            y = F.scaled_dot_product_attention(query=q, key=k, value=v, is_causal=False,
                                               dropout_p=self.attn_pdrop if self.training else 0.0)

        else:
            # causal self-attention; Self-attend: (B, nh, N * T, hs) x (B, nh, hs, N  *T) -> (B, nh, N * T, N *T )
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))  # (B, nh, N * T, N * T)
            if self.positional_bias:
                att = att.view(B, -1, N, T, N, T)  # (B, nh, N, T, N, T)
                if self.max_particles is not None:
                    bias_t, bias_p = self.rel_pos_bias(T, num_particles=N)
                    bias_t = bias_t.view(1, bias_t.shape[1], 1, T, 1, T)
                    bias_p = bias_p.view(1, bias_p.shape[1], N, 1, N, 1)
                    att = att + bias_t + bias_p
                else:
                    bias_t, _ = self.rel_pos_bias(T)
                    bias_t = bias_t.view(1, bias_t.shape[1], 1, T, 1, T)
                    att = att + bias_t
                att = att.view(B, -1, N * T, N * T)  # (B, nh, N * T, N * T)
            att = F.softmax(att, dim=-1)
            att = self.attn_drop(att)
            y = att @ v  # (B, nh, N*T, N*T) x (B, nh, N*T, hs) -> (B, nh, N*T, hs)

        y = y.transpose(1, 2).contiguous().view(B, N * T, C)  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        y = y.view(B, N, T, -1)
        return y


class ParticleCrossAttention(nn.Module):
    """
    A particle-based multi-head masked self-attention layer with a projection at the end.
    """

    def __init__(self, n_embed, n_head, block_size, attn_pdrop=0.1, resid_pdrop=0.1,
                 positional_bias=False, max_particles=None, linear_bias=False, torch_attn=False, particles_first=False):
        super().__init__()
        assert n_embed % n_head == 0
        self.particles_first = particles_first
        self.attn_pdrop = attn_pdrop
        self.resid_pdrop = resid_pdrop
        self.torch_attn = torch_attn
        # key, query, value projections for all heads
        self.key = nn.Linear(n_embed, n_embed, bias=linear_bias)
        self.query = nn.Linear(n_embed, n_embed, bias=linear_bias)
        self.value = nn.Linear(n_embed, n_embed, bias=linear_bias)
        # regularization
        self.attn_drop = nn.Dropout(attn_pdrop) if not self.torch_attn else nn.Identity()
        # output projection
        self.proj = nn.Linear(n_embed, n_embed, bias=linear_bias)

        self.resid_drop = nn.Dropout(resid_pdrop)
        self.n_head = n_head
        self.positional_bias = positional_bias
        self.max_particles = max_particles
        if self.positional_bias:
            self.rel_pos_bias = SimpleRelativePositionalBias(block_size, n_head, max_particles=max_particles)
        else:
            self.rel_pos_bias = nn.Identity()

    def forward(self, x_q, x_kv):
        if self.particles_first:
            B, Nq, Tq, Cq = x_q.size()  # batch size, n_particles, sequence length, embedding dimensionality (n_embd)
            _, Nkv, Tkv, Ckv = x_kv.size()  # batch size, n_particles, sequence length, embedding dimensionality (n_embd)
        else:
            B, Tq, Nq, Cq = x_q.size()  # batch size, n_particles, sequence length, embedding dimensionality (n_embd)
            _, Tkv, Nkv, Ckv = x_kv.size()  # batch size, n_particles, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x_kv).view(B, Nkv * Tkv, self.n_head, Ckv // self.n_head).transpose(1, 2)  # (B, nh, N * T, hs)
        q = self.query(x_q).view(B, Nq * Tq, self.n_head, Cq // self.n_head).transpose(1, 2)  # (B, nh, N * T, hs)
        v = self.value(x_kv).view(B, Nkv * Tkv, self.n_head, Ckv // self.n_head).transpose(1,
                                                                                           2)  # (B, nh, N * T, hs)

        if self.torch_attn:
            y = F.scaled_dot_product_attention(query=q, key=k, value=v, is_causal=False,
                                               dropout_p=self.attn_pdrop if self.training else 0.0)
        else:

            # causal self-attention; Self-attend: (B, nh, N * T, hs) x (B, nh, hs, N  *T) -> (B, nh, N * T, N *T )
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))  # (B, nh, N * T, N * T)
            att = F.softmax(att, dim=-1)
            att = self.attn_drop(att)
            y = att @ v  # (B, nh, N*T, N*T) x (B, nh, N*T, hs) -> (B, nh, N*T, hs)

        y = y.transpose(1, 2).contiguous().view(B, Nq * Tq, Ckv)  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        if self.particles_first:
            y = y.view(B, Nq, Tq, -1)
        else:
            y = y.view(B, Tq, Nq, -1)
        return y


class MLP(nn.Module):
    def __init__(self, n_embed, resid_pdrop=0.1, hidden_dim_multiplier=4, activation='gelu'):
        super().__init__()
        self.fc_1 = nn.Linear(n_embed, hidden_dim_multiplier * n_embed)
        if activation == 'gelu':
            self.act = nn.GELU()
        else:
            self.act = nn.ReLU(True)
        self.proj = nn.Linear(hidden_dim_multiplier * n_embed, n_embed)
        self.dropout = nn.Dropout(resid_pdrop)

    def forward(self, x):
        x = self.dropout(self.proj(self.act(self.fc_1(x))))
        return x


class FinalTransformerLayer(nn.Module):
    def __init__(self, n_embed, output_dim, bias=True, context_cond=False, residual_modulation=False, norm_type='rms'):
        super().__init__()
        if norm_type == 'rms':
            norm_layer = RMSNorm
        else:
            norm_layer = nn.LayerNorm
        self.norm = norm_layer(n_embed)
        self.head = nn.Linear(n_embed, output_dim, bias=bias)
        self.context_cond = context_cond
        self.residual_modulation = residual_modulation
        if self.context_cond:
            self.c_proj = nn.Linear(n_embed, 2 * n_embed)
            nn.init.constant_(self.c_proj.weight, 0.0)
            if self.residual_modulation:
                nn.init.constant_(self.c_proj.bias, 0.0)
            else:
                nn.init.constant_(self.c_proj.bias[:n_embed], 1.0)  # identity
                nn.init.constant_(self.c_proj.bias[n_embed:], 0.0)  # zero shift
        else:
            self.c_proj = None

    def forward(self, x, c=None):
        if self.context_cond and c is not None:
            scale, shift = self.c_proj(c).chunk(2, dim=-1)
            x = self.head(modulate(self.norm(x), scale, shift, self.residual_modulation))
        else:
            x = self.head(self.norm(x))
        return x


class MLPSwiglu(nn.Module):
    def __init__(self, n_embed, resid_pdrop=0.0, hidden_dim_multiplier=4, activation='gelu', bias=False):
        super().__init__()
        self.w1 = nn.Linear(n_embed, hidden_dim_multiplier * n_embed, bias=bias)
        self.w2 = nn.Linear(hidden_dim_multiplier * n_embed, n_embed, bias=bias)
        self.w3 = nn.Linear(n_embed, hidden_dim_multiplier * n_embed, bias=bias)

        self.dropout = nn.Dropout(resid_pdrop)

    def forward(self, x):
        x = self.dropout(self.w2(F.silu((self.w1(x))) * self.w3(x)))
        return x


class CausalBlock(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, n_embed, n_head, block_size, attn_pdrop=0.1, resid_pdrop=0.1, hidden_dim_multiplier=4,
                 positional_bias=False, activation='gelu', max_particles=None, norm_type='rms', context_cond=False,
                 residual_modulation=False, context_gate=False, attn_scale=1.0):
        super().__init__()
        self.max_particles = max_particles
        if norm_type == 'rms':
            norm_layer = RMSNorm
        elif norm_type == 'pn':
            norm_layer = ParticleNorm
        else:
            norm_layer = nn.LayerNorm
        self.ln1 = norm_layer(n_embed)
        self.ln2 = norm_layer(n_embed)
        self.attn = CausalParticleSelfAttention(n_embed, n_head, block_size, attn_pdrop, resid_pdrop,
                                                positional_bias=positional_bias, max_particles=max_particles)
        self.mlp = MLP(n_embed, resid_pdrop, hidden_dim_multiplier, activation=activation)
        self.attn_scale = attn_scale
        self.context_cond = context_cond
        self.residual_modulation = residual_modulation
        self.context_gate = context_gate
        self.c_multiplier = 6 if context_gate else 4
        if self.context_cond:
            self.c_proj = nn.Linear(n_embed, self.c_multiplier * n_embed)
            nn.init.constant_(self.c_proj.weight, 0.0)
            if self.residual_modulation:
                nn.init.constant_(self.c_proj.bias, 0.0)
            else:
                nn.init.constant_(self.c_proj.bias[:2 * n_embed], 1.0)  # identity
                nn.init.constant_(self.c_proj.bias[2 * n_embed: 4 * n_embed], 0.0)  # zero shift
                if self.context_gate:
                    nn.init.constant_(self.c_proj.bias[4 * n_embed:], 0.0)  # zero gate
        else:
            self.c_proj = None

    def forward(self, x, c=None):
        if self.context_cond and c is not None:
            c_proj = self.c_proj(c).chunk(self.c_multiplier, dim=-1)
            scale_a, scale_b, shift_a, shift_b = c_proj[0], c_proj[1], c_proj[2], c_proj[3]
            if self.context_gate:
                gate_a, gate_b = c_proj[4], c_proj[5]
            else:
                gate_a = gate_b = 1.0
            x = x + self.attn_scale * gate_a * self.attn(
                modulate(self.ln1(x), scale_a, shift_a, self.residual_modulation))
            x = x + gate_b * self.mlp(modulate(self.ln2(x), scale_b, shift_b, self.residual_modulation))
        else:
            x = x + self.attn_scale * self.attn(self.ln1(x))
            x = x + self.mlp(self.ln2(x))
        return x


class SpatioTemporalBlock(nn.Module):
    """ spatio-temporal Transformer block """

    def __init__(self, n_embed, n_head, block_size, attn_pdrop=0.1, resid_pdrop=0.1, hidden_dim_multiplier=4,
                 positional_bias=False, activation='gelu', max_particles=None, norm_type='rms', causal=True,
                 context_cond=False, residual_modulation=True, context_gate=False, attn_scale=1.0,
                 cross_attn_cond=False, attention_order=('spatial', 'temporal')):
        super().__init__()
        self.context_cond = context_cond
        self.residual_modulation = residual_modulation
        self.context_gate = context_gate
        spatio_block_size = 1
        self.attn_scale = attn_scale
        self.causal = causal
        self.cross_attn_cond = cross_attn_cond
        self.attention_order = attention_order
        if self.cross_attn_cond and 'cross' not in self.attention_order:
            self.attention_order = [*attention_order, 'cross']
        self.spatio_block = SelfBlock(n_embed, n_head, spatio_block_size, attn_pdrop,
                                      resid_pdrop, hidden_dim_multiplier,
                                      positional_bias, activation=activation, max_particles=max_particles,
                                      norm_type=norm_type, context_cond=context_cond,
                                      residual_modulation=residual_modulation, context_gate=context_gate,
                                      attn_scale=attn_scale)
        temp_block_type = CausalBlock if self.causal else SelfBlock
        self.temp_block = temp_block_type(n_embed, n_head, block_size, attn_pdrop,
                                          resid_pdrop, hidden_dim_multiplier,
                                          positional_bias, activation=activation, max_particles=max_particles,
                                          norm_type=norm_type, context_cond=context_cond,
                                          residual_modulation=residual_modulation, context_gate=context_gate,
                                          attn_scale=attn_scale)
        if self.cross_attn_cond:
            self.cross_block = CrossBlock(n_embed, n_head, spatio_block_size, attn_pdrop,
                                          resid_pdrop, hidden_dim_multiplier, positional_bias, activation=activation,
                                          max_particles=max_particles,
                                          norm_type=norm_type, context_cond=context_cond,
                                          residual_modulation=residual_modulation, context_gate=context_gate,
                                          attn_scale=attn_scale, particles_first=True)
        else:
            self.cross_block = nn.Identity()

    def forward(self, x, c=None, l=None):
        # x: [b, n + 1, t, f]
        # c: context conditioning via AdaLN: [b, n+1, t, f] or None
        # l: language conditioning via cross-attention: [b, t, h, f] or None, h=N_l is the number of lang tokens
        B, N, T, F = x.shape
        for attn_type in self.attention_order:
            if attn_type == 'spatial':
                x = x.permute(0, 2, 1, 3)  # [b, t, n + 1, f]
                x = x.reshape(-1, N, 1, F)  # [b, * t, n + 1, 1, f]
                if c is not None:
                    N_c = c.shape[1]
                    c_s = c.permute(0, 2, 1, 3)  # [b, t, n + 1, f]
                    c_s = c_s.reshape(-1, N_c, 1, F)  # [b, * t, n + 1, 1, f]
                else:
                    c_s = None
                x = self.spatio_block(x, c_s)
                x = x.view(B, T, N, F)  # [b, t, n + 1, f]
                x = x.permute(0, 2, 1, 3)  # [b, n + 1, t, f]
            elif attn_type == 'temporal':
                x = x.reshape(-1, 1, T, F)  # [b * (n + 1), 1, t, f]
                if c is not None:
                    N_c = c.shape[1]
                    c_t = c.reshape(-1, 1, T, F)  # [b * (n + 1), 1, t, f]
                else:
                    c_t = None
                x = self.temp_block(x, c_t)
                x = x.view(B, N, T, F)  # [b, n + 1, t, f]
            elif attn_type == 'cross' and self.cross_attn_cond and l is not None:
                x = x.permute(0, 2, 1, 3)  # [b, t, n + 1, f]
                x = x.reshape(-1, N, 1, F)  # [b * t, n + 1, 1, f]
                N_l = l.shape[2]
                l = l.reshape(-1, N_l, 1, F)  # [b * t, N_l=h, 1, f]
                if c is not None:
                    N_c = c.shape[1]
                    c_s = c.permute(0, 2, 1, 3)  # [b, t, n + 1, f]
                    c_s = c_s.reshape(-1, N_c, 1, F)  # [b, * t, n + 1, 1, f]
                else:
                    c_s = None
                x = self.cross_block(x_q=x, x_kv=l, c=c_s)
                x = x.view(B, T, N, F)  # [b, t, n + 1, f]
                x = x.permute(0, 2, 1, 3)  # [b, n + 1, t, f]
        return x


class SelfBlock(nn.Module):
    """ self-attention Transformer block """

    def __init__(self, n_embed, n_head, block_size, attn_pdrop=0.1, resid_pdrop=0.1, hidden_dim_multiplier=4,
                 positional_bias=False, activation='gelu', max_particles=None, norm_type='ln', context_cond=False,
                 residual_modulation=False, context_gate=False, attn_scale=1.0):
        super().__init__()
        self.max_particles = max_particles
        if norm_type == 'rms':
            norm_layer = RMSNorm
        elif norm_type == 'pn':
            norm_layer = ParticleNorm
        else:
            norm_layer = nn.LayerNorm
        self.ln1 = norm_layer(n_embed)
        self.ln2 = norm_layer(n_embed)
        self.attn = ParticleSelfAttention(n_embed, n_head, block_size, attn_pdrop, resid_pdrop,
                                          positional_bias=positional_bias, max_particles=max_particles)
        self.attn_scale = attn_scale
        self.mlp = MLP(n_embed, resid_pdrop, hidden_dim_multiplier, activation=activation)
        self.context_cond = context_cond
        self.residual_modulation = residual_modulation
        self.context_gate = context_gate
        self.c_multiplier = 6 if context_gate else 4
        if self.context_cond:
            self.c_proj = nn.Linear(n_embed, self.c_multiplier * n_embed)
            nn.init.constant_(self.c_proj.weight, 0.0)
            if self.residual_modulation:
                nn.init.constant_(self.c_proj.bias, 0.0)
            else:
                nn.init.constant_(self.c_proj.bias[:2 * n_embed], 1.0)  # identity
                nn.init.constant_(self.c_proj.bias[2 * n_embed: 4 * n_embed], 0.0)  # zero shift
                if self.context_gate:
                    nn.init.constant_(self.c_proj.bias[4 * n_embed:], 0.0)  # zero gate

    def forward(self, x, c=None):
        if self.context_cond and c is not None:
            c_proj = self.c_proj(c).chunk(self.c_multiplier, dim=-1)
            scale_a, scale_b, shift_a, shift_b = c_proj[0], c_proj[1], c_proj[2], c_proj[3]
            if self.context_gate:
                gate_a, gate_b = c_proj[4], c_proj[5]
            else:
                gate_a = gate_b = 1.0
            x = x + self.attn_scale * gate_a * self.attn(
                modulate(self.ln1(x), scale_a, shift_a, self.residual_modulation))
            x = x + gate_b * self.mlp(modulate(self.ln2(x), scale_b, shift_b, self.residual_modulation))
        else:
            x = x + self.attn_scale * self.attn(self.ln1(x))
            x = x + self.mlp(self.ln2(x))
        return x


class CrossBlock(nn.Module):
    """ cross-attention Transformer block """

    def __init__(self, n_embed, n_head, block_size, attn_pdrop=0.1, resid_pdrop=0.1, hidden_dim_multiplier=4,
                 positional_bias=False, activation='gelu', max_particles=None, norm_type='ln', particles_first=False,
                 norm_kv=False, context_cond=False,
                 residual_modulation=False, context_gate=False, attn_scale=1.0):
        super().__init__()
        self.max_particles = max_particles
        self.norm_kv = norm_kv
        if norm_type == 'rms':
            norm_layer = RMSNorm
        elif norm_type == 'pn':
            norm_layer = ParticleNorm
        else:
            norm_layer = nn.LayerNorm
        self.ln1 = norm_layer(n_embed)
        self.ln2 = norm_layer(n_embed)
        self.ln_kv = self.ln1 if self.norm_kv else nn.Identity()
        self.attn_scale = attn_scale
        self.context_cond = context_cond
        self.residual_modulation = residual_modulation
        self.context_gate = context_gate
        self.c_multiplier = 6 if context_gate else 4
        if self.context_cond:
            self.c_proj = nn.Linear(n_embed, self.c_multiplier * n_embed)
            nn.init.constant_(self.c_proj.weight, 0.0)
            if self.residual_modulation:
                nn.init.constant_(self.c_proj.bias, 0.0)
            else:
                nn.init.constant_(self.c_proj.bias[:2 * n_embed], 1.0)  # identity
                nn.init.constant_(self.c_proj.bias[2 * n_embed: 4 * n_embed], 0.0)  # zero shift
                if self.context_gate:
                    nn.init.constant_(self.c_proj.bias[4 * n_embed:], 0.0)  # zero gate
        self.attn = ParticleCrossAttention(n_embed, n_head, block_size, attn_pdrop, resid_pdrop,
                                           positional_bias=positional_bias, max_particles=max_particles,
                                           particles_first=particles_first)
        self.mlp = MLP(n_embed, resid_pdrop, hidden_dim_multiplier, activation=activation)

    def forward(self, x_q, x_kv, c=None):
        if self.context_cond and c is not None:
            c_proj = self.c_proj(c).chunk(self.c_multiplier, dim=-1)
            scale_a, scale_b, shift_a, shift_b = c_proj[0], c_proj[1], c_proj[2], c_proj[3]
            if self.context_gate:
                gate_a, gate_b = c_proj[4], c_proj[5]
            else:
                gate_a = gate_b = 1.0
            x_q = x_q + self.attn_scale * gate_a * self.attn(
                modulate(self.ln1(x_q), scale_a, shift_a, self.residual_modulation), self.ln_kv(x_kv))
            x_q = x_q + gate_b * self.mlp(modulate(self.ln2(x_q), scale_b, shift_b, self.residual_modulation))
        else:
            # x_q = x_q + self.attn(self.ln1(x_q), self.ln1(x_kv))
            x_q = x_q + self.attn_scale * self.attn(self.ln1(x_q), self.ln_kv(x_kv))
            x_q = x_q + self.mlp(self.ln2(x_q))
        return x_q


class ParticleSpatioTemporalTransformer(nn.Module):
    def __init__(self, n_embed, n_head, n_layer, block_size, output_dim, attn_pdrop=0.1, resid_pdrop=0.1,
                 hidden_dim_multiplier=4, positional_bias=False, activation='gelu', max_particles=None, norm_type='rms',
                 n_registers=0, particles_first=True, init_std=0.02,
                 causal=True, context_cond=False, residual_modulation=True, context_gate=True,
                 attention_order=('spatial', 'temporal'), cond_cross_attn=False,
                 token_pool_adaln=False,
                 pos_embed_t_adaln=False
                 ):
        super().__init__()
        self.positional_bias = positional_bias
        self.max_particles = max_particles  # for positional bias
        self.particles_first = particles_first  # expect [bs, n, t, f], else [bs, t, n, f]
        self.causal = causal
        self.n_head = n_head
        self.init_std = init_std
        self.pos_embed_t_adaln = pos_embed_t_adaln
        self.context_cond = context_cond or pos_embed_t_adaln
        self.residual_modulation = residual_modulation
        self.context_gate = context_gate
        self.attention_order = attention_order
        self.cond_cross_attn = cond_cross_attn
        self.token_pool_adaln = token_pool_adaln
        # self.attn_scale = 1 / math.sqrt(2 * 2 * n_layer)
        self.attn_scale = 1.0
        if norm_type == 'rms':
            norm_layer = RMSNorm
        elif norm_type == 'pn':
            norm_layer = ParticleNorm
        else:
            norm_layer = nn.LayerNorm
        if n_registers > 0:
            self.n_registers = n_registers
            self.registers = nn.Parameter(self.init_std * torch.randn(1, self.n_registers, 1, n_embed))
        else:
            self.n_registers = 0
            self.registers = None
        # input embedding stem
        if self.pos_embed_t_adaln:
            self.pos_embed_t_embedding = nn.Parameter(self.init_std * torch.randn(1, 1, block_size, n_embed))

        if self.positional_bias:
            self.pos_emb = nn.Identity()
        else:
            if self.pos_embed_t_adaln:
                self.pos_emb = nn.Identity()
            else:
                self.pos_emb = nn.Parameter(self.init_std * torch.randn(1, block_size, n_embed))



        attn_context_cond = context_cond or self.token_pool_adaln
        self.blocks = nn.Sequential(*[SpatioTemporalBlock(n_embed, n_head, block_size, attn_pdrop,
                                                          resid_pdrop, hidden_dim_multiplier,
                                                          positional_bias, activation=activation,
                                                          max_particles=max_particles,
                                                          norm_type=norm_type, causal=causal,
                                                          context_cond=attn_context_cond,
                                                          residual_modulation=residual_modulation,
                                                          context_gate=context_gate, attn_scale=self.attn_scale,
                                                          attention_order=self.attention_order,
                                                          cross_attn_cond=self.cond_cross_attn)
                                      for _ in range(n_layer)])

        # decoder head
        self.head = FinalTransformerLayer(n_embed, output_dim, bias=True, context_cond=self.context_cond,
                                          residual_modulation=self.residual_modulation, norm_type=norm_type)
        self.block_size = block_size
        self.n_embed = n_embed
        self.n_layer = n_layer
        # print(f"particle transformer # parameters: {sum(p.numel() for p in self.parameters())}")

    def get_block_size(self):
        return self.block_size

    def init_weights(self):
        # initialize layers
        pass
        # for m in self.modules():
        #     if isinstance(m, nn.Linear):
        #         torch.nn.init.xavier_uniform_(m.weight)
        #         if m.bias is not None:
        #             nn.init.constant_(m.bias, 0)
        # if self.causal:
        #     self.apply(self._init_weights)
        # if self.positional_bias:
        #     for m in self.blocks:
        #         m.attn.rel_pos_bias.reset_parameters()

    def _init_weights(self, module):
        std = 0.02
        if isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(self, x, c=None, l=None):
        if self.particles_first:
            b, n, t, f = x.size()
            if c is not None:
                if len(c.shape) == 3:  # [b, t, f]
                    c = c.unsqueeze(1)  # [b, t, f]
                    n_c = 1
                else:
                    n_c = c.shape[1]
        else:
            b, t, n, f = x.size()
            x = x.permute(0, 2, 1, 3)  # [bs, n, t, f]
            if c is not None:
                if len(c.shape) == 3:  # [b, t, f]
                    c = c.unsqueeze(1)  # [b, t, f]
                    n_c = 1
                else:
                    c = c.permute(0, 2, 1, 3)  # [bs, n_c, t, d]
                    n_c = c.shape[1]
        # n is the number of particles
        assert t <= self.block_size, "Cannot forward, model block size is exhausted."
        assert f == self.n_embed, "invalid particle feature dim"

        # add register tokens
        if self.n_registers > 0:
            x = torch.cat([x, self.registers.repeat(b, 1, t, 1)], dim=1)
            # [bs, n + n_mem_particles, t, f]

        if not self.positional_bias and not self.pos_embed_t_adaln:
            position_embeddings = self.pos_emb[:, None, :t, :]
            x = x + position_embeddings

        # prepare condition
        if self.pos_embed_t_adaln:
            c_t = self.pos_embed_t_embedding[:, :, :t].repeat(x.shape[0], x.shape[1], 1, 1)
            if c is None:
                c = c_t
            else:
                c = c + c_t

        if self.token_pool_adaln:
            token_pool = x[:, -(self.n_registers + 1)].unsqueeze(1).repeat(1, x.shape[1], 1, 1)
            if c is None:
                c_in = token_pool
            else:
                c_in = c + token_pool
        else:
            c_in = c

        for block in self.blocks:
            x = block(x, c_in, l)

        if self.n_registers > 0:
            x = x[:, :-self.n_registers]

        logits = self.head(x, c)
        if not self.particles_first:
            logits = logits.permute(0, 2, 1, 3)  # [bs, t, n, f]
        return logits


class ParticleSelfAttTransformer(nn.Module):
    def __init__(self, n_embed, n_head, n_layer, block_size, output_dim, attn_pdrop=0.1, resid_pdrop=0.1,
                 hidden_dim_multiplier=4, positional_bias=False, activation='gelu', max_particles=None,
                 norm_type='rms', n_registers=0, init_std=0.02):
        super().__init__()
        self.positional_bias = positional_bias
        self.max_particles = max_particles  # for positional bias
        self.n_registers = n_registers  # "vision transformers need registers", balances the attention matrix
        # input embedding stem
        if self.positional_bias:
            self.pos_emb = nn.Identity()
        else:
            self.pos_emb = nn.Parameter(init_std * torch.randn(1, block_size, n_embed))
        if self.n_registers > 0:
            self.registers = nn.Parameter(init_std * torch.randn(1, self.n_registers, 1, n_embed))
        else:
            self.registers = None
        # transformer
        self.blocks = nn.Sequential(*[SelfBlock(n_embed, n_head, block_size, attn_pdrop,
                                                resid_pdrop, hidden_dim_multiplier,
                                                positional_bias, activation=activation, max_particles=max_particles,
                                                norm_type=norm_type)
                                      for _ in range(n_layer)])
        # decoder head
        if norm_type == 'rms':
            norm_layer = RMSNorm
        elif norm_type == 'pn':
            norm_layer = ParticleNorm
        else:
            norm_layer = nn.LayerNorm
        self.ln_f = norm_layer(n_embed)
        self.head = nn.Linear(n_embed, output_dim, bias=False)

        self.block_size = block_size
        self.n_embed = n_embed
        self.n_layer = n_layer
        # print(f"particle transformer # parameters: {sum(p.numel() for p in self.parameters())}")

    def get_block_size(self):
        return self.block_size

    def init_weights(self):
        # initialize layers
        pass
        # self.apply(self._init_weights)
        # if self.positional_bias:
        #     for m in self.blocks:
        #         m.attn.rel_pos_bias.reset_parameters()

    def _init_weights(self, module):
        std = 0.02
        if isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
        # elif isinstance(module, ParticleTransformer):
        #     if not self.positional_bias:
        #         torch.nn.init.normal_(module.pos_emb, mean=0.0, std=std)

    def forward(self, x):
        # x: [b, t, n, f]
        x = x.permute(0, 2, 1, 3)  # [b, n, t, f]
        b, n, t, f = x.size()
        # n is the number of particles
        assert t <= self.block_size, f"Cannot forward, model block size is exhausted: t:{t}, block_size: {self.block_size}"
        assert f == self.n_embed, "invalid particle feature dim"

        if self.n_registers > 0 and self.registers is not None:
            registers = self.registers.repeat(b, 1, t, 1)
            x = torch.cat([x, registers], dim=1)  # [b, n+n_reg, t, f]

        if not self.positional_bias:
            position_embeddings = self.pos_emb[:, None, :t, :]
            x = x + position_embeddings
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.head(x)
        if self.n_registers > 0:
            logits, _ = logits.split([logits.shape[1] - self.n_registers, self.n_registers], dim=1)
        logits = logits.permute(0, 2, 1, 3)  # [b, t, n, f]

        return logits


class ParticleFeatureProjection(torch.nn.Module):
    def __init__(self, in_features_dim, bg_features_dim, hidden_dim, output_dim, context_dim, add_embedding=True,
                 base_dim=32, activation='gelu', max_particles=None,
                 input_is_z=True, particle_positional_embed=True, init_std=0.02,
                 particle_score=False, ctx_cond_mode='adaln', norm_layer=True,
                 mask_inputs=True, use_z_orig=False, obj_on_film=False, mask_obj_on=False):
        super().__init__()
        assert ctx_cond_mode in ['add', 'cat', 'token', 'film', 'adaln']
        self.in_features_dim = in_features_dim
        self.bg_features_dim = bg_features_dim
        self.hidden_dim = hidden_dim
        self.context_dim = context_dim
        self.particle_score = particle_score
        self.add_embedding = add_embedding
        self.output_dim = output_dim
        self.base_dim = base_dim
        self.max_particles = max_particles
        self.input_is_z = input_is_z  # z or [mu, logvar]
        self.init_std = init_std

        self.mask_inputs = mask_inputs
        self.mask_obj_on = mask_obj_on
        self.use_z_orig = use_z_orig
        self.obj_on_film = obj_on_film
        self.ctx_cond_mode = ctx_cond_mode
        activation_f = nn.GELU if activation == 'gelu' else nn.ReLU
        # self.particle_dim = 2 + 2 + 1 + 1 + in_features_dim
        if self.obj_on_film:
            self.n_entities = 4
        else:
            self.n_entities = 5  # [pos, scale, obj_on, depth, features]
        if self.particle_score:
            self.n_entities += 1
        if self.use_z_orig:
            self.n_entities += 1
        if context_dim > 0 and self.ctx_cond_mode == 'cat':
            p_output_dim = 2 * output_dim
        else:
            p_output_dim = output_dim
        self.particle_dim = base_dim * self.n_entities
        # [z, z_scale, z_obj_on, z_depth, z_features]

        input_mult = 1 if self.input_is_z else 2
        self.xy_projection = nn.Sequential(nn.Linear(2 * input_mult, hidden_dim),
                                           RMSNorm(hidden_dim) if norm_layer else nn.Identity(),
                                           activation_f(),
                                           nn.Linear(hidden_dim, base_dim))
        self.scale_projection = nn.Sequential(nn.Linear(2 * input_mult, hidden_dim),
                                              RMSNorm(hidden_dim) if norm_layer else nn.Identity(),
                                              activation_f(),
                                              nn.Linear(hidden_dim, base_dim))
        if self.obj_on_film:
            self.obj_on_projection = nn.Sequential(nn.Linear(1 * input_mult, hidden_dim),
                                                   # RMSNorm(hidden_dim) if norm_layer else nn.Identity(),
                                                   activation_f(),
                                                   nn.Linear(hidden_dim, 2 * hidden_dim))
            nn.init.constant_(self.obj_on_projection[-1].weight, 0.0)
            nn.init.constant_(self.obj_on_projection[-1].bias[:hidden_dim], 1.0)
            nn.init.constant_(self.obj_on_projection[-1].bias[hidden_dim:], 0.0)
        else:
            self.obj_on_projection = nn.Sequential(nn.Linear(1 * input_mult, hidden_dim),
                                                   RMSNorm(hidden_dim) if norm_layer else nn.Identity(),
                                                   activation_f(),
                                                   nn.Linear(hidden_dim, base_dim))
        self.depth_projection = nn.Sequential(nn.Linear(1 * input_mult, hidden_dim),
                                              RMSNorm(hidden_dim) if norm_layer else nn.Identity(),
                                              activation_f(),
                                              nn.Linear(hidden_dim, base_dim))
        self.features_projection = nn.Sequential(nn.Linear(in_features_dim * input_mult, hidden_dim),
                                                 RMSNorm(hidden_dim) if norm_layer else nn.Identity(),
                                                 activation_f(),
                                                 nn.Linear(hidden_dim, base_dim))

        if self.mask_inputs:
            self.xy_mask = nn.Parameter(2 * torch.ones(2 * input_mult))
            self.scale_mask = nn.Parameter(0.1 * torch.ones(2 * input_mult))
            self.depth_mask = nn.Parameter(init_std * torch.randn(1 * input_mult))
            self.features_mask = nn.Parameter(init_std * torch.randn(self.in_features_dim * input_mult))
            if self.mask_obj_on:
                self.obj_on_mask = nn.Parameter(torch.zeros(1))
        if self.particle_score:
            self.score_projection = nn.Sequential(nn.Linear(1 * input_mult, hidden_dim),
                                                  RMSNorm(hidden_dim) if norm_layer else nn.Identity(),
                                                  activation_f(),
                                                  nn.Linear(hidden_dim, base_dim))
        else:
            self.score_projection = nn.Identity()
        if self.use_z_orig:
            self.origin_projection = nn.Sequential(nn.Linear(4, hidden_dim),
                                                   RMSNorm(hidden_dim) if norm_layer else nn.Identity(),
                                                   activation_f(),
                                                   nn.Linear(hidden_dim, base_dim))
            if self.mask_inputs:
                self.orig_mask = nn.Parameter(2 * torch.ones(4))
        else:
            self.origin_projection = nn.Identity()
        if self.obj_on_film:
            self.particle_projection_0 = nn.Sequential(nn.Linear(self.particle_dim, hidden_dim),
                                                       RMSNorm(hidden_dim))
            self.particle_projection = nn.Sequential(activation_f(),
                                                     nn.Linear(hidden_dim, output_dim))
        else:
            self.particle_projection = nn.Sequential(nn.Linear(self.particle_dim, hidden_dim),
                                                     RMSNorm(hidden_dim) if norm_layer else nn.Identity(),
                                                     activation_f(),
                                                     nn.Linear(hidden_dim, output_dim))
        if bg_features_dim > 0:
            bg_output_dim = output_dim
            self.bg_features_projection = nn.Sequential(nn.Linear(bg_features_dim, hidden_dim),
                                                        RMSNorm(hidden_dim) if norm_layer else nn.Identity(),
                                                        activation_f(),
                                                        nn.Linear(hidden_dim, bg_output_dim))
        else:
            self.bg_features_projection = nn.Identity()

        if self.ctx_cond_mode == 'cat' and self.context_dim > 0:
            self.p_final_projection = nn.Sequential(nn.Linear(p_output_dim, 4 * hidden_dim),
                                                    # RMSNorm(2 * hidden_dim),
                                                    activation_f(),
                                                    nn.Linear(4 * hidden_dim, output_dim))
            if bg_features_dim > 0:
                self.bg_final_projection = nn.Sequential(nn.Linear(p_output_dim, 4 * hidden_dim),
                                                         # RMSNorm(2 * hidden_dim),
                                                         activation_f(),
                                                         nn.Linear(4 * hidden_dim, output_dim))
            else:
                self.bg_final_projection = nn.Identity()
        else:
            self.p_final_projection = self.bg_final_projection = nn.Identity()
        if context_dim > 0 and self.ctx_cond_mode in ['token', 'cat']:
            self.context_projection = nn.Sequential(nn.Linear(context_dim, hidden_dim),
                                                    RMSNorm(hidden_dim) if norm_layer else nn.Identity(),
                                                    activation_f(),
                                                    nn.Linear(hidden_dim, output_dim))
            if self.ctx_cond_mode == 'cat' and bg_features_dim > 0:
                self.bg_context_projection = nn.Sequential(nn.Linear(context_dim, hidden_dim),
                                                           RMSNorm(hidden_dim) if norm_layer else nn.Identity(),
                                                           activation_f(),
                                                           nn.Linear(hidden_dim, output_dim))
            else:
                self.bg_context_projection = nn.Identity()
        else:
            self.context_projection = self.bg_context_projection = nn.Identity()
        if self.add_embedding:
            if particle_positional_embed:
                n_particles = 1 if self.max_particles is None else self.max_particles
            else:
                n_particles = 1  # means that all particles get the same "type" embedding
            self.particle_embedding = nn.Parameter(self.init_std * torch.randn(1, n_particles, output_dim))
            if bg_features_dim > 0:
                self.bg_embedding = nn.Parameter(self.init_std * torch.randn(1, output_dim))
            else:
                self.bg_embedding = None
            if context_dim > 0 and self.ctx_cond_mode == 'token':
                self.ctx_embedding = nn.Parameter(self.init_std * torch.randn(1, output_dim))
            else:
                self.ctx_embedding = None
        else:
            self.particle_embedding = None
            self.bg_embedding = None
            self.ctx_embedding = None

        if self.ctx_cond_mode in ['add', 'film'] and self.context_dim > 0:
            ctx_out_dim = 2 * output_dim if self.ctx_cond_mode == 'film' else output_dim
            self.ctx_to_action = nn.Sequential(nn.Linear(context_dim, hidden_dim),
                                               # RMSNorm(hidden_dim) if norm_layer else nn.Identity(),
                                               activation_f(),
                                               nn.Linear(hidden_dim, ctx_out_dim))
            self.ctx_to_action_bg = nn.Sequential(nn.Linear(context_dim, hidden_dim),
                                                  # RMSNorm(hidden_dim) if norm_layer else nn.Identity(),
                                                  activation_f(),
                                                  nn.Linear(hidden_dim, ctx_out_dim))
            if self.ctx_cond_mode == 'film':
                self.ctx_action_norm = RMSNorm(hidden_dim)
                self.ctx_action_mlp = nn.Sequential(activation_f(),
                                                    nn.Linear(hidden_dim, hidden_dim))
                self.ctx_action_bg_norm = RMSNorm(hidden_dim)
                self.ctx_action_bg_mlp = nn.Sequential(activation_f(),
                                                       nn.Linear(hidden_dim, hidden_dim))
            else:
                self.ctx_action_norm = None
                self.ctx_action_bg_norm = None
                self.ctx_action_mlp = None
                self.ctx_action_bg_mlp = None
        else:
            self.ctx_to_action = None
            self.ctx_to_action_bg = None
            self.ctx_action_norm = None
            self.ctx_action_bg_norm = None
            self.ctx_action_mlp = None
            self.ctx_action_bg_mlp = None

        self.init_weights()

    def init_weights(self):
        # pass
        if self.ctx_to_action is not None:
            nn.init.constant_(self.ctx_to_action[-1].weight, 0.0)
            if self.ctx_cond_mode == 'film':
                nn.init.constant_(self.ctx_to_action[-1].bias, 0.0)
            else:
                nn.init.constant_(self.ctx_to_action[-1].bias, 0.0)
        if self.ctx_to_action_bg is not None:
            nn.init.constant_(self.ctx_to_action_bg[-1].weight, 0.0)
            if self.ctx_cond_mode == 'film':
                nn.init.constant_(self.ctx_to_action_bg[-1].bias, 0.0)
            else:
                nn.init.constant_(self.ctx_to_action_bg[-1].bias, 0.0)

    def forward(self, z, z_scale, z_obj_on, z_depth, z_features, z_bg_features=None, z_context=None,
                z_score=None, z_orig=None):
        # z, z_scale, z_velocity: [bs, n_particles, 2]
        # z_depth, z_obj_on: [bs, n_particles, 1]
        # z_features: [bs, n_particles, in_features_dim]
        # z_bg_features: [bs, bg_features_dim]
        # z_context: [bs, context_dim]

        n_particles = z_features.shape[-2]

        # add origin and offset
        if self.use_z_orig and z_orig is not None:
            z_offset = z - z_orig
            z_orig_tot = torch.cat([z_orig, z_offset], dim=-1)
        else:
            z_orig_tot = z_orig
        # apply masks
        if self.mask_inputs:
            z_gate = torch.where(z_obj_on > 0.2, 1.0, 0.0)
            z = z_gate * z + (1 - z_gate) * self.xy_mask
            z_scale = z_gate * z_scale + (1 - z_gate) * self.scale_mask
            z_depth = z_gate * z_depth + (1 - z_gate) * self.depth_mask
            z_features = z_gate * z_features + (1 - z_gate) * self.features_mask
            if self.use_z_orig and z_orig is not None:
                z_orig_mask = self.orig_mask
                z_orig_tot = z_gate * z_orig_tot + (1 - z_gate) * z_orig_mask
            if self.mask_obj_on:
                z_obj_on = z_gate * z_obj_on + (1 - z_gate) * self.obj_on_mask

        z_proj = self.xy_projection(z)
        z_scale_proj = self.scale_projection(z_scale)
        if len(z_obj_on.shape) == 2:
            z_obj_on = z_obj_on.unsqueeze(-1)
        z_obj_on_proj = self.obj_on_projection(z_obj_on)
        z_depth_proj = self.depth_projection(z_depth)
        z_features_proj = self.features_projection(z_features)

        if self.obj_on_film:
            z_all = torch.cat([z_proj, z_scale_proj, z_depth_proj, z_features_proj], dim=-1)
        else:
            z_all = torch.cat([z_proj, z_scale_proj, z_obj_on_proj, z_depth_proj, z_features_proj], dim=-1)
        if self.particle_score and z_score is not None:
            # apply masks
            z_score_proj = self.score_projection(z_score)
            z_all = torch.cat([z_all, z_score_proj], dim=-1)
        if self.use_z_orig and z_orig is not None:
            z_orig_proj = self.origin_projection(z_orig_tot)
            z_all = torch.cat([z_all, z_orig_proj], dim=-1)
        # z_all: [bs, n_particles, 2 + 2 + 1 + 1 + in_features_dim]
        if self.obj_on_film:
            oscale, oshift = z_obj_on_proj.chunk(2, dim=-1)
            z_all_proj = self.particle_projection(oscale * self.particle_projection_0(z_all) + oshift)
        else:
            z_all_proj = self.particle_projection(z_all)
        # [bs, n_particles, output_dim]  or [bs, n_particle, hidden_dim]
        if z_bg_features is not None:
            z_bg_features_proj = self.bg_features_projection(z_bg_features)  # [bs, output_dim] or [bs, hidden_dim]
        else:
            z_bg_features_proj = None

        if z_context is not None and self.ctx_cond_mode in ['cat', 'token']:
            if self.ctx_cond_mode == 'token':
                z_context_proj = self.context_projection(z_context)  # [bs, output_dim] or [bs, hidden_dim]
            elif self.ctx_cond_mode == 'cat':
                if len(z_context.shape) != len(z_all_proj.shape):
                    z_context_fg = z_context.unsqueeze(1)
                    z_context_proj = self.context_projection(z_context_fg)  # [bs, 1, output_dim] or [bs, 1, hidden_dim]
                    z_all_proj = torch.cat([z_all_proj, z_context_proj.repeat(1, n_particles, 1)], dim=-1)
                else:
                    z_context_fg = z_context[:, :-1]
                    z_context_proj = self.context_projection(z_context_fg)
                    # [bs, n_part, output_dim] or [bs, n_part hidden_dim]
                    z_all_proj = torch.cat([z_all_proj, z_context_proj], dim=-1)
                if z_bg_features is not None:
                    if len(z_context.shape) != len(z_all_proj.shape):
                        z_context_bg = z_context
                    else:
                        z_context_bg = z_context[:, -1]
                    z_context_proj_bg = self.bg_context_projection(z_context_bg)
                    z_bg_features_proj = torch.cat([z_bg_features_proj, z_context_proj_bg], dim=-1)
        else:
            z_context_proj = None
        z_all_proj = self.p_final_projection(z_all_proj)
        if z_bg_features is not None:
            z_bg_features_proj = self.bg_final_projection(z_bg_features_proj)
        if self.ctx_cond_mode in ['add', 'film'] and self.context_dim > 0:
            if len(z_context.shape) != len(z_all_proj.shape):
                z_context_fg = z_context.unsqueeze(1)
            else:
                z_context_fg = z_context[:, :-1]
            ctx_act = self.ctx_to_action(z_context_fg)
            if self.ctx_cond_mode == 'film':
                ctx_scale, ctx_shift = ctx_act.chunk(2, dim=-1)
                z_all_proj = (ctx_scale + 1.0) * self.ctx_action_norm(z_all_proj) + ctx_shift
                z_all_proj = self.ctx_action_mlp(z_all_proj)
            else:
                z_all_proj = z_all_proj + ctx_act
            if z_bg_features is not None:
                if len(z_context.shape) != len(z_all_proj.shape):
                    z_context_bg = z_context
                else:
                    z_context_bg = z_context[:, -1]
                ctx_act_bg = self.ctx_to_action_bg(z_context_bg)
                if self.ctx_cond_mode == 'film':
                    ctx_bg_scale, ctx_bg_shift = ctx_act_bg.chunk(2, dim=-1)
                    z_bg_features_proj = (ctx_bg_scale + 1.0) * self.ctx_action_bg_norm(
                        z_bg_features_proj) + ctx_bg_shift
                    z_bg_features_proj = self.ctx_action_bg_mlp(z_bg_features_proj)
                else:
                    z_bg_features_proj = z_bg_features_proj + ctx_act_bg
        if self.add_embedding:
            z_all_proj = z_all_proj + self.particle_embedding
            if z_bg_features is not None:
                z_bg_features_proj = z_bg_features_proj + self.bg_embedding
            # if z_context is not None and self.separate_ctx_token:
            if z_context is not None and self.ctx_cond_mode == 'token':
                z_context_proj = z_context_proj + self.ctx_embedding
        if z_bg_features is not None and z_context is not None and self.ctx_cond_mode == 'token':
            z_context_proj_p = z_context_proj.unsqueeze(1) if len(z_context_proj.shape) == 2 else z_context_proj
            z_processed = torch.cat([z_all_proj, z_bg_features_proj.unsqueeze(1), z_context_proj_p], dim=1)
            # [bs, n_particles + 2, output_dim]
        elif z_bg_features is not None:
            z_processed = torch.cat([z_all_proj, z_bg_features_proj.unsqueeze(1)], dim=1)
        # elif z_context is not None and self.separate_ctx_token:
        elif z_context is not None and self.ctx_cond_mode == 'token':
            z_processed = torch.cat([z_all_proj, z_context_proj.unsqueeze(1)], dim=1)
        else:
            z_processed = z_all_proj
        return z_processed


class ParticleAttributesProjection(torch.nn.Module):
    def __init__(self, n_particles, in_features_dim, hidden_dim, output_dim, bg_features_dim, add_ctx_token=False,
                 base_dim=32, depth=True, obj_on=True, base_var=False, bg=True, activation='gelu', init_std=0.2,
                 cat_particle_num=False, norm_layer=True, particle_score=False,
                 mask_inputs=True, use_z_orig=False, obj_on_film=False, mask_obj_on=False):
        super().__init__()
        self.n_particles = n_particles
        self.in_features_dim = in_features_dim
        self.bg_features_dim = bg_features_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.with_depth = depth
        self.with_obj_on = obj_on
        self.with_var = base_var
        self.with_bg = bg
        self.with_score = particle_score
        self.add_ctx_token = add_ctx_token
        self.cat_particle_num = cat_particle_num
        self.norm_layer = norm_layer
        self.mask_inputs = mask_inputs
        self.mask_obj_on = mask_obj_on
        self.use_z_orig = use_z_orig
        self.obj_on_film = obj_on_film
        activation_f = nn.GELU if activation == 'gelu' else nn.ReLU
        # self.particle_dim = 2 + 2 + 2 + in_features_dim
        # [z, z_scale, z_features]
        self.base_dim = base_dim
        self.n_entities = 3
        if self.with_depth:
            self.n_entities += 1
        if self.with_obj_on and not self.obj_on_film:
            self.n_entities += 1
        if self.with_var:
            self.n_entities += 1
        if self.with_score:
            self.n_entities += 1
        if self.use_z_orig:
            self.n_entities += 1
        if self.cat_particle_num:
            self.n_entities += 1
            self.particle_num_embed = nn.Parameter(0.02 * torch.randn(1, self.n_particles, self.base_dim))
        self.particle_dim = self.base_dim * self.n_entities

        self.xy_projection = nn.Sequential(nn.Linear(2, hidden_dim),
                                           RMSNorm(hidden_dim) if norm_layer else nn.Identity(),
                                           activation_f(),
                                           nn.Linear(hidden_dim, self.base_dim))
        self.scale_projection = nn.Sequential(nn.Linear(2, hidden_dim),
                                              RMSNorm(hidden_dim) if norm_layer else nn.Identity(),
                                              activation_f(),
                                              nn.Linear(hidden_dim, self.base_dim))
        if self.with_var:
            self.var_projection = nn.Sequential(nn.Linear(5, hidden_dim),
                                                RMSNorm(hidden_dim) if norm_layer else nn.Identity(),
                                                activation_f(),
                                                nn.Linear(hidden_dim, self.base_dim))
        if self.with_obj_on:
            if self.obj_on_film:
                self.obj_on_projection = nn.Sequential(nn.Linear(1, hidden_dim),
                                                       activation_f(),
                                                       nn.Linear(hidden_dim, 2 * hidden_dim))
                nn.init.constant_(self.obj_on_projection[-1].weight, 0.0)
                nn.init.constant_(self.obj_on_projection[-1].bias[:hidden_dim], 1.0)
                nn.init.constant_(self.obj_on_projection[-1].bias[hidden_dim:], 0.0)
            else:
                self.obj_on_projection = nn.Sequential(nn.Linear(1, hidden_dim),
                                                       # ParticleNorm2(self.n_particles),
                                                       RMSNorm(hidden_dim) if norm_layer else nn.Identity(),
                                                       activation_f(),
                                                       nn.Linear(hidden_dim, self.base_dim))

            if self.mask_inputs:
                self.xy_mask = nn.Parameter(2 * torch.ones(2))
                self.scale_mask = nn.Parameter(0.1 * torch.ones(2))
                self.features_mask = nn.Parameter(init_std * torch.randn(in_features_dim))
                if self.mask_obj_on:
                    self.obj_on_mask = nn.Parameter(torch.zeros(1))
        if self.with_depth:
            self.depth_projection = nn.Sequential(nn.Linear(1, hidden_dim),
                                                  RMSNorm(hidden_dim) if norm_layer else nn.Identity(),
                                                  activation_f(),
                                                  nn.Linear(hidden_dim, self.base_dim))
            if self.with_obj_on and self.mask_inputs:
                self.depth_mask = nn.Parameter(init_std * torch.randn(1))
        self.features_projection = nn.Sequential(nn.Linear(in_features_dim, hidden_dim),
                                                 RMSNorm(hidden_dim) if norm_layer else nn.Identity(),
                                                 activation_f(),
                                                 nn.Linear(hidden_dim, self.base_dim))
        if self.with_score:
            self.score_projection = nn.Sequential(nn.Linear(1, hidden_dim),
                                                  RMSNorm(hidden_dim) if norm_layer else nn.Identity(),
                                                  activation_f(),
                                                  nn.Linear(hidden_dim, self.base_dim))
        if self.with_bg:
            self.bg_projection = nn.Sequential(nn.Linear(bg_features_dim, hidden_dim),
                                               RMSNorm(hidden_dim) if norm_layer else nn.Identity(),
                                               activation_f(),
                                               nn.Linear(hidden_dim, output_dim))
        if self.use_z_orig:
            self.origin_projection = nn.Sequential(nn.Linear(4, hidden_dim),
                                                   RMSNorm(hidden_dim) if norm_layer else nn.Identity(),
                                                   activation_f(),
                                                   nn.Linear(hidden_dim, base_dim))
            if self.mask_inputs:
                self.orig_mask = nn.Parameter(2 * torch.ones(4))
        if self.obj_on_film:
            self.particle_projection_0 = nn.Sequential(nn.Linear(self.particle_dim, hidden_dim),
                                                       RMSNorm(hidden_dim))
            self.particle_projection = nn.Sequential(activation_f(),
                                                     nn.Linear(hidden_dim, output_dim))
        else:
            self.particle_projection = nn.Sequential(nn.Linear(self.particle_dim, hidden_dim),
                                                     RMSNorm(hidden_dim) if norm_layer else nn.Identity(),
                                                     activation_f(),
                                                     nn.Linear(hidden_dim, output_dim))
        if self.add_ctx_token:
            self.ctx_embedding = nn.Parameter(init_std * torch.randn(1, 1, 1, output_dim))

        self.init_weights()

    def init_weights(self):
        pass

    def forward(self, z, z_scale, z_obj_on, z_depth, z_features, z_bg_features=None, z_base_var=None, z_score=None,
                z_orig=None):
        # def forward(self, z, z_scale, z_obj_on, z_features, z_base_var, z_bg_features):
        # z, z_scale, z_velocity: [bs, n_particles, 2]
        # z_depth, z_obj_on: [bs, n_particles, 1]
        # z_features: [bs, n_particles, in_features_dim]
        # z_bg_features: [bs, bg_features_dim]
        # z_context: [bs, context_dim]
        # bs, n_particles, feat_dim = z_features.shape

        # add origin and offset
        if self.use_z_orig and z_orig is not None:
            z_offset = z - z_orig
            z_orig_tot = torch.cat([z_orig, z_offset], dim=-1)
        else:
            z_orig_tot = z_orig

        if self.with_obj_on and self.mask_inputs:
            z_gate = torch.where(z_obj_on > 0.2, 1.0, 0.0)
            z = z_gate * z + (1 - z_gate) * self.xy_mask
            z_scale = z_gate * z_scale + (1 - z_gate) * self.scale_mask
            z_features = z_gate * z_features + (1 - z_gate) * self.features_mask
            if self.use_z_orig and z_orig is not None:
                z_orig_mask = self.orig_mask
                z_orig_tot = z_gate * z_orig_tot + (1 - z_gate) * z_orig_mask
            if self.mask_obj_on:
                z_obj_on = z_gate * z_obj_on + (1 - z_gate) * self.obj_on_mask

        z_proj = self.xy_projection(z)
        z_scale_proj = self.scale_projection(z_scale)
        z_features_proj = self.features_projection(z_features)
        z_all = torch.cat([z_proj, z_scale_proj, z_features_proj], dim=-1)
        if self.with_obj_on:
            z_obj_on_proj = self.obj_on_projection(z_obj_on)
            if not self.obj_on_film:
                z_all = torch.cat([z_all, z_obj_on_proj], dim=-1)
        if self.with_depth:
            if self.with_obj_on and self.mask_inputs:
                z_depth = z_gate * z_depth + (1 - z_gate) * self.depth_mask
            z_depth_proj = self.depth_projection(z_depth)
            z_all = torch.cat([z_all, z_depth_proj], dim=-1)
        if self.with_var and z_base_var is not None:
            z_var_proj = self.var_projection(z_base_var)
            z_all = torch.cat([z_all, z_var_proj], dim=-1)
        if self.with_score and z_score is not None:
            z_score_proj = self.score_projection(z_score)
            z_all = torch.cat([z_all, z_score_proj], dim=-1)
        if self.use_z_orig and z_orig is not None:
            z_orig_proj = self.origin_projection(z_orig_tot)
            z_all = torch.cat([z_all, z_orig_proj], dim=-1)
        if self.cat_particle_num:
            if len(z.shape) == 4:
                p_embed = self.particle_num_embed.unsqueeze(1).repeat(z.shape[0], z.shape[1], 1, 1)
            else:
                p_embed = self.particle_num_embed.repeat(z.shape[0], 1, 1)
            z_all = torch.cat([z_all, p_embed], dim=-1)

        # z_all: [bs, n_particles, 2 + 2 + in_features_dim]
        if self.with_obj_on and self.obj_on_film:
            oscale, oshift = z_obj_on_proj.chunk(2, dim=-1)
            z_all_proj = self.particle_projection(oscale * self.particle_projection_0(z_all) + oshift)
        else:
            z_all_proj = self.particle_projection(
                z_all)  # [bs, n_particles, output_dim]  or [bs, n_particle, hidden_dim]
        if self.with_bg:
            z_bg_features_proj = self.bg_projection(z_bg_features)  # [bs, output_dim]
            z_all_proj = torch.cat([z_all_proj, z_bg_features_proj.unsqueeze(-2)], dim=-2)
        # [bs, T,  n_particles + 1, output_dim]
        if self.add_ctx_token:
            z_all_proj = torch.cat([z_all_proj,
                                    self.ctx_embedding.repeat(z.shape[0], z.shape[1], 1, 1)], dim=-2)
            # [bs, T,  n_particles + 2, output_dim]
        return z_all_proj


class ParticlePool(nn.Module):
    def __init__(self, pool_mode='mean', pool_dim=-2, keepdim=True):
        super().__init__()
        assert pool_mode in ['mean', 'max', 'sum', 'none', 'last', 'token', 'mlp']
        self.pool_mode = pool_mode
        self.pool_dim = pool_dim
        self.keepdim = keepdim

    def forward(self, x):
        if self.pool_mode == 'mean':
            return x.mean(self.pool_dim, keepdim=self.keepdim)
        elif self.pool_mode == 'sum':
            return x.sum(self.pool_dim, keepdim=self.keepdim)
        elif self.pool_mode == 'max':
            return x.max(self.pool_dim, keepdim=self.keepdim)[0]
        else:
            return x


class ParticleAttributeDecoder(nn.Module):
    def __init__(self, n_particles, input_dim, hidden_dim, features_dim, bg_features_dim=None,
                 depth=False, obj_on=False, features=False, bg_features=False,
                 offset_logvar=False,
                 activation='gelu', dropout=0.0, shared_logvar=False,
                 output_ctx_logvar=True, features_dist='gauss'):
        super().__init__()
        # decoder to map back from PTE's inner dim to the particle's original dimension
        self.n_particles = n_particles
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.features_dist = features_dist
        self.features_dim = features_dim
        self.bg_features_dim = bg_features_dim
        self.offset_logvar = offset_logvar
        self.with_depth = depth
        self.with_obj_on = obj_on
        self.with_features = features
        self.with_bg_features = bg_features
        self.use_fg_backbone = (self.with_obj_on or self.with_depth or self.with_features)
        self.shared_logvar = shared_logvar
        self.output_ctx_logvar = output_ctx_logvar
        activation_f = nn.GELU if activation == 'gelu' else nn.ReLU
        if self.use_fg_backbone:
            self.fg_backbone = nn.Identity()
            if self.with_obj_on:
                self.obj_on_head = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                                 activation_f(),
                                                 nn.Linear(hidden_dim, 1)
                                                 )  # log_a, log_b
            if self.with_depth:
                self.depth_head = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                                activation_f(),
                                                nn.Linear(hidden_dim, 2)
                                                )  # mu_z, logvar_z
            if self.with_features:
                output_feat_dim = 2 * features_dim if (features_dist != 'categorical') else features_dim
                self.features_head = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                                   activation_f(),
                                                   nn.Linear(hidden_dim, output_feat_dim)
                                                   )  # mu_features, logvar_features
        if self.with_bg_features:
            output_bg_feat_dim = 2 * bg_features_dim if (features_dist != 'categorical') else bg_features_dim
            self.bg_backbone = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                             activation_f(),
                                             )
            self.bg_features_head = nn.Linear(hidden_dim, output_bg_feat_dim)  # mu_features, logvar_features

        self.init_weights()

    def init_weights(self):
        if self.with_features and self.features_dist != 'categorical':
            nn.init.constant_(self.features_head[-1].weight[:self.features_dim], 0.0)
            nn.init.constant_(self.features_head[-1].bias[:self.features_dim], 0.0)
            nn.init.constant_(self.features_head[-1].weight[self.features_dim:], 0.0)
            nn.init.constant_(self.features_head[-1].bias[self.features_dim:], math.log(0.001 ** 2))
        if self.with_bg_features and self.features_dist != 'categorical':
            nn.init.constant_(self.bg_features_head.weight[:self.bg_features_dim], 0.0)
            nn.init.constant_(self.bg_features_head.bias[:self.bg_features_dim], 0.0)
            nn.init.constant_(self.bg_features_head.weight[self.bg_features_dim:], 0.0)
            nn.init.constant_(self.bg_features_head.bias[self.bg_features_dim:], math.log(0.001 ** 2))

    def forward(self, x):
        # x: [bs, n_particles, input_dim]
        # bs, n_particles, in_dim = x.shape
        bs, ts, n_particles = x.shape[0], x.shape[1], x.shape[2]
        # the following assumes fg_particles + bg_particle + context particle
        fg_particles = n_particles - 2 if self.with_bg_features else n_particles - 1
        if self.use_fg_backbone:
            x_fg = x[:, :, :fg_particles]
            fg_features = self.fg_backbone(x_fg)
            if self.with_depth:
                depth = self.depth_head(fg_features)
                mu_depth, logvar_depth = torch.chunk(depth, 2, dim=-1)
            else:
                mu_depth = logvar_depth = None
            if self.with_obj_on:
                obj_on = self.obj_on_head(fg_features)
                lobj_on_a = lobj_on_b = obj_on
            else:
                lobj_on_a = lobj_on_b = None
            if self.with_features:
                features = self.features_head(fg_features)
                if self.features_dist != 'categorical':
                    mu_features, logvar_features = torch.chunk(features, 2, dim=-1)
                else:
                    mu_features = logvar_features = features
            else:
                mu_features = logvar_features = None
        else:
            mu_depth = logvar_depth = None
            lobj_on_a = lobj_on_b = None
            mu_features = logvar_features = None

        if self.with_bg_features:
            x_bg = x[:, :, fg_particles]
            bg_features = self.bg_backbone(x_bg)
            bg_features = self.bg_features_head(bg_features)
            if self.features_dist != 'categorical':
                mu_bg_features, logvar_bg_features = torch.chunk(bg_features, 2, dim=-1)
            else:
                mu_bg_features = logvar_bg_features = bg_features
        else:
            mu_bg_features = logvar_bg_features = None

        decoder_out = {'mu_depth': mu_depth, 'logvar_depth': logvar_depth,
                       'lobj_on_a': lobj_on_a, 'lobj_on_b': lobj_on_b,
                       'mu_features': mu_features, 'logvar_features': logvar_features,
                       'mu_bg_features': mu_bg_features, 'logvar_bg_features': logvar_bg_features}

        return decoder_out


class ParticleFeatureDecoderDyn(nn.Module):
    def __init__(self, input_dim, features_dim, bg_features_dim, hidden_dim, kp_activation='tanh', max_delta=1.0,
                 context_dim=7, activation='gelu', shared_logvar=False, logvar_min=-10.0,
                 logvar_max=10.0, ctx_as_token=False, dec_ctx=False,
                 norm_type='rms', dropout=0.0, particle_score=False,
                 features_dist='gauss', n_fg_categories=8, n_fg_classes=4, n_bg_categories=4, n_bg_classes=4,
                 scale_init=None):
        super().__init__()
        # decoder to map back from PTE's inner dim to the particle's original dimension
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.features_dist = features_dist
        self.n_fg_categories = n_fg_categories
        self.n_fg_classes = n_fg_classes
        self.n_bg_categories = n_bg_categories
        self.n_bg_classes = n_bg_classes
        self.features_dim = features_dim
        self.bg_features_dim = bg_features_dim
        self.ctx_dim = context_dim
        self.particle_score = particle_score
        self.kp_activation = kp_activation
        self.max_delta = max_delta
        self.shared_logvar = shared_logvar
        self.logvar_min = logvar_min
        self.logvar_max = logvar_max
        self.ctx_as_token = ctx_as_token
        self.dec_ctx = dec_ctx
        self.n_attributes = 5  # [xy, scale, depth, transp, features]
        self.scale_init = scale_init
        activation_f = nn.GELU if activation == 'gelu' else nn.ReLU

        xy_output_dim = 2 if self.shared_logvar else 4
        scale_output_dim = 2 if self.shared_logvar else 4
        depth_output_dim = 1 if self.shared_logvar else 2
        output_features_logvar = (not self.shared_logvar and self.features_dist != 'categorical')
        feature_output_dim = 2 * features_dim if output_features_logvar else features_dim
        if bg_features_dim > 0:
            bg_output_dim = 2 * bg_features_dim if output_features_logvar else bg_features_dim
        else:
            bg_output_dim = 0
        if self.shared_logvar:
            self.offset_xy_logvar = nn.Parameter(torch.zeros(1, 1, 1))
            self.scale_xy_logvar = nn.Parameter(torch.zeros(1, 1, 1))
            self.depth_logvar = nn.Parameter(torch.zeros(1, 1, 1))
            self.features_logvar = nn.Parameter(torch.zeros(1, 1, 1))
            if bg_features_dim > 0:
                self.bg_features_logvar = nn.Parameter(torch.zeros(1, 1))

        self.offset_xy_head = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                            activation_f(),
                                            nn.Linear(hidden_dim, xy_output_dim)
                                            )  # mu_ox, logvar_ox, mu_oy, logvar_oy

        self.scale_xy_head = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                           activation_f(),
                                           nn.Linear(hidden_dim, scale_output_dim)
                                           )  # mu_sx, logvar_sx, mu_sy, logvar_sy
        self.obj_on_head = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                         activation_f(),
                                         nn.Linear(hidden_dim, 1)
                                         )  # log_a, log_b
        self.depth_head = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                        activation_f(),
                                        nn.Linear(hidden_dim, depth_output_dim)
                                        )  # mu_z, logvar_z
        self.features_head = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                           activation_f(),
                                           nn.Linear(hidden_dim, feature_output_dim)
                                           )  # mu_features, logvar_features
        if self.particle_score:
            self.score_head = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                            activation_f(),
                                            nn.Linear(hidden_dim, 2)
                                            )  # mu_score, logvar_score
            self.n_attributes += 1
        else:
            self.score_head = nn.Identity()

        if self.bg_features_dim > 0:
            self.bg_backbone = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                             # RMSNorm(hidden_dim),
                                             activation_f(),
                                             )
            self.bg_features_head = self.get_mlp_head(bg_output_dim)  # mu_features, logvar_features
        if self.ctx_dim > 0 and self.dec_ctx and self.ctx_as_token:
            self.backbone = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                          activation_f(),
                                          )
            self.context_head = self.get_mlp_head(2 * self.ctx_dim)  # mu_features, logvar_features
        else:
            self.backbone = nn.Identity()
            self.context_head = nn.Identity()
        self.init_weights()

    def init_weights(self):
        # pass

        torch.nn.init.constant_(self.offset_xy_head[-1].weight[:2], 0.0)
        torch.nn.init.constant_(self.offset_xy_head[-1].bias[:2], 0.0)

        if not self.shared_logvar:
            torch.nn.init.constant_(self.offset_xy_head[-1].weight[2:], 0.0)
            torch.nn.init.constant_(self.offset_xy_head[-1].bias[2:], math.log(0.1 ** 2))

        if self.scale_init is not None:
            scale_init = 0.75 * self.scale_init + 1e-5
            torch.nn.init.constant_(self.scale_xy_head[-1].weight[:2], 0.0)
            torch.nn.init.constant_(self.scale_xy_head[-1].bias[:2], np.log(scale_init / (1 - scale_init)))

            torch.nn.init.constant_(self.scale_xy_head[-1].weight[2:], 0.0)
            torch.nn.init.constant_(self.scale_xy_head[-1].bias[2:], math.log(0.2 ** 2))  # 0.1

        # torch.nn.init.constant_(self.obj_on_head[-1].weight, 0.0)
        # torch.nn.init.constant_(self.obj_on_head[-1].bias, -0.3)

        if self.particle_score:
            torch.nn.init.constant_(self.score_head[-1].weight, 0.0)
            torch.nn.init.constant_(self.score_head[-1].bias, 0.0)

    def get_mlp_head(self, output_dim):
        return nn.Linear(self.hidden_dim, output_dim)

    def forward(self, x):
        # x: [bs, n_particles + 2, input_dim]
        bs, n_particles, in_dim = x.shape

        if self.ctx_dim > 0 and self.ctx_as_token:
            fg_features, bg_features, ctx_features = x.split([n_particles - 2, 1, 1], dim=1)
            bg_features = self.bg_backbone(bg_features)
            ctx_features = self.backbone(ctx_features)
        elif self.bg_features_dim > 0:
            fg_features, bg_features = x.split([n_particles - 1, 1], dim=1)
            bg_features = self.bg_backbone(bg_features)
            ctx_features = None
        else:
            fg_features = x
            bg_features = None
            ctx_features = None
        xy = scale = obj_on = depth = features = scores = fg_features

        n, f = xy.shape[1], xy.shape[-1]
        offset_features = xy
        offset_xy = self.offset_xy_head(offset_features)
        if self.shared_logvar:
            mu_offset = offset_xy
            logvar_offset = self.offset_xy_logvar.repeat(mu_offset.shape[0], mu_offset.shape[1], mu_offset.shape[-1])
        else:
            offset_xy = offset_xy.view(bs, -1, offset_xy.shape[-1])
            mu_offset, logvar_offset = torch.chunk(offset_xy, chunks=2, dim=-1)

        if self.kp_activation == "tanh":
            mu_offset = torch.tanh(mu_offset)
        elif self.kp_activation == "sigmoid":
            mu_offset = torch.sigmoid(mu_offset)

        # apply max delta
        mu_offset = self.max_delta * mu_offset

        scale_features = scale
        scale_xy = self.scale_xy_head(scale_features)
        if self.shared_logvar:
            mu_scale = scale_xy
            logvar_scale = self.scale_xy_logvar.repeat(mu_scale.shape[0], mu_scale.shape[1], mu_scale.shape[-1])
        else:
            scale_xy = scale_xy.view(bs, -1, scale_xy.shape[-1])
            mu_scale, logvar_scale = torch.chunk(scale_xy, chunks=2, dim=-1)

        obj_on_1 = self.obj_on_head(obj_on)
        obj_on_1 = obj_on_1.view(bs, -1, 1)
        lobj_on_a = lobj_on_b = obj_on_1
        depth = self.depth_head(depth)
        if self.shared_logvar:
            mu_depth = depth
            logvar_depth = self.depth_logvar.repeat(mu_depth.shape[0], mu_depth.shape[1], 1)
        else:
            depth = depth.view(bs, -1, 2)
            mu_depth, logvar_depth = torch.chunk(depth, 2, dim=-1)

        feat_features = features
        features = self.features_head(feat_features)
        if self.features_dist == 'categorical':
            mu_features = logvar_features = features
        else:
            if self.shared_logvar:
                mu_features = features
                logvar_features = self.features_logvar.repeat(mu_features.shape[0], mu_features.shape[1],
                                                              mu_features.shape[-1])
            else:
                features = features.view(bs, -1, 2 * self.features_dim)
                mu_features, logvar_features = torch.chunk(features, 2, dim=-1)

        if self.particle_score:
            score = self.score_head(scores)
            score = score.view(bs, -1, 2)
            mu_score, logvar_score = torch.chunk(score, chunks=2, dim=-1)
        else:
            mu_score = logvar_score = None

        if self.bg_features_dim > 0:
            f_bg = bg_features.shape[-1]
            bg_features = self.bg_features_head(bg_features.squeeze(1))
            if self.features_dist == 'categorical':
                mu_bg_features = logvar_bg_features = bg_features
            else:
                if self.shared_logvar:
                    mu_bg_features = bg_features
                    logvar_bg_features = self.bg_features_logvar.repeat(mu_bg_features.shape[0],
                                                                        mu_bg_features.shape[-1])
                else:
                    mu_bg_features, logvar_bg_features = torch.chunk(bg_features, 2, dim=-1)
        else:
            mu_bg_features = logvar_bg_features = None

        if self.ctx_dim > 0 and self.ctx_as_token and self.dec_ctx:
            context_features = self.context_head(ctx_features.squeeze(1))
            mu_context, logvar_context = torch.chunk(context_features, 2, dim=-1)
        else:
            mu_context = logvar_context = None

        decoder_out = {'mu_offset': mu_offset,
                       'logvar_offset': logvar_offset, 'lobj_on_a': lobj_on_a, 'lobj_on_b': lobj_on_b,
                       'obj_on': obj_on, 'mu_depth': mu_depth, 'logvar_depth': logvar_depth,
                       'mu_scale': mu_scale, 'logvar_scale': logvar_scale, 'mu_features': mu_features,
                       'logvar_features': logvar_features, 'mu_bg_features': mu_bg_features,
                       'logvar_bg_features': logvar_bg_features, 'mu_context': mu_context,
                       'logvar_context': logvar_context, 'mu_score': mu_score, 'logvar_score': logvar_score}
        return decoder_out


"""
CNN-based modules
"""


class ObjectDecoderCNN(nn.Module):
    def __init__(self, patch_size, num_chans=4, bottleneck_size=128, pad_mode='replicate', embed_position=False,
                 use_resblock=False, context_dim=0, normalize_rgb=False, res_from_fc=8, activation='gelu',
                 ch_mult=(1, 2, 3), base_ch=32, final_cnn_ch=32, num_res_blocks=2, cnn_mid_blocks=False,
                 mlp_hidden_dim=256,
                 # initialization
                 init_zero_bias=True,  # zero bias for conv and linear layers
                 init_conv_layers=True,  # initialize conv layers with normal dist
                 init_conv_fg_std=0.02,  # std for conv fg normal dist
                 ):
        super().__init__()

        # initialization
        self.init_zero_bias = init_zero_bias  # zero bias for conv and linear layers
        self.init_conv_layers = init_conv_layers  # initialize conv layers with normal dist
        self.init_conv_fg_std = init_conv_fg_std  # std for conv fg normal dist

        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        self.patch_size = patch_size
        self.num_chans = num_chans
        self.embed_position = embed_position
        self.use_resblock = use_resblock
        self.features_dim = bottleneck_size
        self.activation = activation
        self.cnn_mid_blocks = cnn_mid_blocks
        self.mlp_hidden_dim = mlp_hidden_dim

        self.context_dim = context_dim
        self.normalize_rgb = normalize_rgb

        self.in_ch = final_cnn_ch
        self.fc_res = res_from_fc
        fc_out_dim = self.in_ch * (self.fc_res ** 2)
        fc_in_dim = bottleneck_size if not self.embed_position else 2 * bottleneck_size

        feature_map_size = self.fc_res ** 2

        if self.features_dim % feature_map_size == 0:
            self.ch_feature_dim = math.ceil(max(self.features_dim / (res_from_fc ** 2), 1))
            output_z_cnn = (self.ch_feature_dim, self.fc_res, self.fc_res)
            flattened_z_cnn = np.prod(output_z_cnn)

            self.projection_mode = 'fcn'
            self.from_latent_lin = nn.Identity()
            # self.from_latent = nn.Conv2d(in_channels=self.ch_feature_dim, out_channels=self.in_ch, kernel_size=1)
            self.from_latent = nn.Identity()
        else:
            self.ch_feature_dim = final_cnn_ch
            output_z_cnn = (self.ch_feature_dim, self.fc_res, self.fc_res)
            flattened_z_cnn = np.prod(output_z_cnn)

            self.projection_mode = 'fc'
            self.from_latent_lin = self.get_mlp(self.features_dim, flattened_z_cnn)
            self.from_latent = nn.Identity()

        self.info = (f'ObjectDecoderCNN: requested latent size: {self.features_dim}, '
                     f'cnn input (h*w): {feature_map_size}, (latent_size / h*w)={self.features_dim / feature_map_size} ->'
                     f' latent projection mode: {self.projection_mode},'
                     f' project {self.features_dim} -> {output_z_cnn} ({flattened_z_cnn})')

        self.num_upsample = max(int(np.log2(patch_size[0])) - int(np.log2(self.fc_res)), 0)
        # print(f'ObjDecCNN: fc to cnn num upsample: {num_upsample}')
        attn_res = [max(self.patch_size[0] // 16, 1)]
        ch_mult = ch_mult[:self.num_upsample + 1]
        z_channels = self.ch_feature_dim
        self.cnn = Decoder(ch=base_ch, out_ch=self.num_chans, ch_mult=ch_mult, num_res_blocks=num_res_blocks,
                           attn_resolutions=attn_res, dropout=0.0, resamp_with_conv=True,
                           resolution=self.patch_size[0], z_channels=z_channels, give_pre_end=False,
                           padding_mode=pad_mode, residual=self.use_resblock, upsample_method='nearest',
                           mid_blocks=cnn_mid_blocks)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if self.init_conv_layers:
                    nn.init.normal_(m.weight, 0.0, self.init_conv_fg_std)
                if self.init_zero_bias and m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def get_mlp(self, in_dim, out_dim, linear=False):
        if linear:
            return nn.Linear(in_dim, out_dim)
        else:
            activation_f = nn.GELU if self.activation == 'gelu' else nn.ReLU
            hidden_dim = self.mlp_hidden_dim
            mlp = nn.Sequential(nn.Linear(in_dim, hidden_dim),
                                activation_f(),
                                nn.Linear(hidden_dim, out_dim))

            return mlp

    def forward(self, x, context=None):
        # x: [bs, n_kp, feat]
        # context: [bs, feat]
        bs, n_kp = x.shape[0], x.shape[1]

        x = self.from_latent_lin(x)
        x = x.view(-1, self.ch_feature_dim, self.fc_res, self.fc_res)
        z = self.from_latent(x)
        conv_in = z
        out = self.cnn(conv_in).view(-1, self.num_chans, *self.patch_size)
        out_a, out_rgb = torch.split(out, [1, out.shape[1] - 1], dim=1)

        rgb_func = torch.tanh if self.normalize_rgb else torch.sigmoid
        out = torch.cat([torch.sigmoid(out_a), rgb_func(out_rgb)], dim=1)

        return out


class ObjectDecoderCNNFILM(nn.Module):
    def __init__(self, patch_size, num_chans=4, bottleneck_size=128, pad_mode='replicate', embed_position=False,
                 use_resblock=False, context_dim=0, normalize_rgb=False, res_from_fc=8, activation='gelu',
                 ch_mult=(1, 2, 3), base_ch=32, final_cnn_ch=32, num_res_blocks=2, cnn_mid_blocks=False,
                 mlp_hidden_dim=256,
                 # initialization
                 init_zero_bias=True,  # zero bias for conv and linear layers
                 init_conv_layers=True,  # initialize conv layers with normal dist
                 init_conv_fg_std=0.02,  # std for conv fg normal dist
                 ):
        super().__init__()

        # initialization
        self.init_zero_bias = init_zero_bias  # zero bias for conv and linear layers
        self.init_conv_layers = init_conv_layers  # initialize conv layers with normal dist
        self.init_conv_fg_std = init_conv_fg_std  # std for conv fg normal dist

        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        self.patch_size = patch_size
        self.num_chans = num_chans
        self.embed_position = embed_position
        self.use_resblock = use_resblock
        self.features_dim = bottleneck_size
        self.activation = activation
        self.cnn_mid_blocks = cnn_mid_blocks
        self.mlp_hidden_dim = mlp_hidden_dim

        self.context_dim = context_dim
        self.normalize_rgb = normalize_rgb

        self.in_ch = final_cnn_ch
        self.fc_res = res_from_fc
        fc_out_dim = self.in_ch * (self.fc_res ** 2)
        fc_in_dim = bottleneck_size if not self.embed_position else 2 * bottleneck_size

        feature_map_size = self.fc_res ** 2

        if self.features_dim % feature_map_size == 0:
            self.ch_feature_dim = math.ceil(max(self.features_dim / (res_from_fc ** 2), 1))
            output_z_cnn = (self.ch_feature_dim, self.fc_res, self.fc_res)
            flattened_z_cnn = np.prod(output_z_cnn)

            self.projection_mode = 'fcn'
            self.from_latent_lin = nn.Identity()
            self.from_latent = nn.Identity()
        else:
            self.ch_feature_dim = final_cnn_ch
            output_z_cnn = (self.ch_feature_dim, self.fc_res, self.fc_res)
            flattened_z_cnn = np.prod(output_z_cnn)

            self.projection_mode = 'fc'
            self.from_latent_lin = self.get_mlp(self.features_dim, flattened_z_cnn)
            self.from_latent = nn.Identity()

        self.info = (f'ObjectDecoderCNN: requested latent size: {self.features_dim}, '
                     f'cnn input (h*w): {feature_map_size}, (latent_size / h*w)={self.features_dim / feature_map_size} ->'
                     f' latent projection mode: {self.projection_mode},'
                     f' project {self.features_dim} -> {output_z_cnn} ({flattened_z_cnn})')

        n_film_layers = 1
        self.film_layer = self.get_mlp(in_dim=self.context_dim, out_dim=n_film_layers * 2 * self.in_ch)

        self.num_upsample = max(int(np.log2(patch_size[0])) - int(np.log2(self.fc_res)), 0)
        # print(f'ObjDecCNN: fc to cnn num upsample: {num_upsample}')
        attn_res = [max(self.patch_size[0] // 16, 1)]
        ch_mult = ch_mult[:self.num_upsample + 1]
        z_channels = self.ch_feature_dim
        self.cnn = Decoder(ch=base_ch, out_ch=self.num_chans, ch_mult=ch_mult, num_res_blocks=num_res_blocks,
                           attn_resolutions=attn_res, dropout=0.0, resamp_with_conv=True,
                           resolution=self.patch_size[0], z_channels=z_channels, give_pre_end=False,
                           padding_mode=pad_mode, residual=self.use_resblock, upsample_method='nearest',
                           mid_blocks=cnn_mid_blocks)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if self.init_conv_layers:
                    nn.init.normal_(m.weight, 0.0, self.init_conv_fg_std)
                if self.init_zero_bias and m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def get_mlp(self, in_dim, out_dim, linear=False):
        if linear:
            return nn.Linear(in_dim, out_dim)
        else:
            activation_f = nn.GELU if self.activation == 'gelu' else nn.ReLU
            hidden_dim = self.mlp_hidden_dim
            mlp = nn.Sequential(nn.Linear(in_dim, hidden_dim),
                                activation_f(),
                                nn.Linear(hidden_dim, out_dim))

            return mlp

    def forward(self, x, context=None):
        # x: [bs, n_kp, feat]
        # context: [bs, feat]
        bs, n_kp = x.shape[0], x.shape[1]

        ctx_param = self.film_layer(context)  # [bs, n_layers * 2 * hidden_size]
        ctx_param = ctx_param.view(ctx_param.shape[0], 1, -1, 2 * self.in_ch)
        ctx_param = ctx_param[:, :, :, :, None, None].repeat(1, n_kp, 1, 1, self.fc_res, self.fc_res)
        ctx_param = ctx_param.view(-1, *ctx_param.shape[2:])
        ctx_gammas, ctx_betas = ctx_param.chunk(2, dim=2)  # [bs, n_layers, hidden_size]

        x = self.from_latent_lin(x)
        x = x.view(-1, self.ch_feature_dim, self.fc_res, self.fc_res)
        z = self.from_latent(x)
        conv_in = ctx_gammas[:, 0] * z + ctx_betas[:, 0]
        out = self.cnn(conv_in).view(-1, self.num_chans, *self.patch_size)
        out_a, out_rgb = torch.split(out, [1, out.shape[1] - 1], dim=1)

        rgb_func = torch.tanh if self.normalize_rgb else torch.sigmoid
        out = torch.cat([torch.sigmoid(out_a), rgb_func(out_rgb)], dim=1)

        return out


class ObjectDecoderCNNConcat(nn.Module):
    def __init__(self, patch_size, num_chans=4, bottleneck_size=128, pad_mode='replicate', embed_position=False,
                 use_resblock=False, context_dim=7, normalize_rgb=False, res_from_fc=8,
                 ch_mult=(1, 2, 3), base_ch=32, final_cnn_ch=32, num_res_blocks=2, cnn_mid_blocks=False,
                 mlp_hidden_dim=256):
        super().__init__()
        assert context_dim > 0, f'ObjectDecoderCNNFILM: context dim - {context_dim} must be > 0'
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        self.patch_size = patch_size
        self.num_chans = num_chans
        self.embed_position = embed_position
        self.use_resblock = use_resblock
        self.features_dim = bottleneck_size
        self.context_dim = context_dim
        self.normalize_rgb = normalize_rgb
        self.cnn_mid_blocks = cnn_mid_blocks
        self.mlp_hidden_dim = mlp_hidden_dim

        self.in_ch = final_cnn_ch
        self.fc_res = res_from_fc
        fc_out_dim = self.in_ch * (self.fc_res ** 2)

        feature_map_size = self.fc_res ** 2

        if self.features_dim % feature_map_size == 0:
            self.ch_feature_dim = math.ceil(max(self.features_dim / (res_from_fc ** 2), 1))
            output_z_cnn = (self.ch_feature_dim, self.fc_res, self.fc_res)
            flattened_z_cnn = np.prod(output_z_cnn)

            self.projection_mode = 'fcn'
            self.from_latent_lin = nn.Identity()
        else:
            self.ch_feature_dim = final_cnn_ch
            output_z_cnn = (self.ch_feature_dim, self.fc_res, self.fc_res)
            flattened_z_cnn = np.prod(output_z_cnn)

        self.info = (f'ObjectDecoderCNNConcat: requested latent size: {self.features_dim}, '
                     f'cnn input (h*w): {feature_map_size}, (latent_size / h*w)={self.features_dim / feature_map_size} ->'
                     f' latent projection mode: {self.projection_mode},'
                     f' project {self.features_dim} -> {output_z_cnn} ({flattened_z_cnn})')

        self.from_latent = nn.Conv2d(in_channels=self.ch_feature_dim, out_channels=self.in_ch, kernel_size=1)
        self.context_projection = nn.Sequential(nn.Linear(self.context_dim, mlp_hidden_dim),
                                                nn.GELU(),
                                                nn.Linear(mlp_hidden_dim, self.in_ch))

        self.num_upsample = max(int(np.log2(patch_size[0])) - int(np.log2(self.fc_res)), 0)
        # print(f'ObjDecCNN: fc to cnn num upsample: {num_upsample}')
        attn_res = [max(self.patch_size[0] // 16, 1)]
        ch_mult = ch_mult[:self.num_upsample + 1]
        self.cnn = Decoder(ch=base_ch, out_ch=self.num_chans, ch_mult=ch_mult, num_res_blocks=num_res_blocks,
                           attn_resolutions=attn_res, dropout=0.0, resamp_with_conv=True,
                           resolution=self.patch_size[0], z_channels=2 * final_cnn_ch, give_pre_end=False,
                           padding_mode=pad_mode, mid_blocks=cnn_mid_blocks)

    def forward(self, x, context):
        # x: [bs, n_kp, feat]
        # context: [bs, feat]
        bs, n_kp = x.shape[0], x.shape[1]

        ctx_param = self.context_projection(context)  # [bs, hidden_size]
        ctx_param = ctx_param.view(ctx_param.shape[0], 1, self.in_ch)
        ctx_param = ctx_param[:, :, :, None, None].repeat(1, n_kp, 1, self.fc_res, self.fc_res)
        ctx_param = ctx_param.view(-1, *ctx_param.shape[2:])

        x = self.from_latent_lin(x)
        x = x.view(-1, self.ch_feature_dim, self.fc_res, self.fc_res)
        z = self.from_latent(x)
        conv_in = torch.cat([z, ctx_param], dim=1)
        out = self.cnn(conv_in).view(-1, self.num_chans, *self.patch_size)
        out_a, out_rgb = torch.split(out, [1, out.shape[1] - 1], dim=1)
        rgb_func = torch.tanh if self.normalize_rgb else torch.sigmoid
        out = torch.cat([torch.sigmoid(out_a), rgb_func(out_rgb)], dim=1)
        return out


class FCToCNN(nn.Module):
    def __init__(self, target_hw=16, n_ch=8, pad_mode='replicate', features_dim=2, use_resblock=False, context_dim=0,
                 res_from_fc=8, activation='gelu', mlp_hidden_dim=256):
        super(FCToCNN, self).__init__()
        # features_dim : 2 [logvar] + additional features
        self.features_dim = features_dim  # logvar, features
        self.n_ch = n_ch
        self.fmap_size = res_from_fc
        self.use_resblock = use_resblock
        self.context_dim = context_dim
        self.activation = activation
        self.mlp_hidden_fim = mlp_hidden_dim
        fc_out_dim = self.n_ch * (self.fmap_size ** 2)

        feature_map_size = self.fmap_size ** 2

        if self.features_dim % feature_map_size == 0:
            self.ch_features_dim = math.ceil(max(self.features_dim / feature_map_size, 1))
            output_z_cnn = (self.ch_features_dim, self.fmap_size, self.fmap_size)
            flattened_z_cnn = np.prod(output_z_cnn)

            self.projection_mode = 'fcn'
            self.from_latent_lin = nn.Identity()
            self.from_latent = nn.Identity()
        else:
            self.ch_features_dim = self.n_ch
            output_z_cnn = (self.ch_features_dim, self.fmap_size, self.fmap_size)
            flattened_z_cnn = np.prod(output_z_cnn)

            self.projection_mode = 'fc'
            self.from_latent_lin = self.get_mlp(self.features_dim, flattened_z_cnn)
            self.from_latent = nn.Identity()

        self.info = (f'FCToCNN: requested latent size: {self.features_dim}, '
                     f'cnn input (h*w): {feature_map_size}, (latent_size / h*w)={self.features_dim / feature_map_size} ->'
                     f' latent projection mode: {self.projection_mode},'
                     f' project {self.features_dim} -> {output_z_cnn} ({flattened_z_cnn})')

    def get_mlp(self, in_dim, out_dim, linear=False):
        if linear:
            return nn.Linear(in_dim, out_dim)
        else:
            activation_f = nn.GELU if self.activation == 'gelu' else nn.ReLU
            hidden_dim = self.mlp_hidden_fim
            mlp = nn.Sequential(nn.Linear(in_dim, hidden_dim),
                                activation_f(),
                                nn.Linear(hidden_dim, out_dim))

            return mlp

    def forward(self, features, context=None):
        # features [batch_size, features_dim]
        batch_size = features.shape[0]
        features = self.from_latent_lin(features)
        x = features.view(-1, self.ch_features_dim, self.fmap_size, self.fmap_size)
        z = self.from_latent(x)
        cnn_out = z
        return cnn_out


class FCToCNNFILM(nn.Module):
    def __init__(self, target_hw=16, n_ch=8, pad_mode='replicate', features_dim=2, use_resblock=False, context_dim=0,
                 res_from_fc=8, activation='gelu', mlp_hidden_dim=256):
        super(FCToCNNFILM, self).__init__()
        # features_dim : 2 [logvar] + additional features
        self.features_dim = features_dim  # logvar, features
        self.n_ch = n_ch
        self.fmap_size = res_from_fc
        self.use_resblock = use_resblock
        self.context_dim = context_dim
        self.activation = activation
        self.mlp_hidden_fim = mlp_hidden_dim
        fc_out_dim = self.n_ch * (self.fmap_size ** 2)

        feature_map_size = self.fmap_size ** 2

        if self.features_dim % feature_map_size == 0:
            self.ch_features_dim = math.ceil(max(self.features_dim / feature_map_size, 1))
            output_z_cnn = (self.ch_features_dim, self.fmap_size, self.fmap_size)
            flattened_z_cnn = np.prod(output_z_cnn)

            self.projection_mode = 'fcn'
            self.from_latent_lin = nn.Identity()
            self.from_latent = nn.Identity()
        else:
            self.ch_features_dim = self.n_ch
            output_z_cnn = (self.ch_features_dim, self.fmap_size, self.fmap_size)
            flattened_z_cnn = np.prod(output_z_cnn)

            self.projection_mode = 'fc'
            self.from_latent_lin = self.get_mlp(self.features_dim, flattened_z_cnn)
            self.from_latent = nn.Identity()

        self.info = (f'FCToCNNFILM: requested latent size: {self.features_dim}, '
                     f'cnn input (h*w): {feature_map_size}, (latent_size / h*w)={self.features_dim / feature_map_size} ->'
                     f' latent projection mode: {self.projection_mode},'
                     f' project {self.features_dim} -> {output_z_cnn} ({flattened_z_cnn})')

        n_film_layers = 1
        self.film_layer = self.get_mlp(in_dim=self.context_dim, out_dim=n_film_layers * 2 * self.ch_features_dim)

    def get_mlp(self, in_dim, out_dim, linear=False):
        if linear:
            return nn.Linear(in_dim, out_dim)
        else:
            activation_f = nn.GELU if self.activation == 'gelu' else nn.ReLU
            hidden_dim = self.mlp_hidden_fim
            mlp = nn.Sequential(nn.Linear(in_dim, hidden_dim),
                                activation_f(),
                                nn.Linear(hidden_dim, out_dim))

            return mlp

    def forward(self, features, context=None):
        # features [batch_size, features_dim]
        batch_size = features.shape[0]

        ctx_param = self.film_layer(context)  # [bs, n_layers * 2 * hidden_size]
        ctx_param = ctx_param.view(ctx_param.shape[0], -1, 2 * self.ch_features_dim)
        ctx_param = ctx_param[:, :, :, None, None].repeat(1, 1, 1, self.fmap_size, self.fmap_size)
        ctx_gammas, ctx_betas = ctx_param.chunk(2, dim=2)  # [bs, n_layers, hidden_size]

        features = self.from_latent_lin(features)
        x = features.view(-1, self.ch_features_dim, self.fmap_size, self.fmap_size)
        z = self.from_latent(x)
        cnn_out = ctx_gammas[:, 0] * z + ctx_betas[:, 0]
        return cnn_out


class FCToCNNConcat(nn.Module):
    def __init__(self, target_hw=16, n_ch=8, pad_mode='replicate', features_dim=2, use_resblock=False, context_dim=0,
                 res_from_fc=8, mlp_hidden_dim=256):
        super(FCToCNNConcat, self).__init__()
        # features_dim : 2 [logvar] + additional features
        assert context_dim > 0, f'FCToCNNFILM: context dim - {context_dim} must be > 0'
        self.features_dim = features_dim  # logvar, features
        self.n_ch = n_ch
        self.fmap_size = res_from_fc
        self.use_resblock = use_resblock
        self.context_dim = context_dim
        self.mlp_hidden_dim = mlp_hidden_dim
        fc_out_dim = self.n_ch * (self.fmap_size ** 2)

        feature_map_size = self.fmap_size ** 2

        if self.features_dim % feature_map_size == 0:
            self.ch_features_dim = math.ceil(max(self.features_dim / feature_map_size, 1))
            output_z_cnn = (self.ch_features_dim, self.fmap_size, self.fmap_size)
            flattened_z_cnn = np.prod(output_z_cnn)

            self.projection_mode = 'fcn'
            self.from_latent_lin = nn.Identity()
        else:
            self.ch_features_dim = self.n_ch
            output_z_cnn = (self.ch_features_dim, self.fmap_size, self.fmap_size)
            flattened_z_cnn = np.prod(output_z_cnn)

            self.projection_mode = 'fc'
            self.from_latent_lin = self.get_mlp(self.features_dim, flattened_z_cnn)

        self.info = (f'FCToCNNConcat: requested latent size: {self.features_dim}, '
                     f'cnn input (h*w): {feature_map_size}, (latent_size / h*w)={self.features_dim / feature_map_size} ->'
                     f' latent projection mode: {self.projection_mode},'
                     f' project {self.features_dim} -> {output_z_cnn} ({flattened_z_cnn})')
        # new
        self.from_latent = nn.Conv2d(in_channels=self.ch_features_dim, out_channels=self.n_ch, kernel_size=1)
        self.context_projection = nn.Sequential(nn.Linear(self.context_dim, mlp_hidden_dim),
                                                nn.ReLU(True),
                                                nn.Linear(mlp_hidden_dim, self.n_ch))

    def forward(self, features, context=None):
        # features [batch_size, features_dim]
        batch_size = features.shape[0]

        # new
        ctx_param = self.context_projection(context)  # [bs, hidden_size]
        ctx_param = ctx_param.view(ctx_param.shape[0], self.n_ch)
        ctx_param = ctx_param[:, :, None, None].repeat(1, 1, self.fmap_size, self.fmap_size)
        features = self.from_latent_lin(features)
        x = features.view(-1, self.ch_features_dim, self.fmap_size, self.fmap_size)
        z = self.from_latent(x)
        cnn_out = torch.cat([z, ctx_param], dim=1)
        return cnn_out


class BgDecoder(nn.Module):
    def __init__(self, cdim=3, image_size=64,
                 pad_mode='replicate', dropout=0.0, learned_bg_feature_dim=16,
                 use_resblock=False, context_dim=0, film=False, timestep_horizon=1,
                 bg_res_from_fc=8, bg_ch_mult=(1, 2, 3), bg_base_ch=32, bg_final_cnn_ch=32, num_res_blocks=2,
                 decode_with_ctx=False, normalize_rgb=False, cnn_mid_blocks=False, mlp_hidden_dim=256,
                 init_zero_bias=True,  # zero bias for conv and linear layers
                 init_conv_layers=True,  # initialize conv layers with normal dist
                 init_conv_bg_std=0.005,  # std for conv fg normal dist
                 ):
        super(BgDecoder, self).__init__()
        """
        DLP Background Module -- decodes bg from latent
        cdim: channels of the input image (3...)
        cnn_channels: channels for the posterior CNN (takes in the whole image)
        n_kp_enc: number of posterior kp to be learned (this is the actual number of kp that will be learnt)
        pad_mode: padding for the CNNs, 'zeros' or  'replicate' (default)
        learned_feature_dim: the latent visual features dimensions extracted from glimpses.
        learned_bg_feature_dim: the latent visual features dimensions extracted from bg.
        anchor_s: defines the glimpse size as a ratio of image_size (e.g., 0.25 for image_size=128 -> glimpse_size=32)
        film: use FiLM conditioning method for context conditioning
        """
        self.image_size = image_size
        self.feature_map_size = image_size
        self.dropout = dropout
        self.features_dim = int(image_size // (2 ** (len(bg_ch_mult) - 1)))
        self.learned_bg_feature_dim = learned_bg_feature_dim
        assert self.learned_bg_feature_dim > 0, "learned_bg_feature_dim must be greater than 0"
        self.context_dim = context_dim
        self.cdim = cdim
        self.use_resblock = use_resblock
        self.film = film
        self.decode_with_ctx = decode_with_ctx
        self.normalize_rgb = normalize_rgb
        self.timestep_horizon = (timestep_horizon + 1) if timestep_horizon > 1 else 1
        self.cnn_mid_blocks = cnn_mid_blocks
        self.mlp_hidden_dim = mlp_hidden_dim

        # initialization
        self.init_zero_bias = init_zero_bias  # zero bias for conv and linear layers
        self.init_conv_layers = init_conv_layers  # initialize conv layers with normal dist
        self.init_conv_bg_std = init_conv_bg_std  # std for conv fg normal dist

        # bg decoder
        decoder_n_kp = bg_final_cnn_ch
        if self.context_dim > 0 and self.decode_with_ctx:
            latent_proj_net = FCToCNNFILM if self.film else FCToCNNConcat
        else:
            latent_proj_net = FCToCNN
        self.latent_to_feat_map = latent_proj_net(target_hw=self.features_dim, n_ch=decoder_n_kp,
                                                  features_dim=self.learned_bg_feature_dim, pad_mode=pad_mode,
                                                  use_resblock=self.use_resblock, context_dim=self.context_dim,
                                                  res_from_fc=bg_res_from_fc, mlp_hidden_dim=mlp_hidden_dim)
        self.num_bg_upsample = max(int(np.log2(image_size)) - int(np.log2(self.features_dim)), 0)
        attn_res = [max(self.image_size // 16, 1)]
        bg_ch_mult = bg_ch_mult[:self.num_bg_upsample + 1]
        if self.decode_with_ctx and self.context_dim > 0 and not film:
            in_z_ch = 2 * self.latent_to_feat_map.ch_features_dim
        else:
            in_z_ch = self.latent_to_feat_map.ch_features_dim
        self.cnn = Decoder(ch=decoder_n_kp, out_ch=self.cdim, ch_mult=bg_ch_mult, num_res_blocks=num_res_blocks,
                           attn_resolutions=attn_res, dropout=0.0, resamp_with_conv=True,
                           resolution=self.image_size, z_channels=in_z_ch, give_pre_end=False,
                           padding_mode=pad_mode, residual=self.use_resblock, upsample_method='nearest',
                           mid_blocks=cnn_mid_blocks)
        self.info = self.latent_to_feat_map.info
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if self.init_conv_layers:
                    nn.init.normal_(m.weight, 0, self.init_conv_bg_std)
                if self.init_zero_bias and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                # use pytorch's default
                pass

    def decode_all(self, z_bg_features, z_ctx=None, warmup=False):
        feature_maps = self.latent_to_feat_map(z_bg_features, z_ctx)
        bg_rec = self.cnn(feature_maps)

        out_func = torch.tanh if self.normalize_rgb else torch.sigmoid
        decoder_out = out_func(bg_rec)

        return decoder_out

    def forward(self, z_bg_features, z_ctx=None, warmup=False):
        return self.decode_all(z_bg_features, z_ctx, warmup)


class BgEncoder(nn.Module):
    def __init__(self, cdim=3, image_size=64, pad_mode='replicate', dropout=0.0,
                 learned_feature_dim=16, use_resblock=False, activation='gelu', cnn_mid_blocks=False,
                 ch_mult=(1, 2, 3), base_ch=32, final_cnn_ch=32, num_res_blocks=2, interaction_features=False,
                 mlp_hidden_dim=256, timestep_horizon=1, add_particle_temp_embed=False, init_std=0.2,
                 features_dist='gauss', n_bg_categories=4, n_bg_classes=4,
                 # initialization
                 init_zero_bias=True,  # zero bias for conv and linear layers
                 init_conv_layers=True,  # initialize conv layers with normal dist
                 init_conv_bg_std=0.005,  # std for conv bg normal dist (<fg -> prioritize fg in learning)
                 ):
        super(BgEncoder, self).__init__()
        """
        DLP Background Module -- encode a latent for the (masked) background, z_bg
        Basically, just a convolutional-based encoder used in standard VAEs
        cdim: channels of the input image (3...)
        enc_channels: channels for the posterior CNN (takes in the whole image)
        pad_mode: padding for the CNNs, 'zeros' or  'replicate' (default)
        learned_feature_dim: the latent visual features dimensions extracted from glimpses.
        """
        self.image_size = image_size
        self.dropout = dropout
        self.output_feat_map_size = int(image_size // (2 ** (len(ch_mult) - 1)))
        self.features_dim = learned_feature_dim
        self.features_dist = features_dist
        self.n_bg_categories = n_bg_categories
        self.n_bg_classes = n_bg_classes
        assert learned_feature_dim > 0, "learned_feature_dim must be greater than 0"
        self.cdim = cdim
        self.n_kp_enc = final_cnn_ch
        self.interaction_features = interaction_features
        self.use_resblock = use_resblock
        self.activation = activation
        self.cnn_mid_blocks = cnn_mid_blocks
        self.mlp_hidden_dim = mlp_hidden_dim
        self.timestep_horizon = (timestep_horizon + 1) if timestep_horizon > 1 else 1
        self.add_particle_temp_embed = add_particle_temp_embed

        # initialization
        self.init_zero_bias = init_zero_bias  # zero bias for conv and linear layers
        self.init_conv_layers = init_conv_layers  # initialize conv layers with normal dist
        self.init_conv_bg_std = init_conv_bg_std  # std for conv bg normal dist

        attn_res = [max(self.image_size // 16, 1)]
        self.bg_cnn_enc = Encoder(ch=base_ch, ch_mult=ch_mult, num_res_blocks=num_res_blocks,
                                  attn_resolutions=attn_res, dropout=0.0, resamp_with_conv=True,
                                  in_channels=self.cdim,
                                  resolution=self.image_size, z_channels=final_cnn_ch, double_z=False,
                                  padding_mode=pad_mode, residual=self.use_resblock, in_conv_kernel_size=3,
                                  mid_blocks=cnn_mid_blocks)
        self.cnn_out_shape = self.get_cnn_shape()

        # new cnn
        feature_map_size = self.output_feat_map_size ** 2
        output_logvar = (not self.interaction_features and self.features_dist != 'categorical')
        self.output_logvar = output_logvar

        # new - FCN
        if self.features_dim % feature_map_size == 0:
            self.ch_learned_feature_dim = math.ceil(max(self.features_dim / feature_map_size, 1))
            out_ch = 2 * self.ch_learned_feature_dim if output_logvar else self.ch_learned_feature_dim
            self.to_latent = nn.Conv2d(in_channels=final_cnn_ch,
                                       out_channels=out_ch, kernel_size=1)
            output_z_cnn = (self.ch_learned_feature_dim, self.cnn_out_shape[-2], self.cnn_out_shape[-1])
            flattened_z_cnn = np.prod(output_z_cnn)
            if self.timestep_horizon > 1 and self.add_particle_temp_embed:
                self.temp_embed = nn.Parameter(
                    init_std * torch.randn(1, self.timestep_horizon, final_cnn_ch, self.cnn_out_shape[-1],
                                           self.cnn_out_shape[-1]))
            else:
                self.temp_embed = None

            self.projection_mode = 'fcn'
            self.to_mu = nn.Identity()
            self.to_logvar = nn.Identity()
        else:
            self.ch_learned_feature_dim = final_cnn_ch
            self.to_latent = nn.Identity()
            output_z_cnn = (self.ch_learned_feature_dim, self.cnn_out_shape[-2], self.cnn_out_shape[-1])
            flattened_z_cnn = np.prod(output_z_cnn)

            if self.timestep_horizon > 1 and self.add_particle_temp_embed:
                self.temp_embed = nn.Parameter(init_std * torch.randn(1, self.timestep_horizon, flattened_z_cnn))
            else:
                self.temp_embed = None

            self.projection_mode = 'fc'
            self.to_mu = self.get_mlp(flattened_z_cnn, self.features_dim)
            self.to_logvar = self.get_mlp(flattened_z_cnn,
                                          self.features_dim) if output_logvar else nn.Identity()

        self.info = (f'BgEncoder: requested latent size: {self.features_dim}, '
                     f'cnn output (h*w): {feature_map_size}, (latent_size / h*w)={self.features_dim / feature_map_size} ->'
                     f' latent projection mode: {self.projection_mode},'
                     f' project {output_z_cnn} ({flattened_z_cnn}) -> {self.features_dim}')

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # pass
                if self.init_conv_layers:
                    nn.init.normal_(m.weight, 0, self.init_conv_bg_std)
                if self.init_zero_bias and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                # use pytorch's default
                pass
        # pass

    def get_mlp(self, in_dim, out_dim, linear=False):
        if linear:
            return nn.Linear(in_dim, out_dim)
        else:
            activation_f = nn.GELU if self.activation == 'gelu' else nn.ReLU
            hidden_dim = self.mlp_hidden_dim
            mlp = nn.Sequential(nn.Linear(in_dim, hidden_dim),
                                activation_f(),
                                nn.Linear(hidden_dim, out_dim))

            return mlp

    def get_cnn_shape(self):
        dummy_input = torch.rand(1, self.cdim, self.image_size, self.image_size)
        out = self.bg_cnn_enc(dummy_input)
        if isinstance(out, tuple):
            out = out[1]
        return out.shape[1:]

    def encode_bg_features(self, x, masks=None, timesteps=None):
        # x: [bs, ch, image_size, image_size]
        # masks: [bs, 1, image_size, image_size]
        batch_size, _, features_dim, _ = x.shape
        # bg features
        if masks is not None:
            x_in = x * masks
        else:
            x_in = x
        enc_out = self.bg_cnn_enc(x_in)
        if isinstance(enc_out, tuple):
            cnn_features = enc_out[1]
        else:
            cnn_features = enc_out

        # new cnn
        if self.projection_mode == 'fcn' and self.temp_embed is not None:
            orig_shape = cnn_features.shape  # [batch_size * n_kp, ch, patch_size, patch_size]
            new_feat = cnn_features.view(-1, timesteps, *cnn_features.shape[1:])
            new_feat = new_feat + self.temp_embed[:, :timesteps]
            cnn_features = new_feat.view(orig_shape)
        features = self.to_latent(cnn_features)
        features = features.view(features.shape[0], -1)
        if self.projection_mode == 'fc' and self.temp_embed is not None:
            orig_shape = features.shape  # [batch_size * n_kp, ch, patch_size, patch_size]
            new_feat = features.view(-1, timesteps, *features.shape[1:])
            new_feat = new_feat + self.temp_embed[:, :timesteps]
            features = new_feat.view(orig_shape)
        if self.interaction_features:
            mu_bg = features
            logvar_bg = None
            mu_bg = self.to_mu(mu_bg)
        else:
            mu_bg = self.to_mu(features)
            logvar_bg = self.to_logvar(features)

        return mu_bg, logvar_bg

    def encode_all(self, x, masks=None, deterministic=False, timesteps=None):
        # encode background
        mu_bg, logvar_bg = self.encode_bg_features(x, masks, timesteps)
        if self.interaction_features:
            z_bg = mu_bg
        else:
            z_bg = reparameterize(mu_bg, logvar_bg) if not deterministic else mu_bg
        z_kp = torch.zeros(mu_bg.shape[0], 1, 2, device=x.device, dtype=torch.float)
        encode_dict = {'mu_bg': mu_bg, 'logvar_bg': logvar_bg, 'z_bg': z_bg, 'z_kp': z_kp}
        return encode_dict

    def forward(self, x, masks=None, deterministic=False, timesteps=None):
        encoder_out = self.encode_all(x, masks, deterministic, timesteps)
        mu_bg = encoder_out['mu_bg']
        logvar_bg = encoder_out['logvar_bg']
        z_bg = encoder_out['z_bg']
        z_kp = encoder_out['z_kp']
        output_dict = {'mu_bg': mu_bg, 'logvar_bg': logvar_bg, 'z_bg': z_bg, 'z_kp': z_kp}
        return output_dict


class ParticleAttributeEncoder(nn.Module):
    """
    Glimpse-encoder: encodes patches visual features in a variational fashion (mu, log-variance).
    Useful for object-based scenes.
    """

    def __init__(self, anchor_size, image_size, n_particles, cnn_channels=(16, 16, 32), margin=0, ch=3, max_offset=1.0,
                 kp_activation='tanh', use_resblock=False, hidden_dim=512, pad_mode='replicate', depth=False,
                 obj_on=True, scale=True, activation='gelu',
                 ch_mult=(1, 2, 3), base_ch=32, final_cnn_ch=32, num_res_blocks=2, cnn_mid_blocks=False,
                 timestep_horizon=1, add_particle_temp_embed=False, init_std=0.2,
                 obj_on_min=1e-4, obj_on_max=100.0,
                 init_zero_bias=True,  # zero bias for conv and linear layers
                 init_conv_layers=True,  # initialize conv layers with normal dist
                 init_conv_fg_std=0.02,  # std for conv fg normal dist
                 ):
        super().__init__()
        self.anchor_size = anchor_size
        self.channels = cnn_channels
        self.image_size = image_size
        self.n_particles = n_particles
        self.patch_size = np.round(anchor_size * (image_size - 1)).astype(int)
        self.margin = margin
        self.crop_size = self.patch_size + 2 * margin
        self.ch = ch
        self.use_resblock = use_resblock
        self.kp_activation = kp_activation
        self.max_offset = max_offset  # max offset of x-y, [-max_offset, +max_offset]
        self.hidden_dim = hidden_dim
        self.with_depth = depth
        self.with_obj_on = obj_on
        self.with_scale = scale
        self.cnn_mid_blocks = cnn_mid_blocks
        self.timestep_horizon = timestep_horizon
        self.add_particle_temp_embed = add_particle_temp_embed
        self.obj_on_min = obj_on_min
        self.obj_on_max = obj_on_max
        self.init_std = init_std
        # initialization
        self.init_zero_bias = init_zero_bias  # zero bias for conv and linear layers
        self.init_conv_layers = init_conv_layers  # initialize conv layers with normal dist
        self.init_conv_fg_std = init_conv_fg_std  # std for conv fg normal dist

        attn_res = [max(self.crop_size // 16, 1)]
        self.cnn = Encoder(ch=base_ch, ch_mult=ch_mult, num_res_blocks=num_res_blocks,
                           attn_resolutions=attn_res, dropout=0.0, resamp_with_conv=True, in_channels=self.ch,
                           resolution=self.crop_size, z_channels=final_cnn_ch, double_z=False, padding_mode=pad_mode,
                           residual=self.use_resblock, mid_blocks=cnn_mid_blocks)

        feature_map_size = (self.crop_size // 2 ** (len(ch_mult) - 1)) ** 2
        fc_in_dim = final_cnn_ch * feature_map_size
        if self.add_particle_temp_embed and self.timestep_horizon > 1:
            self.temp_embed = nn.Parameter(init_std * torch.randn(1, self.timestep_horizon, 1, fc_in_dim))
        else:
            self.temp_embed = None
        activation_f = nn.GELU if activation == 'gelu' else nn.ReLU

        self.backbone = nn.Identity()
        self.xy_head = nn.Sequential(nn.Linear(fc_in_dim, self.hidden_dim),
                                     activation_f(),
                                     nn.Linear(self.hidden_dim, 4))  # mu_x, logvar_s, mu_y, logvar_y
        scale_output = 4 if self.with_scale else 2
        self.scale_xy_head = nn.Sequential(nn.Linear(fc_in_dim, self.hidden_dim),
                                           activation_f(),
                                           nn.Linear(self.hidden_dim,
                                                     scale_output))  # mu_sx, logvar_sx, mu_sy, logvar_sy
        if self.with_obj_on:
            self.obj_on_head = nn.Sequential(nn.Linear(fc_in_dim, self.hidden_dim),
                                             activation_f(),
                                             nn.Linear(self.hidden_dim, 1, bias=False))  # [log_obj_on_a, log_obj_on_b]

        else:
            self.obj_on_head = None
        if self.with_depth:
            self.depth_head = nn.Sequential(nn.Linear(fc_in_dim, self.hidden_dim),
                                            activation_f(),
                                            nn.Linear(self.hidden_dim, 2))  # mu_depth, logvar_depth
        else:
            self.depth_head = None
        self.init_weights()

    def init_weights(self):
        # pass
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # pass
                if self.init_conv_layers:
                    nn.init.normal_(m.weight, 0.0, self.init_conv_fg_std)
                if self.init_zero_bias and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                if self.init_zero_bias and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        torch.nn.init.constant_(self.xy_head[-1].weight[:2], 0.0)
        torch.nn.init.constant_(self.xy_head[-1].bias[:2], 0.0)
        torch.nn.init.constant_(self.xy_head[-1].weight[2:], 0.0)
        torch.nn.init.constant_(self.xy_head[-1].bias[2:], math.log(0.01 ** 2))
        if self.with_scale:
            torch.nn.init.constant_(self.scale_xy_head[-1].bias[:2], 0.0)
            torch.nn.init.constant_(self.scale_xy_head[-1].bias[2:], math.log(0.1 ** 2))
            torch.nn.init.constant_(self.scale_xy_head[-1].weight, 0.0)

        if self.with_obj_on:
            torch.nn.init.constant_(self.obj_on_head[-1].weight, 0.0)
            if self.obj_on_head[-1].bias is not None:
                torch.nn.init.constant_(self.obj_on_head[-1].bias, 0.0)  # beta(a,)

    def forward(self, x, kp, z_scale=None, timesteps=None, deterministic=False):
        # x: [bs, ch, image_size, image_size]
        # kp: [bs, n_kp, 2] in [-1, 1]
        batch_size, _, _, img_size = x.shape
        _, n_kp, _ = kp.shape
        x_repeated = x.unsqueeze(1).repeat(1, n_kp, 1, 1, 1)  # [batch_size, n_kp, ch, image_size, image_size]
        x_repeated = x_repeated.view(-1, *x.shape[1:])  # [batch_size * n_kp, ch, image_size, image_size]
        if z_scale is None:
            z_scale = (self.patch_size / img_size) * torch.ones_like(kp)
        else:
            # assume unnormalized z_scale
            z_scale = torch.sigmoid(z_scale)
        z_pos = kp.reshape(-1, kp.shape[-1])
        z_scale = z_scale.view(-1, z_scale.shape[-1])
        out_dims = (batch_size * n_kp, x.shape[1], self.patch_size, self.patch_size)
        cropped_objects = spatial_transform(x_repeated, z_pos, z_scale, out_dims, inverse=False, padding_mode='border')
        # [batch_size * n_kp, ch, patch_size, patch_size]

        # encode objects - fc
        enc_out = self.cnn(cropped_objects)
        if isinstance(enc_out, tuple):
            cropped_objects_cnn = enc_out[1]
        else:
            cropped_objects_cnn = enc_out

        cropped_objects_flat = cropped_objects_cnn.reshape(batch_size, n_kp, -1)  # flatten
        # backbone features
        backbone_features = cropped_objects_flat
        # projection
        backbone_features = self.backbone(backbone_features)
        if timesteps is not None and self.temp_embed is not None:
            orig_shape = backbone_features.shape
            new_feat = backbone_features.view(-1, timesteps, *backbone_features.shape[1:]) + self.temp_embed[:,
            :timesteps]
            backbone_features = new_feat.view(orig_shape)

        if self.with_obj_on:
            obj_on_feat = backbone_features
            obj_on = self.obj_on_head(obj_on_feat)

            obj_on = obj_on.view(batch_size, n_kp, 1)
            lobj_on_a = lobj_on_b = obj_on
            obj_on_a_gate = lobj_on_a.sigmoid()
            obj_on_a = ((1 - obj_on_a_gate) * self.obj_on_min + obj_on_a_gate * self.obj_on_max).exp()
            obj_on_b_gate = 1 - (lobj_on_b * 0 + lobj_on_a).sigmoid()
            obj_on_b = ((1 - obj_on_b_gate) * self.obj_on_min + obj_on_b_gate * self.obj_on_max).exp()
            obj_on_beta_dist = torch.distributions.Beta(obj_on_a, obj_on_b)
            mu_obj_on = obj_on_beta_dist.mean
            if deterministic:
                z_obj_on = obj_on_beta_dist.mean
            else:
                z_obj_on = obj_on_beta_dist.rsample()
        else:
            lobj_on_a = lobj_on_b = obj_on = None
            obj_on_a = obj_on_b = z_obj_on = mu_obj_on = None

        xy = self.xy_head(backbone_features)
        xy = xy.view(batch_size, n_kp, -1)
        mu, logvar = torch.chunk(xy, chunks=2, dim=-1)

        scale_xy = self.scale_xy_head(backbone_features)
        scale_xy = scale_xy.view(batch_size, n_kp, -1)
        if self.with_scale:
            mu_scale, logvar_scale = torch.chunk(scale_xy, chunks=2, dim=-1)
        else:
            mu_scale = scale_xy
            logvar_scale = None

        if self.kp_activation == "tanh":
            mu = self.max_offset * torch.tanh(mu)
        elif self.kp_activation == "sigmoid":
            mu = self.max_offset * torch.sigmoid(mu)

        if self.with_depth:
            depth = self.depth_head(backbone_features)
            depth = depth.view(batch_size, n_kp, 2)
            mu_depth, logvar_depth = torch.chunk(depth, 2, dim=-1)
        else:
            mu_depth = logvar_depth = None

        spatial_out = {'mu': mu, 'logvar': logvar, 'mu_scale': mu_scale, 'logvar_scale': logvar_scale,
                       'lobj_on_a': lobj_on_a, 'lobj_on_b': lobj_on_b, 'obj_on': obj_on,
                       'mu_depth': mu_depth, 'logvar_depth': logvar_depth, 'obj_on_a': obj_on_a, 'obj_on_b': obj_on_b,
                       'z_obj_on': z_obj_on, 'mu_obj_on': mu_obj_on}
        return spatial_out


class ParticleFeaturesEncoder(nn.Module):
    """
    Glimpse-encoder: encodes patches visual features in a variational fashion (mu, log-variance).
    Useful for object-based scenes.
    """

    def __init__(self, anchor_size, features_dim, image_size, margin=0, ch=3,
                 use_resblock=False, hidden_dim=256, pad_mode='replicate', activation='gelu',
                 ch_mult=(1, 2, 3), base_ch=32, final_cnn_ch=32, num_res_blocks=2, output_logvar=True,
                 cnn_mid_blocks=False, timestep_horizon=1, add_particle_temp_embed=False, init_std=0.2,
                 # initialization
                 init_zero_bias=True,  # zero bias for conv and linear layers
                 init_conv_layers=True,  # initialize conv layers with normal dist
                 init_conv_fg_std=0.02,  # std for conv fg normal dist
                 ):
        super().__init__()
        self.anchor_size = anchor_size
        self.image_size = image_size
        self.patch_size = np.round(anchor_size * (image_size - 1)).astype(int)
        self.margin = margin
        self.crop_size = self.patch_size + 2 * margin
        self.ch = ch
        self.use_resblock = use_resblock
        self.features_dim = features_dim
        self.output_logvar = output_logvar
        self.hidden_dim = hidden_dim
        self.activation = activation
        self.cnn_mid_blocks = cnn_mid_blocks
        self.timestep_horizon = timestep_horizon
        self.add_particle_temp_embed = add_particle_temp_embed
        # initialization
        self.init_zero_bias = init_zero_bias  # zero bias for conv and linear layers
        self.init_conv_layers = init_conv_layers  # initialize conv layers with normal dist
        self.init_conv_fg_std = init_conv_fg_std  # std for conv fg normal dist

        attn_res = [max(self.crop_size // 16, 1)]
        self.cnn = Encoder(ch=base_ch, ch_mult=ch_mult, num_res_blocks=num_res_blocks,
                           attn_resolutions=attn_res, dropout=0.0, resamp_with_conv=True, in_channels=self.ch,
                           resolution=self.crop_size, z_channels=final_cnn_ch, double_z=False, padding_mode=pad_mode,
                           residual=self.use_resblock, mid_blocks=cnn_mid_blocks)

        self.cnn_out_shape = self.get_cnn_shape()
        feature_map_size = (self.crop_size // 2 ** (len(ch_mult) - 1)) ** 2
        # new - FCN
        if self.features_dim % feature_map_size == 0:
            self.ch_feature_dim = math.ceil(max(self.features_dim / feature_map_size, 1))
            z_out_channels = 2 * self.ch_feature_dim if self.output_logvar else self.ch_feature_dim
            self.to_latent = nn.Conv2d(in_channels=final_cnn_ch, out_channels=z_out_channels, kernel_size=1)
            output_z_cnn = (self.ch_feature_dim, self.cnn_out_shape[-2], self.cnn_out_shape[-1])
            flattened_z_cnn = np.prod(output_z_cnn)
            if self.timestep_horizon > 1 and self.add_particle_temp_embed:
                self.temp_embed = nn.Parameter(
                    init_std * torch.randn(1, self.timestep_horizon, 1, final_cnn_ch, self.cnn_out_shape[-1],
                                           self.cnn_out_shape[-1]))
            else:
                self.temp_embed = None

            self.projection_mode = 'fcn'
            self.to_mu = nn.Identity()
            self.to_logvar = nn.Identity()
        else:
            self.ch_feature_dim = final_cnn_ch
            self.to_latent = nn.Identity()
            output_z_cnn = (self.ch_feature_dim, self.cnn_out_shape[-2], self.cnn_out_shape[-1])
            flattened_z_cnn = np.prod(output_z_cnn)
            self.projection_mode = 'fc'
            self.to_mu = self.get_mlp(flattened_z_cnn, self.features_dim)
            self.to_logvar = self.get_mlp(flattened_z_cnn, self.features_dim) if self.output_logvar else nn.Identity()
            if self.timestep_horizon > 1 and self.add_particle_temp_embed:
                self.temp_embed = nn.Parameter(init_std * torch.randn(1, self.timestep_horizon, 1, flattened_z_cnn))
            else:
                self.temp_embed = None
        self.init_weights()

        self.info = (f'ParticleFeaturesEncoder: requested latent size: {self.features_dim}, '
                     f'cnn output (h*w): {feature_map_size}, (latent_size / h*w)={self.features_dim / feature_map_size} ->'
                     f' latent projection mode: {self.projection_mode},'
                     f' project {output_z_cnn} ({flattened_z_cnn}) -> {self.features_dim}')

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if self.init_conv_layers:
                    nn.init.normal_(m.weight, 0.0, self.init_conv_fg_std)
                if self.init_zero_bias and m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def get_mlp(self, in_dim, out_dim, linear=False):
        if linear:
            return nn.Linear(in_dim, out_dim)
        else:
            activation_f = nn.GELU if self.activation == 'gelu' else nn.ReLU
            hidden_dim = self.hidden_dim

            mlp = nn.Sequential(nn.Linear(in_dim, hidden_dim),
                                activation_f(),
                                nn.Linear(hidden_dim, out_dim))
            return mlp

    def get_cnn_shape(self):
        dummy_input = torch.rand(1, self.ch, self.patch_size, self.patch_size)
        out = self.cnn(dummy_input)
        if isinstance(out, tuple):
            out = out[1]
        return out.shape[1:]

    def forward(self, x, kp, z_scale=None, timesteps=None, obj_on=None):
        # x: [bs, ch, image_size, image_size]
        # kp: [bs, n_kp, 2] in [-1, 1]
        batch_size = x.shape[0]
        n_kp = kp.shape[1]
        img_size = x.shape[-1]
        x_repeated = x.unsqueeze(1).repeat(1, n_kp, 1, 1, 1)  # [batch_size, n_kp, ch, image_size, image_size]
        x_repeated = x_repeated.view(-1, *x.shape[1:])  # [batch_size * n_kp, ch, image_size, image_size]
        if z_scale is None:
            z_scale = (self.patch_size / img_size) * torch.ones_like(kp)
        else:
            # assume unnormalized z_scale
            z_scale = torch.sigmoid(z_scale)
        z_pos = kp.reshape(-1, kp.shape[-1])
        z_scale = z_scale.view(-1, z_scale.shape[-1])
        out_dims = (batch_size * n_kp, x.shape[1], self.patch_size, self.patch_size)
        cropped_objects = spatial_transform(x_repeated, z_pos, z_scale, out_dims, inverse=False, padding_mode='border')
        # [batch_size * n_kp, ch, patch_size, patch_size]

        # encode objects - fc
        enc_out = self.cnn(cropped_objects)
        if isinstance(enc_out, tuple):
            cropped_objects_cnn = enc_out[1]
        else:
            cropped_objects_cnn = enc_out

        if obj_on is not None:
            obj_on = obj_on.view(-1)
            cropped_objects_cnn = cropped_objects_cnn * obj_on[:, None, None, None]

        # new with cnn
        if self.projection_mode == 'fcn' and self.temp_embed is not None:
            orig_shape = cropped_objects_cnn.shape  # [batch_size * n_kp, ch, patch_size, patch_size]
            new_feat = cropped_objects_cnn.view(-1, timesteps, n_kp, *cropped_objects_cnn.shape[1:])
            new_feat = new_feat + self.temp_embed[:, :timesteps]
            cropped_objects_cnn = new_feat.view(orig_shape)

        features = self.to_latent(cropped_objects_cnn)
        features = features.view(batch_size, n_kp, -1)
        if self.projection_mode == 'fc' and self.temp_embed is not None:
            orig_shape = features.shape  # [batch_size * n_kp, ch, patch_size, patch_size]
            new_feat = features.view(-1, timesteps, n_kp, *features.shape[2:])
            new_feat = new_feat + self.temp_embed[:, :timesteps]
            features = new_feat.view(orig_shape)
        if self.output_logvar:
            mu_features = self.to_mu(features)
            logvar_features = self.to_logvar(features)
        else:
            mu_features = features
            mu_features = self.to_mu(mu_features)
            logvar_features = None

        cropped_objects = cropped_objects.view(batch_size, -1, *cropped_objects.shape[1:])
        # [batch_size, n_kp, ch, crop_size, crop_size]
        spatial_out = {'mu_features': mu_features, 'logvar_features': logvar_features,
                       'cropped_objects': cropped_objects}
        return spatial_out


"""
DLP components
"""


class DLPPrior(nn.Module):
    def __init__(self, cdim=3, image_size=64, n_kp=1,
                 pad_mode='replicate',
                 patch_size=16, n_kp_prior=64,
                 kp_range=(-1, 1),
                 use_resblock=False,
                 filtering_heuristic='none',
                 ch_mult=(1, 2, 3), base_ch=32, num_res_blocks=2, cnn_mid_blocks=False,
                 init_zero_bias=True,
                 init_ssm_last_layer=True,  # spatial softmax initialization
                 init_conv_layers=True,  # initialize conv layers with normal dist
                 init_conv_fg_std=0.02,  # std for conv fg normal dist
                 ):
        super(DLPPrior, self).__init__()
        """
        DLP Prior Module -- extract object location proposals from an image via SSM
        cdim: channels of the input image (3...)
        prior_channels: channels for prior CNN (takes in patches)
        n_kp: number of kp to extract from each (!) patch
        n_kp_prior: number of kp to filter from the set of prior kp (of size n_kp x num_patches)
        pad_mode: padding for the CNNs, 'zeros' or  'replicate' (default)
        patch_size: patch size for the prior KP proposals network (not to be confused with the glimpse size)
        kp_range: the range of keypoints, can be [-1, 1] (default) or [0,1]
        kp_activation: the type of activation to apply on the keypoints: "tanh" for kp_range [-1, 1], "sigmoid" for [0, 1]
        filtering heuristic: filtering heuristic to filter prior keypoints,['distance', 'variance', 'random', 'none']
        """
        self.image_size = image_size
        self.kp_range = kp_range
        self.num_patches = int((image_size // patch_size) ** 2)
        self.n_kp = n_kp
        self.n_kp_total = self.n_kp * self.num_patches
        self.n_kp_prior = min(self.n_kp_total, n_kp_prior)
        self.patch_size = patch_size
        self.cdim = cdim
        self.use_resblock = use_resblock
        self.cnn_mid_blocks = cnn_mid_blocks
        assert filtering_heuristic in ['distance', 'variance',
                                       'random', 'none'], f'unknown filtering heuristic: {filtering_heuristic}'
        self.filtering_heuristic = filtering_heuristic

        # initialization
        self.init_zero_bias = init_zero_bias  # zero bias for conv and linear layers
        self.init_ssm_last_layer = init_ssm_last_layer  # spatial softmax initialization
        self.init_conv_layers = init_conv_layers  # initialize conv layers with normal dist
        self.init_conv_fg_std = init_conv_fg_std  # std for conv fg normal dist

        # prior
        self.patcher = ImagePatcher(cdim=cdim, image_size=image_size, patch_size=patch_size)
        # self.features_dim = int(patch_size // (2 ** (len(prior_channels) - 1)))
        self.features_dim = int(patch_size // (2 ** (len(ch_mult) - 1)))
        attn_res = [max(self.patch_size // 16, 1)]
        self.enc = Encoder(ch=base_ch, ch_mult=ch_mult, num_res_blocks=num_res_blocks,
                           attn_resolutions=attn_res, dropout=0.0, resamp_with_conv=True, in_channels=cdim,
                           resolution=patch_size, z_channels=n_kp, double_z=False, padding_mode='replicate',
                           residual=self.use_resblock, mid_blocks=cnn_mid_blocks)

        self.ssm = AlternativeSpatialSoftmaxKP(kp_range=kp_range)

        self.init_weights()

    def init_conv_with_spatial_priors(self, conv: nn.Conv2d, gaussian_sigma=0.4, noise_std=0.05):
        """
        Initializes a conv layer with spatially structured filters for RGB or single-channel input.
        Supports Sobel, Prewitt, Laplacian, and Gaussian blobs with noise.
        """
        out_channels, in_channels, H, W = conv.weight.shape

        sobel_x = torch.tensor([[-1, 0, 1],
                                [-2, 0, 2],
                                [-1, 0, 1]], dtype=torch.float32)
        sobel_y = sobel_x.T

        prewitt_x = torch.tensor([[-1, 0, 1],
                                  [-1, 0, 1],
                                  [-1, 0, 1]], dtype=torch.float32)
        prewitt_y = prewitt_x.T

        laplacian = torch.tensor([[0, 1, 0],
                                  [1, -4, 1],
                                  [0, 1, 0]], dtype=torch.float32)

        edge_filters = [sobel_x, sobel_y, prewitt_x, prewitt_y, laplacian]

        def make_gaussian_blob(size, sigma, center):
            x = torch.linspace(-1, 1, size)
            y = torch.linspace(-1, 1, size)
            yy, xx = torch.meshgrid(y, x, indexing="ij")
            grid = torch.stack([xx, yy], dim=0)
            diff = grid - torch.tensor(center).view(2, 1, 1)
            return torch.exp(-torch.sum(diff ** 2, dim=0) / (2 * sigma ** 2))

        weight = torch.zeros_like(conv.weight)

        for i in range(out_channels):
            filter_type = i % (len(edge_filters) + 1)
            if filter_type < len(edge_filters):
                ef = edge_filters[filter_type]
                ef_padded = torch.zeros((H, W))
                h0 = (H - ef.shape[0]) // 2
                w0 = (W - ef.shape[1]) // 2
                ef_padded[h0:h0 + ef.shape[0], w0:w0 + ef.shape[1]] = ef
                base_filter = ef_padded
            else:
                cx, cy = np.random.uniform(-0.5, 0.5, size=2)
                base_filter = make_gaussian_blob(H, gaussian_sigma, (cx, cy))

            noisy_filter = base_filter + noise_std * torch.randn_like(base_filter)

            # Apply the same (or slightly varied) filter across input channels
            for c in range(in_channels):
                # Option 1: same for all channels
                weight[i, c] = noisy_filter.clone()
                # Option 2 (optional): add slight channel-specific noise
                # weight[i, c] = noisy_filter + noise_std * torch.randn_like(noisy_filter)

        with torch.no_grad():
            conv.weight.copy_(weight)
            if conv.bias is not None:
                conv.bias.zero_()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # pass
                if self.init_conv_layers:
                    nn.init.normal_(m.weight, 0.0, self.init_conv_fg_std)
                if self.init_zero_bias and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                # use pytorch's default
                pass
        # initialize input filters with spatial priors
        if self.init_ssm_last_layer:
            m = self.enc.conv_out
            # nn.init.normal_(m.weight, -0.2, 0.02)
            # d = -1 * math.sqrt(1 / (m.in_channels + m.out_channels))
            d = -1.0 * self.init_conv_fg_std
            nn.init.constant_(m.weight, d)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        # self.init_conv_with_spatial_priors(self.enc.conv_in)

    def img_to_patches(self, x):
        return self.patcher.img_to_patches(x)

    def patches_to_img(self, x):
        return self.patcher.patches_to_img(x)

    def get_global_kp(self, local_kp):
        # local_kp: [batch_size, num_patches, n_kp, 2]
        # returns the global coordinates of a KP within the original image.
        batch_size, num_patches, n_kp, _ = local_kp.shape
        global_coor = self.patcher.get_patch_location_idx().to(local_kp.device)  # [num_patches, 2]
        global_coor = global_coor[:, None, :].repeat(1, n_kp, 1)
        global_coor = (((local_kp - self.kp_range[0]) / (self.kp_range[1] - self.kp_range[0])) * (
                self.patcher.patch_size - 1) + global_coor) / (self.image_size - 1)
        global_coor = global_coor * (self.kp_range[1] - self.kp_range[0]) + self.kp_range[0]
        return global_coor

    def get_patch_centers(self):
        # get the resepective coordinates of the patches
        centers = self.patcher.get_patch_centers() / (self.image_size - 1)
        return centers

    def get_distance_from_patch_centers(self, kp, global_kp=False):
        # calculates the distance of a KP from the center of its parent patch. This is useful to understand (and filter)
        # if SSM detected something, otherwise, the KP will probably land in the center of the patch
        # (e.g., a solid-color patch will have the same activation in all pixels).
        if not global_kp:
            global_coor = self.get_global_kp(kp).view(kp.shape[0], -1, 2)
        else:
            global_coor = kp
        centers = 0.5 * (self.kp_range[1] + self.kp_range[0]) * torch.ones_like(kp).to(kp.device)
        global_centers = self.get_global_kp(centers.view(kp.shape[0], -1, self.n_kp, 2)).view(kp.shape[0], -1, 2)
        return ((global_coor - global_centers) ** 2).sum(-1)

    def encode_prior(self, x, filtering_heuristic='none', k=None):
        # encodes prior keypoints by patchifying the image and applying spatial-softmax
        # x: [batch_size, cdim, image_size, image_size]
        # global_kp: set True to get the global coordinates within the image (instead of local KP inside the patch)
        batch_size, cdim, image_size, image_size = x.shape
        x_patches = self.img_to_patches(x)  # [batch_size, cdim, num_patches, patch_size, patch_size]
        x_patches = x_patches.permute(0, 2, 1, 3, 4)  # [batch_size, num_patches, cdim, patch_size, patch_size]
        x_patches = x_patches.contiguous().view(-1, cdim, self.patcher.patch_size, self.patcher.patch_size)
        enc_out = self.enc(x_patches)  # [batch_size*num_patches, n_kp, features_dim, features_dim]
        if isinstance(enc_out, tuple):
            z = enc_out[1]
        else:
            z = enc_out
        kp_p, var_kp = self.ssm(z, probs=False, variance=True)  # [batch_size * num_patches, n_kp, 2]
        kp_p = kp_p.view(batch_size, -1, self.n_kp, 2)  # [batch_size, num_patches, n_kp, 2]
        kp_p = self.get_global_kp(kp_p)
        var_kp = var_kp.view(batch_size, kp_p.shape[1], self.n_kp, -1)  # [batch_size, num_patches, n_kp, 3]

        if k is None:
            k = self.n_kp_prior
        kp_p = kp_p.view(x.shape[0], -1, 2)  # [batch_size, n_kp_total, 2]
        var_kp = var_kp.view(x.shape[0], kp_p.shape[1], -1)  # [batch_size, n_kp_total, 3]
        if filtering_heuristic == 'distance':
            # filter proposals by distance to the patches' center
            dist_from_center = self.prior.get_distance_from_patch_centers(kp_p, global_kp=True)
            _, indices = torch.topk(dist_from_center, k=k, dim=-1, largest=True)
            batch_indices = torch.arange(kp_p.shape[0], device=kp_p.device).view(-1, 1)
            kp_p = kp_p[batch_indices, indices]
            var_kp = var_kp[batch_indices, indices]
        elif filtering_heuristic == 'variance':
            total_var = var_kp.sum(-1)
            _, indices = torch.topk(total_var, k=k, dim=-1, largest=False)
            batch_indices = torch.arange(kp_p.shape[0], device=kp_p.device).view(-1, 1)
            kp_p = kp_p[batch_indices, indices]
        elif filtering_heuristic == 'none':
            return kp_p, var_kp
        else:
            # alternatively, just sample random kp
            kp_p = kp_p[:, torch.randperm(kp_p.shape[1])[:k]]
            var_kp = var_kp[:, torch.randperm(kp_p.shape[1])[:k]]
        return kp_p, var_kp

    def forward(self, x):
        # prior proposals
        kp_p, var_kp = self.encode_prior(x, filtering_heuristic=self.filtering_heuristic)
        return kp_p, var_kp


class ParticleInteractionEncoder(nn.Module):
    def __init__(self, n_kp_enc, dropout=0.0, learned_feature_dim=16, learned_bg_feature_dim=16, embed_init_std=0.2,
                 projection_dim=128, timestep_horizon=1, pte_layers=1, pte_heads=1,
                 attn_norm_type='rms', hidden_dim=256, use_resblock=True, pad_mode='replicate',
                 temporal_interaction=True, interaction_depth=False, interaction_obj_on=False, activation='gelu',
                 scale_anchor=None,
                 interaction_features=False, ch_mult=(1, 2, 3), base_ch=32, final_cnn_ch=32, num_res_blocks=2, cdim=3,
                 image_size=64, n_views=1, bg=True, use_img_input=True, cnn_mid_blocks=False,
                 particle_positional_embed=True,
                 particle_score=False, norm_layer=True, add_particle_temp_embed=False,
                 features_dist='gauss', n_fg_categories=8, n_fg_classes=4, n_bg_categories=4, n_bg_classes=4,
                 obj_on_min=1e-4, obj_on_max=100.0,
                 particle_anchors=None, use_z_orig=False,
                 init_zero_bias=True,  # zero bias for conv and linear layers
                 init_conv_layers=True,  # initialize conv layers with normal dist
                 init_conv_fg_std=0.02,  # std for conv fg normal dist
                 ):
        super(ParticleInteractionEncoder, self).__init__()
        """
        DLP Foreground Module -- extract objects from an image

        """
        self.n_kp_enc = n_kp_enc
        self.dropout = dropout
        self.learned_feature_dim = learned_feature_dim
        self.learned_bg_feature_dim = learned_bg_feature_dim
        self.features_dist = features_dist
        self.n_fg_categories = n_fg_categories
        self.n_fg_classes = n_fg_classes
        self.n_bg_categories = n_bg_categories
        self.n_bg_classes = n_bg_classes
        assert learned_feature_dim > 0, "learned_feature_dim must be greater than 0"
        self.embed_init_std = embed_init_std
        self.projection_dim = projection_dim
        self.timestep_horizon = (timestep_horizon + 1) if timestep_horizon > 1 else 1
        self.attn_norm_type = attn_norm_type
        self.hidden_dim = hidden_dim
        self.temporal_interaction = temporal_interaction
        self.interaction_depth = interaction_depth
        self.interaction_obj_on = interaction_obj_on
        self.interaction_features = interaction_features
        self.with_bg = bg
        self.use_img_input = use_img_input
        self.activation = activation
        self.cnn_mid_blocks = cnn_mid_blocks
        self.particle_score = particle_score
        self.obj_on_min = obj_on_min
        self.obj_on_max = obj_on_max
        self.add_particle_temp_embed = add_particle_temp_embed
        self.scale_anchor = scale_anchor
        self.use_z_orig = use_z_orig
        self.n_views = n_views

        # initialization
        self.init_zero_bias = init_zero_bias  # zero bias for conv and linear layers
        self.init_conv_layers = init_conv_layers  # initialize conv layers with normal dist
        self.init_conv_fg_std = init_conv_fg_std  # std for conv fg normal dist

        if particle_anchors is None:
            self.register_buffer('particles_anchor', torch.zeros(1, 1, self.n_kp_enc))
            self.use_z_orig = False
        else:
            self.register_buffer('particles_anchor', particle_anchors)

        n_particles = self.n_kp_enc  # [n_kp_enc]

        if self.use_img_input:
            # cnn stuff
            self.ctx_pre_pte_latent_dim = projection_dim  # can also be ctx dim
            self.image_size = image_size
            self.output_feat_map_size = int(image_size // (2 ** (len(ch_mult) - 1)))
            self.cdim = cdim

            attn_res = [max(self.image_size // 16, 1)]
            self.ctx_cnn_enc = Encoder(ch=base_ch, ch_mult=ch_mult, num_res_blocks=num_res_blocks,
                                       attn_resolutions=attn_res, dropout=0.0, resamp_with_conv=True,
                                       in_channels=self.cdim,
                                       resolution=self.image_size, z_channels=final_cnn_ch, double_z=False,
                                       padding_mode=pad_mode, residual=use_resblock, in_conv_kernel_size=3,
                                       mid_blocks=cnn_mid_blocks)
            self.cnn_out_shape = self.get_cnn_shape()
            feature_map_size = self.output_feat_map_size ** 2

            # FCN or Linear
            if self.ctx_pre_pte_latent_dim % feature_map_size == 0:
                self.ch_learned_feature_dim = math.ceil(max(self.ctx_pre_pte_latent_dim / feature_map_size, 1))
                out_ch = self.ch_learned_feature_dim
                self.to_latent = nn.Conv2d(in_channels=final_cnn_ch,
                                           out_channels=out_ch, kernel_size=1)
                output_z_cnn = (self.ch_learned_feature_dim, self.cnn_out_shape[-2], self.cnn_out_shape[-1])
                flattened_z_cnn = np.prod(output_z_cnn)

                self.projection_mode = 'fcn'
                self.to_latent_lin = nn.Identity()
            else:
                self.ch_learned_feature_dim = final_cnn_ch
                self.to_latent = nn.Identity()
                output_z_cnn = (self.ch_learned_feature_dim, self.cnn_out_shape[-2], self.cnn_out_shape[-1])
                flattened_z_cnn = np.prod(output_z_cnn)

                self.projection_mode = 'fc'
                self.to_latent_lin = self.get_mlp(flattened_z_cnn, self.ctx_pre_pte_latent_dim)

            self.info = (f'ParticleInteractionEncoder: requested latent size: {self.ctx_pre_pte_latent_dim}, '
                         f'cnn output (h*w): {feature_map_size}, (latent_size / h*w)={self.ctx_pre_pte_latent_dim / feature_map_size} ->'
                         f' latent projection mode: {self.projection_mode},'
                         f' project {output_z_cnn} ({flattened_z_cnn}) -> {self.ctx_pre_pte_latent_dim}')

            # end cnn stuff
            n_particles += 1  # [ctx + n_kp_enc]
            self.ctx_embeddings = nn.Parameter(
                self.embed_init_std * torch.randn(1, 1, 1, projection_dim))
        else:
            self.info = f'ParticleInteractionEncoder: not using image as input context'
        if self.with_bg:
            n_particles += 1
            self.bg_embeddings = nn.Parameter(self.embed_init_std * torch.randn(1, 1, 1, projection_dim))

        # entities positional embeddings
        if particle_positional_embed:
            self.particle_embeddings = nn.Parameter(
                self.embed_init_std * torch.randn(1, 1, self.n_kp_enc, projection_dim))
        else:
            self.particle_embeddings = nn.Parameter(self.embed_init_std * torch.randn(1, 1, 1, projection_dim))

        # interaction encoder
        self.basic_particle_proj = ParticleAttributesProjection(n_particles=self.n_kp_enc,
                                                                in_features_dim=self.learned_feature_dim,
                                                                hidden_dim=self.hidden_dim,
                                                                output_dim=projection_dim,
                                                                bg_features_dim=self.learned_bg_feature_dim,
                                                                add_ctx_token=False,
                                                                depth=not self.interaction_depth,
                                                                obj_on=not self.interaction_obj_on,
                                                                base_var=False, bg=self.with_bg,
                                                                particle_score=self.particle_score,
                                                                norm_layer=norm_layer,
                                                                use_z_orig=self.use_z_orig)
        if self.add_particle_temp_embed and not self.temporal_interaction:
            self.temp_embed = nn.Parameter(
                self.embed_init_std * torch.randn(1, self.timestep_horizon, 1, projection_dim))
        else:
            self.temp_embed = None

        if self.n_views > 1:
            self.view_embeddings = nn.Parameter(
                self.embed_init_std * torch.randn(1, 1, self.n_views, 1, projection_dim))
        else:
            self.view_embeddings = None

        block_size = self.timestep_horizon if self.temporal_interaction else 1
        self.pte = ParticleSelfAttTransformer(n_embed=self.projection_dim, n_head=pte_heads,
                                              n_layer=pte_layers,
                                              block_size=block_size,
                                              output_dim=self.projection_dim, attn_pdrop=dropout,
                                              resid_pdrop=dropout,
                                              hidden_dim_multiplier=4, positional_bias=False,
                                              activation=activation,
                                              max_particles=None, norm_type=attn_norm_type,
                                              init_std=embed_init_std)

        self.particle_decoder = ParticleAttributeDecoder(n_particles=self.n_kp_enc, input_dim=projection_dim,
                                                         hidden_dim=self.hidden_dim,
                                                         features_dim=learned_feature_dim,
                                                         bg_features_dim=learned_bg_feature_dim,
                                                         depth=self.interaction_depth,
                                                         obj_on=self.interaction_obj_on,
                                                         features=self.interaction_features,
                                                         bg_features=(self.interaction_features and self.with_bg),
                                                         features_dist=self.features_dist)
        self.init_weights()

    def init_weights(self):
        # initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if self.init_conv_layers:
                    nn.init.normal_(m.weight, 0, self.init_conv_fg_std)
                if self.init_zero_bias and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                if self.init_zero_bias and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        self.particle_decoder.init_weights()
        self.pte.init_weights()

    def get_mlp(self, in_dim, out_dim, linear=False):
        if linear:
            return nn.Linear(in_dim, out_dim)
        else:
            activation_f = nn.GELU if self.activation == 'gelu' else nn.ReLU
            hidden_dim = self.hidden_dim
            mlp = nn.Sequential(nn.Linear(in_dim, hidden_dim),
                                activation_f(),
                                nn.Linear(hidden_dim, out_dim))
            return mlp

    def get_cnn_shape(self):
        dummy_input = torch.rand(1, self.cdim, self.image_size, self.image_size)
        out = self.ctx_cnn_enc(dummy_input)
        if isinstance(out, tuple):
            out = out[1]
        return out.shape[1:]

    def encode_ctx_features(self, x, masks=None):
        # x: [bs, ch, image_size, image_size]
        # masks: [bs, 1, image_size, image_size]
        batch_size, _, features_dim, _ = x.shape
        # bg features
        if masks is not None:
            x_in = x * masks
        else:
            x_in = x
        enc_out = self.ctx_cnn_enc(x_in)
        if isinstance(enc_out, tuple):
            cnn_features = enc_out[1]
        else:
            cnn_features = enc_out

        # new cnn
        features = self.to_latent(cnn_features)
        features = features.view(features.shape[0], -1)
        features = self.to_latent_lin(features)
        return features

    def encode_all(self, x, z, z_scale, z_obj_on, z_depth, z_features, z_bg_features=None, z_base_var=None,
                   z_score=None, patch_id_embed=None, deterministic=False, warmup=False,
                   detach_before_proj=False):
        """
        output order:
        if with_bg and ctx_pool_mode='token': [n_particles, bg, ctx, ctx_token*]
        else: [n_particles, ctx, ctx_token*]
        """
        # x: [bs * n_views, t, ch, h, w]
        bs, timestep_horizon = z.shape[0], z.shape[1]
        z_v = z.detach() if detach_before_proj else z
        z_scale_v = z_scale.detach() if detach_before_proj else z_scale
        z_obj_on_v = z_obj_on.detach() if (z_obj_on is not None and detach_before_proj) else z_obj_on
        z_depth_v = z_depth.detach() if (z_depth is not None and detach_before_proj) else z_depth
        z_features_v = z_features.detach() if detach_before_proj else z_features
        if not self.with_bg:
            z_bg_features = None
        z_bg_features_v = z_bg_features.detach() if (
                z_bg_features is not None and detach_before_proj) else z_bg_features
        z_base_var_v = z_base_var.detach() if z_base_var is not None else z_base_var
        z_score_v = z_score.detach() if z_score is not None else z_score
        if self.use_z_orig:
            z_orig_v = self.particles_anchor.unsqueeze(0).repeat(z_v.shape[0], z_v.shape[1], 1, 1)
        else:
            z_orig_v = None

        particle_projection = self.basic_particle_proj(z=z_v,
                                                       z_scale=z_scale_v,
                                                       z_obj_on=z_obj_on_v,
                                                       z_depth=z_depth_v,
                                                       z_features=z_features_v,
                                                       z_bg_features=z_bg_features_v,
                                                       z_base_var=z_base_var_v,
                                                       z_score=z_score_v,
                                                       z_orig=z_orig_v)
        # add entity pos embeddings
        if self.particle_embeddings.shape[2] == 1:
            p_embeddings = self.particle_embeddings.repeat(bs, timestep_horizon, z.shape[2], 1)
        else:
            p_embeddings = self.particle_embeddings.repeat(bs, timestep_horizon, 1, 1)
        if patch_id_embed is not None:
            p_embeddings = p_embeddings + patch_id_embed
        if self.with_bg:
            bg_embeddings = self.bg_embeddings.repeat(bs, timestep_horizon, 1, 1)
            p_embeddings = torch.cat([p_embeddings, bg_embeddings], dim=2)
        particle_projection = particle_projection + p_embeddings

        if self.use_img_input:
            # add context
            if len(x.shape) == 5:
                # x: [bs, t, ch, h, w]
                x_in = x.view(-1, *x.shape[2:])
            else:
                x_in = x
            ctx_features = self.encode_ctx_features(x_in)
            ctx_features = ctx_features.view(bs, timestep_horizon, 1, -1)  # [bs, T, 1, projection_dim]
            ctx_features = ctx_features + self.ctx_embeddings.repeat(bs, timestep_horizon, 1, 1)
            particle_projection = torch.cat([particle_projection, ctx_features], dim=2)
            # [bs, t, n_p + 2, proj_dim] if with_bg else [bs, t, n_p + 1, proj_dim]
        #     # [bs, t, n_p + 2, proj_dim]

        if self.n_views > 1:
            # [bs * n_views, t, n, d] -> [bs, t, n_views, n, d] -> [bs, t, n_views * n, d]
            particle_projection = particle_projection.view(-1, self.n_views, *particle_projection.shape[1:])
            particle_projection = particle_projection.permute(0, 2, 1, 3, 4)  # [bs, t, n_views, n, d]
            particle_projection = particle_projection + self.view_embeddings
            particle_projection = particle_projection.reshape(particle_projection.shape[0],
                                                              particle_projection.shape[1],
                                                              -1,
                                                              particle_projection.shape[-1])  # [bs, t, n_views * n, d]

        if timestep_horizon > 1 and not self.temporal_interaction:
            if self.add_particle_temp_embed:
                particle_projection = particle_projection + self.temp_embed[:, :timestep_horizon]
            particle_projection = particle_projection.view(-1, 1, *particle_projection.shape[2:])
            # [bs * ts, 1, n, f]
        particles_out = self.pte(particle_projection)
        particles_out = particles_out.view(-1, timestep_horizon, *particles_out.shape[2:])
        # [bs, ts, n, f]
        if self.n_views > 1:
            # [bs, t, n_views * n, d] -> [bs * n_views, t, n, d]
            particles_out = particles_out.view(particles_out.shape[0], timestep_horizon, self.n_views, -1,
                                               particles_out.shape[-1])
            particles_out = particles_out.permute(0, 2, 1, 3, 4)
            particles_out = particles_out.reshape(-1, *particles_out.shape[2:])
        particle_decoder_out = self.particle_decoder(particles_out)  # [bs * n_views, t, n, d]
        # unpack
        mu_depth = particle_decoder_out['mu_depth']
        logvar_depth = particle_decoder_out['logvar_depth']
        if self.interaction_depth:
            z_depth = reparameterize(mu_depth, logvar_depth) if not deterministic else mu_depth
        else:
            z_depth = None
        mu_features = particle_decoder_out['mu_features']
        logvar_features = particle_decoder_out['logvar_features']
        mu_bg_features = particle_decoder_out['mu_bg_features']
        logvar_bg_features = particle_decoder_out['logvar_bg_features']
        if self.interaction_features:
            mu_features = z_features + mu_features
            if self.features_dist == 'categorical':
                logits = mu_features.view(*mu_features.shape[:-1], self.n_fg_categories, self.n_fg_classes)
                # [bs, T, n_p, n_categories, n_classes]
                probs = logits.softmax(dim=-1)  # [bs, T, n_p, n_categories, n_classes]
                if deterministic:
                    samples = torch.argmax(probs.view(-1, probs.shape[-1]), dim=-1, keepdim=True)
                    samples = F.one_hot(samples.squeeze(-1), num_classes=self.n_fg_classes)
                    samples = samples.view(probs.shape)
                    # straight-through
                    z_features = samples.detach() + (probs - probs.detach())
                    z_features = z_features.view(*mu_features.shape)  # [bs, T, n_p, n_categories * n_classes]
                else:
                    samples = torch.multinomial(probs.view(-1, probs.shape[-1]), num_samples=1)
                    samples = F.one_hot(samples.squeeze(-1), num_classes=self.n_fg_classes)
                    samples = samples.view(probs.shape)
                    # straight-through
                    z_features = samples.detach() + (probs - probs.detach())
                    z_features = z_features.view(*mu_features.shape)  # [bs, T, n_p, n_categories * n_classes]
            else:
                # logvar_features = logvar_features.clamp_max(math.log(0.2 ** 2))
                z_features = reparameterize(mu_features, logvar_features) if not deterministic else mu_features
            if self.with_bg:
                mu_bg_features = z_bg_features + mu_bg_features
                if self.features_dist == 'categorical':
                    logits_bg = mu_bg_features.view(*mu_bg_features.shape[:-1], self.n_bg_categories, self.n_bg_classes)
                    # [bs, T, n_p, n_categories, n_classes]
                    probs_bg = logits_bg.softmax(dim=-1)  # [bs, T, n_p, n_categories, n_classes]
                    if deterministic:
                        samples_bg = torch.argmax(probs_bg.view(-1, probs_bg.shape[-1]), dim=-1, keepdim=True)
                        samples_bg = F.one_hot(samples_bg.squeeze(-1), num_classes=self.n_bg_classes)
                        samples_bg = samples_bg.view(probs_bg.shape)
                        # straight-through
                        z_bg_features = samples_bg.detach() + (probs_bg - probs_bg.detach())
                        z_bg_features = z_bg_features.view(
                            *mu_bg_features.shape)  # [bs, T, n_p, n_categories * n_classes]
                    else:
                        samples_bg = torch.multinomial(probs_bg.view(-1, probs_bg.shape[-1]), num_samples=1)
                        samples_bg = F.one_hot(samples_bg.squeeze(-1), num_classes=self.n_bg_classes)
                        samples_bg = samples_bg.view(probs_bg.shape)
                        # straight-through
                        z_bg_features = samples_bg.detach() + (probs_bg - probs_bg.detach())
                        z_bg_features = z_bg_features.view(
                            *mu_bg_features.shape)  # [bs, T, n_p, n_categories * n_classes]
                else:
                    # logvar_bg_features = logvar_bg_features.clamp_max(math.log(0.2 ** 2))
                    z_bg_features = reparameterize(mu_bg_features,
                                                   logvar_bg_features) if not deterministic else mu_bg_features
        else:
            z_features = z_bg_features = None
        lobj_on_a = particle_decoder_out['lobj_on_a']
        lobj_on_b = particle_decoder_out['lobj_on_b']
        if self.interaction_obj_on:
            obj_on_a_gate = (lobj_on_a).sigmoid()
            obj_on_a = ((1 - obj_on_a_gate) * self.obj_on_min + obj_on_a_gate * self.obj_on_max).exp()
            obj_on_b_gate = 1 - (lobj_on_b * 0 + lobj_on_a).sigmoid()
            obj_on_b = ((1 - obj_on_b_gate) * self.obj_on_min + obj_on_b_gate * self.obj_on_max).exp()
            obj_on_beta_dist = torch.distributions.Beta(obj_on_a, obj_on_b)
            mu_obj_on = obj_on_beta_dist.mean
            z_obj_on = obj_on_beta_dist.rsample() if not deterministic else obj_on_beta_dist.mean
        else:
            obj_on_a = obj_on_b = z_obj_on = mu_obj_on = None

        encode_dict = {'mu_depth': mu_depth, 'logvar_depth': logvar_depth, 'z_depth': z_depth,
                       'obj_on_a': obj_on_a, 'obj_on_b': obj_on_b, 'z_obj_on': z_obj_on, 'mu_obj_on': mu_obj_on,
                       'mu_features': mu_features, 'logvar_features': logvar_features, 'z_features': z_features,
                       'mu_bg_features': mu_bg_features, 'logvar_bg_features': logvar_bg_features,
                       'z_bg_features': z_bg_features, 'z_scale': z_scale, 'z': z}
        return encode_dict

    def forward(self, x, z, z_scale, z_obj_on, z_depth, z_features, z_bg_features=None, z_base_var=None, z_score=None,
                patch_id_embed=None, deterministic=False, warmup=False):
        output_dict = self.encode_all(x, z, z_scale, z_obj_on, z_depth, z_features, z_bg_features, z_base_var, z_score,
                                      patch_id_embed, deterministic=deterministic, warmup=warmup)
        return output_dict


class ParticleContextEncoder(nn.Module):
    def __init__(self, n_kp_enc, dropout=0.1, learned_feature_dim=16, learned_bg_feature_dim=16, embed_init_std=0.02,
                 projection_dim=128, timestep_horizon=1, pte_layers=1, pte_heads=1,
                 attn_norm_type='rms', context_dim=7, hidden_dim=256,
                 activation='gelu',
                 ctx_pool_mode='none', bg=True, causal=True, particle_positional_embed=True,
                 particle_score=False, norm_layer=True,
                 shared_logvar=False, ctx_dist='gauss', n_ctx_categories=4, n_ctx_classes=4,
                 particle_anchors=None, use_z_orig=False,
                 ctx_pool_dim=256, n_pool_ctx_categories=8, n_pool_ctx_classes=8, global_ctx_pool=False):
        super(ParticleContextEncoder, self).__init__()
        """
        This module takes in temporal sequence of particles and outputs latent context,
        which can be per-particle, or global, depending on the pooling type.

        """
        assert ctx_pool_mode in ['none', 'mean', 'max', 'token', 'last', 'mlp']
        self.ctx_pool_mode = ctx_pool_mode
        self.n_kp_enc = n_kp_enc
        self.dropout = dropout
        self.learned_feature_dim = learned_feature_dim
        self.learned_bg_feature_dim = learned_bg_feature_dim
        assert learned_feature_dim > 0, "learned_feature_dim must be greater than 0"
        self.embed_init_std = embed_init_std
        self.projection_dim = projection_dim
        self.timestep_horizon = (timestep_horizon + 1) if timestep_horizon > 1 else 1
        self.attn_norm_type = attn_norm_type
        self.context_dist = ctx_dist
        self.n_ctx_categories = n_ctx_categories
        self.n_ctx_classes = n_ctx_classes
        self.context_dim = context_dim
        self.learned_ctx_token = (ctx_pool_mode == 'token')
        self.n_pool_ctx_categories = n_pool_ctx_categories
        self.n_pool_ctx_classes = n_pool_ctx_classes
        self.ctx_pool_dim = ctx_pool_dim
        if self.context_dist == 'categorical':
            self.ctx_pool_dim = int(self.n_pool_ctx_categories * self.n_pool_ctx_classes)
        self.global_ctx_pool = global_ctx_pool
        self.hidden_dim = hidden_dim
        self.with_bg = bg
        self.activation = activation
        self.is_causal = causal
        # assert not (ctx_pool_mode == 'none' and not self.use_img_input), \
        #     f'context pooling mode can not be "{ctx_pool_mode}" without using image encoder!'
        self.particle_score = particle_score
        self.shared_logvar = shared_logvar
        self.use_z_orig = use_z_orig
        if particle_anchors is None:
            self.register_buffer('particles_anchor', torch.zeros(1, 1, self.n_kp_enc))
            self.use_z_orig = False
        else:
            self.register_buffer('particles_anchor', particle_anchors)

        n_particles = self.n_kp_enc  # [n_kp_enc]
        # entities in attn: [bg*, n_particles, ctx, ctx_tokens*]
        if self.learned_ctx_token:
            n_particles += 1
            self.ctx_token_embeddings = nn.Parameter(
                self.embed_init_std * torch.randn(1, 1, 1, projection_dim))
        if self.learned_ctx_token or self.ctx_pool_mode == 'last':
            block_size = 1  # this means token pooling does not depend on the temporal horizon
            self.cross_attn_block = CrossBlock(n_embed=self.projection_dim, n_head=pte_heads,
                                               block_size=block_size,
                                               attn_pdrop=dropout,
                                               resid_pdrop=dropout,
                                               hidden_dim_multiplier=4, positional_bias=False,
                                               activation='gelu',
                                               max_particles=None, norm_type=attn_norm_type)
        else:
            self.cross_attn_block = None
        if self.with_bg:
            n_particles += 1
            self.bg_embeddings = nn.Parameter(self.embed_init_std * torch.randn(1, 1, 1, projection_dim))

        # entities positional embeddings
        if particle_positional_embed:
            self.particle_embeddings = nn.Parameter(
                self.embed_init_std * torch.randn(1, 1, self.n_kp_enc, projection_dim))
        else:
            self.particle_embeddings = nn.Parameter(self.embed_init_std * torch.randn(1, 1, 1, projection_dim))

        # interaction encoder
        proj_out_dim = projection_dim
        self.basic_particle_proj = ParticleAttributesProjection(n_particles=self.n_kp_enc,
                                                                in_features_dim=self.learned_feature_dim,
                                                                hidden_dim=self.hidden_dim,
                                                                output_dim=proj_out_dim,
                                                                bg_features_dim=self.learned_bg_feature_dim,
                                                                add_ctx_token=False,
                                                                depth=True,
                                                                obj_on=True,
                                                                base_var=False, bg=self.with_bg,
                                                                norm_layer=norm_layer,
                                                                particle_score=self.particle_score,
                                                                use_z_orig=self.use_z_orig)

        block_size = self.timestep_horizon
        self.pte = ParticleSpatioTemporalTransformer(n_embed=self.projection_dim, n_head=pte_heads,
                                                     n_layer=pte_layers,
                                                     block_size=block_size,
                                                     output_dim=self.projection_dim, attn_pdrop=dropout,
                                                     resid_pdrop=dropout,
                                                     hidden_dim_multiplier=4, positional_bias=False,
                                                     activation='gelu',
                                                     max_particles=None, norm_type=attn_norm_type,
                                                     particles_first=False, init_std=embed_init_std,
                                                     causal=self.is_causal)

        self.particle_decoder = ParticleContextDecoder(n_particles=self.n_kp_enc, input_dim=projection_dim,
                                                       hidden_dim=self.hidden_dim,
                                                       context_dim=self.context_dim,
                                                       context_dist=self.context_dist,
                                                       n_ctx_categories=self.n_ctx_categories,
                                                       n_ctx_classes=self.n_ctx_classes,
                                                       learned_ctx_token=self.learned_ctx_token,
                                                       ctx_pool_mode=self.ctx_pool_mode,
                                                       shared_logvar=self.shared_logvar,
                                                       output_ctx_logvar=(ctx_dist != 'categorical'))
        self.init_weights()

    def init_weights(self):
        self.particle_decoder.init_weights()
        self.pte.init_weights()

    def encode_all(self, z, z_scale, z_obj_on, z_depth, z_features, z_bg_features=None, z_base_var=None,
                   z_score=None, patch_id_embed=None, deterministic=False, warmup=False,
                   detach_before_proj=False):
        """
        output order:
        if with_bg and ctx_pool_mode='token': [n_particles, bg, ctx, ctx_token*]
        else: [n_particles, ctx, ctx_token*]
        """
        bs, timestep_horizon = z.shape[0], z.shape[1]
        z_v = z.detach() if detach_before_proj else z
        z_scale_v = z_scale.detach() if detach_before_proj else z_scale
        z_obj_on_v = z_obj_on.detach() if (z_obj_on is not None and detach_before_proj) else z_obj_on
        z_depth_v = z_depth.detach() if (z_depth is not None and detach_before_proj) else z_depth
        z_features_v = z_features.detach() if detach_before_proj else z_features
        if not self.with_bg:
            z_bg_features = None
        z_bg_features_v = z_bg_features.detach() if (
                z_bg_features is not None and detach_before_proj) else z_bg_features
        z_base_var_v = z_base_var.detach() if z_base_var is not None else z_base_var
        z_score_v = z_score.detach() if z_score is not None else z_score
        if self.use_z_orig:
            z_orig_v = self.particles_anchor.unsqueeze(0).repeat(z_v.shape[0], z_v.shape[1], 1, 1)
        else:
            z_orig_v = None

        particle_projection = self.basic_particle_proj(z=z_v,
                                                       z_scale=z_scale_v,
                                                       z_obj_on=z_obj_on_v,
                                                       z_depth=z_depth_v,
                                                       z_features=z_features_v,
                                                       z_bg_features=z_bg_features_v,
                                                       z_base_var=z_base_var_v,
                                                       z_score=z_score_v,
                                                       z_orig=z_orig_v)
        # [bs, T, n_kp + 1, projection_dim or 2 * pctx_dim]

        # add entity pos embeddings
        if self.particle_embeddings.shape[2] == 1:
            p_embeddings = self.particle_embeddings.repeat(bs, timestep_horizon, self.n_kp_enc, 1)
        else:
            p_embeddings = self.particle_embeddings.repeat(bs, timestep_horizon, 1, 1)
        if patch_id_embed is not None:
            p_embeddings = p_embeddings + patch_id_embed
        if self.with_bg:
            bg_embeddings = self.bg_embeddings.repeat(bs, timestep_horizon, 1, 1)
            p_embeddings = torch.cat([p_embeddings, bg_embeddings], dim=2)
        particle_projection = particle_projection + p_embeddings

        particles_out = self.pte(particle_projection)
        particles_out = particles_out.view(bs, timestep_horizon, *particles_out.shape[2:])
        # [bs, ts, n, f]

        if self.learned_ctx_token or self.ctx_pool_mode == 'last':
            if self.learned_ctx_token:
                q_particles = self.ctx_token_embeddings.repeat(bs, timestep_horizon, 1, 1)
                q_particles = q_particles.view(bs * timestep_horizon, 1, *q_particles.shape[2:])
                # [bs * t, 1, 1, embed_dim]
                kv_particles = particles_out[:, :, :self.n_kp_enc + 1]  # only fg + bg particles
                kv_particles = kv_particles.reshape(bs * timestep_horizon, 1, *kv_particles.shape[2:])
                # [bs * t, 1, n_particles + 1, embed_dim]
            else:
                # 'last' pooling
                kv_particles, q_particles = particles_out.split([particles_out.shape[2] - 1, 1], dim=2)
            ctx_ca = self.cross_attn_block(q_particles, kv_particles)
            # [bs * t, 1, 1, embed_dim]
            particles_out = torch.cat([kv_particles, ctx_ca], dim=2)
            particles_out = particles_out.view(bs, timestep_horizon, *particles_out.shape[2:])

        particle_decoder_out = self.particle_decoder(particles_out, deterministic=deterministic)
        # unpack
        mu_context = particle_decoder_out['mu_context']
        logvar_context = particle_decoder_out['logvar_context']
        z_context = particle_decoder_out['z_context']

        encode_dict = {'mu_context': mu_context, 'logvar_context': logvar_context, 'z_context': z_context}
        return encode_dict

    def forward(self, z, z_scale, z_obj_on, z_depth, z_features, z_bg_features=None, z_base_var=None,
                z_score=None, patch_id_embed=None, deterministic=False, warmup=False):
        output_dict = self.encode_all(z, z_scale, z_obj_on, z_depth, z_features, z_bg_features, z_base_var, z_score,
                                      patch_id_embed, deterministic=deterministic, warmup=warmup)
        return output_dict


class ParticleContextDecoder(nn.Module):
    def __init__(self, n_particles, input_dim, hidden_dim,
                 context_dist='gauss',
                 context_dim=7,
                 n_ctx_categories=4,
                 n_ctx_classes=4,
                 learned_ctx_token=False,
                 ctx_pool_mode='none',
                 activation='gelu',
                 shared_logvar=False,
                 output_ctx_logvar=True,
                 projection_base_dim=32,
                 conditional=False,
                 cond_dim=512):
        super().__init__()
        # decoder to map back from PTE's inner dim to the particle's original dimension
        self.n_particles = n_particles
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.context_dist = context_dist
        self.n_ctx_categories = n_ctx_categories
        self.n_ctx_classes = n_ctx_classes
        self.ctx_dim = context_dim
        self.learned_ctx_token = learned_ctx_token
        self.ctx_pool_mode = ctx_pool_mode
        self.shared_logvar = shared_logvar
        self.output_ctx_logvar = output_ctx_logvar
        self.projection_base_dim = projection_base_dim
        self.conditional = conditional
        self.cond_dim = cond_dim
        activation_f = nn.GELU if activation == 'gelu' else nn.ReLU
        base_dim = self.projection_base_dim
        ctx_output_dim = self.ctx_dim if (self.shared_logvar or not output_ctx_logvar) else 2 * self.ctx_dim
        if self.shared_logvar and self.output_ctx_logvar:
            self.ctx_logvar = nn.Parameter(torch.zeros(1, 1, self.ctx_dim))

        if self.conditional:
            # cond projection to FiLM parameters
            self.cond_projection = nn.Sequential(nn.Linear(input_dim, self.hidden_dim),
                                                 activation_f(),
                                                 nn.Linear(self.hidden_dim, 2 * self.hidden_dim))
            # init to zeros (=identity)
            nn.init.constant_(self.cond_projection[-1].weight, 0.0)
            nn.init.constant_(self.cond_projection[-1].bias, 0.0)
            self.ctx_ln = RMSNorm(self.hidden_dim)

            # ctx projection
            if self.ctx_pool_mode == 'mlp':
                self.context_projection = nn.Sequential(nn.Linear(self.cond_dim, base_dim),
                                                        activation_f(),
                                                        nn.Flatten(start_dim=-2, end_dim=-1),
                                                        nn.Linear((self.n_particles + 1) * base_dim, self.hidden_dim),
                                                        activation_f())
            else:
                self.context_projection = nn.Sequential(nn.Linear(self.cond_dim, hidden_dim),
                                                        activation_f(),
                                                        ParticlePool(pool_mode=self.ctx_pool_mode, pool_dim=-2),
                                                        nn.Linear(self.hidden_dim, self.hidden_dim))

            self.context_head = nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim),
                                              activation_f(),
                                              nn.Linear(self.hidden_dim, ctx_output_dim))

        else:
            self.cond_projection = self.ctx_ln = nn.Identity()
            if self.ctx_pool_mode == 'mlp':
                ctx_head_in_dim = self.hidden_dim
                self.context_projection = nn.Sequential(nn.Linear(input_dim, base_dim),
                                                        activation_f(),
                                                        nn.Flatten(start_dim=-2, end_dim=-1),
                                                        nn.Linear((self.n_particles + 1) * base_dim, ctx_head_in_dim),
                                                        activation_f())
            else:
                self.context_projection = nn.Identity()
                ctx_head_in_dim = input_dim

            self.context_head = nn.Sequential(ParticlePool(pool_mode=self.ctx_pool_mode, pool_dim=-2),
                                              nn.Linear(ctx_head_in_dim, ctx_output_dim))

        self.init_weights()

    def init_weights(self):
        pass

    def reparameterize(self, mu_context, logvar_context, deterministic=False):
        if self.context_dist == 'beta':
            mu_context = torch.exp(mu_context)
            logvar_context = torch.exp(logvar_context)
            beta_context = Beta(mu_context, logvar_context)
            z_context = beta_context.rsample() if not deterministic else beta_context.mean
        elif self.context_dist == 'categorical':
            # raise NotImplementedError(f'context dist: {self.context_dist}')
            logits = mu_context.view(*mu_context.shape[:-1], self.n_ctx_categories, self.n_ctx_classes)
            # [bs, T, n_p, n_categories, n_classes]
            probs = logits.softmax(dim=-1)  # [bs, T, n_p, n_categories, n_classes]
            if deterministic:
                samples = torch.argmax(probs.view(-1, probs.shape[-1]), dim=-1, keepdim=True)
                samples = F.one_hot(samples.squeeze(-1), num_classes=self.n_ctx_classes)
                samples = samples.view(probs.shape)
                # straight-through
                z_context = samples.detach() + (probs - probs.detach())
                z_context = z_context.view(*mu_context.shape)  # [bs, T, n_p, n_categories * n_classes]
            else:
                samples = torch.multinomial(probs.view(-1, probs.shape[-1]), num_samples=1)
                samples = F.one_hot(samples.squeeze(-1), num_classes=self.n_ctx_classes)
                samples = samples.view(probs.shape)
                # straight-through
                z_context = samples.detach() + (probs - probs.detach())
                z_context = z_context.view(*mu_context.shape)  # [bs, T, n_p, n_categories * n_classes]
        else:
            z_context = reparameterize(mu_context, logvar_context) if not deterministic else mu_context

        return z_context

    def forward(self, x, c=None, deterministic=False):
        # x: [bs, n_particles, input_dim]
        # bs, n_particles, in_dim = x.shape
        bs, ts, n_particles = x.shape[0], x.shape[1], x.shape[2]
        if self.ctx_pool_mode == 'last' or self.learned_ctx_token:
            # same kl-weight as per-particles ctx
            if self.ctx_pool_mode == 'token':
                in_x = x[:, :, -1:].repeat(1, 1, n_particles - 1, 1)
            else:
                in_x = x[:, :, -1:].repeat(1, 1, n_particles, 1)
            if self.conditional and c is not None:
                cond_proj = self.cond_projection(x[:, :, -1])
                scale, shift = cond_proj.chunk(2, dim=-1)

                ctx_proj = self.context_projection(c)
                ctx_feat = self.context_head(modulate(self.ctx_ln(ctx_proj), scale, shift, residual=True))
            else:
                ctx_feat = self.context_head(in_x)  # [bs, T, dim]
        else:
            # consider only fg + bg particles for pooling
            ctx_feat = x[:, :, :self.n_particles + 1]
            if self.conditional and c is not None:
                cond_proj = self.cond_projection(ctx_feat)
                scale, shift = cond_proj.chunk(2, dim=-1)

                ctx_proj = self.context_projection(c)
                if len(ctx_proj.shape) == 3:
                    # [bs, t, d] -> [bs, t, 1, d]
                    ctx_proj = ctx_proj.unsqueeze(-2)
                ctx_feat = self.context_head(modulate(self.ctx_ln(ctx_proj), scale, shift, residual=True))
            else:
                # [bs, ts, hidden_dim]
                ctx_feat = self.context_head(self.context_projection(ctx_feat))

        context_features = ctx_feat
        if self.shared_logvar and self.output_ctx_logvar:
            mu_context = context_features
            if len(mu_context.shape) == 3:
                # [bs, t, dim]
                logvar_context = self.ctx_logvar.repeat(mu_context.shape[0], mu_context.shape[1], 1)
            else:
                logvar_context = self.ctx_logvar.unsqueeze(1).repeat(mu_context.shape[0],
                                                                     mu_context.shape[1],
                                                                     mu_context.shape[2],
                                                                     1)
        elif not self.output_ctx_logvar:
            mu_context = context_features
            logvar_context = None
        else:
            mu_context, logvar_context = torch.chunk(context_features, 2, dim=-1)

        z_context = self.reparameterize(mu_context, logvar_context, deterministic)
        decoder_out = {'mu_context': mu_context, 'logvar_context': logvar_context, 'z_context': z_context}

        return decoder_out


class ParticleEncoder(nn.Module):
    def __init__(self, cdim=3, image_size=64,
                 pad_mode='replicate', dropout=0.0, n_kp_per_patch=1, n_kp_prior=20,
                 patch_size=16, n_kp_enc=20, n_kp_dec=None, learned_feature_dim=16,
                 kp_range=(-1, 1), kp_activation="tanh", anchor_s=0.25,
                 use_resblock=True, embed_init_std=0.2, projection_dim=128, timestep_horizon=1,
                 filtering_heuristic='none', obj_ch_mult_prior=(1, 2),
                 obj_ch_mult=(1, 2, 3), obj_base_ch=32, obj_final_cnn_ch=32, num_res_blocks=2,
                 interaction_features=False, interaction_obj_on=False, interaction_depth=True,
                 temporal_interaction=True, cnn_mid_blocks=False, mlp_hidden_dim=256,
                 embed_prior_patch_pos=False, add_particle_temp_embed=False,
                 features_dist='gauss', n_fg_categories=8, n_fg_classes=4,
                 use_null_features_embed=True, obj_on_min=1e-4, obj_on_max=100.0, warmup_n_kp_ratio=0.35,
                 # initialization
                 init_zero_bias=True,  # zero bias for conv and linear layers
                 init_ssm_last_layer=True,  # spatial softmax initialization
                 init_conv_layers=True,  # initialize conv layers with normal dist
                 init_conv_fg_std=0.02,  # std for conv fg normal dist
                 ):
        super(ParticleEncoder, self).__init__()
        """
        DLP Foreground Module – Extracts objects from an image using keypoints and learned features. 
        Combines posterior CNN for full image processing and prior CNN for patch-based keypoint proposals.
        
        Args:
        cdim (int, default=3): Number of channels in the input image.
        image_size (int, default=64): Resolution of the input image (assumes square images).
        pad_mode (str, default='replicate'): Padding mode for CNNs, options are 'zeros' or 'replicate'.
        dropout (float, default=0.0): Dropout rate for CNNs (not used in practice).
        n_kp_per_patch (int, default=1): Number of keypoints proposed per patch.
        n_kp_prior (int, default=20): Number of keypoints filtered from prior proposals.
        patch_size (int, default=16): Size of patches for the prior keypoint proposal network.
        n_kp_enc (int, default=20): Number of posterior keypoints to learn.
        n_kp_dec (int, optional): Number of keypoints for decoder (if different from encoder).
        learned_feature_dim (int, default=16): Dimensionality of latent visual features for glimpses.
        kp_range (tuple, default=(-1, 1)): Range for keypoints; options are (-1, 1) or (0, 1).
        kp_activation (str, default='tanh'): Activation function for keypoints; 'tanh' for range (-1, 1), 'sigmoid' for range (0, 1).
        anchor_s (float, default=0.25): Glimpse size as a ratio of image size (e.g., 0.25 → glimpse size is 0.25 * image_size).
        use_resblock (bool, default=True): Whether to use residual blocks in CNNs.
        embed_init_std (float, default=0.2): Standard deviation for initializing learned tokens.
        projection_dim (int, default=128): Dimensionality of embeddings for transformer input.
        timestep_horizon (int, default=1): Maximum timesteps the model processes at once.
        filtering_heuristic (str, default='none'): Method for filtering prior keypoints. Options: 'distance', 'variance', 'random', 'none'.
        obj_ch_mult (tuple, default=(1, 2, 3)): Multiplicative factors for object feature channels at each CNN stage.
        obj_base_ch (int, default=32): Base number of channels in object feature extractor.
        obj_final_cnn_ch (int, default=32): Number of channels in the final object CNN layer.
        num_res_blocks (int, default=2): Number of residual blocks in object feature extractor.
        interaction_features (bool, default=False): Whether to compute interaction-based features.
        interaction_obj_on (bool, default=False): Whether to include "object-on" features for interactions.
        interaction_depth (bool, default=True): Whether to compute depth information for interactions.
        temporal_interaction (bool, default=True): Whether to model temporal interactions between features.
        cnn_mid_blocks (bool, default=False): Whether to include intermediate blocks in the CNN.
        mlp_hidden_dim (int, default=256): Hidden dimensionality for MLP layers.
        embed_prior_patch_pos (bool, default=False): Whether to embed positional information for prior patches.
        add_particle_temp_embed (bool, default=False): Whether to add temporal embeddings to particles.
        features_dist (str, default='gauss'): Distribution type for keypoint features. Options: 'gauss'.
        n_fg_categories (int, default=8): Number of foreground categories for classification.
        n_fg_classes (int, default=4): Number of foreground classes for classification.
        use_null_features_embed (bool, default=True): Whether to use a learned embedding for filtered-out particles.
        obj_on_min (float, default=1e-4): Minimum concentration value in Beta dist for transparency" probabilities.
        obj_on_max (float, default=100.0): Maximum concentration value in Beta dist for transparency" probabilities.
        """
        self.image_size = image_size
        self.dropout = dropout
        self.kp_range = kp_range
        self.n_kp_per_patch = n_kp_per_patch
        self.n_kp_enc = n_kp_enc
        self.n_kp_dec = self.n_kp_enc if n_kp_dec is None else n_kp_dec
        self.n_kp_prior = n_kp_prior
        self.kp_activation = kp_activation
        self.patch_size = patch_size
        self.anchor_patch_s = patch_size / image_size
        self.features_dim = int(image_size // (2 ** (len(obj_ch_mult) - 1)))
        self.learned_feature_dim = learned_feature_dim
        self.features_dist = features_dist
        self.n_fg_categories = n_fg_categories
        self.n_fg_classes = n_fg_classes
        assert learned_feature_dim > 0, "learned_feature_dim must be greater than 0"
        self.anchor_s = anchor_s
        self.obj_patch_size = np.round(anchor_s * (image_size - 1)).astype(int)
        self.cdim = cdim
        self.use_resblock = use_resblock
        self.embed_init_std = embed_init_std
        self.projection_dim = projection_dim
        self.timestep_horizon = (timestep_horizon + 1) if timestep_horizon > 1 else 1
        self.num_patches = int((image_size // self.patch_size) ** 2)
        self.interaction_features = interaction_features
        self.interaction_depth = interaction_depth
        self.interaction_obj_on = interaction_obj_on
        self.temporal_interaction = temporal_interaction
        self.add_particle_temp_embed = add_particle_temp_embed
        self.cnn_mid_blocks = cnn_mid_blocks
        self.mlp_hidden_dim = mlp_hidden_dim
        self.embed_prior_patch_pos = embed_prior_patch_pos
        self.obj_on_min = obj_on_min
        self.obj_on_max = obj_on_max
        self.use_null_features_embed = use_null_features_embed
        self.warmup_n_kp_ratio = warmup_n_kp_ratio
        # initialization
        self.init_zero_bias = init_zero_bias  # zero bias for conv and linear layers
        self.init_ssm_last_layer = init_ssm_last_layer  # spatial softmax initialization
        self.init_conv_layers = init_conv_layers  # initialize conv layers with normal dist
        self.init_conv_fg_std = init_conv_fg_std  # std for conv fg normal dist

        self.prior_encoder = DLPPrior(cdim=cdim, image_size=image_size, n_kp=self.n_kp_per_patch,
                                      patch_size=patch_size, kp_range=kp_range, pad_mode=pad_mode,
                                      n_kp_prior=n_kp_prior,
                                      filtering_heuristic=filtering_heuristic,
                                      ch_mult=obj_ch_mult_prior, base_ch=obj_base_ch, num_res_blocks=num_res_blocks,
                                      use_resblock=use_resblock, cnn_mid_blocks=cnn_mid_blocks,
                                      init_ssm_last_layer=init_ssm_last_layer, init_conv_layers=init_conv_layers,
                                      init_conv_fg_std=init_conv_fg_std)

        # attribute encoder - anchor (z_a), offset (z_o), scale (z_s)
        anchor_s_att = patch_size / image_size
        self.particle_attribute_enc = ParticleAttributeEncoder(anchor_size=anchor_s, image_size=image_size,
                                                               n_particles=self.n_kp_prior,
                                                               margin=0, ch=cdim,
                                                               kp_activation=kp_activation,
                                                               use_resblock=use_resblock,
                                                               max_offset=1.0,
                                                               pad_mode=pad_mode, depth=not self.interaction_depth,
                                                               obj_on=not self.interaction_obj_on,
                                                               ch_mult=obj_ch_mult, base_ch=obj_base_ch,
                                                               final_cnn_ch=obj_final_cnn_ch,
                                                               num_res_blocks=num_res_blocks,
                                                               cnn_mid_blocks=cnn_mid_blocks,
                                                               hidden_dim=mlp_hidden_dim,
                                                               timestep_horizon=self.timestep_horizon,
                                                               add_particle_temp_embed=add_particle_temp_embed,
                                                               init_std=embed_init_std,
                                                               obj_on_min=self.obj_on_min,
                                                               obj_on_max=self.obj_on_max,
                                                               init_zero_bias=init_zero_bias,
                                                               init_conv_layers=init_conv_layers,
                                                               init_conv_fg_std=init_conv_fg_std)
        # appearance encoder - visual features encoder (z_f)
        output_logvar = (not self.interaction_features and self.features_dist != 'categorical')
        self.particle_features_enc = ParticleFeaturesEncoder(anchor_s, learned_feature_dim,
                                                             image_size,
                                                             margin=0, pad_mode=pad_mode,
                                                             ch_mult=obj_ch_mult, base_ch=obj_base_ch,
                                                             final_cnn_ch=obj_final_cnn_ch,
                                                             num_res_blocks=num_res_blocks,
                                                             output_logvar=output_logvar,
                                                             use_resblock=use_resblock, cnn_mid_blocks=cnn_mid_blocks,
                                                             hidden_dim=mlp_hidden_dim,
                                                             timestep_horizon=self.timestep_horizon,
                                                             add_particle_temp_embed=add_particle_temp_embed,
                                                             init_zero_bias=init_zero_bias,
                                                             init_conv_layers=init_conv_layers,
                                                             init_conv_fg_std=init_conv_fg_std
                                                             )
        # embed the source patch of the particles
        if self.embed_prior_patch_pos:
            self.patch_id_embed = nn.Parameter(self.embed_init_std * torch.randn(1, self.n_kp_prior, mlp_hidden_dim))
        else:
            self.patch_id_embed = None
        patch_centers = self.prior_encoder.get_patch_centers().unsqueeze(0) * (
                self.kp_range[1] - self.kp_range[0]) + self.kp_range[0]
        # append null particle
        patch_centers = torch.cat([patch_centers, torch.zeros(1, 1, 2)], dim=1)
        if self.n_kp_enc != self.n_kp_dec and self.interaction_features and self.use_null_features_embed:
            self.null_feature_embed = nn.Parameter(self.embed_init_std * torch.randn(1, 1, self.learned_feature_dim))
        self.register_buffer('patch_centers', patch_centers)
        self.register_buffer('mu_scale_prior', torch.tensor(np.log(self.anchor_s / (1 - self.anchor_s + 1e-5))))
        self.init_weights()

    def init_weights(self):
        self.prior_encoder.init_weights()

    def encode_prior(self, x):
        return self.prior_encoder(x)

    def encode_pos_scale_with_prior(self, x, deterministic=False, warmup=False, timesteps=None):
        batch_size, ch, h, w = x.shape
        kp_p, var_kp = self.encode_prior(x)
        # kp_init: [batch_size, n_kp, 2] in [-1, 1]
        kp_init = kp_p
        # 0. create or filter anchors
        if kp_init is None:
            # randomly sample n_kp_enc kp
            mu = torch.rand(batch_size, self.n_kp_prior, 2, device=x.device) * 2 - 1  # in [-1, 1]
        else:
            mu = kp_init
        logvar = torch.zeros_like(mu)
        z_base = mu + 0.0 * logvar  # deterministic value for chamfer-kl
        # 1. posterior offsets and scale, it is okay of scale_prev is None
        particle_stats_dict = self.particle_attribute_enc(x, z_base, timesteps=timesteps, deterministic=deterministic)

        mu_offset = particle_stats_dict['mu']
        logvar_offset = particle_stats_dict['logvar']
        mu_scale = particle_stats_dict['mu_scale']
        logvar_scale = particle_stats_dict['logvar_scale']
        if not self.interaction_obj_on:
            lobj_on_a = particle_stats_dict['lobj_on_a']
            lobj_on_b = particle_stats_dict['lobj_on_b']
            obj_on_a = particle_stats_dict['obj_on_a']
            obj_on_b = particle_stats_dict['obj_on_b']
            mu_obj_on = particle_stats_dict['mu_obj_on']
            z_obj_on = particle_stats_dict['z_obj_on']
        else:
            obj_on_a = obj_on_b = z_obj_on = mu_obj_on = None
        if not self.interaction_depth:
            mu_depth = particle_stats_dict['mu_depth']
            logvar_depth = particle_stats_dict['logvar_depth']
            if deterministic:
                z_depth = mu_depth
            else:
                z_depth = reparameterize(mu_depth, logvar_depth)
        else:
            mu_depth = logvar_depth = z_depth = None

        # final position
        mu_tot = z_base + mu_offset
        logvar_tot = logvar_offset
        mu_scale = self.mu_scale_prior + mu_scale

        # reparameterize
        if deterministic:
            z_offset = mu_offset
            z_scale = mu_scale
        else:
            z_offset = reparameterize(mu_offset, logvar_offset)
            z_scale = reparameterize(mu_scale, logvar_scale)

        z = z_base + z_offset
        z_base_var = var_kp.detach()
        confidence_score = particle_stats_dict['logvar'].detach()
        z_base_var = torch.cat([z_base_var, confidence_score], dim=-1)
        z_base_id = torch.arange(z_base.shape[-2], device=z_base.device)[None, :, None]  # [1, n_patches, 1]
        z_base_id = z_base_id.repeat(z_base.shape[0], 1, 1)  # [bs, n_patches, 1]

        if self.embed_prior_patch_pos:
            patch_id_embed = self.patch_id_embed.repeat(mu_tot.shape[0], 1, 1)
        else:
            patch_id_embed = None

        mu_score = (z_base_var.sum(-1, keepdim=True) / 30) * 2 - 1  # [bs * T, n_patches, 1]
        logvar_score = math.log(0.2 ** 2) * torch.ones_like(mu_score)  # 0.1, original without normalization: 1.0
        z_score = mu_score

        # variance filtering
        total_var = z_base_var.sum(-1)  # [bs * T, n_kp]
        # for single-image settings (self.timestep_horizon == 1), we can filter in the encoder
        # if self.n_kp_enc < self.n_kp_prior or (warmup and self.timestep_horizon == 1):
        if self.n_kp_enc < self.n_kp_prior:
            n_filter = self.n_kp_enc if not warmup else min(self.n_kp_enc,
                                                            int(self.warmup_n_kp_ratio * self.n_kp_prior))
            _, embed_ind = torch.topk(total_var, k=n_filter, dim=-1, largest=False)
            # make selection
            batch_ind = torch.arange(batch_size, device=x.device)[:, None]
            mu_tot = mu_tot[batch_ind, embed_ind]  # [bs * T, n_kp_enc, 2]
            z_base = z_base[batch_ind, embed_ind]  # [bs * T, n_kp_enc, 2]
            z_base_var = z_base_var[batch_ind, embed_ind]  # [bs * T, n_kp_enc, 2]
            z_base_id = z_base_id[batch_ind, embed_ind]  # [bs * T, n_kp_enc, 1]
            mu_offset = mu_offset[batch_ind, embed_ind]  # [bs * T, n_kp_enc, 2]
            logvar_offset = logvar_offset[batch_ind, embed_ind]  # [bs * T, n_kp_enc, 2]
            z = z[batch_ind, embed_ind]  # [bs * T, n_kp_enc, 2]
            z_offset = z_offset[batch_ind, embed_ind]  # [bs * T, n_kp_enc, 2]
            z_scale = z_scale[batch_ind, embed_ind]  # [bs * T, n_kp_enc, 2]
            mu_scale = mu_scale[batch_ind, embed_ind]  # [bs * T, n_kp_enc, 2]
            mu_score = mu_score[batch_ind, embed_ind]  # [bs * T, n_kp_enc, 1]
            logvar_score = logvar_score[batch_ind, embed_ind]  # [bs * T, n_kp_enc, 1]
            z_score = z_score[batch_ind, embed_ind]  # [bs * T, n_kp_enc, 1]
            if logvar_scale is not None:
                logvar_scale = logvar_scale[batch_ind, embed_ind]  # [bs * T, n_kp_enc, 2]
            if not self.interaction_obj_on:
                obj_on_a = obj_on_a[batch_ind, embed_ind]  # [bs * T, n_kp_enc, 1]
                obj_on_b = obj_on_b[batch_ind, embed_ind]  # [bs * T, n_kp_enc, 1]
                mu_obj_on = mu_obj_on[batch_ind, embed_ind]  # [bs * T, n_kp_enc, 1]
                z_obj_on = z_obj_on[batch_ind, embed_ind]  # [bs * T, n_kp_enc, 1]
            if not self.interaction_depth:
                z_depth = z_depth[batch_ind, embed_ind]  # [bs * T, n_kp_enc, 1]
                mu_depth = mu_depth[batch_ind, embed_ind]  # [bs * T, n_kp_enc, 1]
                logvar_depth = logvar_depth[batch_ind, embed_ind]  # [bs * T, n_kp_enc, 1]
            if self.embed_prior_patch_pos:
                patch_id_embed = patch_id_embed[batch_ind, embed_ind]

        out_dict = {'mu': mu, 'logvar': logvar, 'z_base': z_base, 'z': z, 'mu_tot': mu_tot,
                    'patch_id_embed': patch_id_embed,
                    'mu_scale': mu_scale, 'logvar_scale': logvar_scale, 'z_scale': z_scale,
                    'mu_depth': mu_depth, 'logvar_depth': logvar_depth, 'z_depth': z_depth,
                    'mu_offset': mu_offset, 'logvar_offset': logvar_offset, 'z_offset': z_offset,
                    'kp_p': kp_p, 'var_kp': var_kp, 'z_base_var': z_base_var, 'total_var': total_var,
                    'obj_on_a': obj_on_a, 'obj_on_b': obj_on_b, 'z_obj_on': z_obj_on, 'mu_obj_on': mu_obj_on,
                    'z_base_id': z_base_id, 'mu_score': mu_score, 'logvar_score': logvar_score, 'z_score': z_score}
        return out_dict

    def encode_appearance(self, x, z, z_scale, deterministic=False, timesteps=None, obj_on=None):
        # 2. posterior attributes: obj_on, depth and visual features
        obj_enc_out = self.particle_features_enc(x, z, z_scale=z_scale, timesteps=timesteps)

        mu_features = obj_enc_out['mu_features']
        logvar_features = obj_enc_out['logvar_features']
        cropped_objects = obj_enc_out['cropped_objects']

        if obj_on is not None:
            z_gate = torch.where(obj_on > 0.2, 1.0, 0.0)
            mu_features = z_gate * mu_features + (1 - z_gate) * self.null_feature_embed

        if not self.interaction_features:
            # reparameterize
            if self.features_dist == 'categorical':
                logits = mu_features.view(*mu_features.shape[:-1], self.n_fg_categories, self.n_fg_classes)
                # [bs, T, n_p, n_categories, n_classes]
                probs = logits.softmax(dim=-1)  # [bs, T, n_p, n_categories, n_classes]
                if deterministic:
                    samples = torch.argmax(probs.view(-1, probs.shape[-1]), dim=-1, keepdim=True)
                    samples = F.one_hot(samples.squeeze(-1), num_classes=self.n_fg_classes)
                    samples = samples.view(probs.shape)
                    # straight-through
                    z_features = samples.detach() + (probs - probs.detach())
                    z_features = z_features.view(*mu_features.shape)  # [bs, T, n_p, n_categories * n_classes]
                else:
                    samples = torch.multinomial(probs.view(-1, probs.shape[-1]), num_samples=1)
                    samples = F.one_hot(samples.squeeze(-1), num_classes=self.n_fg_classes)
                    samples = samples.view(probs.shape)
                    # straight-through
                    z_features = samples.detach() + (probs - probs.detach())
                    z_features = z_features.view(*mu_features.shape)  # [bs, T, n_p, n_categories * n_classes]
            else:
                z_features = reparameterize(mu_features, logvar_features) if not deterministic else mu_features
        else:
            z_features = mu_features

        out_dict = {'mu_features': mu_features, 'logvar_features': logvar_features, 'z_features': z_features,
                    'cropped_objects': cropped_objects}
        return out_dict

    def encode_all(self, x, deterministic=False, warmup=False):
        # make sure x is [bs, T, ch, h, w]
        if len(x.shape) == 4:
            # that means x: [bs, ch, h, w]
            x = x.unsqueeze(1)  # -> [bs, T=1, ch, h, w]
        bs, timestep_horizon, ch, h, w = x.shape
        x = x.view(bs * timestep_horizon, *x.shape[2:])  # [bs * T, ch, h, w]
        # encode particles position and scale
        stage1_dict = self.encode_pos_scale_with_prior(x, deterministic=deterministic, warmup=warmup,
                                                       timesteps=timestep_horizon)
        # unpack
        kp_p = stage1_dict['kp_p']
        var_kp = stage1_dict['var_kp']
        z_base_var = stage1_dict['z_base_var']
        total_var = stage1_dict['total_var']
        patch_id_embed = stage1_dict['patch_id_embed']

        z_base = stage1_dict['z_base']
        mu_offset = stage1_dict['mu_offset']
        logvar_offset = stage1_dict['logvar_offset']
        z_offset = stage1_dict['z_offset']
        mu_tot = stage1_dict['mu_tot']
        z = stage1_dict['z']
        mu_scale = stage1_dict['mu_scale']
        logvar_scale = stage1_dict['logvar_scale']
        z_scale = stage1_dict['z_scale']
        # the following may be None if they are modeled by the interaction module
        mu_depth = stage1_dict['mu_depth']
        logvar_depth = stage1_dict['logvar_depth']
        z_depth = stage1_dict['z_depth']
        obj_on_a = stage1_dict['obj_on_a']
        obj_on_b = stage1_dict['obj_on_b']
        mu_obj_on = stage1_dict['mu_obj_on']
        z_obj_on = stage1_dict['z_obj_on']

        mu_score = stage1_dict['mu_score']
        logvar_score = stage1_dict['logvar_score']
        z_score = stage1_dict['z_score']

        if self.n_kp_enc != self.n_kp_dec and self.interaction_features and self.use_null_features_embed:
            total_var = z_base_var.sum(-1)
            n_filter = self.n_kp_dec if not warmup else min(self.n_kp_dec, int(self.warmup_n_kp_ratio * self.n_kp_enc))
            _, embed_ind = torch.topk(total_var, k=n_filter, dim=-1, largest=False)
            # make selection
            batch_ind = torch.arange(z.shape[0], device=z.device)[:, None]
            z_app = z[batch_ind, embed_ind].contiguous()
            z_scale_app = z_scale[batch_ind, embed_ind].contiguous()
            stage2_dict = self.encode_appearance(x, z_app, z_scale_app, deterministic=deterministic,
                                                 timesteps=timestep_horizon, obj_on=None)
            # unpack
            cropped_objects = stage2_dict['cropped_objects']
            mu_features_app = stage2_dict['mu_features']
            logvar_features = stage2_dict['logvar_features']  # None
            z_features_app = stage2_dict['z_features']

            mu_features = self.null_feature_embed.repeat(z.shape[0], self.n_kp_enc, 1)
            mu_features[batch_ind, embed_ind] = mu_features_app

            z_features = mu_features

        else:
            stage2_dict = self.encode_appearance(x, z, z_scale, deterministic=deterministic, timesteps=timestep_horizon,
                                                 obj_on=None)
            # unpack
            cropped_objects = stage2_dict['cropped_objects']
            mu_features = stage2_dict['mu_features']
            logvar_features = stage2_dict['logvar_features']
            z_features = stage2_dict['z_features']

        # reshape to [bs, T, ...]
        z_base = z_base.view(bs, timestep_horizon, *z_base.shape[1:])
        z_base_var = z_base_var.view(bs, timestep_horizon, *z_base_var.shape[1:])
        if patch_id_embed is not None:
            patch_id_embed = patch_id_embed.view(bs, timestep_horizon, *patch_id_embed.shape[1:])
        mu_offset = mu_offset.view(bs, timestep_horizon, *mu_offset.shape[1:])
        logvar_offset = logvar_offset.view(bs, timestep_horizon, *logvar_offset.shape[1:])
        z_offset = z_offset.view(bs, timestep_horizon, *z_offset.shape[1:])
        mu_tot = mu_tot.view(bs, timestep_horizon, *mu_tot.shape[1:])
        z = z.view(bs, timestep_horizon, *z.shape[1:])
        mu_scale = mu_scale.view(bs, timestep_horizon, *mu_scale.shape[1:])
        if logvar_scale is not None:
            logvar_scale = logvar_scale.view(bs, timestep_horizon, *logvar_scale.shape[1:])
        z_scale = z_scale.view(bs, timestep_horizon, *z_scale.shape[1:])
        if not self.interaction_features:
            mu_features = mu_features.view(bs, timestep_horizon, *mu_features.shape[1:])
            logvar_features = logvar_features.view(bs, timestep_horizon, *logvar_features.shape[1:])
        z_features = z_features.view(bs, timestep_horizon, *z_features.shape[1:])
        cropped_objects = cropped_objects.view(-1, *cropped_objects.shape[2:])
        if not self.interaction_depth:
            mu_depth = mu_depth.view(bs, timestep_horizon, *mu_depth.shape[1:])
            logvar_depth = logvar_depth.view(bs, timestep_horizon, *logvar_depth.shape[1:])
            z_depth = z_depth.view(bs, timestep_horizon, *z_depth.shape[1:])
        if not self.interaction_obj_on:
            obj_on_a = obj_on_a.view(bs, timestep_horizon, *obj_on_a.shape[1:])
            obj_on_b = obj_on_b.view(bs, timestep_horizon, *obj_on_b.shape[1:])
            mu_obj_on = mu_obj_on.view(bs, timestep_horizon, *mu_obj_on.shape[1:])
            z_obj_on = z_obj_on.view(bs, timestep_horizon, *z_obj_on.shape[1:])
        mu_score = mu_score.view(bs, timestep_horizon, *mu_score.shape[1:])
        logvar_score = logvar_score.view(bs, timestep_horizon, *logvar_score.shape[1:])
        z_score = z_score.view(bs, timestep_horizon, *z_score.shape[1:])

        encode_dict = {'mu_anchor': z_base, 'logvar_anchor': torch.zeros_like(z_base), 'z_base': z_base, 'z': z,
                       'mu_offset': mu_offset, 'logvar_offset': logvar_offset, 'z_offset': z_offset, 'mu_tot': mu_tot,
                       'mu_features': mu_features, 'logvar_features': logvar_features, 'z_features': z_features,
                       'cropped_objects': cropped_objects.detach(), 'patch_id_embed': patch_id_embed,
                       'obj_on_a': obj_on_a, 'obj_on_b': obj_on_b, 'z_obj_on': z_obj_on, 'mu_obj_on': mu_obj_on,
                       'mu_depth': mu_depth, 'logvar_depth': logvar_depth, 'z_depth': z_depth,
                       'mu_scale': mu_scale, 'logvar_scale': logvar_scale, 'z_scale': z_scale,
                       'kp_p': kp_p, 'var_kp': var_kp, 'z_base_var': z_base_var, 'mu_score': mu_score,
                       'logvar_score': logvar_score, 'z_score': z_score}
        return encode_dict

    def forward(self, x, deterministic=False, warmup=False):
        output_dict = self.encode_all(x, deterministic, warmup)
        return output_dict


class DLPEncoder(nn.Module):
    def __init__(self,
                 # Input configuration
                 cdim=3,  # Number of input image channels
                 image_size=64,  # Input image size (assumed square)
                 n_views=1,  # number of input views (e.g., multiple cameras)
                 pad_mode='replicate',  # Padding mode for CNNs
                 dropout=0.0,  # Dropout rate (not typically used)

                 # Keypoint and patch configuration
                 n_kp_per_patch=1,  # Number of keypoints per patch
                 n_kp_prior=20,  # Number of keypoints to filter from proposals
                 patch_size=16,  # Patch size for keypoint proposal network
                 n_kp_enc=20,  # Number of posterior keypoints to learn
                 n_kp_dec=None,  # Number of keypoints for decoder (if different from encoder)
                 warmup_n_kp_ratio=0.35,
                 mask_bg_in_enc=True,  # before encoding the bg, mask with the particles' obj_on

                 # Feature dimensions
                 learned_feature_dim=16,  # Dimension of learned visual features
                 learned_bg_feature_dim=16,  # Dimension of background features
                 kp_range=(-1, 1),  # Range for keypoint coordinates
                 kp_activation="tanh",  # Activation for keypoint coordinates
                 anchor_s=0.25,  # Glimpse size ratio

                 # Network architecture
                 use_resblock=True,  # Use residual blocks
                 embed_init_std=0.02,  # Standard deviation for embedding initialization
                 projection_dim=128,  # Embedding dimension for transformer

                 # Transformer configuration
                 timestep_horizon=1,  # Maximum timesteps to process at once
                 pte_layers=1,  # Number of particle transformer encoder layers
                 pte_heads=1,  # Number of particle transformer encoder heads
                 context_dim=16,  # Context latent dimension
                 filtering_heuristic='none',  # Method to filter prior keypoints
                 attn_norm_type='rms',  # Normalization type for attention

                 # Object encoder configuration
                 obj_ch_mult_prior=(1, 2,),  # Channel multipliers for prior patch encoder (kp proposals)
                 obj_ch_mult=(1, 2, 3),  # Channel multipliers for object encoder
                 obj_base_ch=32,  # Base channels for object encoder
                 obj_final_cnn_ch=32,  # Final CNN channels for object encoder
                 cnn_mid_blocks=False,  # Use middle blocks in CNN
                 mlp_hidden_dim=256,  # Hidden dimension for MLPs
                 pte_inner_dim=256,  # Inner dimension for particle transformer

                 # Background decoder configuration
                 bg_ch_mult=(1, 2, 3),  # Channel multipliers for background encoder
                 bg_base_ch=32,  # Base channels for background encoder
                 bg_final_cnn_ch=32,  # Final CNN channels for background encoder
                 num_res_blocks=2,  # Number of residual blocks

                 # Interaction configuration
                 ctx_pool_mode='none',  # Mode for pooling context features
                 interaction_depth=True,  # Enable depth interaction between particles
                 interaction_obj_on=False,  # Enable transparency interaction
                 interaction_features=True,  # Enable feature interaction
                 particle_score=False,  # Use particle confidence scores

                 # Embedding options
                 add_particle_temp_embed=False,  # Add temporal embeddings to particles
                 particle_positional_embed=True,  # Add positional embeddings to particles

                 # Context modeling
                 ctx_enc=None,
                 causal_ctx=True,  # Use causal attention for context
                 pte_ctx_layers=1,  # Number of context transformer layers
                 pte_ctx_heads=1,  # Number of context transformer heads
                 ctx_dist='gauss',  # Distribution type for context
                 n_ctx_categories=4,  # Number of context categories
                 n_ctx_classes=4,  # Number of context classes per category
                 global_ctx_pool=False,  # learn global latent context in addition to per-particle context
                 pool_ctx_dim=256,  # pool dimension for the global ctx latent
                 n_pool_ctx_categories=8,  # Number of global context categories (if categorical)
                 n_pool_ctx_classes=4,  # Number of global context classes per category
                 global_local_fuse_mode='none',  # concatenate/add global and local z_ctx to condition the dynamics
                 condition_local_on_global=True,  # condition z_context on z_context_global

                 # Distribution configuration
                 features_dist='gauss',  # Distribution type for features
                 n_fg_categories=8,  # Number of foreground categories, 'categorical' dist
                 n_fg_classes=4,  # Number of foreground classes per category, 'categorical' dist
                 n_bg_categories=4,  # Number of background categories, 'categorical' dist
                 n_bg_classes=4,  # Number of background classes per category, 'categorical' dist
                 obj_on_min=1e-4,  # Minimum concentration in Beta dist transparency value
                 obj_on_max=100,  # Maximum concentration in Beta dist transparency value
                 use_z_orig=True,  # Use original patch center coordinates as features

                 # initialization
                 init_zero_bias=True,  # zero bias for conv and linear layers
                 init_ssm_last_layer=True,  # spatial softmax initialization
                 init_conv_layers=True,  # initialize conv layers with normal dist
                 init_conv_fg_std=0.02,  # std for conv fg normal dist
                 init_conv_bg_std=0.005,  # std for conv bg normal dist (<fg -> prioritize fg in learning)
                 ):
        """
        DLP Encoder Module

        A neural network module that extracts object-centric representations from images using
        the Deep Latent Particles (DLP) approach. This encoder processes images to identify
        and represent objects as particles with learned attributes.

        Args:
            cdim (int): Number of input image channels. Defaults to 3.
            image_size (int): Size of input images (assumed square). Defaults to 64.
            pad_mode (str): Padding mode for CNNs ('zeros' or 'replicate'). Defaults to 'replicate'.
            dropout (float): Dropout rate for CNNs (typically unused). Defaults to 0.0.
            n_kp_per_patch (int): Number of keypoints to extract per patch. Defaults to 1.
            n_kp_prior (int): Number of keypoints to filter from proposals. Defaults to 20.
            patch_size (int): Size of patches for keypoint proposal network. Defaults to 16.
            n_kp_enc (int): Number of posterior keypoints to learn. Defaults to 20.
            n_kp_dec (Optional[int]): Number of keypoints for decoder. If None, equals n_kp_enc. Defaults to None.
            learned_feature_dim (int): Dimension of learned visual features. Defaults to 16.
            learned_bg_feature_dim (int): Dimension of background features. Defaults to 16.
            kp_range (tuple): Range for keypoint coordinates, either (-1, 1) or (0, 1). Defaults to (-1, 1).
            kp_activation (str): Activation for keypoint coordinates ('tanh' or 'sigmoid'). Defaults to 'tanh'.
            anchor_s (float): Glimpse size as ratio of image_size. Defaults to 0.25.
            use_resblock (bool): Use residual blocks in network. Defaults to True.
            embed_init_std (float): Standard deviation for embedding initialization. Defaults to 0.02.
            projection_dim (int): Embedding dimension for transformer. Defaults to 128.
            timestep_horizon (int): Maximum number of timesteps to process at once. Defaults to 1.
            pte_layers (int): Number of particle transformer encoder layers. Defaults to 1.
            pte_heads (int): Number of particle transformer encoder heads. Defaults to 1.
            context_dim (int): Dimension of context latent space. Defaults to 16.
            filtering_heuristic (str): Method to filter prior keypoints. Defaults to 'none'.
            attn_norm_type (str): Normalization type for attention blocks. Defaults to 'rms'.
            obj_ch_mult_prior (tuple): Channel multipliers for prior patch encoder. Defaults to (1, 2, 3).
            obj_ch_mult (tuple): Channel multipliers for object encoder. Defaults to (1, 2, 3).
            obj_base_ch (int): Base channels for object encoder. Defaults to 32.
            obj_final_cnn_ch (int): Final CNN channels for object encoder. Defaults to 32.
            cnn_mid_blocks (bool): Use middle blocks in CNN. Defaults to False.
            mlp_hidden_dim (int): Hidden dimension for MLPs. Defaults to 256.
            pte_inner_dim (int): Inner dimension for particle transformer. Defaults to 256.
            bg_ch_mult (tuple): Channel multipliers for background encoder. Defaults to (1, 2, 3).
            bg_base_ch (int): Base channels for background encoder. Defaults to 32.
            bg_final_cnn_ch (int): Final CNN channels for background encoder. Defaults to 32.
            num_res_blocks (int): Number of residual blocks. Defaults to 2.
            ctx_pool_mode (str): Mode for pooling context features. Defaults to 'none'.
            interaction_depth (bool): Enable modeling depth by interaction between particles. Defaults to True.
            interaction_obj_on (bool): Enable modeling transparency by interaction. Defaults to False.
            interaction_features (bool): Enable modeling features by interaction. Defaults to True.
            particle_score (bool): Use particle confidence scores. Defaults to False.
            add_particle_temp_embed (bool): Add temporal embeddings to particles. Defaults to False.
            particle_positional_embed (bool): Add positional embeddings to particles. Defaults to True.
            causal_ctx (bool): Use causal attention for context. Defaults to True.
            pte_ctx_layers (int): Number of context transformer layers. Defaults to 1.
            pte_ctx_heads (int): Number of context transformer heads. Defaults to 1.
            ctx_dist (str): Distribution type for context ('gauss' or 'categorical'). Defaults to 'gauss'.
            n_ctx_categories (int): Number of context categories if categorical. Defaults to 4.
            n_ctx_classes (int): Number of context classes per category. Defaults to 4.
            features_dist (str): Distribution type for features ('gauss' or 'categorical'). Defaults to 'gauss'.
            n_fg_categories (int): Number of foreground categories if categorical. Defaults to 8.
            n_fg_classes (int): Number of foreground classes per category. Defaults to 4.
            n_bg_categories (int): Number of background categories if categorical. Defaults to 4.
            n_bg_classes (int): Number of background classes per category. Defaults to 4.
            obj_on_min (float): Minimum concentration value in Beta dist for transparency value. Defaults to 1e-4.
            obj_on_max (float): Maximum concentration value in Beta dist transparency value. Defaults to 100.
            use_z_orig (bool): Use original patch center coordinates. Defaults to True.

        Notes:
            The encoder operates in several stages:
            1. Patch Processing: Divides input image into patches and processes each
            2. Keypoint Proposal: Generates candidate keypoints using spatial softmax
            3. Feature Extraction: Learns visual features around each keypoint
            4. Particle Interaction: Models relationships between particles
            5. Context Modeling: Captures dynamics for the latent context (if enabled)

            The module supports both Gaussian and categorical distributions for
            features and context variables.

        The architecture uses a combination of CNNs and transformers:
            - CNNs for initial feature extraction from patches
            - Transformer encoders for modeling particle interactions
            - Separate pathways for foreground and background processing
            - Optional causal attention for temporal modeling
        """
        super(DLPEncoder, self).__init__()
        self.cdim = cdim
        self.image_size = image_size
        self.n_views = n_views
        self.dropout = dropout
        self.kp_range = kp_range
        self.n_kp_per_patch = n_kp_per_patch
        self.n_kp_enc = n_kp_enc
        self.n_kp_prior = n_kp_prior
        self.n_kp_dec = self.n_kp_enc if n_kp_dec is None else n_kp_dec
        self.warmup_n_kp_ratio = warmup_n_kp_ratio
        self.kp_activation = kp_activation
        self.patch_size = patch_size
        self.anchor_patch_s = patch_size / image_size
        self.features_dim = int(image_size // (2 ** (len(bg_ch_mult) - 1)))
        self.learned_feature_dim = learned_feature_dim
        self.learned_bg_feature_dim = learned_bg_feature_dim
        assert learned_feature_dim > 0, "learned_feature_dim must be greater than 0"
        self.features_dist = features_dist
        self.n_fg_categories = n_fg_categories
        self.n_fg_classes = n_fg_classes
        self.n_bg_categories = n_bg_categories
        self.n_bg_classes = n_bg_classes

        self.context_dim = context_dim
        self.mask_bg_in_enc = mask_bg_in_enc  # before encoding the bg, mask with the particles' obj_on
        self.anchor_s = anchor_s
        self.obj_patch_size = np.round(anchor_s * (image_size - 1)).astype(int)
        self.obj_on_min = obj_on_min
        self.obj_on_max = obj_on_max
        self.use_resblock = use_resblock
        self.embed_init_std = embed_init_std
        self.projection_dim = projection_dim
        self.timestep_horizon = (timestep_horizon + 1) if timestep_horizon > 1 else 1
        self.num_patches = int((image_size // self.patch_size) ** 2)
        self.attn_norm_type = attn_norm_type
        self.use_z_orig = use_z_orig
        self.interaction_depth = interaction_depth
        self.interaction_obj_on = interaction_obj_on
        self.interaction_features = interaction_features
        self.use_particle_inter_enc = (self.interaction_features or self.interaction_depth or self.interaction_obj_on)
        self.add_particle_temp_embed = add_particle_temp_embed
        self.temporal_interaction = False  # True=allow to attend over timesteps

        self.use_ctx_enc = (self.context_dim > 0)
        self.particle_score = particle_score
        self.cnn_mid_blocks = cnn_mid_blocks
        self.mlp_hidden_dim = mlp_hidden_dim

        # initialization
        self.init_zero_bias = init_zero_bias  # zero bias for conv and linear layers
        self.init_ssm_last_layer = init_ssm_last_layer  # spatial softmax initialization
        self.init_conv_layers = init_conv_layers  # initialize conv layers with normal dist
        self.init_conv_fg_std = init_conv_fg_std  # std for conv fg normal dist
        self.init_conv_bg_std = init_conv_bg_std  # std for conv bg normal dist

        self.register_buffer('scale_anchor', torch.tensor(np.log(anchor_s / (1 - anchor_s + 1e-5))))
        use_norm_layer = True  # norm layer in the pre-attention projections modules
        self.particle_enc = ParticleEncoder(cdim=cdim,
                                            image_size=image_size,
                                            pad_mode=pad_mode,
                                            n_kp_per_patch=self.n_kp_per_patch,
                                            n_kp_prior=self.n_kp_prior,
                                            patch_size=self.patch_size, n_kp_enc=self.n_kp_enc, n_kp_dec=self.n_kp_dec,
                                            learned_feature_dim=learned_feature_dim,
                                            kp_range=kp_range, kp_activation=kp_activation, anchor_s=anchor_s,
                                            use_resblock=use_resblock, embed_init_std=embed_init_std,
                                            projection_dim=projection_dim, timestep_horizon=timestep_horizon,
                                            filtering_heuristic=filtering_heuristic,
                                            obj_ch_mult_prior=obj_ch_mult_prior,
                                            obj_ch_mult=obj_ch_mult,
                                            obj_base_ch=obj_base_ch,
                                            obj_final_cnn_ch=obj_final_cnn_ch, num_res_blocks=num_res_blocks,
                                            interaction_features=interaction_features,
                                            interaction_obj_on=interaction_obj_on,
                                            interaction_depth=interaction_depth,
                                            temporal_interaction=self.temporal_interaction,
                                            cnn_mid_blocks=cnn_mid_blocks,
                                            mlp_hidden_dim=mlp_hidden_dim, embed_prior_patch_pos=False,
                                            add_particle_temp_embed=self.add_particle_temp_embed,
                                            features_dist=self.features_dist, n_fg_categories=n_fg_categories,
                                            n_fg_classes=n_fg_classes, obj_on_min=self.obj_on_min,
                                            obj_on_max=self.obj_on_max, warmup_n_kp_ratio=self.warmup_n_kp_ratio,
                                            init_zero_bias=init_zero_bias,
                                            init_ssm_last_layer=init_ssm_last_layer,
                                            init_conv_layers=init_conv_layers,
                                            init_conv_fg_std=init_conv_fg_std)

        self.prior_encoder = self.particle_enc.prior_encoder
        self.bg_encoder = BgEncoder(cdim=cdim, image_size=image_size, pad_mode=pad_mode,
                                    learned_feature_dim=learned_bg_feature_dim, use_resblock=use_resblock,
                                    ch_mult=bg_ch_mult, base_ch=bg_base_ch, final_cnn_ch=bg_final_cnn_ch,
                                    num_res_blocks=num_res_blocks, interaction_features=interaction_features,
                                    cnn_mid_blocks=cnn_mid_blocks, mlp_hidden_dim=mlp_hidden_dim,
                                    timestep_horizon=timestep_horizon,
                                    add_particle_temp_embed=self.add_particle_temp_embed,
                                    features_dist=self.features_dist, n_bg_categories=n_bg_categories,
                                    n_bg_classes=n_bg_classes,
                                    init_zero_bias=init_zero_bias,
                                    init_conv_layers=init_conv_layers,
                                    init_conv_bg_std=init_conv_bg_std)

        patch_centers = self.prior_encoder.get_patch_centers().unsqueeze(0) * (
                self.kp_range[1] - self.kp_range[0]) + self.kp_range[0]
        # append null particle
        patch_centers = torch.cat([patch_centers, torch.zeros(1, 1, 2)], dim=1)
        # self.patch_centers = patch_centers
        self.register_buffer('patch_centers', patch_centers)
        particle_anchors = patch_centers[:, :-1]  # [1, 1, n_kp_enc], no need for (0,0)-the bg
        particle_anchors = particle_anchors.unsqueeze(-2).repeat(1, 1, self.n_kp_per_patch, 1).view(1, -1, 2)

        if self.use_particle_inter_enc:
            self.particle_inter_enc = ParticleInteractionEncoder(n_kp_enc=n_kp_enc, dropout=0.0,
                                                                 learned_feature_dim=learned_feature_dim,
                                                                 learned_bg_feature_dim=learned_bg_feature_dim,
                                                                 embed_init_std=embed_init_std,
                                                                 projection_dim=projection_dim,
                                                                 timestep_horizon=timestep_horizon,
                                                                 pte_layers=pte_layers,
                                                                 pte_heads=pte_heads,
                                                                 attn_norm_type=attn_norm_type, pad_mode=pad_mode,
                                                                 use_resblock=use_resblock,
                                                                 hidden_dim=mlp_hidden_dim,
                                                                 temporal_interaction=self.temporal_interaction,
                                                                 interaction_features=interaction_features,
                                                                 interaction_depth=interaction_depth,
                                                                 interaction_obj_on=interaction_obj_on,
                                                                 cdim=cdim, image_size=image_size, n_views=self.n_views,
                                                                 ch_mult=bg_ch_mult, base_ch=bg_base_ch,
                                                                 final_cnn_ch=bg_final_cnn_ch,
                                                                 num_res_blocks=num_res_blocks,
                                                                 bg=True, use_img_input=True,
                                                                 cnn_mid_blocks=cnn_mid_blocks,
                                                                 particle_score=True,
                                                                 particle_positional_embed=particle_positional_embed,
                                                                 norm_layer=use_norm_layer,
                                                                 add_particle_temp_embed=self.add_particle_temp_embed,
                                                                 features_dist=self.features_dist,
                                                                 n_fg_categories=n_fg_categories,
                                                                 n_fg_classes=n_fg_classes,
                                                                 n_bg_categories=n_bg_categories,
                                                                 n_bg_classes=n_bg_classes,
                                                                 scale_anchor=self.scale_anchor,
                                                                 obj_on_min=self.obj_on_min,
                                                                 obj_on_max=self.obj_on_max,
                                                                 particle_anchors=particle_anchors,
                                                                 use_z_orig=self.use_z_orig,
                                                                 init_zero_bias=init_zero_bias,
                                                                 init_conv_layers=init_conv_layers,
                                                                 init_conv_fg_std=init_conv_fg_std
                                                                 )
        else:
            self.particle_inter_enc = None

        self.ctx_enc = ctx_enc

        self.init_weights()

    def init_weights(self):
        self.particle_enc.init_weights()
        self.bg_encoder.init_weights()
        self.prior_encoder.init_weights()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                pass
            elif isinstance(m, nn.Linear):
                pass

    def get_bg_mask_from_particle_glimpses(self, z, z_obj_on, mask_size, z_scale=None, detach_grad=True):
        """
        generates a mask based on particles position and the scale. Masks are squares.
        """
        if detach_grad:
            with torch.no_grad():
                if z_scale is None:
                    obj_fmap_masks = create_masks_fast(z.detach(), anchor_s=self.anchor_s, feature_dim=mask_size)
                else:
                    obj_fmap_masks = create_masks_with_scale(z.detach(), anchor_s=self.anchor_s, image_size=mask_size,
                                                             scale=z_scale.detach())
                z_gate = torch.where(z_obj_on.detach() > 0.2, 1.0, 0.0)[:, :, None, None, None]
                obj_fmap_masks = obj_fmap_masks.clamp(0, 1) * z_gate
                # obj_fmap_masks = obj_fmap_masks.clamp(0, 1) * z_obj_on[:, :, None, None, None].detach()
                bg_mask = 1 - obj_fmap_masks.squeeze(2).sum(1, keepdim=True).clamp(0, 1)
        else:
            with torch.no_grad():
                if z_scale is None:
                    obj_fmap_masks = create_masks_fast(z, anchor_s=self.anchor_s, feature_dim=mask_size)
                else:
                    obj_fmap_masks = create_masks_with_scale(z, anchor_s=self.anchor_s, image_size=mask_size,
                                                             scale=z_scale)
            obj_fmap_masks = obj_fmap_masks.clamp(0, 1) * z_obj_on[:, :, None, None, None]
            bg_mask = 1 - obj_fmap_masks.squeeze(2).sum(1, keepdim=True).clamp(0, 1)
        return bg_mask

    def encode_all(self, x, deterministic=False, warmup=False, actions=None, actions_mask=None, lang_embed=None,
                   x_goal=None, deterministic_goal=True):
        """
        encoding steps:
        1. encode bg: x -> bg_enc -> [bs * T, projection_dim]
        2. encode patches: x -> patch_enc -> [bs * T, n_patches, projection_dim ]
        3. encode particles: [patches, bg, particle_tokens, bg_token, ctx_token] -> pte -> [bs, T, n_particles + 2, dim]
        """
        # make sure x is [bs, T, ch, h, w]
        if len(x.shape) == 4:
            # that means x: [bs, ch, h, w]
            x = x.unsqueeze(1)  # -> [bs, T=1, ch, h, w]
        bs, timestep_horizon, ch, h, w = x.shape
        if x_goal is not None:
            if len(x_goal.shape) == 4:
                # that means x: [bs, ch, h, w]
                x_goal = x_goal.unsqueeze(1)  # -> [bs, T=1, ch, h, w]
            x = torch.cat([x, x_goal], dim=1)  # [bs, T+1, ...]
        # x = x.view(bs * timestep_horizon, *x.shape[2:])  # [bs * T, ch, h, w]
        # encode particles
        particle_dict = self.particle_enc(x, deterministic, warmup)
        # unpack
        kp_p = particle_dict['kp_p']
        var_kp = particle_dict['var_kp']
        patch_id_embed = particle_dict['patch_id_embed']
        z_base = particle_dict['z_base']
        z = particle_dict['z']
        mu_offset = particle_dict['mu_offset']
        logvar_offset = particle_dict['logvar_offset']
        z_offset = particle_dict['z_offset']
        mu_tot = particle_dict['mu_tot']
        z_base_var = particle_dict['z_base_var']
        mu_scale = particle_dict['mu_scale']
        logvar_scale = particle_dict['logvar_scale']
        z_scale = particle_dict['z_scale']
        mu_depth = particle_dict['mu_depth']
        logvar_depth = particle_dict['logvar_depth']
        z_depth = particle_dict['z_depth']
        obj_on_a = particle_dict['obj_on_a']
        obj_on_b = particle_dict['obj_on_b']
        mu_obj_on = particle_dict['mu_obj_on']
        z_obj_on = particle_dict['z_obj_on']
        mu_features = particle_dict['mu_features']
        logvar_features = particle_dict['logvar_features']
        z_features = particle_dict['z_features']
        cropped_objects = particle_dict['cropped_objects']

        z_score = particle_dict['z_score']
        mu_score = particle_dict['mu_score']
        logvar_score = particle_dict['logvar_score']

        if x_goal is not None and deterministic_goal:
            z = torch.cat([z[:, :-1], mu_tot[:, -1:]], dim=1)
            if z_obj_on is not None:
                z_obj_on = torch.cat([z_obj_on[:, :-1], Beta(obj_on_a[:, -1:], obj_on_b[:, -1:]).mean], dim=1)
            z_scale = torch.cat([z_scale[:, :-1], mu_scale[:, -1:]], dim=1)
            if z_depth is not None:
                z_depth = torch.cat([z_depth[:, :-1], mu_depth[:, -1:]], dim=1)
            if not self.interaction_features:
                z_features = torch.cat([z_features[:, :-1], mu_features[:, -1:]], dim=1)

        # encode bg
        # x = x.view(bs * timestep_horizon, *x.shape[2:])  # [bs * T, ch, h, w]
        x = x.view(-1, *x.shape[2:])  # [bs * T, ch, h, w]
        z_v = z.view(-1, *z.shape[2:])
        if self.n_kp_dec != self.n_kp_enc:
            # variance filtering
            total_var = z_base_var.view(-1, *z_base_var.shape[2:]).sum(-1)
            n_filter = self.n_kp_dec if not warmup else min(self.n_kp_dec, int(self.warmup_n_kp_ratio * self.n_kp_enc))
            _, embed_ind = torch.topk(total_var, k=n_filter, dim=-1, largest=False)
            # make selection
            batch_ind = torch.arange(z_v.shape[0], device=z_v.device)[:, None]
            z_v = z_v[batch_ind, embed_ind]  # [bs * T, n_kp_enc, 2]

        if self.interaction_obj_on:
            z_obj_on_v = torch.ones(-1, z_v.shape[1], device=x.device, dtype=torch.float)
        else:
            z_obj_on_v = z_obj_on.view(-1, *z_obj_on.shape[2:]).squeeze(-1)
            if self.n_kp_dec != self.n_kp_enc:
                z_obj_on_v = z_obj_on_v[batch_ind, embed_ind]  # [bs * T, n_kp_enc, 1]

        if self.mask_bg_in_enc:
            bg_enc_mask = self.get_bg_mask_from_particle_glimpses(z_v, z_obj_on_v, mask_size=x.shape[-1])
            bg_dict = self.bg_encoder(x, bg_enc_mask, deterministic, timestep_horizon)
        else:
            bg_enc_mask = None
            bg_dict = self.bg_encoder(x, None, deterministic, timestep_horizon)  # unmasked bg
        mu_bg_features = bg_dict['mu_bg']
        mu_bg_features = mu_bg_features.view(bs, -1, mu_bg_features.shape[-1])
        logvar_bg_features = bg_dict['logvar_bg']
        if logvar_bg_features is not None:
            logvar_bg_features = logvar_bg_features.view(bs, -1, logvar_bg_features.shape[-1])
        z_bg_features = bg_dict['z_bg']
        z_bg_features = z_bg_features.view(bs, -1, z_bg_features.shape[-1])
        if x_goal is not None and deterministic_goal and not self.interaction_features:
            z_bg_features = torch.cat([z_bg_features[:, :-1], mu_bg_features[:, -1:]], dim=1)

        if self.use_particle_inter_enc:
            z_in_inter = z_base + z_offset  # so we can detach z_base (ssm) if more stable
            inter_dict = self.particle_inter_enc(x, z_in_inter, z_scale, z_obj_on, z_depth, z_features, z_bg_features,
                                                 z_base_var, z_score, patch_id_embed,
                                                 deterministic=deterministic, warmup=warmup)
            if self.interaction_features:
                mu_features = inter_dict['mu_features']
                logvar_features = inter_dict['logvar_features']
                z_features = inter_dict['z_features']

                if x_goal is not None and deterministic_goal:
                    z_features = torch.cat([z_features[:, :-1], mu_features[:, -1:]], dim=1)

                if inter_dict.get('mu_bg_features') is not None:
                    mu_bg_features = inter_dict['mu_bg_features']
                    logvar_bg_features = inter_dict['logvar_bg_features']
                    z_bg_features = inter_dict['z_bg_features']

                    if x_goal is not None and deterministic_goal:
                        z_bg_features = torch.cat([z_bg_features[:, :-1], mu_bg_features[:, -1:]], dim=1)
            if self.interaction_obj_on:
                obj_on_a = inter_dict['obj_on_a']
                obj_on_b = inter_dict['obj_on_b']
                mu_obj_on = inter_dict['mu_obj_on']
                z_obj_on = inter_dict['z_obj_on']
                if x_goal is not None and deterministic_goal:
                    z_obj_on = torch.cat([z_obj_on[:, :-1], Beta(obj_on_a[:, -1:], obj_on_b[:, -1:]).mean], dim=1)
            if self.interaction_depth:
                mu_depth = inter_dict['mu_depth']
                logvar_depth = inter_dict['logvar_depth']
                z_depth = inter_dict['z_depth']

                if x_goal is not None and deterministic_goal:
                    z_depth = torch.cat([z_depth[:, :-1], mu_depth[:, -1:]], dim=1)

        if self.use_ctx_enc:
            z_in_ctx = z_base + z_offset  # so we can detach z_base (ssm) if more stable
            if x_goal is not None and deterministic_goal:
                z_in_ctx = torch.cat([z_in_ctx[:, :-1], mu_tot[:, -1:]], dim=1)
            z_scale_in_ctx = z_scale
            z_obj_on_in_ctx = z_obj_on
            z_depth_in_ctx = z_depth
            z_features_in_ctx = z_features
            z_bg_features_in_ctx = z_bg_features

            ctx_dict = self.ctx_enc(z_in_ctx, z_scale_in_ctx, z_obj_on_in_ctx, z_depth_in_ctx,
                                    z_features_in_ctx, z_bg_features_in_ctx, z_base_var,
                                    z_score, patch_id_embed, deterministic=deterministic, warmup=warmup,
                                    actions=actions, actions_mask=actions_mask, lang_embed=lang_embed)
            z_goal_proj = ctx_dict['z_goal_proj']
            # global context
            mu_context_global = ctx_dict['mu_context_global']
            logvar_context_global = ctx_dict['logvar_context_global']
            z_context_global = ctx_dict['z_context_global']

            mu_context_global_dyn = ctx_dict['mu_context_global_dyn']
            logvar_context_global_dyn = ctx_dict['logvar_context_global_dyn']
            z_context_global_dyn = ctx_dict['z_context_global_dyn']

            # local context
            mu_context = ctx_dict['mu_context']
            logvar_context = ctx_dict['logvar_context']
            z_context = ctx_dict['z_context']

            mu_context_dyn = ctx_dict['mu_context_dyn']
            logvar_context_dyn = ctx_dict['logvar_context_dyn']
            z_context_dyn = ctx_dict['z_context_dyn']
        else:
            mu_context_global = logvar_context_global = z_context_global = None
            mu_context_global_dyn = logvar_context_global_dyn = z_context_global_dyn = None
            mu_context = logvar_context = z_context = None
            mu_context_dyn = logvar_context_dyn = z_context_dyn = None
            z_goal_proj = None

        if x_goal is not None:
            # remove last timestep
            z_base = z_base[:, :-1].contiguous()
            z = z[:, :-1].contiguous()
            mu_offset = mu_offset[:, :-1].contiguous()
            logvar_offset = logvar_offset[:, :-1].contiguous()
            z_offset = z_offset[:, :-1].contiguous()
            mu_tot = mu_tot[:, :-1].contiguous()
            mu_features = mu_features[:, :-1].contiguous()
            logvar_features = logvar_features[:, :-1].contiguous()
            z_features = z_features[:, :-1].contiguous()
            mu_bg_features = mu_bg_features[:, :-1].contiguous()
            logvar_bg_features = logvar_bg_features[:, :-1].contiguous()
            z_bg_features = z_bg_features[:, :-1].contiguous()
            obj_on_a = obj_on_a[:, :-1].contiguous()
            obj_on_b = obj_on_b[:, :-1].contiguous()
            z_obj_on = z_obj_on[:, :-1].contiguous()
            if mu_obj_on is not None:
                mu_obj_on = mu_obj_on[:, :-1].contiguous()
            z_base_var = z_base_var[:, :-1].contiguous()
            mu_depth = mu_depth[:, :-1].contiguous()
            logvar_depth = logvar_depth[:, :-1].contiguous()
            z_depth = z_depth[:, :-1].contiguous()
            mu_scale = mu_scale[:, :-1].contiguous()
            logvar_scale = logvar_scale[:, :-1].contiguous()
            z_scale = z_scale[:, :-1].contiguous()
            kp_p = kp_p.view(bs, -1, *kp_p.shape[1:])[:, :-1].reshape(-1, *kp_p.shape[1:])  # orig: [bs * T, N, 2]
            var_kp = var_kp.view(bs, -1, *var_kp.shape[1:])[:, :-1].reshape(-1,
                                                                            *var_kp.shape[1:])  # orig: [bs * T, N, 2]
            bg_enc_mask = bg_enc_mask.view(bs, -1, *bg_enc_mask.shape[1:])[:, :-1].reshape(-1, *bg_enc_mask.shape[
                1:])  # orig: [bs * t, 1, im_size, im_size]
            mu_score = mu_score[:, :-1].contiguous()
            logvar_score = logvar_score[:, :-1].contiguous()
            z_score = z_score[:, :-1].contiguous()

        encode_dict = {'mu_anchor': z_base, 'logvar_anchor': torch.zeros_like(z_base), 'z_base': z_base, 'z': z,
                       'mu_offset': mu_offset, 'logvar_offset': logvar_offset, 'z_offset': z_offset, 'mu_tot': mu_tot,
                       'mu_features': mu_features, 'logvar_features': logvar_features, 'z_features': z_features,
                       'mu_bg_features': mu_bg_features, 'logvar_bg_features': logvar_bg_features,
                       'z_bg_features': z_bg_features, 'mu_context': mu_context, 'logvar_context': logvar_context,
                       'z_context': z_context,
                       'mu_context_global': mu_context_global, 'logvar_context_global': logvar_context_global,
                       'z_context_global': z_context_global,
                       'cropped_objects': cropped_objects.detach(), 'patch_id_embed': patch_id_embed,
                       'obj_on_a': obj_on_a, 'obj_on_b': obj_on_b, 'obj_on': z_obj_on, 'mu_obj_on': mu_obj_on,
                       'z_base_var': z_base_var,
                       'mu_depth': mu_depth, 'logvar_depth': logvar_depth, 'z_depth': z_depth,
                       'mu_scale': mu_scale, 'logvar_scale': logvar_scale, 'z_scale': z_scale,
                       'kp_p': kp_p, 'var_kp': var_kp, 'bg_enc_mask': bg_enc_mask,
                       'mu_score': mu_score, 'logvar_score': logvar_score, 'z_score': z_score,
                       'mu_context_dyn': mu_context_dyn, 'logvar_context_dyn': logvar_context_dyn,
                       'z_context_dyn': z_context_dyn,
                       'mu_context_global_dyn': mu_context_global_dyn,
                       'logvar_context_global_dyn': logvar_context_global_dyn,
                       'z_context_global_dyn': z_context_global_dyn,
                       'z_goal_proj': z_goal_proj
                       }
        return encode_dict

    def forward(self, x, deterministic=False, warmup=False, actions=None, actions_mask=None, lang_embed=None,
                x_goal=None):
        output_dict = self.encode_all(x, deterministic, warmup, actions=actions, actions_mask=actions_mask,
                                      lang_embed=lang_embed, x_goal=x_goal)
        return output_dict


class DLPDecoder(nn.Module):
    def __init__(self,
                 # Input configuration
                 cdim=3,  # Number of input image channels
                 image_size=64,  # Input image size (assumed square)
                 pad_mode='replicate',  # Padding mode for CNNs
                 dropout=0.0,  # Dropout rate
                 normalize_rgb=False,  # Normalize RGB output to [-1, 1]

                 # Feature dimensions
                 learned_feature_dim=16,  # Dimension of learned visual features
                 learned_bg_feature_dim=16,  # Dimension of background features
                 anchor_s=0.25,  # Glimpse size ratio
                 n_kp_enc=16,  # Number of keypoints to decode
                 context_dim=0,  # Dimension of context features

                 # Network architecture
                 use_resblock=True,  # Use residual blocks
                 timestep_horizon=1,  # Maximum timesteps to process
                 decode_with_ctx=False,  # Use context in decoding
                 cnn_mid_blocks=False,  # Use middle blocks in CNN
                 mlp_hidden_dim=256,  # Hidden dimension for MLPs

                 # Object decoder configuration
                 obj_res_from_fc=8,  # Initial resolution for object decoder
                 obj_ch_mult=(1, 2, 3),  # Channel multipliers for object decoder
                 obj_base_ch=32,  # Base channels for object decoder
                 obj_final_cnn_ch=32,  # Final CNN channels for object decoder

                 # Background decoder configuration
                 bg_res_from_fc=8,  # Initial resolution for background decoder
                 bg_ch_mult=(1, 2, 3),  # Channel multipliers for background decoder
                 bg_base_ch=32,  # Base channels for background decoder
                 bg_final_cnn_ch=32,  # Final CNN channels for background decoder
                 num_res_blocks=2,  # Number of residual blocks

                 # initialization
                 init_zero_bias=True,  # zero bias for conv and linear layers
                 init_conv_layers=True,  # initialize conv layers with normal dist
                 init_conv_fg_std=0.02,  # std for conv fg normal dist
                 init_conv_bg_std=0.005,  # std for conv bg normal dist (<fg -> prioritize fg in learning)
                 ):
        """
        DLP Decoder Module

        A neural network module that reconstructs images from object-centric representations using
        the Deep Latent Particles (DLP) approach. This decoder transforms particle representations
        back into image space, handling both foreground objects and background separately.

        Args:
            cdim (int): Number of input image channels. Defaults to 3.
            image_size (int): Size of input images (assumed square). Defaults to 64.
            pad_mode (str): Padding mode for CNNs ('zeros' or 'replicate'). Defaults to 'replicate'.
            dropout (float): Dropout rate for networks. Defaults to 0.0.
            normalize_rgb (bool): Normalize RGB output to [-1, 1] range. Defaults to False.
            learned_feature_dim (int): Dimension of learned visual features. Defaults to 16.
            learned_bg_feature_dim (int): Dimension of background features. Defaults to 16.
            anchor_s (float): Glimpse size as ratio of image_size (e.g., 0.25 for 32px glimpse on 128px image).
                            Defaults to 0.25.
            n_kp_enc (int): Number of keypoints to decode. Defaults to 16.
            context_dim (int): Dimension of context features. Set to 0 to disable context. Defaults to 0.
            use_resblock (bool): Use residual blocks in decoders. Defaults to True.
            timestep_horizon (int): Maximum number of timesteps to process at once. Defaults to 1.
            decode_with_ctx (bool): Use context information during decoding. Defaults to False.
            cnn_mid_blocks (bool): Use middle blocks in CNN decoders. Defaults to False.
            mlp_hidden_dim (int): Hidden dimension for MLPs. Defaults to 256.
            obj_res_from_fc (int): Initial resolution for object decoder from fully connected layer.
                                 Defaults to 8.
            obj_ch_mult (tuple): Channel multipliers for progressive object decoder stages.
                               Defaults to (1, 2, 3).
            obj_base_ch (int): Base number of channels for object decoder. Defaults to 32.
            obj_final_cnn_ch (int): Number of channels in final object CNN layer. Defaults to 32.
            bg_res_from_fc (int): Initial resolution for background decoder from fully connected layer.
                                Defaults to 8.
            bg_ch_mult (tuple): Channel multipliers for progressive background decoder stages.
                              Defaults to (1, 2, 3).
            bg_base_ch (int): Base number of channels for background decoder. Defaults to 32.
            bg_final_cnn_ch (int): Number of channels in final background CNN layer. Defaults to 32.
            num_res_blocks (int): Number of residual blocks per resolution level. Defaults to 2.

        Architecture Details:
            The decoder consists of two main pathways:
            1. Object Decoder:
               - Processes each particle independently
               - Progressively upsamples from initial resolution (obj_res_from_fc)
               - Optionally incorporates context information

            2. Background Decoder:
               - Processes background features
               - Similar progressive upsampling architecture

        Notes:
            - The decoder uses spatial transformer networks (STN) for differentiable
              rendering of particles
        """
        super(DLPDecoder, self).__init__()
        self.image_size = image_size
        self.feature_map_size = image_size
        self.n_kp_enc = n_kp_enc
        self.dropout = dropout
        self.learned_feature_dim = learned_feature_dim
        self.learned_bg_feature_dim = learned_bg_feature_dim
        assert learned_feature_dim > 0, "learned_feature_dim must be greater than 0"
        self.anchor_s = anchor_s
        self.context_dim = context_dim
        self.obj_patch_size = np.round(anchor_s * (image_size - 1)).astype(int)
        self.cdim = cdim
        self.use_resblock = use_resblock
        self.decode_with_ctx = decode_with_ctx
        self.normalize_rgb = normalize_rgb
        self.timestep_horizon = (timestep_horizon + 1) if timestep_horizon > 1 else 1
        self.cnn_mid_blocks = cnn_mid_blocks
        self.mlp_hidden_dim = mlp_hidden_dim
        self.context_dim = context_dim

        # initialization
        self.init_zero_bias = init_zero_bias  # zero bias for conv and linear layers
        self.init_conv_layers = init_conv_layers  # initialize conv layers with normal dist
        self.init_conv_fg_std = init_conv_fg_std  # std for conv fg normal dist
        self.init_conv_bg_std = init_conv_bg_std  # std for conv bg normal dist

        # object decoder
        if self.context_dim > 0 and self.decode_with_ctx:
            particle_dec_net = ObjectDecoderCNNFILM
        else:
            particle_dec_net = ObjectDecoderCNN
        self.particle_dec = particle_dec_net(patch_size=(self.obj_patch_size, self.obj_patch_size), num_chans=4,
                                             bottleneck_size=learned_feature_dim,
                                             use_resblock=self.use_resblock,
                                             pad_mode='replicate', context_dim=context_dim, normalize_rgb=normalize_rgb,
                                             res_from_fc=obj_res_from_fc,
                                             ch_mult=obj_ch_mult, base_ch=obj_base_ch, final_cnn_ch=obj_final_cnn_ch,
                                             num_res_blocks=num_res_blocks, cnn_mid_blocks=cnn_mid_blocks,
                                             mlp_hidden_dim=mlp_hidden_dim,
                                             init_zero_bias=init_zero_bias,
                                             init_conv_layers=init_conv_layers,
                                             init_conv_fg_std=init_conv_fg_std
                                             )

        self.num_obj_upsample = self.particle_dec.num_upsample
        # bg decoder
        self.bg_dec = BgDecoder(cdim=cdim, image_size=image_size,
                                pad_mode='replicate', learned_bg_feature_dim=learned_bg_feature_dim,
                                use_resblock=use_resblock, context_dim=context_dim, film=decode_with_ctx,
                                timestep_horizon=timestep_horizon,
                                bg_res_from_fc=bg_res_from_fc, bg_ch_mult=bg_ch_mult, bg_base_ch=bg_base_ch,
                                bg_final_cnn_ch=bg_final_cnn_ch, num_res_blocks=num_res_blocks,
                                decode_with_ctx=decode_with_ctx, normalize_rgb=normalize_rgb,
                                cnn_mid_blocks=cnn_mid_blocks, mlp_hidden_dim=mlp_hidden_dim,
                                init_zero_bias=init_zero_bias, init_conv_layers=init_conv_layers,
                                init_conv_bg_std=init_conv_bg_std
                                )
        self.num_bg_upsample = self.bg_dec.num_bg_upsample
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                pass
            elif isinstance(m, (nn.GroupNorm, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                if self.init_zero_bias and m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def translate_patches(self, kp_batch, patches_batch, scale=None, translation=None, scale_normalized=False):
        """
        translate patches to be centered around given keypoints
        kp_batch: [bs, n_kp, 2] in [-1, 1]
        patches: [bs, n_kp, ch_patches, patch_size, patch_size]
        scale: None or [bs, n_kp, 2] or [bs, n_kp, 1]
        translation: None or [bs, n_kp, 2] or [bs, n_kp, 1] (delta from kp)
        scale_normalized: False if scale is not in [0, 1]
        :return: translated_padded_patches [bs, n_kp, ch, img_size, img_size]
        """
        batch_size, n_kp, ch_patch, patch_size, _ = patches_batch.shape
        # img_size = self.image_size
        img_size = self.feature_map_size
        if scale is None:
            z_scale = (patch_size / img_size) * torch.ones_like(kp_batch)
        else:
            # normalize to [0, 1]
            if scale_normalized:
                z_scale = scale
            else:
                z_scale = torch.sigmoid(scale)  # -> [0, 1]
        z_pos = kp_batch.reshape(-1, kp_batch.shape[-1])  # [bs * n_kp, 2]
        z_scale = z_scale.view(-1, z_scale.shape[-1])  # [bs * n_kp, 2]
        patches_batch = patches_batch.reshape(-1, *patches_batch.shape[2:])
        out_dims = (batch_size * n_kp, ch_patch, img_size, img_size)
        trans_patches_batch = spatial_transform(patches_batch, z_pos, z_scale, out_dims, inverse=True)
        trans_padded_patches_batch = trans_patches_batch.view(batch_size, n_kp, *trans_patches_batch.shape[1:])
        # [bs, n_kp, ch, img_size, img_size]
        return trans_padded_patches_batch

    def get_objects_alpha_rgb(self, z_kp, z_features, z_scale=None, z_ctx=None, translation=None):
        # decode the latent particles into RGBA glimpses and place them on the canvas
        dec_objects = self.particle_dec(z_features, context=z_ctx)  # [bs * n_kp, 4, patch_size, patch_size]
        dec_objects = dec_objects.view(-1, z_kp.shape[1],
                                       *dec_objects.shape[1:])  # [bs, n_kp, 4, patch_size, patch_size]
        # translate patches - place the decoded glimpses on the canvas
        dec_objects_trans = self.translate_patches(z_kp, dec_objects, z_scale, translation)
        # dec_objects_trans: [bs, n_kp, 4, im_size, im_size]
        # multiply by alpha channel
        a_obj, rgb_obj = torch.split(dec_objects_trans, [1, dec_objects_trans.shape[2] - 1], dim=2)
        return dec_objects, a_obj, rgb_obj

    def get_objects_alpha_rgb_with_depth(self, a_obj, rgb_obj, obj_on, z_depth, eps=1e-5):
        # stitching the glimpses by factoring the alpha maps and the particle's inferred depth
        # obj_on: [bs, n_kp, 1]
        # z_depth: [bs, n_kp, 1]
        n_kp = a_obj.shape[1]
        # turn off inactive particles
        a_obj = obj_on[:, :, None, None, None] * a_obj  # [bs, n_kp, 1, im_size, im_size]
        rgba_obj = a_obj * rgb_obj
        # normalize
        importance_map = a_obj * torch.sigmoid(-z_depth[:, :, :, None, None])
        importance_map = importance_map / (torch.sum(importance_map, dim=1, keepdim=True) + eps)
        # this imitates softmax to move objects on the depth axis
        dec_objects_trans = (rgba_obj * importance_map).sum(dim=1)
        alpha_mask = 1.0 - (importance_map * a_obj).sum(dim=1)
        a_obj = importance_map * a_obj
        return a_obj, alpha_mask, dec_objects_trans

    def decode_objects(self, z_kp, z_features, obj_on, z_scale=None, translation=None, z_depth=None,
                       z_ctx=None):
        # stitching the decoded latent particles -> RGB, factoring the alpha maps and depths
        dec_objects, a_obj, rgb_obj = self.get_objects_alpha_rgb(z_kp, z_features, z_scale=z_scale, z_ctx=z_ctx,
                                                                 translation=translation)
        alpha_masks, bg_mask, dec_objects_trans = self.get_objects_alpha_rgb_with_depth(a_obj, rgb_obj, obj_on=obj_on,
                                                                                        z_depth=z_depth)
        return dec_objects, dec_objects_trans, alpha_masks, bg_mask

    def decode_all(self, z, z_scale, z_features, obj_on, z_depth, z_bg_features, z_ctx=None,
                   warmup=False):
        if len(z.shape) == 4:
            # z: [bs, T, n_kp, 2]
            batch_size = z.shape[0]
            timesteps = z.shape[1]
            z = z.view(-1, *z.shape[2:])  # [bs * T, n_kp, 2]
            z_scale = z_scale.view(-1, *z_scale.shape[2:])  # [bs * T, n_kp, 2]
            obj_on = obj_on.view(-1, *obj_on.shape[2:])  # [bs * T, n_kp, 1]
            z_depth = z_depth.view(-1, *z_depth.shape[2:])  # [bs * T, n_kp, 1]
            z_features = z_features.view(-1, *z_features.shape[2:])  # [bs * T, n_kp, feature_dim]
            z_bg_features = z_bg_features.view(-1, *z_bg_features.shape[2:])  # [bs * T, feature_dim]
            if z_ctx is not None:
                z_ctx = z_ctx.view(-1, *z_ctx.shape[2:])  # [bs * T, feature_dim]
        else:
            timesteps = 1
        # z: [bs * T, ...]
        # squeeze the last dimension of `obj_on`:
        if len(obj_on.shape) == 3:
            obj_on = obj_on.squeeze(-1)
        # a wrapper function to decode latent particles into and RGB image
        object_dec_out = self.decode_objects(z, z_features, obj_on, z_depth=z_depth, z_scale=z_scale,
                                             z_ctx=z_ctx)
        dec_objects, dec_objects_trans, alpha_masks, bg_mask = object_dec_out
        bg_rec = self.bg_dec(z_bg_features, z_ctx)
        rec = bg_mask * bg_rec + dec_objects_trans
        decoder_out = {'rec': rec, 'dec_objects': dec_objects, 'dec_objects_trans': dec_objects_trans,
                       'bg_mask': bg_mask, 'alpha_masks': alpha_masks, 'bg_rec': bg_rec}

        return decoder_out

    def forward(self, z, z_scale, z_features, obj_on_sample, z_depth, z_bg_features, z_ctx=None,
                warmup=False):
        return self.decode_all(z, z_scale, z_features, obj_on_sample, z_depth, z_bg_features, z_ctx, warmup)


class DLPContext(nn.Module):
    def __init__(self, n_kp_enc, dropout=0.1, learned_feature_dim=16, learned_bg_feature_dim=16, embed_init_std=0.02,
                 projection_dim=128, timestep_horizon=1, pte_layers=1, pte_heads=1,
                 attn_norm_type='rms', context_dim=7, hidden_dim=256,
                 activation='gelu',
                 ctx_pool_mode='none', bg=True, n_views=1, causal=True, particle_positional_embed=True,
                 particle_score=False, norm_layer=True,
                 shared_logvar=False, ctx_dist='gauss', n_ctx_categories=4, n_ctx_classes=4,
                 particle_anchors=None, use_z_orig=False,
                 ctx_pool_dim=256, n_pool_ctx_categories=8, n_pool_ctx_classes=8, global_ctx_pool=False,
                 token_pool_cross_attn=False, global_local_fuse_mode='none', condition_local_on_global=True,
                 # external conditioning
                 action_condition=False,  # condition on actions
                 action_dim=0,  # dimension of input actions
                 random_action_condition=False,  # condition on random actions
                 random_action_dim=0,  # dimension of sampled random actions
                 null_action_embed=False,  # learn a "no-input-action" embedding, to learn on action-free videos as well
                 action_as_particle=False,  # if False, use AdaLN conditioning
                 language_condition=False,  # condition on language embedding
                 language_embed_dim=0,  # embedding dimension for each token
                 language_max_len=64,  # maximum tokens per prompt
                 language_condition_type='self',  # cross-attention ('cross') or self-attention ('self')
                 img_goal_condition=False,
                 img_goal_condition_type='adaln',
                 # cross-attention ('cross'), adaptive LN ('adaln') or self-attention ('self')
                 pos_embed_t_adaln=True,  # pos embeddings for timesteps using adaln
                 pos_embed_p_adaln=True,  # pos embeddings for particles using adaln
                 pos_embed_objon_adaln=False,  # pos embeddings for particles transparency using adaln
                 particle_pool_adaln=False
                 ):
        super(DLPContext, self).__init__()
        """
        This module takes in temporal sequence of particles and outputs latent context,
        which can be per-particle, or global, depending on the pooling type.
        This module shares attention layers and has different head for posterior latent context (inverse model)
        and prior latent context (policy).
        """
        assert ctx_pool_mode in ['none', 'mean', 'max', 'token', 'last', 'mlp']
        self.ctx_pool_mode = ctx_pool_mode
        self.n_kp_enc = n_kp_enc
        self.dropout = dropout
        self.learned_feature_dim = learned_feature_dim
        self.learned_bg_feature_dim = learned_bg_feature_dim
        assert learned_feature_dim > 0, "learned_feature_dim must be greater than 0"
        self.embed_init_std = embed_init_std
        self.projection_dim = projection_dim
        self.timestep_horizon = (timestep_horizon + 1) if timestep_horizon > 1 else 1
        self.attn_norm_type = attn_norm_type
        self.context_dist = ctx_dist
        self.n_ctx_categories = n_ctx_categories
        self.n_ctx_classes = n_ctx_classes
        self.context_dim = context_dim
        self.learned_ctx_token = (ctx_pool_mode == 'token' and self.context_dim > 0)
        self.n_pool_ctx_categories = n_pool_ctx_categories
        self.n_pool_ctx_classes = n_pool_ctx_classes
        self.ctx_pool_dim = ctx_pool_dim
        if self.context_dist == 'categorical':
            self.ctx_pool_dim = int(self.n_pool_ctx_categories * self.n_pool_ctx_classes)
        self.global_ctx_pool = global_ctx_pool
        self.global_local_fuse_mode = global_local_fuse_mode
        self.condition_local_on_global = condition_local_on_global
        self.token_pool_cross_attn = token_pool_cross_attn
        self.hidden_dim = hidden_dim
        self.with_bg = bg
        self.n_views = n_views
        self.activation = activation
        self.is_causal = causal
        self.particle_score = particle_score
        self.shared_logvar = shared_logvar
        self.use_z_orig = use_z_orig
        self.pos_embed_t_adaln = pos_embed_t_adaln
        self.pos_embed_p_adaln = pos_embed_p_adaln
        self.pos_embed_objon_adaln = pos_embed_objon_adaln
        self.particle_pool_adaln = particle_pool_adaln

        # actions
        self.action_condition = action_condition
        self.action_dim = action_dim
        self.random_action_condition = random_action_condition
        self.random_action_dim = random_action_dim
        self.learn_null_action_embed = null_action_embed
        self.action_as_particle = action_as_particle
        # language
        self.language_condition = language_condition
        self.language_embed_dim = language_embed_dim
        self.language_max_len = language_max_len
        self.language_condition_type = language_condition_type
        assert self.language_condition_type in ['cross', 'self'], \
            f'lang condition type {self.language_condition_type} is not supported'
        # image goal
        self.img_goal_condition = img_goal_condition
        self.img_goal_condition_type = img_goal_condition_type
        assert self.img_goal_condition_type in ['cross', 'self', 'adaln'], \
            f'img goal condition type {self.img_goal_condition_type} is not supported'

        if self.learn_null_action_embed:
            self.null_action_embeddings = nn.Parameter(
                self.embed_init_std * torch.randn(1, 1, self.action_dim))
        else:
            self.null_action_embeddings = None

        if self.action_condition and self.action_dim > 0:
            if self.action_as_particle:
                self.action_proj = nn.Sequential(nn.Linear(self.action_dim, hidden_dim),
                                                 RMSNorm(hidden_dim))
            else:
                self.action_proj = nn.Sequential(nn.Linear(self.action_dim, hidden_dim),
                                                 RMSNorm(hidden_dim),
                                                 nn.GELU())
        else:
            self.action_proj = None

        if self.random_action_condition and self.random_action_dim > 0:
            if self.action_as_particle:
                self.random_action_proj = nn.Sequential(nn.Linear(self.random_action_dim, hidden_dim),
                                                        RMSNorm(hidden_dim))
            else:
                self.random_action_proj = nn.Sequential(nn.Linear(self.random_action_dim, hidden_dim),
                                                        RMSNorm(hidden_dim),
                                                        nn.GELU())
        else:
            self.random_action_proj = None

        if self.language_condition and self.language_embed_dim > 0:
            # self.lang_proj = nn.Sequential(nn.Linear(self.language_embed_dim, hidden_dim),
            #                                RMSNorm(hidden_dim),
            #                                nn.GELU(),
            #                                nn.Linear(hidden_dim, hidden_dim))
            self.lang_proj = nn.Linear(self.language_embed_dim, hidden_dim)
            if self.language_condition_type == 'self':
                self.lang_pos_embed = nn.Parameter(
                    self.embed_init_std * torch.randn(1, 1, 1, hidden_dim))
            else:
                # self.lang_pos_embed = nn.Parameter(
                #     self.embed_init_std * torch.randn(1, 1, self.language_max_len, hidden_dim))
                self.lang_pos_embed = None
        else:
            self.lang_proj = None
            self.lang_pos_embed = None

        if self.img_goal_condition:
            if self.img_goal_condition_type == 'adaln':
                self.goal_proj = nn.Sequential(nn.Linear(projection_dim, hidden_dim),
                                               RMSNorm(hidden_dim),
                                               nn.GELU())
            else:
                self.goal_proj = nn.Sequential(nn.Linear(projection_dim, hidden_dim),
                                               RMSNorm(hidden_dim),
                                               nn.GELU(),
                                               nn.Linear(hidden_dim, hidden_dim))
            # self.goal_proj = nn.Linear(projection_dim, hidden_dim)
            # self.goal_proj = nn.Identity()
        else:
            self.goal_proj = None

        if self.particle_pool_adaln:
            self.particle_pool_proj = nn.Sequential(nn.Linear(projection_dim, self.hidden_dim),
                                                    nn.GELU(),
                                                    ParticlePool(pool_mode='mean', pool_dim=-2),
                                                    nn.Linear(self.hidden_dim, self.hidden_dim),
                                                    RMSNorm(hidden_dim),
                                                    nn.GELU())
        else:
            self.particle_pool_proj = None

        if particle_anchors is None:
            self.register_buffer('particles_anchor', torch.zeros(1, 1, self.n_kp_enc))
            self.use_z_orig = False
        else:
            self.register_buffer('particles_anchor', particle_anchors)

        n_particles = self.n_kp_enc  # [n_kp_enc]

        # entities in attn: [bg*, n_particles, ctx, ctx_tokens*]
        if (self.learned_ctx_token or self.ctx_pool_mode == 'last') and self.token_pool_cross_attn:
            if self.learned_ctx_token:
                n_particles += 1
                self.ctx_token_embeddings = nn.Parameter(
                    self.embed_init_std * torch.randn(1, 1, 1, projection_dim))
            block_size = 1  # this means token pooling does not depend on the temporal horizon
            self.cross_attn_block = CrossBlock(n_embed=self.projection_dim, n_head=pte_heads,
                                               block_size=block_size,
                                               attn_pdrop=dropout,
                                               resid_pdrop=dropout,
                                               hidden_dim_multiplier=4, positional_bias=False,
                                               activation='gelu',
                                               max_particles=None, norm_type=attn_norm_type)
        elif self.learned_ctx_token:
            n_particles += 1
            self.ctx_token_embeddings = nn.Parameter(
                self.embed_init_std * torch.randn(1, 1, 1, projection_dim))
            self.cross_attn_block = None
        else:
            self.ctx_token_embeddings = None
            self.cross_attn_block = None
        if self.with_bg:
            n_particles += 1
            self.bg_embeddings = nn.Parameter(self.embed_init_std * torch.randn(1, 1, 1, projection_dim))
        if self.img_goal_condition and self.img_goal_condition_type == 'self':
            n_particles += self.n_kp_enc
            if self.with_bg:
                n_particles += 1
            self.goal_embeddings = nn.Parameter(self.embed_init_std * torch.randn(1, 1, 1, projection_dim))
        if self.action_condition and self.action_as_particle:
            self.action_embeddings = nn.Parameter(self.embed_init_std * torch.randn(1, 1, 1, projection_dim))
            n_particles += 1
        if self.random_action_condition and self.action_as_particle:
            self.random_action_embeddings = nn.Parameter(self.embed_init_std * torch.randn(1, 1, 1, projection_dim))
            n_particles += 1
        if self.language_condition and self.language_condition_type == 'self':
            n_particles += self.language_max_len


        # entities positional embeddings
        self.particle_pos_embed = particle_positional_embed and not self.pos_embed_p_adaln
        if self.particle_pos_embed:
            self.particle_embeddings = nn.Parameter(
                self.embed_init_std * torch.randn(1, 1, self.n_kp_enc, projection_dim))
        else:
            self.particle_embeddings = nn.Parameter(self.embed_init_std * torch.randn(1, 1, 1, projection_dim))

        if self.n_views > 1:
            self.view_embeddings = nn.Parameter(
                self.embed_init_std * torch.randn(1, 1, self.n_views, 1, projection_dim))
        else:
            self.view_embeddings = None

        if self.pos_embed_p_adaln:
            n_particles += (self.n_views - 1) * (self.n_kp_enc + 1)
            self.pos_p_embeddings = nn.Parameter(
                self.embed_init_std * torch.randn(1, 1, n_particles, self.hidden_dim))

        if self.pos_embed_objon_adaln:
            self.objon_embeddings = nn.Sequential(nn.Linear(1, self.hidden_dim),
                                                  RMSNorm(hidden_dim),
                                                  nn.GELU())

        # interaction encoder
        proj_out_dim = projection_dim
        self.basic_particle_proj = ParticleAttributesProjection(n_particles=self.n_kp_enc,
                                                                in_features_dim=self.learned_feature_dim,
                                                                hidden_dim=self.hidden_dim,
                                                                output_dim=proj_out_dim,
                                                                bg_features_dim=self.learned_bg_feature_dim,
                                                                add_ctx_token=False,
                                                                depth=True,
                                                                obj_on=True,
                                                                base_var=False, bg=self.with_bg,
                                                                norm_layer=norm_layer,
                                                                particle_score=self.particle_score,
                                                                use_z_orig=self.use_z_orig)

        block_size = self.timestep_horizon
        pte_action_cond = (self.action_condition or self.random_action_condition) and not self.action_as_particle
        pte_goal_cond = (self.img_goal_condition and self.img_goal_condition_type == 'adaln')
        pte_context_cond = pte_action_cond or pte_goal_cond or self.particle_pool_adaln or self.pos_embed_p_adaln
        lang_cross_attn = (self.language_condition and self.language_condition_type == 'cross')
        img_goal_cross_attn = (self.img_goal_condition and self.img_goal_condition_type == 'cross')
        cross_attn = lang_cross_attn or img_goal_cross_attn
        self.pte = ParticleSpatioTemporalTransformer(n_embed=self.projection_dim, n_head=pte_heads,
                                                     n_layer=pte_layers,
                                                     block_size=block_size,
                                                     output_dim=self.projection_dim, attn_pdrop=dropout,
                                                     resid_pdrop=dropout,
                                                     hidden_dim_multiplier=4, positional_bias=False,
                                                     activation='gelu',
                                                     max_particles=None, norm_type=attn_norm_type,
                                                     particles_first=False, init_std=embed_init_std,
                                                     causal=self.is_causal,
                                                     context_cond=pte_context_cond,
                                                     residual_modulation=pte_context_cond,
                                                     context_gate=pte_context_cond,
                                                     cond_cross_attn=cross_attn,
                                                     pos_embed_t_adaln=self.pos_embed_t_adaln)

        if self.global_ctx_pool:
            # global
            global_ctx_pool = 'token' if self.ctx_pool_mode == 'none' else self.ctx_pool_mode
            self.global_posterior_decoder = ParticleContextDecoder(n_particles=self.n_kp_enc, input_dim=projection_dim,
                                                                   hidden_dim=self.hidden_dim,
                                                                   context_dim=self.ctx_pool_dim,
                                                                   context_dist=self.context_dist,
                                                                   n_ctx_categories=self.n_pool_ctx_categories,
                                                                   n_ctx_classes=self.n_pool_ctx_classes,
                                                                   learned_ctx_token=self.learned_ctx_token,
                                                                   ctx_pool_mode=global_ctx_pool,
                                                                   shared_logvar=self.shared_logvar,
                                                                   output_ctx_logvar=(ctx_dist != 'categorical'),
                                                                   conditional=False, cond_dim=0)
            self.global_prior_decoder = ParticleContextDecoder(n_particles=self.n_kp_enc, input_dim=projection_dim,
                                                               hidden_dim=self.hidden_dim,
                                                               context_dim=self.ctx_pool_dim,
                                                               context_dist=self.context_dist,
                                                               n_ctx_categories=self.n_pool_ctx_categories,
                                                               n_ctx_classes=self.n_pool_ctx_classes,
                                                               learned_ctx_token=self.learned_ctx_token,
                                                               ctx_pool_mode=global_ctx_pool,
                                                               shared_logvar=self.shared_logvar,
                                                               output_ctx_logvar=(ctx_dist != 'categorical'),
                                                               conditional=False, cond_dim=0)

            # local
            self.posterior_decoder = ParticleContextDecoder(n_particles=self.n_kp_enc, input_dim=projection_dim,
                                                            hidden_dim=self.hidden_dim,
                                                            context_dim=self.context_dim,
                                                            context_dist=self.context_dist,
                                                            n_ctx_categories=self.n_ctx_categories,
                                                            n_ctx_classes=self.n_ctx_classes,
                                                            learned_ctx_token=False,
                                                            ctx_pool_mode="none",
                                                            shared_logvar=self.shared_logvar,
                                                            output_ctx_logvar=(ctx_dist != 'categorical'),
                                                            conditional=self.condition_local_on_global,
                                                            cond_dim=self.ctx_pool_dim)
            self.prior_decoder = ParticleContextDecoder(n_particles=self.n_kp_enc, input_dim=projection_dim,
                                                        hidden_dim=self.hidden_dim,
                                                        context_dim=self.context_dim,
                                                        context_dist=self.context_dist,
                                                        n_ctx_categories=self.n_ctx_categories,
                                                        n_ctx_classes=self.n_ctx_classes,
                                                        learned_ctx_token=False,
                                                        ctx_pool_mode="none",
                                                        shared_logvar=self.shared_logvar,
                                                        output_ctx_logvar=(ctx_dist != 'categorical'),
                                                        conditional=self.condition_local_on_global,
                                                        cond_dim=self.ctx_pool_dim)
        else:
            self.global_posterior_decoder = self.global_prior_decoder = nn.Identity()
            self.posterior_decoder = ParticleContextDecoder(n_particles=self.n_kp_enc, input_dim=projection_dim,
                                                            hidden_dim=self.hidden_dim,
                                                            context_dim=self.context_dim,
                                                            context_dist=self.context_dist,
                                                            n_ctx_categories=self.n_ctx_categories,
                                                            n_ctx_classes=self.n_ctx_classes,
                                                            learned_ctx_token=self.learned_ctx_token,
                                                            ctx_pool_mode=self.ctx_pool_mode,
                                                            shared_logvar=self.shared_logvar,
                                                            output_ctx_logvar=(ctx_dist != 'categorical'),
                                                            conditional=False, cond_dim=0)
            self.prior_decoder = ParticleContextDecoder(n_particles=self.n_kp_enc, input_dim=projection_dim,
                                                        hidden_dim=self.hidden_dim,
                                                        context_dim=self.context_dim,
                                                        context_dist=self.context_dist,
                                                        n_ctx_categories=self.n_ctx_categories,
                                                        n_ctx_classes=self.n_ctx_classes,
                                                        learned_ctx_token=self.learned_ctx_token,
                                                        ctx_pool_mode=self.ctx_pool_mode,
                                                        shared_logvar=self.shared_logvar,
                                                        output_ctx_logvar=(ctx_dist != 'categorical'),
                                                        conditional=False, cond_dim=0)
        self.init_weights()

    def init_weights(self):
        self.posterior_decoder.init_weights()
        self.prior_decoder.init_weights()
        self.pte.init_weights()

    def encode_all(self, z, z_scale, z_obj_on, z_depth, z_features, z_bg_features=None, z_base_var=None,
                   z_score=None, patch_id_embed=None, deterministic=False, warmup=False,
                   detach_before_proj=False, encode_posterior=True, encode_prior=True, actions=None, actions_mask=None,
                   lang_embed=None, z_goal=None, detach_z_goal=False):
        """
        output order:
        if with_bg and ctx_pool_mode='token': [n_particles, bg, ctx, ctx_token*]
        else: [n_particles, ctx, ctx_token*]
        """
        # x: [bs, t, ch, h, w]
        bs, timestep_horizon = z.shape[0], z.shape[1]
        z_v = z.detach() if detach_before_proj else z
        z_scale_v = z_scale.detach() if detach_before_proj else z_scale
        z_obj_on_v = z_obj_on.detach() if (z_obj_on is not None and detach_before_proj) else z_obj_on
        z_depth_v = z_depth.detach() if (z_depth is not None and detach_before_proj) else z_depth
        z_features_v = z_features.detach() if detach_before_proj else z_features
        if not self.with_bg:
            z_bg_features = None
        z_bg_features_v = z_bg_features.detach() if (
                z_bg_features is not None and detach_before_proj) else z_bg_features
        z_base_var_v = z_base_var.detach() if z_base_var is not None else z_base_var
        z_score_v = z_score.detach() if z_score is not None else z_score
        if self.use_z_orig:
            z_orig_v = self.particles_anchor.unsqueeze(0).repeat(z_v.shape[0], z_v.shape[1], 1, 1)
        else:
            z_orig_v = None

        particle_projection = self.basic_particle_proj(z=z_v,
                                                       z_scale=z_scale_v,
                                                       z_obj_on=z_obj_on_v,
                                                       z_depth=z_depth_v,
                                                       z_features=z_features_v,
                                                       z_bg_features=z_bg_features_v,
                                                       z_base_var=z_base_var_v,
                                                       z_score=z_score_v,
                                                       z_orig=z_orig_v)
        # [bs, T, n_kp + 1, projection_dim or 2 * pctx_dim]

        # add entity pos embeddings
        if self.particle_embeddings.shape[2] == 1:
            p_embeddings = self.particle_embeddings.repeat(bs, timestep_horizon, self.n_kp_enc, 1)
        else:
            p_embeddings = self.particle_embeddings.repeat(bs, timestep_horizon, 1, 1)
        if patch_id_embed is not None:
            p_embeddings = p_embeddings + patch_id_embed
        if self.with_bg:
            bg_embeddings = self.bg_embeddings.repeat(bs, timestep_horizon, 1, 1)
            p_embeddings = torch.cat([p_embeddings, bg_embeddings], dim=2)
        particle_projection = particle_projection + p_embeddings

        c = c_a = c_r = c_g = l = None  # conditions
        if self.img_goal_condition and z_goal is None:
            particle_projection, goal_projection = particle_projection.split([particle_projection.shape[1] - 1, 1],
                                                                             dim=1)
            # goal_projection: [bs, 1, N, d]
            if detach_z_goal:
                goal_projection = goal_projection.detach()
            z_goal = self.goal_proj(goal_projection)
            l_or_cg = z_goal.repeat(1, particle_projection.shape[1], 1, 1)
            timestep_horizon = timestep_horizon - 1
            if self.img_goal_condition_type == 'cross':
                l = l_or_cg
            elif self.img_goal_condition_type == 'self':
                p_g = l_or_cg + self.goal_embeddings
                particle_projection = torch.cat([particle_projection, p_g], dim=2)  # [bs, T, n_p * 2]
            elif self.img_goal_condition_type == 'adaln':
                c_g = l_or_cg
                c = c_g
        elif self.img_goal_condition and z_goal is not None:
            l_or_cg = z_goal.repeat(1, particle_projection.shape[1], 1, 1)
            if self.img_goal_condition_type == 'cross':
                l = l_or_cg
            elif self.img_goal_condition_type == 'self':
                p_g = l_or_cg + self.goal_embeddings
                particle_projection = torch.cat([particle_projection, p_g], dim=2)  # [bs, T, n_p * 2]
            elif self.img_goal_condition_type == 'adaln':
                c_g = l_or_cg
                c = c_g

        if self.learned_ctx_token and not self.token_pool_cross_attn:
            if self.img_goal_condition and self.img_goal_condition_type == 'self':
                n_goal_particles = self.n_kp_enc
                if self.with_bg:
                    n_goal_particles += 1
                pp, pg = particle_projection.split([particle_projection.shape[2] - n_goal_particles, n_goal_particles],
                                                   dim=2)
                pc = self.ctx_token_embeddings.repeat(bs, timestep_horizon, 1, 1)
                particle_projection = torch.cat([pp, pc, pg], dim=1)
            else:
                particle_projection = torch.cat([particle_projection,
                                                 self.ctx_token_embeddings.repeat(bs, timestep_horizon, 1, 1)], dim=2)

        if self.random_action_condition:
            rand_action_horizon = timestep_horizon
            random_actions = torch.rand(bs, rand_action_horizon, self.random_action_dim,
                                        device=particle_projection.device)
            c_r = self.random_action_proj(random_actions)
            if self.action_as_particle:
                if len(c_r.shape) == 3:
                    c_r = c_r.unsqueeze(2)  # [bs, t, 1, f]
                    action_embeddings = self.random_action_embeddings.repeat(bs, timestep_horizon, 1, 1)
                    c_r = c_r + action_embeddings
                    particle_projection = torch.cat([particle_projection, c_r], dim=2)
            else:
                if len(c_r.shape) == 3:
                    n_particles = particle_projection.shape[2]
                    c_r = c_r.unsqueeze(2).repeat(1, 1, n_particles, 1)  # [bs, t, n, f]
                if c is None:
                    c = c_r
                else:
                    c = c + c_r

        if self.action_condition and actions is not None:
            if self.learn_null_action_embed and actions_mask is not None:
                # action_mask: [batch_size, T] or [batch_size, T, 1], 1 where use action, 0 replace action
                # Expand mask
                if len(actions_mask.shape) == 2:
                    actions_mask = actions_mask.bool().unsqueeze(-1)  # (batch_size, seq_len, 1)
                # Expand null embedding to match
                null_action_embeds = self.null_action_embeddings.expand(actions.size(0), actions.size(1), -1)

                # Blend
                actions = actions * actions_mask + null_action_embeds * (~actions_mask)

            c_a = self.action_proj(actions)
            if self.action_as_particle:
                if len(c_a.shape) == 3:
                    c_a = c_a.unsqueeze(2)  # [bs, t, 1, f]
                    action_embeddings = self.action_embeddings.repeat(bs, timestep_horizon, 1, 1)
                    c_a = c_a + action_embeddings
                    particle_projection = torch.cat([particle_projection, c_a], dim=2)
            else:
                if len(c_a.shape) == 3:
                    n_particles = particle_projection.shape[2]
                    c_a = c_a.unsqueeze(2).repeat(1, 1, n_particles, 1)  # [bs, t, n, f]
                if c is None:
                    c = c_a
                else:
                    c = c + c_a

        # views
        if self.n_views > 1:
            # [bs * n_views, t, n, d] -> [bs, n_views, t, n, d] -> [bs, t, n_views * n, d]
            particle_projection = particle_projection.view(-1, self.n_views, *particle_projection.shape[1:])
            # [bs, n_views, t, n, d]
            particle_projection = particle_projection.permute(0, 2, 1, 3, 4)  # [bs, t, n_views, n, d]
            # add view embeddings
            particle_projection = particle_projection + self.view_embeddings
            particle_projection = particle_projection.reshape(particle_projection.shape[0],
                                                              particle_projection.shape[1],
                                                              -1,
                                                              particle_projection.shape[-1])  # [bs, t, n_views * n, d]
            if c is not None:
                c = c.view(-1, self.n_views, *c.shape[1:])
                c = c.permute(0, 2, 1, 3, 4)  # [bs, t, n_views, n, d]
                c = c.reshape(c.shape[0], c.shape[1], -1, c.shape[-1])  # [bs, t, n_views * n, d]

        if self.language_condition and lang_embed is not None and l is None:
            l = self.lang_proj(lang_embed)
            if len(l.shape) == 3:
                # [bs, h=N_l, f]
                l = l.unsqueeze(1).repeat(1, timestep_horizon, 1, 1)  # [bs, t, h=N_l, f]
            elif l.shape[1] != timestep_horizon:
                # [bs, 1, h=N_l, f]
                l = l.repeat(1, timestep_horizon, 1, 1)  # [bs, t, h=N_l, f]
            # clip max tokens
            l = l[:, :, :self.language_max_len].contiguous()
            # add positional embeddings
            if self.lang_pos_embed is not None:
                l = l + self.lang_pos_embed[:, :, :self.language_max_len]
            if self.language_condition_type == 'self':
                particle_projection = torch.cat([particle_projection, l[:particle_projection.shape[0]]], dim=2)
                # [bs, T, n_p + n_l, dim], if n_views > 1, we make sure the effective batch size is the same
                # note that we assume the language instruction is the same for all views here
                l = None

        if self.particle_pool_adaln:
            c_p = self.particle_pool_proj(particle_projection)  # [bs, T, 1, dim]
            c_p = c_p.repeat(1, 1, particle_projection.shape[2], 1)
            if c is not None:
                c = c + c_p
            else:
                c = c_p

        if self.pos_embed_p_adaln:
            c_pe = self.pos_p_embeddings[:, :, :particle_projection.shape[2]].repeat(particle_projection.shape[0],
                                                                                     particle_projection.shape[1],
                                                                                     1, 1)
            if c is not None:
                c = c + c_pe
            else:
                c = c_pe

        if self.pos_embed_objon_adaln:
            z_obj_on_proj = z_obj_on_v[:, :timestep_horizon]
            c_objon = self.objon_embeddings(z_obj_on_proj)  # [bs, t, n, dim]
            # add zeros for the bg particle
            c_objon_bg = torch.zeros([c_objon.shape[0], c_objon.shape[1], 1, c_objon.shape[-1]], device=c_objon.device)
            c_objon = torch.cat([c_objon, c_objon_bg], dim=2)  # [bs, t, n + 1, dim]
            if self.n_views > 1:
                c_objon = c_objon.view(-1, self.n_views, *c_objon.shape[1:])
                c_objon = c_objon.permute(0, 2, 1, 3, 4)  # [bs, t, n_views, n, d]
                c_objon = c_objon.reshape(c_objon.shape[0], c_objon.shape[1], -1,
                                          c_objon.shape[-1])  # [bs, t, n_views * n, d]
            total_particles = particle_projection.shape[2]
            c_objon_other = torch.zeros(c_objon.shape[0], c_objon.shape[1], total_particles - c_objon.shape[2],
                                        c_objon.shape[3], device=c_objon.device)
            c_objon = torch.cat([c_objon, c_objon_other], dim=2)
            if c is not None:
                c = c + c_objon
            else:
                c = c_objon


        particles_out = self.pte(particle_projection, c=c, l=l)
        # [bs, ts, n, f]
        if self.language_condition and self.language_condition_type == 'self':
            particles_out = particles_out[:, :, :-self.language_max_len].contiguous()
        if self.n_views > 1:
            # [bs, t, n_views * n, d] -> [bs, n_views, t, n, d] -> [bs * n_views, t, n, d]
            particles_out = particles_out.view(particles_out.shape[0], particles_out.shape[1],
                                               self.n_views, -1, particles_out.shape[-1])
            particles_out = particles_out.permute(0, 2, 1, 3, 4)  # [bs, n_views, t, n, d]
            particles_out = particles_out.reshape(-1, *particles_out.shape[2:])  # [bs * n_views, t, n, d]
        if self.action_condition and self.action_as_particle:
            particles_out = particles_out[:, :, :-1].contiguous()
        if self.random_action_condition and self.action_as_particle:
            particles_out = particles_out[:, :, :-1].contiguous()
        if self.img_goal_condition and self.img_goal_condition_type == 'self':
            n_goal_particles = self.n_kp_enc
            if self.with_bg:
                n_goal_particles += 1
            particles_out = particles_out[:, :, :-n_goal_particles].contiguous()

        if (self.learned_ctx_token or self.ctx_pool_mode == 'last') and self.token_pool_cross_attn:
            if self.learned_ctx_token:
                q_particles = self.ctx_token_embeddings.repeat(bs, timestep_horizon, 1, 1)
                q_particles = q_particles.view(bs * timestep_horizon, 1, *q_particles.shape[2:])
                # [bs * t, 1, 1, embed_dim]
                kv_particles = particles_out[:, :, :self.n_kp_enc + 1]  # only fg + bg particles
                kv_particles = kv_particles.reshape(bs * timestep_horizon, 1, *kv_particles.shape[2:])
                # [bs * t, 1, n_particles + 1, embed_dim]
            else:
                # 'last' pooling
                kv_particles, q_particles = particles_out.split([particles_out.shape[2] - 1, 1], dim=2)
            ctx_ca = self.cross_attn_block(q_particles, kv_particles)
            # [bs * t, 1, 1, embed_dim]
            particles_out = torch.cat([kv_particles, ctx_ca], dim=2)
            particles_out = particles_out.view(bs, timestep_horizon, *particles_out.shape[2:])

        if encode_posterior:
            if self.global_ctx_pool:
                # EXPERIMENTAL, NOT USED IN THE PAPER
                # global
                particle_decoder_out_global = self.global_posterior_decoder(particles_out, deterministic=deterministic)
                # unpack
                mu_context_global = particle_decoder_out_global['mu_context']
                logvar_context_global = particle_decoder_out_global['logvar_context']
                z_context_global = particle_decoder_out_global['z_context']
                # local
                if self.condition_local_on_global:
                    c = z_context_global
                else:
                    c = None
                particle_decoder_out = self.posterior_decoder(particles_out, c=c, deterministic=deterministic)
                # unpack
                mu_context = particle_decoder_out['mu_context']
                logvar_context = particle_decoder_out['logvar_context']
                z_context = particle_decoder_out['z_context']

                if self.global_local_fuse_mode != 'none':
                    if len(z_context_global.shape) != len(z_context.shape):
                        z_context_global = z_context_global.unsqueeze(2).repeat(1, 1, z_context.shape[2], 1)
                    elif z_context_global.shape[2] != z_context.shape[2]:
                        z_context_global = z_context_global.repeat(1, 1, z_context.shape[2], 1)
                    if self.global_local_fuse_mode == 'concat':
                        z_context = torch.cat([z_context, z_context_global], dim=-1)
                    else:
                        # add
                        z_context = z_context + z_context_global
            else:
                mu_context_global = logvar_context_global = z_context_global = None
                particle_decoder_out = self.posterior_decoder(particles_out, deterministic=deterministic)
                # unpack
                mu_context = particle_decoder_out['mu_context']
                logvar_context = particle_decoder_out['logvar_context']
                z_context = particle_decoder_out['z_context']
        else:
            mu_context = logvar_context = z_context = None
            mu_context_global = logvar_context_global = z_context_global = None

        if encode_prior:
            if self.global_ctx_pool:
                # EXPERIMENTAL, NOT USED IN THE PAPER
                # global
                prior_decoder_out_global = self.global_prior_decoder(particles_out, deterministic=deterministic)
                # unpack
                mu_context_global_dyn = prior_decoder_out_global['mu_context']
                logvar_context_global_dyn = prior_decoder_out_global['logvar_context']
                z_context_global_dyn = prior_decoder_out_global['z_context']

                # local
                if self.condition_local_on_global:
                    if z_context_global is None:
                        # sampling
                        c = z_context_global_dyn
                    else:
                        # teacher-forcing: shift inverse-model output by one timestep
                        c = torch.cat([z_context_global[:, 1:], z_context_global[:, -1:]], dim=1)
                else:
                    c = None
                prior_decoder_out = self.prior_decoder(particles_out, c=c, deterministic=deterministic)
                # unpack
                mu_context_dyn = prior_decoder_out['mu_context']
                logvar_context_dyn = prior_decoder_out['logvar_context']
                z_context_dyn = prior_decoder_out['z_context']

                if self.global_local_fuse_mode != 'none':
                    if len(z_context_global_dyn.shape) != len(z_context_dyn.shape):
                        z_context_global_dyn = z_context_global_dyn.unsqueeze(2).repeat(1, 1, z_context_dyn.shape[2], 1)
                    elif z_context_global_dyn.shape[2] != z_context_dyn.shape[2]:
                        z_context_global_dyn = z_context_global_dyn.repeat(1, 1, z_context_dyn.shape[2], 1)
                    if self.global_local_fuse_mode == 'concat':
                        z_context_dyn = torch.cat([z_context_dyn, z_context_global_dyn], dim=-1)
                    else:
                        # add
                        z_context_dyn = z_context_dyn + z_context_global_dyn
            else:
                mu_context_global_dyn = logvar_context_global_dyn = z_context_global_dyn = None
                prior_decoder_out = self.prior_decoder(particles_out, deterministic=deterministic)
                # unpack
                mu_context_dyn = prior_decoder_out['mu_context']
                logvar_context_dyn = prior_decoder_out['logvar_context']
                z_context_dyn = prior_decoder_out['z_context']
        else:
            mu_context_dyn = logvar_context_dyn = z_context_dyn = None
            mu_context_global_dyn = logvar_context_global_dyn = z_context_global_dyn = None

        encode_dict = {'mu_context': mu_context, 'logvar_context': logvar_context, 'z_context': z_context,
                       'mu_context_dyn': mu_context_dyn, 'logvar_context_dyn': logvar_context_dyn,
                       'z_context_dyn': z_context_dyn,
                       'mu_context_global': mu_context_global, 'logvar_context_global': logvar_context_global,
                       'z_context_global': z_context_global,
                       'mu_context_global_dyn': mu_context_global_dyn,
                       'logvar_context_global_dyn': logvar_context_global_dyn,
                       'z_context_global_dyn': z_context_global_dyn,
                       'z_goal_proj': z_goal,
                       }
        return encode_dict

    def forward(self, z, z_scale, z_obj_on, z_depth, z_features, z_bg_features=None, z_base_var=None,
                z_score=None, patch_id_embed=None, deterministic=False, warmup=False,
                encode_posterior=True, encode_prior=True, actions=None, actions_mask=None, lang_embed=None,
                z_goal=None):
        output_dict = self.encode_all(z, z_scale, z_obj_on, z_depth, z_features, z_bg_features, z_base_var, z_score,
                                      patch_id_embed, deterministic=deterministic, warmup=warmup,
                                      encode_posterior=encode_posterior, encode_prior=encode_prior, actions=actions,
                                      actions_mask=actions_mask, lang_embed=lang_embed, z_goal=z_goal)
        return output_dict


class DLPDynamics(nn.Module):
    def __init__(self,
                 features_dim,
                 bg_features_dim,
                 hidden_dim,
                 projection_dim,
                 n_head=8,  # Number of attention heads
                 n_layer=2,  # Number of attention layers
                 block_size=12,  # Timestep horizon
                 dropout=0.1,
                 kp_activation='tanh',  # Keypoint activation function
                 predict_delta=False,  # Predict position deltas instead of absolute positions
                 max_delta=1.5,  # Maximum delta value for predictions
                 positional_bias=False,  # Use positional bias in dynamics
                 max_particles=None,  # Maximum particles for positional bias
                 context_dim=7,  # Context latent dimension
                 attn_norm_type='rms',  # Normalization type for attention
                 n_fg_particles=None,  # Number of foreground particles
                 ctx_pool_mode='none',  # Context pooling mode
                 ctx_mode='adaln',  # Conditioning type for latent context
                 particle_score=False,  # Include particle confidence scores
                 particle_positional_embed=True,  # Use positional embeddings for particles
                 scale_anchor=None,  # Anchor scale for particle dynamics
                 init_std=0.02,  # Standard deviation for initialization
                 pint_ctx_layers=6,  # Number of PINT context transformer layers
                 pint_ctx_heads=8,  # Number of PINT context transformer heads
                 ctx_dist='gauss',  # Context distribution type ('gauss' or 'categorical')
                 n_ctx_categories=4,  # Number of context categories
                 n_ctx_classes=4,  # Number of context classes per category
                 residual_modulation=True,  # Use residual modulation for dynamics
                 context_gate=True,  # Use gating for context features
                 context_decoder=None,  # Decoder configuration for context
                 features_dist='gauss',  # Distribution type for features
                 n_fg_categories=8,  # Number of foreground feature categories
                 n_fg_classes=4,  # Number of foreground feature classes per category
                 n_bg_categories=4,  # Number of background feature categories
                 n_bg_classes=4,  # Number of background feature classes per category
                 particle_anchors=None,  # Anchors for particles
                 scale_init=None,  # Initial scale for particles
                 obj_on_min=1e-4,  # Minimum transparency concentration value
                 obj_on_max=100,  # Maximum transparency concentration value
                 use_z_orig=True,  # Include original patch coordinates in features
                 n_views=1,  # number of input views (e.g., multiple cameras)
                 # external conditioning
                 action_condition=False,  # condition on actions
                 action_dim=0,  # dimension of input actions
                 random_action_condition=False,  # condition on random actions
                 random_action_dim=0,  # dimension of sampled random actions
                 null_action_embed=False,  # learn a "no-input-action" embedding, to learn on action-free videos as well
                 pos_embed_t_adaln=True,  # pos embeddings for timesteps using adaln
                 pos_embed_p_adaln=True,  # pos embeddings for particles using adaln
                 pos_embed_objon_adaln=False,  # pos embeddings for particles transparency using adaln
                 # language_condition=False,  # condition on language embedding
                 # language_embed_dim=0,  # embedding dimension for each token
                 # language_max_len=64,  # maximum tokens per prompt
                 ):
        super(DLPDynamics, self).__init__()
        """
        Args:
        features_dim (int): Dimension of visual features.
        bg_features_dim (int): Dimension of background features.
        hidden_dim (int): Hidden dimension for dynamics layers.
        projection_dim (int): Projection dimension for dynamics.
        n_head (int): Number of attention heads. Defaults to 8.
        n_layer (int): Number of attention layers. Defaults to 2.
        block_size (int): Timestep horizon for dynamics. Defaults to 12.
        dropout (float): Dropout rate for transformer layers. Defaults to 0.1.
        kp_activation (str): Activation function for keypoints ('tanh' or 'relu'). Defaults to 'tanh'.
        predict_delta (bool): Predict position deltas instead of absolute positions. Defaults to False.
        max_delta (float): Maximum value for delta predictions. Defaults to 1.5.
        positional_bias (bool): Use positional bias in dynamics computations. Defaults to False.
        max_particles (Optional[int]): Maximum number of particles for positional bias. Defaults to None.
        context_dim (int): Dimension of the latent context. Defaults to 7.
        attn_norm_type (str): Normalization type for attention ('rms' or 'layer'). Defaults to 'rms'.
        n_fg_particles (Optional[int]): Number of foreground particles. Defaults to None.
        ctx_pool_mode (str): Pooling mode for context ('none', 'mean', etc.). Defaults to 'none'.
        ctx_mode (str): Conditioning mode for latent context ('adaln', etc.). Defaults to 'adaln'.
        particle_score (bool): Include particle confidence scores as features. Defaults to False.
        particle_positional_embed (bool): Use positional embeddings for particles. Defaults to True.
        scale_anchor (Optional[float]): Anchor scale for dynamics. Defaults to None.
        init_std (float): Standard deviation for parameter initialization. Defaults to 0.02.
        pint_ctx_layers (int): Number of PINT context transformer layers. Defaults to 6.
        pint_ctx_heads (int): Number of PINT context transformer heads. Defaults to 8.
        ctx_dist (str): Distribution type for context ('gauss' or 'categorical'). Defaults to 'gauss'.
        n_ctx_categories (int): Number of context categories if categorical. Defaults to 4.
        n_ctx_classes (int): Number of context classes per category. Defaults to 4.
        residual_modulation (bool): Apply residual modulation to dynamics features. Defaults to True.
        context_gate (bool): Use gating mechanisms for context integration. Defaults to True.
        context_decoder (Optional[str]): Configuration of context decoder. Defaults to None.
        features_dist (str): Distribution type for features ('gauss' or 'categorical'). Defaults to 'gauss'.
        n_fg_categories (int): Number of foreground feature categories. Defaults to 8.
        n_fg_classes (int): Number of foreground feature classes per category. Defaults to 4.
        n_bg_categories (int): Number of background feature categories. Defaults to 4.
        n_bg_classes (int): Number of background feature classes per category. Defaults to 4.
        particle_anchors (Optional[Any]): Anchors for particle initialization. Defaults to None.
        scale_init (Optional[float]): Initial scale value for particles. Defaults to None.
        obj_on_min (float): Minimum concentration for Beta distribution in transparency. Defaults to 1e-4.
        obj_on_max (float): Maximum concentration for Beta distribution in transparency. Defaults to 100.
        use_z_orig (bool): Include original patch coordinates in particle features. Defaults to True.

        DLP Dynamics with Context:
        This module predicts particle dynamics across timesteps using a PINT-based transformer. Each particle's attributes,
        such as position, scale, and transparency, evolve over time, guided by latent context variables.

        """

        self.predict_delta = predict_delta
        self.projection_dim = projection_dim
        self.hidden_dim = hidden_dim
        self.max_delta = max_delta
        self.max_particles = max_particles  # for positional bias
        self.n_fg_particles = n_fg_particles
        self.learned_feature_dim = features_dim
        self.learned_bg_feature_dim = bg_features_dim
        self.features_dist = features_dist
        self.n_fg_categories = n_fg_categories
        self.n_fg_classes = n_fg_classes
        self.n_bg_categories = n_bg_categories
        self.n_bg_classes = n_bg_classes
        self.context_dist = ctx_dist
        self.n_ctx_categories = n_ctx_categories
        self.n_ctx_classes = n_ctx_classes
        self.context_dim = context_dim
        self.particle_score = particle_score
        self.attn_norm_type = attn_norm_type
        assert ctx_mode in ['add', 'cat', 'token', 'film', 'adaln']
        self.ctx_mode = ctx_mode
        self.ctx_pool_mode = ctx_pool_mode
        # ['last'-last token is ctx, otherwise, use pool op over the particles to generate context]
        self.init_std = init_std
        self.obj_on_min = obj_on_min
        self.obj_on_max = obj_on_max
        self.use_z_orig = use_z_orig  # use the origin of the particles (the center of the source patch) as attribute
        self.n_views = n_views  # number of input views (e.g., multiple cameras)
        use_norm_layer = True  # norm layer in the projections modules

        # actions
        self.action_condition = action_condition
        self.action_dim = action_dim
        self.random_action_condition = random_action_condition
        self.random_action_dim = random_action_dim
        self.learn_null_action_embed = null_action_embed

        # token adaln
        self.pos_embed_t_adaln = pos_embed_t_adaln
        self.pos_embed_p_adaln = pos_embed_p_adaln
        self.pos_embed_objon_adaln = pos_embed_objon_adaln

        if self.learn_null_action_embed and self.action_condition:
            self.null_action_embeddings = nn.Parameter(
                self.init_std * torch.randn(1, 1, self.action_dim))
        else:
            self.null_action_embeddings = None

        if scale_anchor is None:
            self.register_buffer('scale_anchor', torch.tensor(0.0))
        else:
            self.register_buffer('scale_anchor',
                                 torch.tensor(np.log(0.75 * scale_anchor / (1 - 0.75 * scale_anchor + 1e-5))))
        if particle_anchors is None:
            self.register_buffer('particles_anchor', torch.zeros(1, 1, self.n_fg_particles))
            self.use_z_orig = False
        else:
            self.register_buffer('particles_anchor', particle_anchors)

        self.particle_pos_embed = particle_positional_embed and not self.pos_embed_p_adaln

        proj_max_particles = self.n_fg_particles
        self.particle_projection = ParticleFeatureProjection(features_dim, bg_features_dim,
                                                             hidden_dim, self.projection_dim, context_dim=context_dim,
                                                             max_particles=proj_max_particles, add_embedding=True,
                                                             ctx_cond_mode=self.ctx_mode,
                                                             particle_positional_embed=self.particle_pos_embed,
                                                             init_std=self.init_std, particle_score=self.particle_score,
                                                             norm_layer=use_norm_layer,
                                                             use_z_orig=self.use_z_orig)
        if self.ctx_mode == 'adaln' and self.context_dim > 0:
            self.context_proj = nn.Linear(self.context_dim, hidden_dim)
            # self.context_proj = nn.Sequential(nn.Linear(self.context_dim, hidden_dim),
            #                                   RMSNorm(hidden_dim),
            #                                   nn.GELU())
            if self.action_condition and self.action_dim > 0:
                self.action_proj = nn.Linear(self.action_dim, hidden_dim)
            else:
                self.action_proj = None
            if self.random_action_condition and self.random_action_dim > 0:
                self.random_action_proj = nn.Linear(self.random_action_dim, hidden_dim)
            else:
                self.random_action_proj = None
            self.cond_activation = nn.GELU()
        else:
            self.context_proj = None
            self.action_proj = None
            self.cond_activation = None

        if self.n_views > 1:
            self.view_embeddings = nn.Parameter(self.init_std * torch.randn(1, self.n_views, 1, 1, self.projection_dim))
        else:
            self.view_embeddings = None

        if self.pos_embed_p_adaln and (self.ctx_mode == 'adaln'):
            n_particles = self.n_views * (self.n_fg_particles + 1)
            self.pos_p_embeddings = nn.Parameter(
                self.init_std * torch.randn(1, n_particles, 1, hidden_dim))
        if self.pos_embed_objon_adaln:
            self.objon_embeddings = nn.Sequential(nn.Linear(1, hidden_dim),
                                                  RMSNorm(hidden_dim),
                                                  nn.GELU())

        self.particle_transformer = ParticleSpatioTemporalTransformer(self.projection_dim, n_head, n_layer,
                                                                      block_size, self.projection_dim,
                                                                      attn_pdrop=dropout, resid_pdrop=dropout,
                                                                      hidden_dim_multiplier=4,
                                                                      positional_bias=positional_bias,
                                                                      activation='gelu',
                                                                      max_particles=max_particles,
                                                                      norm_type=attn_norm_type,
                                                                      init_std=self.init_std, causal=True,
                                                                      context_cond=(self.ctx_mode == 'adaln'),
                                                                      residual_modulation=residual_modulation,
                                                                      context_gate=context_gate,
                                                                      pos_embed_t_adaln=self.pos_embed_t_adaln)

        self.particle_decoder = ParticleFeatureDecoderDyn(self.projection_dim, features_dim, bg_features_dim,
                                                          hidden_dim, kp_activation=kp_activation, max_delta=max_delta,
                                                          context_dim=context_dim,
                                                          ctx_as_token=(self.ctx_mode == 'token'),
                                                          dec_ctx=False, norm_type=attn_norm_type, dropout=dropout,
                                                          particle_score=self.particle_score,
                                                          features_dist=self.features_dist,
                                                          n_fg_categories=n_fg_categories,
                                                          n_fg_classes=n_fg_classes, n_bg_categories=n_bg_categories,
                                                          n_bg_classes=n_bg_classes, scale_init=scale_init)
        self.context_decoder = context_decoder

    def init_weights(self):
        self.particle_projection.init_weights()
        self.particle_transformer.init_weights()
        self.particle_decoder.init_weights()

    def sample(self, z, z_scale, z_obj_on, z_depth, z_features, z_bg_features, z_context=None,
               z_score=None, steps=10, deterministic=False, deterministic_particles=True, actions=None,
               actions_mask=None, lang_embed=None, z_goal=None, return_context_posterior=False):
        """
        Samples a sequence of particle states based on the given conditioning inputs and internal model dynamics.

        Args:
            z (torch.Tensor): Initial particle positions, shape `(batch_size, timesteps, n_particles, 2)`.
            z_scale (torch.Tensor): Scale of particles, shape `(batch_size, timesteps, n_particles, 2)`.
            z_obj_on (torch.Tensor): transparency probabilities, shape `(batch_size, timesteps, n_particles, 1)`.
            z_depth (torch.Tensor): Depth of particles, shape `(batch_size, timesteps, n_particles, 1)`.
            z_features (torch.Tensor): Particle features, shape `(batch_size, timesteps, n_particles, in_features_dim)`.
            z_bg_features (torch.Tensor): Background features, shape `(batch_size, timesteps, bg_features_dim)`.
            z_context (torch.Tensor, optional): Dynamic context encoding, shape `(batch_size, timesteps, context_dim)`.
            z_score (torch.Tensor, optional): Particle scores, shape `(batch_size, timesteps, n_particles, 1)`.
                If not provided, it defaults to zeros.
            steps (int): Number of forward sampling steps. Defaults to 10.
            deterministic (bool): If True, the sampling is deterministic. Defaults to False.
            deterministic_particles (bool): If True, uses deterministic particles during sampling. Defaults to True.

        Returns:
            dict: A dictionary containing sampled outputs:
                - `z` (torch.Tensor): Updated particle positions.
                - `z_scale` (torch.Tensor): Updated particle scales.
                - `z_obj_on` (torch.Tensor): Updated transparency probabilities.
                - `z_depth` (torch.Tensor): Updated particle depths.
                - `z_features` (torch.Tensor): Updated particle features.
                - `z_bg_features` (torch.Tensor): Updated background features.
                - `z_context` (torch.Tensor): Generated or updated context.
                - `z_score` (torch.Tensor): Updated particle scores.

        Notes:
            - The function iteratively generates future particle states using a transformer-based architecture.
            - Reparameterization techniques are employed for stochastic sampling when `deterministic=False`.
            - Quadratic complexity is involved in the sampling process due to the block size and transformer operations.
        """
        block_size = self.particle_transformer.get_block_size()
        # z, z_scale: [bs, T, n_particles, 2]
        # z_depth, z_obj_on: [bs, T, n_particles, 1]
        # z_features: [bs, T, n_particles, in_features_dim]
        # z_bg_features: [bs, T, bg_features_dim]
        # z_context: [bs, T, context_dim]
        if z_score is None:
            z_score = torch.zeros(z.shape[0], z.shape[1], z.shape[2], 1, dtype=torch.float, device=z.device)

        mu_context_posterior = z_context_posterior = z_context  # initialize in case they are needed
        bs, timestep_horizon, n_particles, _ = z.shape
        for k in range(steps):
            # first generate context, then use the context with the current particles
            if self.context_dim > 0:
                start_step = max(z.shape[1] - block_size, 0)
                end_step = min(start_step + block_size, z.shape[1])
                # check if context was provided
                if z_context is None or z_context.shape[1] < z.shape[1]:
                    # generate context
                    if actions is not None:
                        actions_in = actions[:, start_step:end_step]
                    else:
                        actions_in = None
                    if actions_mask is not None:
                        actions_mask_in = actions_mask[:, start_step:end_step]
                    else:
                        actions_mask_in = None
                    ctx_dec_out = self.context_decoder(z=z[:, -block_size:],
                                                       z_scale=z_scale[:, -block_size:],
                                                       z_obj_on=z_obj_on[:, -block_size:],
                                                       z_depth=z_depth[:, -block_size:],
                                                       z_features=z_features[:, -block_size:],
                                                       z_bg_features=z_bg_features[:, -block_size:],
                                                       z_score=z_score[:, -block_size:],
                                                       deterministic=deterministic,
                                                       encode_posterior=return_context_posterior,
                                                       encode_prior=True,
                                                       actions=actions_in,
                                                       actions_mask=actions_mask_in,
                                                       lang_embed=lang_embed,
                                                       z_goal=z_goal)
                    z_context_last = ctx_dec_out['z_context_dyn'][:, -1:]

                    new_z_context = z_context_last
                    if z_context is None:
                        # that means that it the very first step
                        z_context = new_z_context
                    else:
                        z_context = torch.cat([z_context, new_z_context], dim=1)
                        if return_context_posterior:
                            mu_context_posterior_last = ctx_dec_out['mu_context']
                            z_context_posterior_last = ctx_dec_out['z_context']
                            if z_context_posterior_last.shape[1] > 1:
                                new_mu_context_posterior = mu_context_posterior_last[:, -1:]
                                new_z_context_posterior = z_context_posterior_last[:, -1:]
                                if z_context_posterior is None:
                                    mu_context_posterior = new_mu_context_posterior
                                    z_context_posterior = new_z_context_posterior
                                else:
                                    mu_context_posterior = torch.cat([mu_context_posterior,
                                                                      new_mu_context_posterior], dim=1)
                                    z_context_posterior = torch.cat([z_context_posterior,
                                                                     new_z_context_posterior], dim=1)

                # prepare input to dyn module
                # start_step = max(z.shape[1] - block_size, 0)
                # end_step = min(start_step + block_size, z.shape[1])
                z_context_v = z_context[:, start_step:end_step].reshape(-1, *z_context.shape[2:])
            else:
                z_context_v = None

            # project particles
            z_v = z[:, -block_size:].reshape(-1, *z.shape[2:])
            z_scale_v = z_scale[:, -block_size:].reshape(-1, *z_scale.shape[2:])
            z_obj_on_v = z_obj_on[:, -block_size:].reshape(-1, *z_obj_on.shape[2:])
            z_depth_v = z_depth[:, -block_size:].reshape(-1, *z_depth.shape[2:])
            z_features_v = z_features[:, -block_size:].reshape(-1, *z_features.shape[2:])
            z_bg_features_v = z_bg_features[:, -block_size:].reshape(-1, *z_bg_features.shape[2:])
            z_score_v = z_score[:, -block_size:].reshape(-1, *z_score.shape[2:])
            if self.use_z_orig:
                z_orig_v = self.particles_anchor.repeat(z_v.shape[0], 1, 1)
            else:
                z_orig_v = None

            particle_projection = self.particle_projection(z_v, z_scale_v, z_obj_on_v, z_depth_v, z_features_v,
                                                           z_bg_features_v, z_context_v, z_score_v, z_orig_v)
            # [bs * T, n_particles + 1, projection_dim]
            particle_proj_int = particle_projection
            # unroll forward
            particle_proj_int = particle_proj_int.view(bs, -1, *particle_proj_int.shape[1:])
            # [bs, T, n_particles + 2, projection_dim]
            particle_proj_int = particle_proj_int.permute(0, 2, 1, 3)
            # [bs, n_particles + 2, T, projection_dim]
            if self.ctx_mode == 'adaln':
                if self.random_action_condition:
                    random_actions = torch.rand(particle_proj_int.shape[0], particle_proj_int.shape[2],
                                                self.random_action_dim, device=particle_proj_int.device)
                    c_random_action = self.random_action_proj(random_actions)
                    if len(c_random_action.shape) == 3:
                        c_random_action = c_random_action.unsqueeze(1).repeat(1, particle_proj_int.shape[1], 1,
                                                                              1)  # [bs, n, t, f]
                else:
                    c_random_action = 0

                if self.action_condition and actions is not None:
                    start_step = max(z.shape[1] - block_size, 0)
                    end_step = min(start_step + block_size, z.shape[1])
                    actions_v = actions[:, start_step:end_step]
                    if self.learn_null_action_embed and actions_mask is not None:
                        # action_mask: [batch_size, T] or [batch_size, T, 1], 1 where use action, 0 replace action
                        # Expand mask
                        if len(actions_mask.shape) == 2:
                            actions_mask_v = actions_mask[:, start_step:end_step].bool().unsqueeze(
                                -1)  # (batch_size, seq_len, 1)
                        # Expand null embedding to match
                        null_action_embeds = self.null_action_embeddings.expand(actions_v.size(0), actions_v.size(1),
                                                                                -1)

                        # Blend
                        actions_v = actions_v * actions_mask_v + null_action_embeds * (~actions_mask_v)

                    c_action = self.action_proj(actions_v)
                    if len(c_action.shape) == 3:
                        c_action = c_action.unsqueeze(1).repeat(1, particle_proj_int.shape[1], 1, 1)  # [bs, n, t, f]
                else:
                    c_action = 0
                c = self.context_proj(z_context_v)
                c = c.reshape(bs, -1, *c.shape[1:])
                if len(c.shape) == 3:
                    c = c.unsqueeze(1)  # [bs, 1, t, f]
                elif c.shape[2] != particle_proj_int.shape[1]:
                    c = c.permute(0, 2, 1, 3)  # [bs, n + 1, t, f]
                    c = c.repeat(1, particle_proj_int.shape[1], 1, 1)  # [bs, 1, t, f]
                else:
                    c = c.permute(0, 2, 1, 3)  # [bs, n + 1, t, f]
                c = c + c_action + c_random_action
                c = self.cond_activation(c)
            else:
                c = None

            if self.n_views > 1:
                # [bs * n_views, n, T, d] -> [bs, n_views, n, T, d] -> [bs, n_views * n, T, d]
                particle_proj_int = particle_proj_int.view(-1, self.n_views, particle_proj_int.shape[1],
                                                           *particle_proj_int.shape[2:])  # [bs, n_views, n, T, d]
                particle_proj_int = particle_proj_int + self.view_embeddings
                particle_proj_int = particle_proj_int.reshape(particle_proj_int.shape[0], -1,
                                                              *particle_proj_int.shape[3:])  # [bs, n_views * n, T, d]
                if c is not None:
                    c = c.reshape(-1, self.n_views * c.shape[1], *c.shape[2:])

            if c is not None and self.pos_embed_p_adaln:
                c_pe = self.pos_p_embeddings.repeat(c.shape[0], 1, c.shape[2], 1)
                c = c + c_pe

            if c is not None and self.pos_embed_objon_adaln:
                c_objon = self.objon_embeddings(z_obj_on[:, -block_size:])  # [bs, t, n, dim]
                c_objon_bg = torch.zeros(c_objon.shape[0], c_objon.shape[1], 1, c_objon.shape[-1],
                                         device=c_objon.device)
                c_objon = torch.cat([c_objon, c_objon_bg], dim=2)  # [bs, t, n + 1, dim]
                c_objon = c_objon.permute(0, 2, 1, 3)  # [bs, n + 1, t, dim]
                if self.n_views > 1:
                    c_objon = c_objon.reshape(-1, self.n_views * c_objon.shape[1], c_objon.shape[2], c_objon.shape[-1])
                    # [bs, n_views * (n + 1), t, dim]
                c = c + c_objon

            particles_trans = self.particle_transformer(particle_proj_int, c)
            if self.n_views > 1:
                # [bs, n_views * n, T, d] -> [bs * n_views, n, T, d]
                particles_trans = particles_trans.reshape(bs, -1, *particles_trans.shape[2:])
            particles_trans = particles_trans[:, :, -1]  # [bs, (n_particles + 1), projection_dim]
            # [bs, n_particles + 1, projection_dim]
            # decode transformer output
            # [bs, n_particles + 1, projection_dim]
            particle_decoder_out = self.particle_decoder(particles_trans)
            mu = particle_decoder_out['mu_offset']
            logvar = particle_decoder_out['logvar_offset']

            obj_on_a_gate = (particle_decoder_out['lobj_on_a']).sigmoid()
            obj_on_a = ((1 - obj_on_a_gate) * self.obj_on_min + obj_on_a_gate * self.obj_on_max).exp()
            obj_on_b_gate = 1 - (
                    particle_decoder_out['lobj_on_b'] * 0 + particle_decoder_out['lobj_on_a']).sigmoid()
            obj_on_b = ((1 - obj_on_b_gate) * self.obj_on_min + obj_on_b_gate * self.obj_on_max).exp()

            mu_depth = particle_decoder_out['mu_depth']
            logvar_depth = particle_decoder_out['logvar_depth']
            mu_scale = particle_decoder_out['mu_scale']
            logvar_scale = particle_decoder_out['logvar_scale']
            mu_features = particle_decoder_out['mu_features']
            logvar_features = particle_decoder_out['logvar_features']
            mu_bg_features = particle_decoder_out['mu_bg_features']
            logvar_bg_features = particle_decoder_out['logvar_bg_features']
            mu_score = particle_decoder_out['mu_score']
            logvar_score = particle_decoder_out['logvar_score']

            # reshape to [bs, t, ...]
            mu = mu.view(bs, 1, *mu.shape[1:])
            logvar = logvar.view(bs, 1, *logvar.shape[1:])
            obj_on_a = obj_on_a.view(bs, 1, *obj_on_a.shape[1:])
            obj_on_b = obj_on_b.view(bs, 1, *obj_on_b.shape[1:])
            mu_depth = mu_depth.view(bs, 1, *mu_depth.shape[1:])
            logvar_depth = logvar_depth.view(bs, 1, *logvar_depth.shape[1:])
            mu_scale = mu_scale.view(bs, 1, *mu_scale.shape[1:])
            logvar_scale = logvar_scale.view(bs, 1, *logvar_scale.shape[1:])
            mu_features = mu_features.view(bs, 1, *mu_features.shape[1:])
            logvar_features = logvar_features.view(bs, 1, *logvar_features.shape[1:])
            mu_bg_features = mu_bg_features.view(bs, 1, *mu_bg_features.shape[1:])
            logvar_bg_features = logvar_bg_features.view(bs, 1, *logvar_bg_features.shape[1:])
            if self.particle_score and mu_score is not None:
                mu_score = mu_score.view(bs, 1, *mu_score.shape[1:])
                logvar_score = logvar_score.view(bs, 1, *logvar_score.shape[1:])

            mu_scale = mu_scale + self.scale_anchor
            if self.use_z_orig:
                mu = self.particles_anchor.unsqueeze(1) + mu

            if self.predict_delta:
                mu = z[:, -1].unsqueeze(1) + mu
                # mu_scale = z_scale[:, -1].unsqueeze(1) + mu_scale
                # mu_depth = z_depth[:, -1].unsqueeze(1) + mu_depth
                # mu_features = z_features[:, -1].unsqueeze(1) + mu_features
                # mu_bg_features = z_bg_features[:, -1].unsqueeze(1) + mu_bg_features

            beta_dist = Beta(obj_on_a, obj_on_b)

            if deterministic:
                new_z = mu
                new_z_depth = mu_depth
                new_z_scale = mu_scale
                if self.features_dist == 'categorical':
                    logits = mu_features.view(*mu_features.shape[:-1], self.n_fg_categories, self.n_fg_classes)
                    # [bs, T, n_p, n_categories, n_classes]
                    probs = logits.softmax(dim=-1)  # [bs, T, n_p, n_categories, n_classes]
                    samples = torch.argmax(probs.view(-1, probs.shape[-1]), dim=-1, keepdim=True)
                    samples = F.one_hot(samples.squeeze(-1), num_classes=self.n_fg_classes)
                    samples = samples.view(probs.shape)
                    # straight-through
                    new_z_features = samples.detach() + (probs - probs.detach())
                    new_z_features = new_z_features.view(*mu_features.shape)  # [bs, T, n_p, n_categories * n_classes]

                    logits_bg = mu_bg_features.view(*mu_bg_features.shape[:-1], self.n_bg_categories, self.n_bg_classes)
                    # [bs, T, n_p, n_categories, n_classes]
                    probs_bg = logits_bg.softmax(dim=-1)  # [bs, T, n_p, n_categories, n_classes]
                    samples_bg = torch.argmax(probs_bg.view(-1, probs_bg.shape[-1]), dim=-1, keepdim=True)
                    samples_bg = F.one_hot(samples_bg.squeeze(-1), num_classes=self.n_bg_classes)
                    samples_bg = samples_bg.view(probs_bg.shape)
                    # straight-through
                    new_z_bg_features = samples_bg.detach() + (probs_bg - probs_bg.detach())
                    new_z_bg_features = new_z_bg_features.view(*mu_bg_features.shape)
                    # [bs, T, n_p, n_categories * n_classes]
                else:
                    new_z_features = mu_features
                    new_z_bg_features = mu_bg_features
                new_z_obj_on = beta_dist.mean
                if self.particle_score and mu_score is not None:
                    new_z_score = mu_score
                else:
                    new_z_score = logvar.sum(-1, keepdim=True)
            else:
                if deterministic_particles:
                    new_z = mu
                    new_z_depth = mu_depth
                    new_z_scale = mu_scale
                    if self.features_dist == 'categorical':
                        logits = mu_features.view(*mu_features.shape[:-1], self.n_fg_categories, self.n_fg_classes)
                        # [bs, T, n_p, n_categories, n_classes]
                        probs = logits.softmax(dim=-1)  # [bs, T, n_p, n_categories, n_classes]
                        samples = torch.argmax(probs.view(-1, probs.shape[-1]), dim=-1, keepdim=True)
                        samples = F.one_hot(samples.squeeze(-1), num_classes=self.n_fg_classes)
                        samples = samples.view(probs.shape)
                        # straight-through
                        new_z_features = samples.detach() + (probs - probs.detach())
                        new_z_features = new_z_features.view(
                            *mu_features.shape)  # [bs, T, n_p, n_categories * n_classes]

                        logits_bg = mu_bg_features.view(*mu_bg_features.shape[:-1], self.n_bg_categories,
                                                        self.n_bg_classes)
                        # [bs, T, n_p, n_categories, n_classes]
                        probs_bg = logits_bg.softmax(dim=-1)  # [bs, T, n_p, n_categories, n_classes]
                        samples_bg = torch.argmax(probs_bg.view(-1, probs_bg.shape[-1]), dim=-1, keepdim=True)
                        samples_bg = F.one_hot(samples_bg.squeeze(-1), num_classes=self.n_bg_classes)
                        samples_bg = samples_bg.view(probs_bg.shape)
                        # straight-through
                        new_z_bg_features = samples_bg.detach() + (probs_bg - probs_bg.detach())
                        new_z_bg_features = new_z_bg_features.view(*mu_bg_features.shape)
                        # [bs, T, n_p, n_categories * n_classes]
                    else:
                        new_z_features = mu_features
                        new_z_bg_features = mu_bg_features
                    new_z_obj_on = beta_dist.mean
                    if self.particle_score and mu_score is not None:
                        new_z_score = mu_score
                    else:
                        new_z_score = logvar.sum(-1, keepdim=True)
                else:
                    new_z = reparameterize(mu, logvar)
                    new_z_depth = reparameterize(mu_depth, logvar_depth)
                    new_z_scale = reparameterize(mu_scale, logvar_scale)
                    if self.features_dist == 'categorical':
                        logits = mu_features.view(*mu_features.shape[:-1], self.n_fg_categories, self.n_fg_classes)
                        # [bs, T, n_p, n_categories, n_classes]
                        probs = logits.softmax(dim=-1)  # [bs, T, n_p, n_categories, n_classes]
                        samples = torch.multinomial(probs.view(-1, probs.shape[-1]), num_samples=1)
                        samples = F.one_hot(samples.squeeze(-1), num_classes=self.n_fg_classes)
                        samples = samples.view(probs.shape)
                        # straight-through
                        new_z_features = samples.detach() + (probs - probs.detach())
                        new_z_features = new_z_features.view(*mu_features.shape)
                        # [bs, T, n_p, n_categories * n_classes]

                        logits_bg = mu_bg_features.view(*mu_bg_features.shape[:-1],
                                                        self.n_bg_categories, self.n_bg_classes)
                        # [bs, T, n_p, n_categories, n_classes]
                        probs_bg = logits_bg.softmax(dim=-1)  # [bs, T, n_p, n_categories, n_classes]
                        samples_bg = torch.multinomial(probs_bg.view(-1, probs_bg.shape[-1]), num_samples=1)
                        samples_bg = F.one_hot(samples_bg.squeeze(-1), num_classes=self.n_bg_classes)
                        samples_bg = samples_bg.view(probs.shape)
                        # straight-through
                        new_z_bg_features = samples_bg.detach() + (probs_bg - probs_bg.detach())
                        new_z_bg_features = new_z_bg_features.view(*mu_bg_features.shape)
                        # [bs, T, n_p, n_categories * n_classes]
                    else:
                        new_z_features = reparameterize(mu_features, logvar_features)
                        new_z_bg_features = reparameterize(mu_bg_features, logvar_bg_features)
                    new_z_obj_on = beta_dist.sample()
                    if self.particle_score and mu_score is not None:
                        new_z_score = reparameterize(mu_score, logvar_score)
                    else:
                        new_z_score = logvar.sum(-1, keepdim=True)

            z = torch.cat([z, new_z], dim=1)
            z_depth = torch.cat([z_depth, new_z_depth], dim=1)
            z_scale = torch.cat([z_scale, new_z_scale], dim=1)
            z_features = torch.cat([z_features, new_z_features], dim=1)
            z_bg_features = torch.cat([z_bg_features, new_z_bg_features], dim=1)
            z_obj_on = torch.cat([z_obj_on, new_z_obj_on], dim=1)
            z_score = torch.cat([z_score, new_z_score], dim=1)

        out_dict = {'z': z, 'z_scale': z_scale, 'z_obj_on': z_obj_on, 'z_depth': z_depth,
                    'z_features': z_features, 'z_bg_features': z_bg_features, 'z_context': z_context,
                    'z_score': z_score,
                    'z_context_posterior': z_context_posterior, 'mu_context_posterior': mu_context_posterior}
        return out_dict

    def forward(self, z, z_scale, z_obj_on, z_depth, z_features, z_bg_features, z_context, z_score=None, actions=None,
                actions_mask=None):
        # forward dynamics
        # z, z_scale: [bs, T, n_particles, 2]
        # z_depth, z_obj_on: [bs, T, n_particles, 1]
        # z_features: [bs, T, n_particles, in_features_dim]
        # z_bg_features: [bs, T, bg_features_dim]
        # z_bg_features: [bs, T, action_dim]
        # z_context: [bs, T, context_dim]
        bs, timestep_horizon, n_particles, _ = z.shape

        # policy: state -> context
        mu_context = logvar_context = None

        # dynamics: prev_state + context -> next_state

        # project particles
        z_v = z.reshape(bs * timestep_horizon, *z.shape[2:])
        z_scale_v = z_scale.reshape(bs * timestep_horizon, *z_scale.shape[2:])
        z_obj_on_v = z_obj_on.reshape(bs * timestep_horizon, *z_obj_on.shape[2:])
        z_depth_v = z_depth.reshape(bs * timestep_horizon, *z_depth.shape[2:])
        z_features_v = z_features.reshape(bs * timestep_horizon, *z_features.shape[2:])
        z_bg_features_v = z_bg_features.reshape(bs * timestep_horizon, *z_bg_features.shape[2:])
        z_context_v = z_context.reshape(bs * timestep_horizon, *z_context.shape[2:])
        if self.use_z_orig:
            z_orig_v = self.particles_anchor.repeat(bs * timestep_horizon, 1, 1)
        else:
            z_orig_v = None
        if z_score is not None:
            z_score_v = z_score.reshape(bs * timestep_horizon, *z_score.shape[2:])
        else:
            z_score_v = z_score

        detach_dyn_inputs = False
        if detach_dyn_inputs:
            z_v = z_v.detach()
            z_scale_v = z_scale_v.detach()
            z_obj_on_v = z_obj_on_v.detach()
            z_depth_v = z_depth_v.detach()
            z_features_v = z_features_v.detach()
            z_bg_features_v = z_bg_features_v.detach()

        particle_projection = self.particle_projection(z_v,
                                                       z_scale_v,
                                                       z_obj_on_v,
                                                       z_depth_v,
                                                       z_features_v,
                                                       z_bg_features_v,
                                                       z_context_v,
                                                       z_score_v,
                                                       z_orig_v)
        # [bs * T, n_particles + 2, projection_dim]
        particle_proj_int = particle_projection

        # unroll forward
        particle_proj_int = particle_proj_int.view(bs, timestep_horizon, *particle_proj_int.shape[1:])
        # [bs, T, n_particles + 2, projection_dim]

        particle_proj_int = particle_proj_int.permute(0, 2, 1, 3)
        # [bs, n_particles + 2, T, projection_dim]
        if self.ctx_mode == 'adaln':
            if self.random_action_condition:
                random_actions = torch.rand(particle_proj_int.shape[0], particle_proj_int.shape[2],
                                            self.random_action_dim, device=particle_proj_int.device)
                c_random_action = self.random_action_proj(random_actions)
                if len(c_random_action.shape) == 3:
                    c_random_action = c_random_action.unsqueeze(1).repeat(1, particle_proj_int.shape[1], 1,
                                                                          1)  # [bs, n, t, f]
            else:
                c_random_action = 0

            if self.action_condition and actions is not None:
                if self.learn_null_action_embed and actions_mask is not None:
                    # action_mask: [batch_size, T] or [batch_size, T, 1], 1 where use action, 0 replace action
                    # Expand mask
                    if len(actions_mask.shape) == 2:
                        actions_mask = actions_mask.bool().unsqueeze(-1)  # (batch_size, seq_len, 1)
                    # Expand null embedding to match
                    null_action_embeds = self.null_action_embeddings.expand(actions.size(0), actions.size(1), -1)

                    # Blend
                    actions = actions * actions_mask + null_action_embeds * (~actions_mask)

                c_action = self.action_proj(actions)
                if len(c_action.shape) == 3:
                    c_action = c_action.unsqueeze(1).repeat(1, particle_proj_int.shape[1], 1, 1)  # [bs, n, t, f]
            else:
                c_action = 0

            c = self.context_proj(z_context_v)
            c = c.reshape(bs, timestep_horizon, *c.shape[1:])
            if len(c.shape) == 3:
                c = c.unsqueeze(1).repeat(1, particle_proj_int.shape[1], 1, 1)  # [bs, 1, t, f]
            elif c.shape[2] != particle_proj_int.shape[1]:
                c = c.permute(0, 2, 1, 3)  # [bs, n + 1, t, f]
                c = c.repeat(1, particle_proj_int.shape[1], 1, 1)  # [bs, 1, t, f]
            else:
                c = c.permute(0, 2, 1, 3)  # [bs, n + 1, t, f]
            c = c + c_action + c_random_action
            c = self.cond_activation(c)
        else:
            c = None

        if self.n_views > 1:
            # [bs * n_views, n, T, d] -> [bs, n_views, n, T, d] -> [bs, n_views * n, T, d]
            particle_proj_int = particle_proj_int.view(-1, self.n_views, particle_proj_int.shape[1],
                                                       *particle_proj_int.shape[2:])  # [bs, n_views, n, T, d]
            particle_proj_int = particle_proj_int + self.view_embeddings
            particle_proj_int = particle_proj_int.reshape(particle_proj_int.shape[0], -1,
                                                          *particle_proj_int.shape[3:])  # [bs, n_views * n, T, d]
            if c is not None:
                c = c.reshape(-1, self.n_views * c.shape[1], *c.shape[2:])

        if c is not None and self.pos_embed_p_adaln:
            c_pe = self.pos_p_embeddings.repeat(c.shape[0], 1, c.shape[2], 1)
            c = c + c_pe

        if c is not None and self.pos_embed_objon_adaln:
            c_objon = self.objon_embeddings(z_obj_on)  # [bs, t, n, dim]
            c_objon_bg = torch.zeros(c_objon.shape[0], c_objon.shape[1], 1, c_objon.shape[-1], device=c_objon.device)
            c_objon = torch.cat([c_objon, c_objon_bg], dim=2)  # [bs, t, n + 1, dim]
            c_objon = c_objon.permute(0, 2, 1, 3)  # [bs, n + 1, t, dim]
            if self.n_views > 1:
                c_objon = c_objon.reshape(-1, self.n_views * c_objon.shape[1], c_objon.shape[2], c_objon.shape[-1])
                # [bs, n_views * (n + 1), t, dim]
            c = c + c_objon

        particles_trans = self.particle_transformer(particle_proj_int, c)
        # [bs, n_particles + 2, T, projection_dim]
        if self.n_views > 1:
            # [bs, n_views * n, T, d] -> [bs * n_views, n, T, d]
            particles_trans = particles_trans.reshape(bs, -1, *particles_trans.shape[2:])
        particles_trans = particles_trans.permute(0, 2, 1, 3)
        # [bs, T, n_particles + 2, projection_dim]

        # decode transformer output
        particles_trans = particles_trans.reshape(-1, *particles_trans.shape[2:])
        # [bs * T, n_particles + 2, projection_dim]
        particle_decoder_out = self.particle_decoder(particles_trans)
        mu = particle_decoder_out['mu_offset']
        logvar = particle_decoder_out['logvar_offset']

        obj_on_a_gate = (particle_decoder_out['lobj_on_a']).sigmoid()
        obj_on_a = ((1 - obj_on_a_gate) * self.obj_on_min + obj_on_a_gate * self.obj_on_max).exp()
        obj_on_b_gate = 1 - (
                particle_decoder_out['lobj_on_b'] * 0 + particle_decoder_out['lobj_on_a']).sigmoid()
        obj_on_b = ((1 - obj_on_b_gate) * self.obj_on_min + obj_on_b_gate * self.obj_on_max).exp()

        mu_depth = particle_decoder_out['mu_depth']
        logvar_depth = particle_decoder_out['logvar_depth']
        mu_scale = particle_decoder_out['mu_scale']
        logvar_scale = particle_decoder_out['logvar_scale']
        mu_features = particle_decoder_out['mu_features']
        logvar_features = particle_decoder_out['logvar_features']
        mu_bg_features = particle_decoder_out['mu_bg_features']
        logvar_bg_features = particle_decoder_out['logvar_bg_features']
        mu_score = particle_decoder_out['mu_score']
        logvar_score = particle_decoder_out['logvar_score']

        mu_scale = mu_scale + self.scale_anchor
        if self.use_z_orig:
            mu = self.particles_anchor + mu

        if self.predict_delta:
            mu = z_v + mu
            # mu_scale = z_scale_v + mu_scale
            # mu_depth = z_depth_v + mu_depth
            # mu_features = z_features_v + mu_features
            # mu_bg_features = z_bg_features_v + mu_bg_features

        # reshape to [bs, t, ...]
        mu = mu.view(bs, timestep_horizon, *mu.shape[1:])
        logvar = logvar.view(bs, timestep_horizon, *logvar.shape[1:])
        obj_on_a = obj_on_a.view(bs, timestep_horizon, *obj_on_a.shape[1:])
        obj_on_b = obj_on_b.view(bs, timestep_horizon, *obj_on_b.shape[1:])
        mu_depth = mu_depth.view(bs, timestep_horizon, *mu_depth.shape[1:])
        logvar_depth = logvar_depth.view(bs, timestep_horizon, *logvar_depth.shape[1:])
        mu_scale = mu_scale.view(bs, timestep_horizon, *mu_scale.shape[1:])
        logvar_scale = logvar_scale.view(bs, timestep_horizon, *logvar_scale.shape[1:])
        mu_features = mu_features.view(bs, timestep_horizon, *mu_features.shape[1:])
        logvar_features = logvar_features.view(bs, timestep_horizon, *logvar_features.shape[1:])
        mu_bg_features = mu_bg_features.view(bs, timestep_horizon, *mu_bg_features.shape[1:])
        logvar_bg_features = logvar_bg_features.view(bs, timestep_horizon, *logvar_bg_features.shape[1:])
        if self.particle_score and mu_score is not None:
            mu_score = mu_score.view(bs, timestep_horizon, *mu_score.shape[1:])
            logvar_score = logvar_score.view(bs, timestep_horizon, *logvar_score.shape[1:])

        output_dict = {'mu': mu, 'logvar': logvar, 'mu_features': mu_features, 'logvar_features': logvar_features,
                       'obj_on_a': obj_on_a.squeeze(-1), 'obj_on_b': obj_on_b.squeeze(-1), 'mu_depth': mu_depth,
                       'logvar_depth': logvar_depth, 'mu_scale': mu_scale, 'logvar_scale': logvar_scale,
                       'mu_bg_features': mu_bg_features, 'logvar_bg_features': logvar_bg_features,
                       'mu_context': mu_context, 'logvar_context': logvar_context,
                       'mu_score': mu_score, 'logvar_score': logvar_score}

        return output_dict
