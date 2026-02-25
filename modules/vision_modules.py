"""
based on: https://github.com/CompVis/taming-transformers/blob/master/taming
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from torchvision import models
from collections import namedtuple
import os
import hashlib
import requests
from PIL import Image
from tqdm import tqdm

URL_MAP = {
    "vgg_lpips": "https://heibox.uni-heidelberg.de/f/607503859c864bc1b30b/?dl=1"
}

CKPT_MAP = {
    "vgg_lpips": "vgg.pth"
}

MD5_MAP = {
    "vgg_lpips": "d507d7349b931f0638a25a48a722f98a"
}

"""
Functions
"""


def calc_model_size(model):
    num_trainable_params = sum([p.numel() for p in model.parameters() if p.requires_grad])
    # estimate model size on disk: https://discuss.pytorch.org/t/finding-model-size/130275/2
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    size_all_mb = (param_size + buffer_size) / 1024 ** 2
    return {'n_params': num_trainable_params, 'size_mb': size_all_mb}


def nonlinearity(x):
    # lrelu
    # return F.leaky_relu(x, negative_slope=0.01)
    # relu
    # return F.relu(x)
    # gelu
    return F.gelu(x)
    # swish
    # return x * torch.sigmoid(x)


def download(url, local_path, chunk_size=1024):
    os.makedirs(os.path.split(local_path)[0], exist_ok=True)
    with requests.get(url, stream=True) as r:
        total_size = int(r.headers.get("content-length", 0))
        with tqdm(total=total_size, unit="B", unit_scale=True) as pbar:
            with open(local_path, "wb") as f:
                for data in r.iter_content(chunk_size=chunk_size):
                    if data:
                        f.write(data)
                        pbar.update(chunk_size)


def md5_hash(path):
    with open(path, "rb") as f:
        content = f.read()
    return hashlib.md5(content).hexdigest()


def get_ckpt_path(name, root, check=False):
    assert name in URL_MAP
    path = os.path.join(root, CKPT_MAP[name])
    if not os.path.exists(path) or (check and not md5_hash(path) == MD5_MAP[name]):
        print("Downloading {} model from {} to {}".format(name, URL_MAP[name], path))
        download(URL_MAP[name], path)
        md5 = md5_hash(path)
        assert md5 == MD5_MAP[name], md5
    return path


def normalize_tensor(x, eps=1e-10):
    norm_factor = torch.sqrt(torch.sum(x ** 2, dim=1, keepdim=True))
    return x / (norm_factor + eps)


def spatial_average(x, keepdim=True):
    return x.mean([2, 3], keepdim=keepdim)


def norm_layer(in_channels, num_groups=4, eps=1e-5):
    # base_groups = num_groups
    # if in_channels <= 32:
    #     num_groups = base_groups
    # elif in_channels == 64:
    #     num_groups = base_groups * 2  # 8
    # elif num_groups == 128:
    #     num_groups = base_groups * 4  # 16
    # else:
    #     num_groups = base_groups * 8  # 32
    return torch.nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=eps, affine=True)


def rgb_to_minusoneone(x):
    x = 2. * x - 1.
    return x


def minusoneone_to_rgb(x):
    x = torch.clamp(x, -1., 1.)
    x = (x + 1.) / 2.
    return x


def custom_to_pil(x):
    x = x.detach().cpu()
    x = torch.clamp(x, -1., 1.)
    x = (x + 1.) / 2.
    x = x.permute(1, 2, 0).numpy()
    x = (255 * x).astype(np.uint8)
    x = Image.fromarray(x)
    if not x.mode == "RGB":
        x = x.convert("RGB")
    return x


class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv, padding_mode='zeros', mode='nearest'):
        super().__init__()
        self.with_conv = with_conv
        self.mode = mode
        if self.with_conv:
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1, padding_mode=padding_mode)

    def forward(self, x):
        if self.mode == 'bilinear':
            x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode=self.mode, align_corners=False)
        else:
            x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode=self.mode)
        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv, use_conv_block=False, padding_mode='constant'):
        super().__init__()
        self.with_conv = with_conv
        self.use_conv_block = use_conv_block
        self.padding_mode = 'constant' if padding_mode == 'zeros' else padding_mode
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            if self.use_conv_block:
                self.conv = ConvBlock(in_channels=in_channels, out_channels=in_channels, dropout=0.0, padding=0,
                                      stride=2, kernel_size=3)
            else:
                self.conv = torch.nn.Conv2d(in_channels,
                                            in_channels,
                                            kernel_size=3,
                                            stride=2,
                                            padding=0)

    def forward(self, x):
        if self.with_conv:
            pad = (0, 1, 0, 1)
            x = torch.nn.functional.pad(x, pad, mode=self.padding_mode, value=0)
            x = self.conv(x)
        else:
            x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        return x


class ConvBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False,
                 dropout=0.0, temb_channels=0, padding_mode='zeros', padding=1, stride=1, kernel_size=3):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.conv = torch.nn.Conv2d(in_channels,
                                    out_channels,
                                    kernel_size=kernel_size,
                                    stride=stride,
                                    padding=padding, padding_mode=padding_mode)
        if temb_channels > 0:
            self.temb_proj = torch.nn.Linear(temb_channels,
                                             out_channels)
        self.norm = norm_layer(out_channels)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x, temb=None):
        h = x
        h = self.conv(h)
        if temb is not None:
            h = h + self.temb_proj(nonlinearity(temb))[:, :, None, None]

        h = self.norm(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        return h


class ResnetBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False,
                 dropout, temb_channels=0, padding_mode='zeros'):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = norm_layer(in_channels)
        self.conv1 = torch.nn.Conv2d(in_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1, padding_mode=padding_mode)
        if temb_channels > 0:
            self.temb_proj = torch.nn.Linear(temb_channels,
                                             out_channels)
        self.norm2 = norm_layer(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv2d(out_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1, padding_mode=padding_mode)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv2d(in_channels,
                                                     out_channels,
                                                     kernel_size=3,
                                                     stride=1,
                                                     padding=1, padding_mode=padding_mode)
            else:
                self.nin_shortcut = torch.nn.Conv2d(in_channels,
                                                    out_channels,
                                                    kernel_size=1,
                                                    stride=1,
                                                    padding=0)

    def forward(self, x, temb):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        if temb is not None:
            h = h + self.temb_proj(nonlinearity(temb))[:, :, None, None]

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x + h


class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = norm_layer(in_channels)
        self.q = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b, c, h, w = q.shape
        q = q.reshape(b, c, h * w)
        q = q.permute(0, 2, 1)  # b,hw,c
        k = k.reshape(b, c, h * w)  # b,c,hw
        w_ = torch.bmm(q, k)  # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c) ** (-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b, c, h * w)
        w_ = w_.permute(0, 2, 1)  # b,hw,hw (first hw of k, second of q)
        h_ = torch.bmm(v, w_)  # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = h_.reshape(b, c, h, w)

        h_ = self.proj_out(h_)

        return x + h_


class Encoder(nn.Module):
    def __init__(self, *, ch, ch_mult=(1, 2, 4, 8), num_res_blocks, residual=True,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels,
                 resolution, z_channels, double_z=True, padding_mode='zeros', attention=False,
                 mid_blocks=True, in_conv_kernel_size=3, **ignore_kwargs):
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.padding_mode = padding_mode
        self.use_attention = attention
        self.residual = residual
        self.mid_blocks = mid_blocks
        block_nn = ResnetBlock if self.residual else ConvBlock

        # downsampling
        # if self.residual:
        #     self.conv_in = torch.nn.Conv2d(in_channels,
        #                                    self.ch,
        #                                    kernel_size=3,
        #                                    stride=1,
        #                                    padding=1, padding_mode=self.padding_mode)
        # else:
        #     self.conv_in = ConvBlock(in_channels=in_channels, out_channels=self.ch, padding_mode=self.padding_mode,
        #                              temb_channels=self.temb_ch, dropout=dropout)

        first_conv_pad = in_conv_kernel_size // 2
        self.conv_in = torch.nn.Conv2d(in_channels,
                                       self.ch,
                                       kernel_size=in_conv_kernel_size,
                                       stride=1,
                                       padding=first_conv_pad, padding_mode=self.padding_mode)

        curr_res = resolution
        in_ch_mult = (1,) + tuple(ch_mult)
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(block_nn(in_channels=block_in,
                                      out_channels=block_out,
                                      temb_channels=self.temb_ch,
                                      dropout=dropout, padding_mode=self.padding_mode))
                block_in = block_out
                if curr_res in attn_resolutions:
                    if attention:
                        attn.append(AttnBlock(block_in))
                    else:
                        attn.append(nn.Identity())
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                # down.downsample = Downsample(block_in, resamp_with_conv, padding_mode=padding_mode,
                #                              use_conv_block=not self.residual)
                down.downsample = Downsample(block_in, resamp_with_conv, padding_mode=padding_mode)
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()

        if self.mid_blocks:
            self.mid.block_1 = block_nn(in_channels=block_in,
                                        out_channels=block_in,
                                        temb_channels=self.temb_ch,
                                        dropout=dropout, padding_mode=self.padding_mode)
            if attention:
                self.mid.attn_1 = AttnBlock(block_in)
            else:
                self.mid.attn_1 = nn.Identity()
            self.mid.block_2 = block_nn(in_channels=block_in,
                                        out_channels=block_in,
                                        temb_channels=self.temb_ch,
                                        dropout=dropout, padding_mode=self.padding_mode)
        else:
            self.mid.block_1 = nn.Identity()
            self.mid.attn_1 = nn.Identity()
            self.mid.block_2 = nn.Identity()

        # if attention:
        #     self.mid.block_1 = block_nn(in_channels=block_in,
        #                                 out_channels=block_in,
        #                                 temb_channels=self.temb_ch,
        #                                 dropout=dropout, padding_mode=self.padding_mode)
        #     self.mid.attn_1 = AttnBlock(block_in)
        #     self.mid.block_2 = block_nn(in_channels=block_in,
        #                                 out_channels=block_in,
        #                                 temb_channels=self.temb_ch,
        #                                 dropout=dropout, padding_mode=self.padding_mode)
        # else:
        #     self.mid.block_1 = nn.Identity()
        #     self.mid.attn_1 = nn.Identity()
        #     self.mid.block_2 = nn.Identity()

        # end
        self.norm_out = norm_layer(block_in) if self.residual else nn.Identity()
        self.conv_out = torch.nn.Conv2d(block_in,
                                        2 * z_channels if double_z else z_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1, padding_mode=self.padding_mode)
        self.conv_output_size = self.calc_conv_output_size()

    def calc_conv_output_size(self):
        dummy_input = torch.zeros(1, self.in_channels, self.resolution, self.resolution)
        dummy_input = self(dummy_input)
        return dummy_input[0].shape

    def forward(self, x):
        # assert x.shape[2] == x.shape[3] == self.resolution, "{}, {}, {}".format(x.shape[2], x.shape[3], self.resolution)

        # timestep embedding
        temb = None

        # downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1]
        if self.mid_blocks:
            h = self.mid.block_1(h, temb)
            h = self.mid.attn_1(h)
            h = self.mid.block_2(h, temb)

        # end
        if self.residual:
            h = self.norm_out(h)
            h = nonlinearity(h)
        h = self.conv_out(h)
        return h

class Decoder(nn.Module):
    def __init__(self, *, ch, out_ch, ch_mult=(1, 2, 4, 8), num_res_blocks,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True, residual=True,
                 resolution, z_channels, give_pre_end=False, padding_mode='zeros', attention=False,
                 mid_blocks=True, upsample_method='nearest', **ignorekwargs):
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        # self.in_channels = in_channels
        self.give_pre_end = give_pre_end
        self.padding_mode = padding_mode
        self.use_attention = attention
        self.residual = residual
        self.mid_blocks = mid_blocks
        self.upsample_method = upsample_method
        block_nn = ResnetBlock if self.residual else ConvBlock

        # compute in_ch_mult, block_in and curr_res at lowest res
        in_ch_mult = (1,) + tuple(ch_mult)
        block_in = ch * ch_mult[self.num_resolutions - 1]
        curr_res = resolution // 2 ** (self.num_resolutions - 1)
        self.z_shape = (1, z_channels, curr_res, curr_res)
        # print("Working with z of shape {} = {} dimensions.".format(
        #     self.z_shape, np.prod(self.z_shape)))

        # z to block_in
        # if self.residual:
        #     self.conv_in = torch.nn.Conv2d(z_channels,
        #                                    block_in,
        #                                    kernel_size=3,
        #                                    stride=1,
        #                                    padding=1, padding_mode=self.padding_mode)
        # else:
        #     self.conv_in = ConvBlock(in_channels=z_channels, out_channels=block_in, padding_mode=self.padding_mode,
        #                              temb_channels=self.temb_ch, dropout=dropout)

        self.conv_in = torch.nn.Conv2d(z_channels,
                                       block_in,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1, padding_mode=self.padding_mode)

        # middle
        self.mid = nn.Module()

        if self.mid_blocks:
            self.mid.block_1 = block_nn(in_channels=block_in,
                                        out_channels=block_in,
                                        temb_channels=self.temb_ch,
                                        dropout=dropout, padding_mode=self.padding_mode)
            if attention:
                self.mid.attn_1 = AttnBlock(block_in)
            else:
                self.mid.attn_1 = nn.Identity()
            self.mid.block_2 = block_nn(in_channels=block_in,
                                        out_channels=block_in,
                                        temb_channels=self.temb_ch,
                                        dropout=dropout, padding_mode=self.padding_mode)
        else:
            self.mid.block_1 = nn.Identity()
            self.mid.attn_1 = nn.Identity()
            self.mid.block_2 = nn.Identity()

        # if attention:
        #     self.mid.block_1 = block_nn(in_channels=block_in,
        #                                 out_channels=block_in,
        #                                 temb_channels=self.temb_ch,
        #                                 dropout=dropout, padding_mode=self.padding_mode)
        #     self.mid.attn_1 = AttnBlock(block_in)
        #     self.mid.block_2 = block_nn(in_channels=block_in,
        #                                 out_channels=block_in,
        #                                 temb_channels=self.temb_ch,
        #                                 dropout=dropout, padding_mode=self.padding_mode)
        # else:
        #     self.mid.block_1 = nn.Identity()
        #     self.mid.attn_1 = nn.Identity()
        #     self.mid.block_2 = nn.Identity()

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks + 1):
                block.append(block_nn(in_channels=block_in,
                                      out_channels=block_out,
                                      temb_channels=self.temb_ch,
                                      dropout=dropout, padding_mode=self.padding_mode))
                block_in = block_out
                if curr_res in attn_resolutions:
                    if attention:
                        attn.append(AttnBlock(block_in))
                    else:
                        attn.append(nn.Identity())
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                # with_conv = resamp_with_conv if residual else False
                with_conv = resamp_with_conv
                up.upsample = Upsample(block_in, with_conv, padding_mode=self.padding_mode, mode=self.upsample_method)
                curr_res = curr_res * 2
            self.up.insert(0, up)  # prepend to get consistent order

        # end
        self.norm_out = norm_layer(block_in) if self.residual else nn.Identity()
        self.conv_out = torch.nn.Conv2d(block_in,
                                        out_ch,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1, padding_mode=self.padding_mode)

    def forward(self, z):
        # assert z.shape[1:] == self.z_shape[1:]
        self.last_z_shape = z.shape

        # timestep embedding
        temb = None

        # z to block_in
        h = self.conv_in(z)

        # middle
        if self.mid_blocks:
            h = self.mid.block_1(h, temb)
            h = self.mid.attn_1(h)
            h = self.mid.block_2(h, temb)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h, temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        if self.give_pre_end:
            return h

        if self.residual:
            h = self.norm_out(h)
            h = nonlinearity(h)
        h = self.conv_out(h)
        return h

class LPIPS(nn.Module):
    # Learned perceptual metric
    def __init__(self, use_dropout=True):
        super().__init__()
        self.scaling_layer = ScalingLayer()
        self.chns = [64, 128, 256, 512, 512]  # vg16 features
        self.net = vgg16(pretrained=True, requires_grad=False)
        self.lin0 = NetLinLayer(self.chns[0], use_dropout=use_dropout)
        self.lin1 = NetLinLayer(self.chns[1], use_dropout=use_dropout)
        self.lin2 = NetLinLayer(self.chns[2], use_dropout=use_dropout)
        self.lin3 = NetLinLayer(self.chns[3], use_dropout=use_dropout)
        self.lin4 = NetLinLayer(self.chns[4], use_dropout=use_dropout)
        self.load_from_pretrained()
        for param in self.parameters():
            param.requires_grad = False

    def load_from_pretrained(self, name="vgg_lpips"):
        ckpt = get_ckpt_path(name, "eval/lpips")
        self.load_state_dict(torch.load(ckpt, map_location=torch.device("cpu")), strict=False)
        print("loaded pretrained LPIPS loss from {}".format(ckpt))

    @classmethod
    def from_pretrained(cls, name="vgg_lpips"):
        if name != "vgg_lpips":
            raise NotImplementedError
        model = cls()
        ckpt = get_ckpt_path(name)
        model.load_state_dict(torch.load(ckpt, map_location=torch.device("cpu")), strict=False)
        return model

    def forward(self, input, target):
        in0_input, in1_input = (self.scaling_layer(input), self.scaling_layer(target))
        outs0, outs1 = self.net(in0_input), self.net(in1_input)
        feats0, feats1, diffs = {}, {}, {}
        lins = [self.lin0, self.lin1, self.lin2, self.lin3, self.lin4]
        for kk in range(len(self.chns)):
            feats0[kk], feats1[kk] = normalize_tensor(outs0[kk]), normalize_tensor(outs1[kk])
            diffs[kk] = (feats0[kk] - feats1[kk]) ** 2

        res = [spatial_average(lins[kk].model(diffs[kk]), keepdim=True) for kk in range(len(self.chns))]
        val = res[0]
        for l in range(1, len(self.chns)):
            val += res[l]
        return val


class ScalingLayer(nn.Module):
    def __init__(self):
        super(ScalingLayer, self).__init__()
        self.register_buffer('shift', torch.Tensor([-.030, -.088, -.188])[None, :, None, None])
        self.register_buffer('scale', torch.Tensor([.458, .448, .450])[None, :, None, None])

    def forward(self, inp):
        return (inp - self.shift) / self.scale


class NetLinLayer(nn.Module):
    """ A single linear layer which does a 1x1 conv """

    def __init__(self, chn_in, chn_out=1, use_dropout=False):
        super(NetLinLayer, self).__init__()
        layers = [nn.Dropout(), ] if (use_dropout) else []
        layers += [nn.Conv2d(chn_in, chn_out, 1, stride=1, padding=0, bias=False), ]
        self.model = nn.Sequential(*layers)


class vgg16(torch.nn.Module):
    def __init__(self, requires_grad=False, pretrained=True):
        super(vgg16, self).__init__()
        vgg_pretrained_features = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        self.N_slices = 5
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(23, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        h = self.slice5(h)
        h_relu5_3 = h
        vgg_outputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3', 'relu5_3'])
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3)
        return out


class LossLPIPS(nn.Module):
    def __init__(self, pixelloss_weight=1.0, perceptual_weight=1.0):
        super().__init__()
        self.pixel_weight = pixelloss_weight
        self.perceptual_loss = LPIPS().eval()
        self.perceptual_weight = perceptual_weight

    def forward(self, inputs, reconstructions, split="train"):
        rec_loss = torch.abs(inputs.contiguous() - reconstructions.contiguous())
        if self.perceptual_weight > 0:
            p_loss = self.perceptual_loss(inputs.contiguous(), reconstructions.contiguous())
            rec_loss = rec_loss + self.perceptual_weight * p_loss
        else:
            p_loss = torch.tensor([0.0])

        nll_loss = rec_loss
        # nll_loss = torch.sum(nll_loss) / nll_loss.shape[0]
        loss = torch.mean(nll_loss)

        log = {"total_loss".format(split): loss.clone().detach().mean(),
               "nll_loss".format(split): nll_loss.detach().mean(),
               "rec_loss".format(split): rec_loss.detach().mean(),
               "p_loss".format(split): p_loss.detach().mean(),
               }
        return loss, log

