"""
Loss functions implementations used in the optimization of DLP.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms

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


# functions
def batch_pairwise_kl(mu_x, logvar_x, mu_y, logvar_y, reverse_kl=False):
    """
    Calculate batch-wise KL-divergence
    mu_x, logvar_x: [batch_size, n_x, points_dim]
    mu_y, logvar_y: [batch_size, n_y, points_dim]
    kl = -0.5 * Σ_points_dim (1 + logvar_x - logvar_y - exp(logvar_x)/exp(logvar_y)
                    - ((mu_x - mu_y) ** 2)/exp(logvar_y))
    """
    if reverse_kl:
        mu_a, logvar_a = mu_y, logvar_y
        mu_b, logvar_b = mu_x, logvar_x
    else:
        mu_a, logvar_a = mu_x, logvar_x
        mu_b, logvar_b = mu_y, logvar_y
    bs, n_a, points_dim = mu_a.size()
    _, n_b, _ = mu_b.size()
    logvar_aa = logvar_a.unsqueeze(2).expand(-1, -1, n_b, -1)  # [batch_size, n_a, n_b, points_dim]
    logvar_bb = logvar_b.unsqueeze(1).expand(-1, n_a, -1, -1)  # [batch_size, n_a, n_b, points_dim]
    mu_aa = mu_a.unsqueeze(2).expand(-1, -1, n_b, -1)  # [batch_size, n_a, n_b, points_dim]
    mu_bb = mu_b.unsqueeze(1).expand(-1, n_a, -1, -1)  # [batch_size, n_a, n_b, points_dim]
    p_kl = -0.5 * (1 + logvar_aa - logvar_bb - logvar_aa.exp() / logvar_bb.exp()
                   - ((mu_aa - mu_bb) ** 2) / logvar_bb.exp()).sum(-1)  # [batch_size, n_x, n_y]
    return p_kl


def batch_pairwise_dist(x, y, metric='l2'):
    assert metric in ['l2', 'l2_simple', 'l1', 'cosine'], f'metric {metric} unrecognized'
    bs, num_points_x, points_dim = x.size()
    _, num_points_y, _ = y.size()
    if metric == 'cosine':
        dist_func = torch.nn.functional.cosine_similarity
        P = -dist_func(x.unsqueeze(2), y.unsqueeze(1), dim=-1, eps=1e-8)
    elif metric == 'l1':
        P = torch.abs(x.unsqueeze(2) - y.unsqueeze(1)).sum(-1)
    elif metric == 'l2_simple':
        P = ((x.unsqueeze(2) - y.unsqueeze(1)) ** 2).sum(-1)
    else:
        xx = torch.bmm(x, x.transpose(2, 1))
        yy = torch.bmm(y, y.transpose(2, 1))
        zz = torch.bmm(x, y.transpose(2, 1))
        diag_ind_x = torch.arange(0, num_points_x, device=x.device)
        diag_ind_y = torch.arange(0, num_points_y, device=y.device)
        rx = xx[:, diag_ind_x, diag_ind_x].unsqueeze(1).expand_as(zz.transpose(2, 1))
        ry = yy[:, diag_ind_y, diag_ind_y].unsqueeze(1).expand_as(zz)
        P = rx.transpose(2, 1) + ry - 2 * zz
    return P


def calc_reconstruction_loss(x, recon_x, loss_type='mse', reduction='sum'):
    """

    :param x: original inputs
    :param recon_x:  reconstruction of the VAE's input
    :param loss_type: "mse", "l1", "bce"
    :param reduction: "sum", "mean", "none"
    :return: recon_loss
    """
    if reduction not in ['sum', 'mean', 'none']:
        raise NotImplementedError
    recon_x = recon_x.view(recon_x.size(0), -1)
    x = x.view(x.size(0), -1)
    if loss_type == 'mse':
        recon_error = F.mse_loss(recon_x, x, reduction='none')
        recon_error = recon_error.sum(1)
        if reduction == 'sum':
            recon_error = recon_error.sum()
        elif reduction == 'mean':
            recon_error = recon_error.mean()
    elif loss_type == 'l1':
        recon_error = F.l1_loss(recon_x, x, reduction='none')
        recon_error = recon_error.sum(1)
        if reduction == 'sum':
            recon_error = recon_error.sum()
        elif reduction == 'mean':
            recon_error = recon_error.mean()
    elif loss_type == 'bce':
        recon_error = F.binary_cross_entropy(recon_x, x, reduction=reduction)
    else:
        raise NotImplementedError
    return recon_error


def calc_kl(logvar, mu, mu_o=0.0, logvar_o=0.0, reduce='sum', balance=0.5):
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
    if not isinstance(mu_o, torch.Tensor):
        mu_o = torch.tensor(mu_o).to(mu.device)
    if not isinstance(logvar_o, torch.Tensor):
        logvar_o = torch.tensor(logvar_o).to(mu.device)
    if balance == 0.5:
        # kl = -0.5 * (1 + logvar - logvar_o - logvar.exp() / (torch.exp(logvar_o) + eps) - (mu - mu_o).pow(2) / (
        #             torch.exp(logvar_o) + eps)).sum(-1)
        kl = -0.5 * (1 + logvar - logvar_o - torch.exp(logvar - logvar_o) - (mu - mu_o).pow(2) * torch.exp(
            -logvar_o)).sum(-1)
    else:
        # detach post
        mu_post = mu.detach()
        logvar_post = logvar.detach()
        mu_prior = mu_o
        logvar_prior = logvar_o
        # kl_a = -0.5 * (1 + logvar_post - logvar_prior - logvar_post.exp() / (torch.exp(logvar_prior) + eps) - (
        #         mu_post - mu_prior).pow(2) / (torch.exp(logvar_prior) + eps)).sum(-1)
        kl_a = -0.5 * (1 + logvar_post - logvar_prior - torch.exp(logvar_post - logvar_prior) - (
                mu_post - mu_prior).pow(2) * torch.exp(-logvar_prior)).sum(-1)
        # detach prior
        mu_post = mu
        logvar_post = logvar
        mu_prior = mu_o.detach()
        logvar_prior = logvar_o.detach()
        # kl_b = -0.5 * (1 + logvar_post - logvar_prior - logvar_post.exp() / (torch.exp(logvar_prior) + eps) - (
        #         mu_post - mu_prior).pow(2) / (torch.exp(logvar_prior) + eps)).sum(-1)
        kl_b = -0.5 * (1 + logvar_post - logvar_prior - torch.exp(logvar_post - logvar_prior) - (
                mu_post - mu_prior).pow(2) * torch.exp(-logvar_prior)).sum(-1)
        kl = (1 - balance) * kl_a + balance * kl_b
    if reduce == 'sum':
        kl = torch.sum(kl)
    elif reduce == 'mean':
        kl = torch.mean(kl)
    return kl


def calc_kl_bern(post_prob, prior_prob, eps=1e-15, reduce='none'):
    """
    Compute kl divergence of Bernoulli variable
    :param post_prob [batch_size, 1], in [0,1]
    :param prior_prob [batch_size, 1], in [0,1]
    :return: kl divergence, (B, ...)
    """
    kl = post_prob * (torch.log(post_prob + eps) - torch.log(prior_prob + eps)) + (1 - post_prob) * (
            torch.log(1 - post_prob + eps) - torch.log(1 - prior_prob + eps))
    if reduce == 'sum':
        kl = kl.sum()
    elif reduce == 'mean':
        kl = kl.mean()
    else:
        kl = kl.squeeze(-1)
    return kl


def log_beta_function(alpha, beta, eps: float = 1e-5):
    """
    B(alpha, beta) = gamma(alpha) * gamma(beta) / gamma(alpha + beta)
    logB = loggamma(alpha) + loggamma(beta) - loggamaa(alpha + beta)
    """
    # return torch.special.gammaln(alpha) + torch.special.gammaln(beta) - torch.special.gammaln(alpha + beta)
    return torch.lgamma(alpha + eps) + torch.lgamma(beta + eps) - torch.lgamma(alpha + beta + eps)


def calc_kl_beta_dist(alpha_post, beta_post, alpha_prior, beta_prior, reduce: str = 'none', eps: float = 1e-5,
                      balance: float = 0.5):
    """
    Compute kl divergence of Beta variable
    https://en.wikipedia.org/wiki/Beta_distribution
    :param alpha_post, beta_post [batch_size, 1]
    :param alpha_prior,  beta_prior  [batch_size, 1]
    :param balance kl balance between posterior and prior
    :return: kl divergence, (B, ...)
    """
    if balance == 0.5:
        log_bettas = log_beta_function(alpha_prior, beta_prior) - log_beta_function(alpha_post, beta_post)
        alpha = (alpha_post - alpha_prior) * torch.digamma(alpha_post + eps)
        beta = (beta_post - beta_prior) * torch.digamma(beta_post + eps)
        alpha_beta = (alpha_prior - alpha_post + beta_prior - beta_post) * torch.digamma(alpha_post + beta_post + eps)
        kl = log_bettas + alpha + beta + alpha_beta
    else:
        # detach post
        log_bettas = log_beta_function(alpha_prior, beta_prior) - log_beta_function(alpha_post.detach(),
                                                                                    beta_post.detach())
        alpha = (alpha_post - alpha_prior) * torch.digamma(alpha_post.detach() + eps)
        beta = (beta_post.detach() - beta_prior) * torch.digamma(beta_post.detach() + eps)
        alpha_beta = (alpha_prior - alpha_post.detach() + beta_prior - beta_post.detach()) * torch.digamma(
            alpha_post.detach() + beta_post.detach() + eps)
        kl_a = log_bettas + alpha + beta + alpha_beta

        # detach prior
        log_bettas = log_beta_function(alpha_prior.detach(), beta_prior.detach()) - log_beta_function(alpha_post,
                                                                                                      beta_post)
        alpha = (alpha_post - alpha_prior.detach()) * torch.digamma(alpha_post + eps)
        beta = (beta_post - beta_prior.detach()) * torch.digamma(beta_post + eps)
        alpha_beta = (alpha_prior.detach() - alpha_post + beta_prior.detach() - beta_post) * torch.digamma(
            alpha_post + beta_post + eps)
        kl_b = log_bettas + alpha + beta + alpha_beta
        kl = (1 - balance) * kl_a + balance * kl_b
    if reduce == 'sum':
        kl = kl.sum()
    elif reduce == 'mean':
        kl = kl.mean()
    else:
        kl = kl.squeeze(-1)
    return kl


def calc_kl_categorical(logits_post, logits_prior, num_classes: int = 4, reduce: str = 'none', balance: float = 0.5):
    """
    Compute kl divergence of categorical variable
    :param logits_post, beta_post [batch_size, num_categories * num_classes]
    :param logits_prior,  beta_prior  [batch_size, num_categories * num_classes]
    :param balance kl balance between posterior and prior
    :return: kl divergence, (B, ...)
    """
    orig_shape = logits_post.shape
    logits_post = logits_post.view(-1, num_classes)
    logits_prior = logits_prior.view(-1, num_classes)
    post_logprobs = torch.log_softmax(logits_post, dim=-1)
    prior_logprobs = torch.log_softmax(logits_prior, dim=-1)
    if balance == 0.5:
        kl = F.kl_div(prior_logprobs, post_logprobs, reduction='none', log_target=True)
    else:
        kl_a = F.kl_div(prior_logprobs, post_logprobs.detach(), reduction='none', log_target=True)
        kl_b = F.kl_div(prior_logprobs.detach(), post_logprobs, reduction='none', log_target=True)
        kl = (1 - balance) * kl_a + balance * kl_b
    kl = kl.view(orig_shape).sum(-1)
    if reduce == 'sum':
        kl = kl.sum()
    elif reduce == 'mean':
        kl = kl.mean()
    return kl


# classes
class ChamferLossKL(nn.Module):
    """
    Calculates the KL-divergence between two sets of (R.V.) particle coordinates.
    """

    def __init__(self, use_reverse_kl=False):
        super(ChamferLossKL, self).__init__()
        self.use_reverse_kl = use_reverse_kl

    def forward(self, mu_preds, logvar_preds, mu_gts, logvar_gts, posterior_mask=None):
        """
        mu_preds, logvar_preds: [bs, n_x, feat_dim]
        mu_gts, logvar_gts: [bs, n_y, feat_dim]
        posterior_mask: [bs, n_x]
        """
        p_kl = batch_pairwise_kl(mu_preds, logvar_preds, mu_gts, logvar_gts, reverse_kl=False)
        # [bs, n_x, n_y]
        if self.use_reverse_kl:
            p_rkl = batch_pairwise_kl(mu_preds, logvar_preds, mu_gts, logvar_gts, reverse_kl=True)
            p_kl = 0.5 * (p_kl + p_rkl.transpose(2, 1))
        mins, _ = torch.min(p_kl, 1)  # [bs, n_y]
        loss_1 = torch.sum(mins, 1)
        mins, _ = torch.min(p_kl, 2)  # [bs, n_x]
        if posterior_mask is not None:
            mins = mins * posterior_mask
        loss_2 = torch.sum(mins, 1)
        return loss_1 + loss_2


class NetVGGFeatures(nn.Module):

    def __init__(self, layer_ids):
        super().__init__()

        self.vggnet = models.vgg16(pretrained=True)
        self.vggnet.eval()
        self.vggnet.requires_grad_(False)
        self.layer_ids = layer_ids

    def forward(self, x):
        output = []
        for i in range(self.layer_ids[-1] + 1):
            x = self.vggnet.features[i](x)

            if i in self.layer_ids:
                output.append(x)

        return output


class VGGDistance(nn.Module):

    def __init__(self, layer_ids=(2, 7, 12, 21, 30), accumulate_mode='sum', device=torch.device("cpu"),
                 normalize=True, use_loss_scale=False, vgg_coeff=0.12151):
        super().__init__()

        self.vgg = NetVGGFeatures(layer_ids).to(device)
        self.layer_ids = layer_ids
        self.accumulate_mode = accumulate_mode
        self.device = device
        self.use_normalization = normalize
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])
        self.use_loss_scale = use_loss_scale
        self.vgg_coeff = vgg_coeff

    def forward(self, I1, I2, reduction='sum', only_image=False):
        b_sz = I1.size(0)
        num_ch = I1.size(1)

        if self.accumulate_mode == 'sum':
            loss = ((I1 - I2) ** 2).view(b_sz, -1).sum(1)
            # if normalized, effectively: (1 / (std ** 2)) * (I_1 - I_2) ** 2
        elif self.accumulate_mode == 'ch_mean':
            loss = ((I1 - I2) ** 2).view(b_sz, I1.shape[1], -1).mean(1).sum(-1)
        else:
            loss = ((I1 - I2) ** 2).view(b_sz, -1).mean(1)

        if self.use_normalization:
            I1, I2 = self.normalize(I1), self.normalize(I2)

        if num_ch == 1:
            I1 = I1.repeat(1, 3, 1, 1)
            I2 = I2.repeat(1, 3, 1, 1)

        f1 = self.vgg(I1)
        f2 = self.vgg(I2)

        if not only_image:
            for i in range(len(self.layer_ids)):
                if self.accumulate_mode == 'sum':
                    layer_loss = ((f1[i] - f2[i]) ** 2).view(b_sz, -1).sum(1)
                elif self.accumulate_mode == 'ch_mean':
                    layer_loss = ((f1[i] - f2[i]) ** 2).view(b_sz, f1[i].shape[1], -1).mean(1).sum(-1)
                else:
                    layer_loss = ((f1[i] - f2[i]) ** 2).view(b_sz, -1).mean(1)
                c = self.vgg_coeff if self.use_normalization else 1.0
                loss = loss + c * layer_loss

        if self.use_loss_scale:
            # by using `sum` for the features, and using scaling instead of `mean` we maintain the weight
            # of each dimension contribution to the loss
            max_dim = max([np.product(f.shape[1:]) for f in f1])
            scale = 1 / max_dim
            loss = scale * loss
        if reduction == 'mean':
            return loss.mean()
        elif reduction == 'sum':
            return loss.sum()
        else:
            return loss

    def get_dimensions(self, device=torch.device("cpu")):
        dims = []
        dummy_input = torch.zeros(1, 3, 128, 128).to(device)
        dims.append(dummy_input.view(1, -1).size(1))
        f = self.vgg(dummy_input)
        for i in range(len(self.layer_ids)):
            dims.append(f[i].view(1, -1).size(1))
        return dims


class ChamferLoss(nn.Module):

    def __init__(self):
        super(ChamferLoss, self).__init__()
        # self.use_cuda = torch.cuda.is_available()

    def forward(self, preds, gts):
        P = self.batch_pairwise_dist(gts, preds)
        mins, _ = torch.min(P, 1)
        loss_1 = torch.sum(mins, 1)
        mins, _ = torch.min(P, 2)
        loss_2 = torch.sum(mins, 1)
        return loss_1 + loss_2

    def batch_pairwise_dist(self, x, y):
        bs, num_points_x, points_dim = x.size()
        _, num_points_y, _ = y.size()
        xx = torch.bmm(x, x.transpose(2, 1))
        yy = torch.bmm(y, y.transpose(2, 1))
        zz = torch.bmm(x, y.transpose(2, 1))
        diag_ind_x = torch.arange(0, num_points_x, device=x.device, dtype=torch.long)
        diag_ind_y = torch.arange(0, num_points_y, device=y.device, dtype=torch.long)
        rx = xx[:, diag_ind_x, diag_ind_x].unsqueeze(1).expand_as(
            zz.transpose(2, 1))
        ry = yy[:, diag_ind_y, diag_ind_y].unsqueeze(1).expand_as(zz)
        P = rx.transpose(2, 1) + ry - 2 * zz
        return P


"""
LPIPS
based on: https://github.com/CompVis/taming-transformers/blob/master/taming
"""


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
    def __init__(self, pixelloss_weight=1.0, perceptual_weight=0.1, normalized_rgb=True):
        super().__init__()
        self.pixel_weight = pixelloss_weight
        self.perceptual_loss = LPIPS().eval()
        self.perceptual_weight = perceptual_weight
        self.normalized_rgb = normalized_rgb

    def scale_input(self, x):
        if self.normalized_rgb:
            return x
        else:
            return 2 * x - 1

    def forward(self, inputs, reconstructions, reduction='mean', split="train", p_loss=True):
        # rec_loss = torch.abs(inputs.contiguous() - reconstructions.contiguous())
        rec_loss = (inputs.contiguous() - reconstructions.contiguous()) ** 2
        if p_loss and self.perceptual_weight > 0:
            p_loss = self.perceptual_loss(self.scale_input(inputs.contiguous()),
                                          self.scale_input(reconstructions.contiguous()))
            rec_loss = rec_loss + self.perceptual_weight * p_loss
        else:
            p_loss = torch.tensor([0.0])

        nll_loss = rec_loss
        # nll_loss = torch.sum(nll_loss) / nll_loss.shape[0]
        if reduction == 'mean':
            loss = torch.mean(nll_loss)
        elif reduction == 'sum':
            loss = torch.sum(nll_loss)
        else:
            loss = nll_loss.view(inputs.shape[0], -1).mean(-1, keepdim=True)

        # log = {"total_loss".format(split): loss.clone().detach().mean(),
        #        "nll_loss".format(split): nll_loss.detach().mean(),
        #        "rec_loss".format(split): rec_loss.detach().mean(),
        #        "p_loss".format(split): p_loss.detach().mean(),
        #        }
        # return loss, log
        return loss


if __name__ == '__main__':
    bs = 32
    n_points_x = 10
    n_points_y = 15
    dim = 8
    x = torch.randn(bs, n_points_x, dim)
    y = torch.randn(bs, n_points_y, dim)
    for metric in ['cosine', 'l1', 'l2', 'l2_simple']:
        P = batch_pairwise_dist(x, y, metric)
        print(f'metric: {metric}, P: {P.shape}, max: {P.max()}, min: {P.min()}')
