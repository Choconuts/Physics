import os
import copy
import functools
import numpy as np
from tqdm import tqdm
from typing import Optional
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import numbers

import torch
from torch import optim
from torch import autograd

from utils import io_util
from utils.checkpoints import CheckpointIO
from utils.print_fn import log
from utils.logger import Logger
from utils import rend_util, train_util


class Embedder(nn.Module):
    def __init__(self, input_dim, max_freq_log2, N_freqs,
                 log_sampling=True, include_input=True,
                 periodic_fns=(torch.sin, torch.cos)):
        '''
        :param input_dim: dimension of input to be embedded
        :param max_freq_log2: log2 of max freq; min freq is 1 by default
        :param N_freqs: number of frequency bands
        :param log_sampling: if True, frequency bands are linerly sampled in log-space
        :param include_input: if True, raw input is included in the embedding
        :param periodic_fns: periodic functions used to embed input
        '''
        super(Embedder, self).__init__()

        self.input_dim = input_dim
        self.include_input = include_input
        self.periodic_fns = periodic_fns

        self.out_dim = 0
        if self.include_input:
            self.out_dim += self.input_dim

        self.out_dim += self.input_dim * N_freqs * len(self.periodic_fns)

        if log_sampling:
            self.freq_bands = 2. ** torch.linspace(0., max_freq_log2, N_freqs)
        else:
            self.freq_bands = torch.linspace(
                2. ** 0., 2. ** max_freq_log2, N_freqs)

        self.freq_bands = self.freq_bands.numpy().tolist()

    def forward(self, input: torch.Tensor):
        '''
        :param input: tensor of shape [..., self.input_dim]
        :return: tensor of shape [..., self.out_dim]
        '''
        assert (input.shape[-1] == self.input_dim)

        out = []
        if self.include_input:
            out.append(input)

        for i in range(len(self.freq_bands)):
            freq = self.freq_bands[i]
            for p_fn in self.periodic_fns:
                out.append(p_fn(input * freq))
        out = torch.cat(out, dim=-1)

        assert (out.shape[-1] == self.out_dim)
        return out


def get_embedder(multires, input_dim=3):
    if multires < 0:
        return nn.Identity(), input_dim

    embed_kwargs = {
        "include_input": True,  # needs to be True for ray_bending to work properly
        "input_dim": input_dim,
        "max_freq_log2": multires - 1,
        "N_freqs": multires,
        "log_sampling": True,
        "periodic_fns": [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)
    return embedder_obj, embedder_obj.out_dim


class Sine(nn.Module):
    def __init__(self, w0):
        super().__init__()
        self.w0 = w0

    def forward(self, x):
        return torch.sin(self.w0 * x)


class SirenLayer(nn.Linear):
    def __init__(self, input_dim, out_dim, *args, is_first=False, **kwargs):
        self.is_first = is_first
        self.input_dim = input_dim
        self.w0 = 30
        self.c = 6
        super().__init__(input_dim, out_dim, *args, **kwargs)
        self.activation = Sine(self.w0)

    # override
    def reset_parameters(self) -> None:
        # NOTE: in offical SIREN, first run linear's original initialization, then run custom SIREN init.
        #       hence the bias is initalized in super()'s reset_parameters()
        super().reset_parameters()
        with torch.no_grad():
            dim = self.input_dim
            w_std = (1 / dim) if self.is_first else (math.sqrt(self.c / dim) / self.w0)
            self.weight.uniform_(-w_std, w_std)

    def forward(self, x):
        out = super().forward(x)
        out = self.activation(out)
        return out


class DenseLayer(nn.Linear):
    def __init__(self, input_dim: int, out_dim: int, *args, activation=None, **kwargs):
        super().__init__(input_dim, out_dim, *args, **kwargs)
        if activation is None:
            self.activation = nn.ReLU(inplace=True)
        else:
            self.activation = activation

    def forward(self, x):
        out = super().forward(x)
        out = self.activation(out)
        return out


class ImplicitSurface(nn.Module):
    def __init__(self,
                 W=256,
                 D=8,
                 skips=[4],
                 W_geo_feat=256,
                 input_ch=3,
                 radius_init=1.0,
                 obj_bounding_size=2.0,
                 geometric_init=True,
                 embed_multires=6,
                 weight_norm=True,
                 use_siren=False,
                 ):
        """
        W_geo_feat: to set whether to use nerf-like geometry feature or IDR-like geometry feature.
            set to -1: nerf-like, the output feature is the second to last level's feature of the geometry network.
            set to >0: IDR-like ,the output feature is the last part of the geometry network's output.
        """
        super().__init__()
        # occ_net_list = [
        #     nn.Sequential(
        #         nn.Linear(input_ch, W),
        #         nn.Softplus(),
        #     )
        # ] + [
        #     nn.Sequential(
        #         nn.Linear(W, W),
        #         nn.Softplus()
        #     ) for _ in range(D-2)
        # ] + [
        #     nn.Linear(W, 1)
        # ]
        self.radius_init = radius_init
        self.register_buffer('obj_bounding_size', torch.tensor([obj_bounding_size]).float())
        self.geometric_init = geometric_init
        self.D = D
        self.W = W
        self.W_geo_feat = W_geo_feat
        if use_siren:
            assert len(skips) == 0, "do not use skips for siren"
            self.register_buffer('is_pretrained', torch.tensor([False], dtype=torch.bool))
        self.skips = skips
        self.use_siren = use_siren
        self.embed_fn, input_ch = get_embedder(embed_multires)

        surface_fc_layers = []
        # NOTE: as in IDR/NeuS, the network's has D+1 layers
        for l in range(D + 1):
            # decide out_dim
            if l == D:
                if W_geo_feat > 0:
                    out_dim = 1 + W_geo_feat
                else:
                    out_dim = 1
            elif (l + 1) in self.skips:
                out_dim = W - input_ch  # recude output dim before the skips layers, as in IDR / NeuS
            else:
                out_dim = W

            # decide in_dim
            if l == 0:
                in_dim = input_ch
            else:
                in_dim = W

            if l != D:
                if use_siren:
                    layer = SirenLayer(in_dim, out_dim, is_first=(l == 0))
                else:
                    # NOTE: beta=100 is important! Otherwise, the initial output would all be > 10, and there is not initial sphere.
                    layer = DenseLayer(in_dim, out_dim, activation=nn.Softplus(beta=100))
            else:
                layer = nn.Linear(in_dim, out_dim)

            # if true preform preform geometric initialization
            if geometric_init and not use_siren:
                # --------------
                # sphere init, as in SAL / IDR.
                # --------------
                if l == D:
                    nn.init.normal_(layer.weight, mean=np.sqrt(np.pi) / np.sqrt(in_dim), std=0.0001)
                    nn.init.constant_(layer.bias, -radius_init)
                elif embed_multires > 0 and l == 0:
                    torch.nn.init.constant_(layer.bias, 0.0)
                    torch.nn.init.constant_(layer.weight[:, 3:], 0.0)  # let the initial weights for octaves to be 0.
                    torch.nn.init.normal_(layer.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(out_dim))
                elif embed_multires > 0 and l in self.skips:
                    torch.nn.init.constant_(layer.bias, 0.0)
                    torch.nn.init.normal_(layer.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    torch.nn.init.constant_(layer.weight[:, -(input_ch - 3):],
                                            0.0)  # NOTE: this contrains the concat order to be  [h, x_embed]
                else:
                    nn.init.constant_(layer.bias, 0.0)
                    nn.init.normal_(layer.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

            if weight_norm:
                layer = nn.utils.weight_norm(layer)

            surface_fc_layers.append(layer)

        self.surface_fc_layers = nn.ModuleList(surface_fc_layers)

    def pretrain_hook(self, configs={}):
        configs['target_radius'] = self.radius_init
        # TODO: more flexible, bbox-like scene bound.
        configs['obj_bounding_size'] = self.obj_bounding_size.item()
        if self.geometric_init and self.use_siren and not self.is_pretrained:
            pretrain_siren_sdf(self, **configs)
            self.is_pretrained = ~self.is_pretrained
            return True
        return False

    def forward(self, x: torch.Tensor, return_h=False):
        x = self.embed_fn(x)

        h = x
        for i in range(self.D):
            if i in self.skips:
                # NOTE: concat order can not change! there are special operations taken in intialization.
                h = torch.cat([h, x], dim=-1) / np.sqrt(2)
            h = self.surface_fc_layers[i](h)

        out = self.surface_fc_layers[-1](h)

        if self.W_geo_feat > 0:
            h = out[..., 1:]
            out = out[..., :1].squeeze(-1)
        else:
            out = out.squeeze(-1)
        if return_h:
            return out, h
        else:
            return out

    def forward_with_nablas(self, x: torch.Tensor, has_grad_bypass: bool = None):
        has_grad = torch.is_grad_enabled() if has_grad_bypass is None else has_grad_bypass
        # force enabling grad for normal calculation
        with torch.enable_grad():
            x = x.requires_grad_(True)
            implicit_surface_val, h = self.forward(x, return_h=True)
            nabla = autograd.grad(
                implicit_surface_val,
                x,
                torch.ones_like(implicit_surface_val, device=x.device),
                create_graph=has_grad,
                retain_graph=has_grad,
                only_inputs=True)[0]
        if not has_grad:
            implicit_surface_val = implicit_surface_val.detach()
            nabla = nabla.detach()
            h = h.detach()
        return implicit_surface_val, nabla, h


def pretrain_siren_sdf(
        implicit_surface: ImplicitSurface,
        num_iters=5000, lr=1.0e-4, batch_points=5000,
        target_radius=0.5, obj_bounding_size=3.0,
        logger=None):
    # --------------
    # pretrain SIREN-sdf to be a sphere, as in SIREN and Neural Lumigraph Rendering
    # --------------
    from tqdm import tqdm
    from torch import optim
    device = next(implicit_surface.parameters()).device
    optimizer = optim.Adam(implicit_surface.parameters(), lr=lr)

    with torch.enable_grad():
        for it in tqdm(range(num_iters), desc="=> pretraining SIREN..."):
            pts = torch.empty([batch_points, 3]).uniform_(-obj_bounding_size, obj_bounding_size).float().to(device)
            sdf_gt = pts.norm(dim=-1) - target_radius
            sdf_pred = implicit_surface.forward(pts)

            loss = F.l1_loss(sdf_pred, sdf_gt, reduction='mean')

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if logger is not None:
                logger.add('pretrain_siren', 'loss_l1', loss.item(), it)


class RadianceNet(nn.Module):
    def __init__(self,
                 D=4,
                 W=256,
                 skips=[],
                 W_geo_feat=256,
                 embed_multires=6,
                 embed_multires_view=4,
                 use_view_dirs=True,
                 weight_norm=True,
                 use_siren=False, ):
        super().__init__()

        input_ch_pts = 3
        input_ch_views = 3
        if use_siren:
            assert len(skips) == 0, "do not use skips for siren"
        self.skips = skips
        self.D = D
        self.W = W
        self.use_view_dirs = use_view_dirs
        self.embed_fn, input_ch_pts = get_embedder(embed_multires)
        if use_view_dirs:
            self.embed_fn_view, input_ch_views = get_embedder(embed_multires_view)
            in_dim_0 = input_ch_pts + input_ch_views + 3 + W_geo_feat
        else:
            in_dim_0 = input_ch_pts + W_geo_feat

        fc_layers = []
        # NOTE: as in IDR/NeuS, the network's has D+1 layers
        for l in range(D + 1):
            # decicde out_dim
            if l == D:
                out_dim = 3
            else:
                out_dim = W

            # decide in_dim
            if l == 0:
                in_dim = in_dim_0
            elif l in self.skips:
                in_dim = in_dim_0 + W
            else:
                in_dim = W

            if l != D:
                if use_siren:
                    layer = SirenLayer(in_dim, out_dim, is_first=(l == 0))
                else:
                    layer = DenseLayer(in_dim, out_dim, activation=nn.ReLU(inplace=True))
            else:
                layer = DenseLayer(in_dim, out_dim, activation=nn.Sigmoid())

            if weight_norm:
                layer = nn.utils.weight_norm(layer)

            fc_layers.append(layer)

        self.layers = nn.ModuleList(fc_layers)

    def forward(
            self,
            x: torch.Tensor,
            view_dirs: torch.Tensor,
            normals: torch.Tensor,
            geometry_feature: torch.Tensor):
        # calculate radiance field
        x = self.embed_fn(x)
        if self.use_view_dirs:
            view_dirs = self.embed_fn_view(view_dirs)
            radiance_input = torch.cat([x, view_dirs, normals, geometry_feature], dim=-1)
        else:
            radiance_input = torch.cat([x, geometry_feature], dim=-1)

        h = radiance_input
        for i in range(self.D + 1):
            if i in self.skips:
                h = torch.cat([h, radiance_input], dim=-1)
            h = self.layers[i](h)
        return h


# modified from https://github.com/yenchenlin/nerf-pytorch
class NeRF(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, input_ch_view=3, multires=-1, multires_view=-1, output_ch=4, skips=[4],
                 use_view_dirs=False):
        """
        """
        super(NeRF, self).__init__()
        self.D = D
        self.W = W
        self.skips = skips
        self.use_view_dirs = use_view_dirs

        self.embed_fn, input_ch = get_embedder(multires, input_dim=input_ch)
        self.embed_fn_view, input_ch_view = get_embedder(multires_view, input_dim=input_ch_view)

        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch, W)] + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W) for i in
                                        range(D - 1)])

        ### Implementation according to the official code release (https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py#L104-L105)
        self.views_linears = nn.ModuleList([nn.Linear(input_ch_view + W, W // 2)])

        ### Implementation according to the paper
        # self.views_linears = nn.ModuleList(
        #     [nn.Linear(input_ch_views + W, W//2)] + [nn.Linear(W//2, W//2) for i in range(D//2)])

        if use_view_dirs:
            self.feature_linear = nn.Linear(W, W)
            self.alpha_linear = nn.Linear(W, 1)
            self.rgb_linear = nn.Linear(W // 2, 3)
        else:
            self.output_linear = nn.Linear(W, output_ch)

    def forward(self, input_pts, input_views):
        input_pts = self.embed_fn(input_pts)
        input_views = self.embed_fn_view(input_views)

        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu_(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], dim=-1)

        if self.use_view_dirs:
            sigma = self.alpha_linear(h)
            feature = self.feature_linear(h)
            h = torch.cat([feature, input_views], dim=-1)

            for i in range(len(self.views_linears)):
                h = self.views_linears[i](h)
                h = F.relu_(h)

            rgb = self.rgb_linear(h)
        else:
            outputs = self.output_linear(h)
            rgb = outputs[..., :3]
            sigma = outputs[..., 3:]

        rgb = torch.sigmoid(rgb)
        return sigma.squeeze(-1), rgb


class ScalarField(nn.Module):
    # TODO: should re-use some feature/parameters from implicit-surface net.
    def __init__(self, input_ch=3, W=128, D=4, skips=[], init_val=-2.0):
        super().__init__()
        self.skips = skips

        pts_linears = [nn.Linear(input_ch, W)] + \
                      [nn.Linear(W, W) if i not in skips
                       else nn.Linear(W + input_ch, W) for i in range(D - 1)]
        for linear in pts_linears:
            nn.init.kaiming_uniform_(linear.weight, a=0, mode='fan_in', nonlinearity='relu')
            nn.init.zeros_(linear.bias)

        self.pts_linears = nn.ModuleList(pts_linears)
        self.output_linear = nn.Linear(W, 1)
        nn.init.zeros_(self.output_linear.weight)
        nn.init.constant_(self.output_linear.bias, init_val)

    def forward(self, x: torch.Tensor):
        h = x
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu_(h)
            if i in self.skips:
                h = torch.cat([x, h], dim=-1)
        out = self.output_linear(h).squeeze(-1)
        return out


def get_optimizer(args, model):
    if isinstance(args.training.lr, numbers.Number):
        optimizer = optim.Adam(model.parameters(), lr=args.training.lr)
    elif isinstance(args.training.lr, dict):
        lr_dict = args.training.lr
        default_lr = lr_dict.pop('default')

        param_groups = []
        select_params_names = []
        for name, lr in lr_dict.items():
            if name in model._parameters.keys():
                select_params_names.append(name)
                param_groups.append({
                    'params': getattr(model, name),
                    'lr': lr
                })
            elif name in model._modules.keys():
                select_params_names.extend(
                    ["{}.{}".format(name, param_name) for param_name, _ in getattr(model, name).named_parameters()])
                param_groups.append({
                    'params': getattr(model, name).parameters(),
                    'lr': lr
                })
            else:
                raise RuntimeError('wrong lr key:', name)

        # NOTE: parameters() is just calling named_parameters without returning name.
        other_params = [param for name, param in model.named_parameters() if name not in select_params_names]
        param_groups.insert(0, {
            'params': other_params,
            'lr': default_lr
        })

        optimizer = optim.Adam(params=param_groups, lr=default_lr)
    else:
        raise NotImplementedError
    return optimizer


def CosineAnnealWarmUpSchedulerLambda(total_steps, warmup_steps, min_factor=0.1):
    assert 0 <= min_factor < 1

    def lambda_fn(epoch):
        """
        modified from https://github.com/Totoro97/NeuS/blob/main/exp_runner.py
        """
        if epoch < warmup_steps:
            learning_factor = epoch / warmup_steps
        else:
            learning_factor = (np.cos(np.pi * ((epoch - warmup_steps) / (total_steps - warmup_steps))) + 1.0) * 0.5 * (
                        1 - min_factor) + min_factor
        return learning_factor

    return lambda_fn


def ExponentialSchedulerLambda(total_steps, min_factor=0.1):
    assert 0 <= min_factor < 1

    def lambda_fn(epoch):
        t = np.clip(epoch / total_steps, 0, 1)
        learning_factor = np.exp(t * np.log(min_factor))
        return learning_factor

    return lambda_fn


def get_scheduler(args, optimizer, last_epoch=-1):
    stype = args.training.scheduler.type
    if stype == 'multistep':
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer,
            args.training.scheduler.milestones,
            gamma=args.training.scheduler.gamma,
            last_epoch=last_epoch)
    elif stype == 'warmupcosine':
        # NOTE: this do not support per-parameter lr
        # from cosine_annealing_warmup import CosineAnnealingWarmupRestarts
        # scheduler = CosineAnnealingWarmupRestarts(
        #     optimizer,
        #     args.training.num_iters,
        #     max_lr=args.training.lr,
        #     min_lr=args.training.scheduler.setdefault('min_lr', 0.1*args.training.lr),
        #     warmup_steps=args.training.scheduler.warmup_steps,
        #     last_epoch=last_epoch)
        # NOTE: support per-parameter lr
        scheduler = optim.lr_scheduler.LambdaLR(
            optimizer,
            CosineAnnealWarmUpSchedulerLambda(
                total_steps=args.training.num_iters,
                warmup_steps=args.training.scheduler.warmup_steps,
                min_factor=args.training.scheduler.setdefault('min_factor', 0.1)
            ),
            last_epoch=last_epoch)
    elif stype == 'exponential_step':
        scheduler = optim.lr_scheduler.LambdaLR(
            optimizer,
            ExponentialSchedulerLambda(
                total_steps=args.training.num_iters,
                min_factor=args.training.scheduler.setdefault('min_factor', 0.1)
            )
        )
    else:
        raise NotImplementedError
    return scheduler


# def pdf_phi_s(x: torch.Tensor, s):
#     esx = torch.exp(-s*x)
#     y = s*esx / ((1+esx) ** 2)
#     return y


def cdf_Phi_s(x, s):
    # den = 1 + torch.exp(-s*x)
    # y = 1./den
    # return y
    return torch.sigmoid(x * s)


def sdf_to_alpha(sdf: torch.Tensor, s):
    # [(B), N_rays, N_pts]
    cdf = cdf_Phi_s(sdf, s)
    # [(B), N_rays, N_pts-1]
    # TODO: check sanity.
    opacity_alpha = (cdf[..., :-1] - cdf[..., 1:]) / (cdf[..., :-1] + 1e-10)
    opacity_alpha = torch.clamp_min(opacity_alpha, 0)
    return cdf, opacity_alpha


def sdf_to_w(sdf: torch.Tensor, s):
    device = sdf.device
    # [(B), N_rays, N_pts-1]
    cdf, opacity_alpha = sdf_to_alpha(sdf, s)

    # [(B), N_rays, N_pts]
    shifted_transparency = torch.cat(
        [
            torch.ones([*opacity_alpha.shape[:-1], 1], device=device),
            1.0 - opacity_alpha + 1e-10,
        ], dim=-1)

    # [(B), N_rays, N_pts-1]
    visibility_weights = opacity_alpha * \
                         torch.cumprod(shifted_transparency, dim=-1)[..., :-1]

    return cdf, opacity_alpha, visibility_weights


def alpha_to_w(alpha: torch.Tensor):
    device = alpha.device
    # [(B), N_rays, N_pts]
    shifted_transparency = torch.cat(
        [
            torch.ones([*alpha.shape[:-1], 1], device=device),
            1.0 - alpha + 1e-10,
        ], dim=-1)

    # [(B), N_rays, N_pts-1]
    visibility_weights = alpha * \
                         torch.cumprod(shifted_transparency, dim=-1)[..., :-1]

    return visibility_weights


class NeuS(nn.Module):
    def __init__(self,
                 variance_init=0.05,
                 speed_factor=1.0,

                 input_ch=3,
                 W_geo_feat=-1,
                 use_outside_nerf=False,
                 obj_bounding_radius=1.0,

                 surface_cfg=dict(),
                 radiance_cfg=dict()):
        super().__init__()

        self.ln_s = nn.Parameter(data=torch.Tensor([-np.log(variance_init) / speed_factor]), requires_grad=True)
        self.speed_factor = speed_factor

        # ------- surface network
        self.implicit_surface = ImplicitSurface(
            W_geo_feat=W_geo_feat, input_ch=input_ch, obj_bounding_size=obj_bounding_radius, **surface_cfg)

        # ------- radiance network
        if W_geo_feat < 0:
            W_geo_feat = self.implicit_surface.W
        self.radiance_net = RadianceNet(
            W_geo_feat=W_geo_feat, **radiance_cfg)

        # -------- outside nerf++
        if use_outside_nerf:
            self.nerf_outside = NeRF(input_ch=4, multires=10, multires_view=4, use_view_dirs=True)

    def forward_radiance(self, x: torch.Tensor, view_dirs: torch.Tensor):
        _, nablas, geometry_feature = self.implicit_surface.forward_with_nablas(x)
        radiance = self.radiance_net.forward(x, view_dirs, nablas, geometry_feature)
        return radiance

    def forward_s(self):
        return torch.exp(self.ln_s * self.speed_factor)

    def forward(self, x: torch.Tensor, view_dirs: torch.Tensor):
        sdf, nablas, geometry_feature = self.implicit_surface.forward_with_nablas(x)
        radiances = self.radiance_net.forward(x, view_dirs, nablas, geometry_feature)
        return radiances, sdf, nablas


def volume_render(
        rays_o,
        rays_d,
        model: NeuS,

        obj_bounding_radius=1.0,

        batched=False,
        batched_info={},

        # render algorithm config
        calc_normal=False,
        use_view_dirs=True,
        rayschunk=65536,
        netchunk=1048576,
        white_bkgd=False,
        near_bypass: Optional[float] = None,
        far_bypass: Optional[float] = None,

        # render function config
        detailed_output=True,
        show_progress=False,

        # sampling related
        perturb=False,  # config whether do stratified sampling
        fixed_s_recp=1 / 64.,
        N_samples=64,
        N_importance=64,
        N_outside=0,  # whether to use outside nerf

        # upsample related
        upsample_algo='official_solution',
        N_nograd_samples=2048,
        N_upsample_iters=4,

        **dummy_kwargs  # just place holder
):
    """
    input:
        rays_o: [(B,) N_rays, 3]
        rays_d: [(B,) N_rays, 3] NOTE: not normalized. contains info about ratio of len(this ray)/len(principle ray)
    """
    device = rays_o.device
    if batched:
        DIM_BATCHIFY = 1
        B = rays_d.shape[0]  # batch_size
        flat_vec_shape = [B, -1, 3]
    else:
        DIM_BATCHIFY = 0
        flat_vec_shape = [-1, 3]

    rays_o = torch.reshape(rays_o, flat_vec_shape).float()
    rays_d = torch.reshape(rays_d, flat_vec_shape).float()
    # NOTE: already normalized
    rays_d = F.normalize(rays_d, dim=-1)

    batchify_query = functools.partial(train_util.batchify_query, chunk=netchunk, dim_batchify=DIM_BATCHIFY)

    # ---------------
    # Render a ray chunk
    # ---------------
    def render_rayschunk(rays_o: torch.Tensor, rays_d: torch.Tensor):
        # rays_o: [(B), N_rays, 3]
        # rays_d: [(B), N_rays, 3]

        # [(B), N_rays] x 2
        near, far = rend_util.near_far_from_sphere(rays_o, rays_d, r=obj_bounding_radius)
        if near_bypass is not None:
            near = near_bypass * torch.ones_like(near).to(device)
        if far_bypass is not None:
            far = far_bypass * torch.ones_like(far).to(device)

        if use_view_dirs:
            view_dirs = rays_d
        else:
            view_dirs = None

        prefix_batch = [B] if batched else []
        N_rays = rays_o.shape[-2]

        # ---------------
        # Sample points on the rays
        # ---------------

        # ---------------
        # Coarse Points

        # [(B), N_rays, N_samples]
        # d_coarse = torch.linspace(near, far, N_samples).float().to(device)
        # d_coarse = d_coarse.view([*[1]*len(prefix_batch), 1, N_samples]).repeat([*prefix_batch, N_rays, 1])
        _t = torch.linspace(0, 1, N_samples).float().to(device)
        d_coarse = near * (1 - _t) + far * _t

        # ---------------
        # Up Sampling
        with torch.no_grad():
            # -------- option 1: directly use
            if upsample_algo == 'direct_use':  # nerf-like
                # [(B), N_rays, N_samples, 3]
                pts_coarse = rays_o.unsqueeze(-2) + d_coarse.unsqueeze(-1) * rays_d.unsqueeze(-2)
                # query network to get sdf
                # [(B), N_rays, N_samples]
                sdf_coarse = model.implicit_surface.forward(pts_coarse)
                # [(B), N_rays, N_samples-1]
                *_, w_coarse = sdf_to_w(sdf_coarse, 1. / fixed_s_recp)
                # Fine points
                # [(B), N_rays, N_importance]
                d_fine = rend_util.sample_pdf(d_coarse, w_coarse, N_importance, det=not perturb)
                # Gather points
                d_all = torch.cat([d_coarse, d_fine], dim=-1)
                d_all, d_sort_indices = torch.sort(d_all, dim=-1)

            # -------- option 2: just using more points to calculate visibility weights for upsampling
            # used config: N_nograd_samples
            elif upsample_algo == 'direct_more':
                _t = torch.linspace(0, 1, N_nograd_samples).float().to(device)
                _d = near * (1 - _t) + far * _t
                _pts = rays_o.unsqueeze(-2) + _d.unsqueeze(-1) * rays_d.unsqueeze(-2)
                # _sdf = model.implicit_surface.forward(_pts)
                _sdf = batchify_query(model.implicit_surface.forward, _pts)
                *_, _w = sdf_to_w(_sdf, 1. / fixed_s_recp)
                d_fine = rend_util.sample_pdf(_d, _w, N_importance, det=not perturb)
                # Gather points
                d_all = torch.cat([d_coarse, d_fine], dim=-1)
                d_all, d_sort_indices = torch.sort(d_all, dim=-1)


            # -------- option 3: modified from NeuS official implementation: estimate sdf slopes and middle points' sdf
            # https://github.com/Totoro97/NeuS/blob/9dc9275d3a8c7266994a3b9cf9f36071621987dd/models/renderer.py#L131
            # used config: N_upsample_iters
            elif upsample_algo == 'official_solution':
                _d = d_coarse
                _sdf = batchify_query(model.implicit_surface.forward,
                                      rays_o.unsqueeze(-2) + _d.unsqueeze(-1) * rays_d.unsqueeze(-2))
                for i in range(N_upsample_iters):
                    prev_sdf, next_sdf = _sdf[..., :-1], _sdf[..., 1:]
                    prev_z_vals, next_z_vals = _d[..., :-1], _d[..., 1:]
                    mid_sdf = (prev_sdf + next_sdf) * 0.5
                    dot_val = (next_sdf - prev_sdf) / (next_z_vals - prev_z_vals + 1e-5)
                    prev_dot_val = torch.cat([torch.zeros_like(dot_val[..., :1], device=device), dot_val[..., :-1]],
                                             dim=-1)  # jianfei: prev_slope, right shifted
                    dot_val = torch.stack([prev_dot_val, dot_val], dim=-1)  # jianfei: concat prev_slope with slope
                    dot_val, _ = torch.min(dot_val, dim=-1,
                                           keepdim=False)  # jianfei: find the minimum of prev_slope and current slope. (forward diff vs. backward diff., or the prev segment's slope vs. this segment's slope)
                    dot_val = dot_val.clamp(-10.0, 0.0)

                    dist = (next_z_vals - prev_z_vals)
                    prev_esti_sdf = mid_sdf - dot_val * dist * 0.5
                    next_esti_sdf = mid_sdf + dot_val * dist * 0.5

                    prev_cdf = cdf_Phi_s(prev_esti_sdf, 64 * (2 ** i))
                    next_cdf = cdf_Phi_s(next_esti_sdf, 64 * (2 ** i))
                    alpha = (prev_cdf - next_cdf + 1e-5) / (prev_cdf + 1e-5)
                    _w = alpha_to_w(alpha)
                    d_fine = rend_util.sample_pdf(_d, _w, N_importance // N_upsample_iters, det=not perturb)
                    _d = torch.cat([_d, d_fine], dim=-1)

                    sdf_fine = batchify_query(model.implicit_surface.forward,
                                              rays_o.unsqueeze(-2) + d_fine.unsqueeze(-1) * rays_d.unsqueeze(-2))
                    _sdf = torch.cat([_sdf, sdf_fine], dim=-1)
                    _d, d_sort_indices = torch.sort(_d, dim=-1)
                    _sdf = torch.gather(_sdf, DIM_BATCHIFY + 1, d_sort_indices)
                d_all = _d
            else:
                raise NotImplementedError

        # ------------------
        # Calculate Points
        # [(B), N_rays, N_samples+N_importance, 3]
        pts = rays_o[..., None, :] + rays_d[..., None, :] * d_all[..., :, None]
        # [(B), N_rays, N_pts-1, 3]
        # pts_mid = 0.5 * (pts[..., 1:, :] + pts[..., :-1, :])
        d_mid = 0.5 * (d_all[..., 1:] + d_all[..., :-1])
        pts_mid = rays_o[..., None, :] + rays_d[..., None, :] * d_mid[..., :, None]

        # ------------------
        # Inside Scene
        # ------------------
        # sdf, nablas, _ = model.implicit_surface.forward_with_nablas(pts)
        sdf, nablas, _ = batchify_query(model.implicit_surface.forward_with_nablas, pts)
        # [(B), N_ryas, N_pts], [(B), N_ryas, N_pts-1]
        cdf, opacity_alpha = sdf_to_alpha(sdf, model.forward_s())
        # radiances = model.forward_radiance(pts_mid, view_dirs_mid)
        radiances = batchify_query(model.forward_radiance, pts_mid,
                                   view_dirs.unsqueeze(-2).expand_as(pts_mid) if use_view_dirs else None)

        # ------------------
        # Outside Scene
        # ------------------
        if N_outside > 0:
            _t = torch.linspace(0, 1, N_outside + 2)[..., 1:-1].float().to(device)
            d_vals_out = far / torch.flip(_t, dims=[-1])
            if perturb:
                _mids = .5 * (d_vals_out[..., 1:] + d_vals_out[..., :-1])
                _upper = torch.cat([_mids, d_vals_out[..., -1:]], -1)
                _lower = torch.cat([d_vals_out[..., :1], _mids], -1)
                _t_rand = torch.rand(_upper.shape).float().to(device)
                d_vals_out = _lower + (_upper - _lower) * _t_rand

            d_vals_out = torch.cat([d_mid, d_vals_out], dim=-1)  # already sorted
            pts_out = rays_o[..., None, :] + rays_d[..., None, :] * d_vals_out[..., :, None]
            r = pts_out.norm(dim=-1, keepdim=True)
            x_out = torch.cat([pts_out / r, 1. / r], dim=-1)
            views_out = view_dirs.unsqueeze(-2).expand_as(x_out[..., :3]) if use_view_dirs else None

            sigma_out, radiance_out = batchify_query(model.nerf_outside.forward, x_out, views_out)
            dists = d_vals_out[..., 1:] - d_vals_out[..., :-1]
            dists = torch.cat([dists, 1e10 * torch.ones(dists[..., :1].shape).to(device)], dim=-1)
            alpha_out = 1 - torch.exp(
                -F.softplus(sigma_out) * dists)  # use softplus instead of relu as NeuS's official repo

        # --------------
        # Ray Integration
        # --------------
        # [(B), N_rays, N_pts-1]
        if N_outside > 0:
            N_pts_1 = d_mid.shape[-1]
            # [(B), N_ryas, N_pts-1]
            mask_inside = (pts_mid.norm(dim=-1) <= obj_bounding_radius)
            # [(B), N_ryas, N_pts-1]
            alpha_in = opacity_alpha * mask_inside.float() + alpha_out[..., :N_pts_1] * (~mask_inside).float()
            # [(B), N_ryas, N_pts-1 + N_outside]
            opacity_alpha = torch.cat([alpha_in, alpha_out[..., N_pts_1:]], dim=-1)

            # [(B), N_ryas, N_pts-1, 3]
            radiance_in = radiances * mask_inside.float()[..., None] + radiance_out[..., :N_pts_1, :] * \
                          (~mask_inside).float()[..., None]
            # [(B), N_ryas, N_pts-1 + N_outside, 3]
            radiances = torch.cat([radiance_in, radiance_out[..., N_pts_1:, :]], dim=-2)
            d_final = d_vals_out
        else:
            d_final = d_mid

        # [(B), N_ryas, N_pts-1 + N_outside]
        visibility_weights = alpha_to_w(opacity_alpha)
        # [(B), N_rays]
        rgb_map = torch.sum(visibility_weights[..., None] * radiances, -2)
        # depth_map = torch.sum(visibility_weights * d_mid, -1)
        # NOTE: to get the correct depth map, the sum of weights must be 1!
        depth_map = torch.sum(visibility_weights / (visibility_weights.sum(-1, keepdim=True) + 1e-10) * d_final, -1)
        acc_map = torch.sum(visibility_weights, -1)

        if white_bkgd:
            rgb_map = rgb_map + (1.0 - acc_map[..., None])

        ret_i = OrderedDict([
            ('rgb', rgb_map),  # [(B), N_rays, 3]
            ('depth_volume', depth_map),  # [(B), N_rays]
            # ('depth_surface', d_pred_out),    # [(B), N_rays]
            ('mask_volume', acc_map)  # [(B), N_rays]
        ])

        if calc_normal:
            normals_map = F.normalize(nablas, dim=-1)
            N_pts = min(visibility_weights.shape[-1], normals_map.shape[-2])
            normals_map = (normals_map[..., :N_pts, :] * visibility_weights[..., :N_pts, None]).sum(dim=-2)
            ret_i['normals_volume'] = normals_map

        if detailed_output:
            ret_i['implicit_nablas'] = nablas
            ret_i['implicit_surface'] = sdf
            ret_i['radiance'] = radiances
            ret_i['alpha'] = opacity_alpha
            ret_i['cdf'] = cdf
            ret_i['visibility_weights'] = visibility_weights
            ret_i['d_final'] = d_final
            if N_outside > 0:
                ret_i['sigma_out'] = sigma_out
                ret_i['radiance_out'] = radiance_out

        return ret_i

    ret = {}
    for i in tqdm(range(0, rays_o.shape[DIM_BATCHIFY], rayschunk), disable=not show_progress):
        ret_i = render_rayschunk(
            rays_o[:, i:i + rayschunk] if batched else rays_o[i:i + rayschunk],
            rays_d[:, i:i + rayschunk] if batched else rays_d[i:i + rayschunk]
        )
        for k, v in ret_i.items():
            if k not in ret:
                ret[k] = []
            ret[k].append(v)
    for k, v in ret.items():
        ret[k] = torch.cat(v, DIM_BATCHIFY)

    return ret['rgb'], ret['depth_volume'], ret


class SingleRenderer(nn.Module):
    def __init__(self, model: NeuS):
        super().__init__()
        self.model = model

    def forward(self, rays_o, rays_d, **kwargs):
        return volume_render(rays_o, rays_d, self.model, **kwargs)


class Trainer(nn.Module):
    def __init__(self, model: NeuS, device_ids=[0], batched=True):
        super().__init__()
        self.model = model
        self.renderer = SingleRenderer(model)
        if len(device_ids) > 1:
            self.renderer = nn.DataParallel(self.renderer, device_ids=device_ids, dim=1 if batched else 0)
        self.device = device_ids[0]

    def forward(self,
                args,
                indices,
                model_input,
                ground_truth,
                render_kwargs_train: dict,
                it: int,
                device='cuda'):

        intrinsics = model_input["intrinsics"].to(device)
        c2w = model_input['c2w'].to(device)
        H = render_kwargs_train['H']
        W = render_kwargs_train['W']
        rays_o, rays_d, select_inds = rend_util.get_rays(
            c2w, intrinsics, H, W, N_rays=args.data.N_rays)
        # [B, N_rays, 3]
        target_rgb = torch.gather(ground_truth['rgb'].to(device), 1, torch.stack(3 * [select_inds], -1))

        if "mask_ignore" in model_input:
            mask_ignore = torch.gather(model_input["mask_ignore"].to(device), 1, select_inds)
        else:
            mask_ignore = None

        rgb, depth_v, extras = self.renderer(rays_o, rays_d, detailed_output=True, **render_kwargs_train)

        # [B, N_rays, N_pts, 3]
        nablas: torch.Tensor = extras['implicit_nablas']
        # [B, N_rays, N_pts]
        nablas_norm = torch.norm(nablas, dim=-1)
        # [B, N_rays]
        mask_volume: torch.Tensor = extras['mask_volume']
        # NOTE: when predicted mask is close to 1 but GT is 0, exploding gradient.
        # mask_volume = torch.clamp(mask_volume, 1e-10, 1-1e-10)
        mask_volume = torch.clamp(mask_volume, 1e-3, 1 - 1e-3)
        extras['mask_volume_clipped'] = mask_volume

        losses = OrderedDict()

        # [B, N_rays, 3]
        losses['loss_img'] = F.l1_loss(rgb, target_rgb, reduction='none')
        # [B, N_rays, N_pts]
        losses['loss_eikonal'] = args.training.w_eikonal * F.mse_loss(nablas_norm,
                                                                      nablas_norm.new_ones(nablas_norm.shape),
                                                                      reduction='mean')

        if args.training.with_mask:
            # [B, N_rays]
            target_mask = torch.gather(model_input["object_mask"].to(device), 1, select_inds)
            losses['loss_mask'] = args.training.w_mask * F.binary_cross_entropy(mask_volume, target_mask.float(),
                                                                                reduction='mean')
            if mask_ignore is not None:
                target_mask = torch.logical_and(target_mask, mask_ignore)
            # [N_masked, 3]
            losses['loss_img'] = (losses['loss_img'] * target_mask[..., None].float()).sum() / (
                        target_mask.sum() + 1e-10)
        else:
            if mask_ignore is not None:
                losses['loss_img'] = (losses['loss_img'] * mask_ignore[..., None].float()).sum() / (
                            mask_ignore.sum() + 1e-10)
            else:
                losses['loss_img'] = losses['loss_img'].mean()

        loss = 0
        for k, v in losses.items():
            loss += losses[k]

        losses['total'] = loss
        extras['implicit_nablas_norm'] = nablas_norm
        extras['scalars'] = {'1/s': 1. / self.model.forward_s().data}
        extras['select_inds'] = select_inds

        return OrderedDict(
            [('losses', losses),
             ('extras', extras)])


def get_model(args):
    if not args.training.with_mask:
        assert 'N_outside' in args.model.keys() and args.model.N_outside > 0, \
            "Please specify a positive model:N_outside for neus with nerf++"

    model_config = {
        'obj_bounding_radius': args.model.obj_bounding_radius,
        'W_geo_feat': args.model.setdefault('W_geometry_feature', 256),
        'use_outside_nerf': not args.training.with_mask,
        'speed_factor': args.training.setdefault('speed_factor', 1.0),
        'variance_init': args.model.setdefault('variance_init', 0.05)
    }

    surface_cfg = {
        'use_siren': args.model.surface.setdefault('use_siren', args.model.setdefault('use_siren', False)),
        'embed_multires': args.model.surface.setdefault('embed_multires', 6),
        'radius_init': args.model.surface.setdefault('radius_init', 1.0),
        'geometric_init': args.model.surface.setdefault('geometric_init', True),
        'D': args.model.surface.setdefault('D', 8),
        'W': args.model.surface.setdefault('W', 256),
        'skips': args.model.surface.setdefault('skips', [4]),
    }

    radiance_cfg = {
        'use_siren': args.model.radiance.setdefault('use_siren', args.model.setdefault('use_siren', False)),
        'embed_multires': args.model.radiance.setdefault('embed_multires', -1),
        'embed_multires_view': args.model.radiance.setdefault('embed_multires_view', -1),
        'use_view_dirs': args.model.radiance.setdefault('use_view_dirs', True),
        'D': args.model.radiance.setdefault('D', 4),
        'W': args.model.radiance.setdefault('W', 256),
        'skips': args.model.radiance.setdefault('skips', []),
    }

    model_config['surface_cfg'] = surface_cfg
    model_config['radiance_cfg'] = radiance_cfg

    model = NeuS(**model_config)

    ## render kwargs
    render_kwargs_train = {
        # upsample config
        'upsample_algo': args.model.setdefault('upsample_algo', 'official_solution'),
        # [official_solution, direct_more, direct_use]
        'N_nograd_samples': args.model.setdefault('N_nograd_samples', 2048),
        'N_upsample_iters': args.model.setdefault('N_upsample_iters', 4),

        'N_outside': args.model.setdefault('N_outside', 0),
        'obj_bounding_radius': args.data.setdefault('obj_bounding_radius', 1.0),
        'batched': args.data.batch_size is not None,
        'perturb': args.model.setdefault('perturb', True),  # config whether do stratified sampling
        'white_bkgd': args.model.setdefault('white_bkgd', False),
    }
    render_kwargs_test = copy.deepcopy(render_kwargs_train)
    render_kwargs_test['rayschunk'] = args.data.val_rayschunk
    render_kwargs_test['perturb'] = False

    trainer = Trainer(model, device_ids=args.device_ids, batched=render_kwargs_train['batched'])

    return model, trainer, render_kwargs_train, render_kwargs_test, trainer.renderer


def load_model(device="cuda") -> NeuS:
    parser = io_util.create_args_parser()
    args, unknown = parser.parse_known_args()
    exp_dir = args.resume_dir
    config = io_util.load_config(args, unknown)
    model, trainer, render_kwargs_train, render_kwargs_test, volume_render_fn = get_model(config)
    model.to(device)

    # checkpoints
    checkpoint_io = CheckpointIO(checkpoint_dir=os.path.join(exp_dir, 'ckpts'), allow_mkdir=False)
    # Register modules to checkpoint
    checkpoint_io.register_modules(
        model=model,
    )

    # Load checkpoints
    load_dict = checkpoint_io.load_file(
        config.training.ckpt_file,
        ignore_keys=config.training.ckpt_ignore_keys,
        only_use_keys=config.training.ckpt_only_use_keys,
        map_location=device)

    return model


class ImplicitNetworkMy(nn.Module):
    def __init__(
            self,
            feature_vector_size=None,
            d_in=None,
            d_out=None,
            dims=None,
            geometric_init=True,
            bias=1.0,
            skip_in=(),
            weight_norm=True,
            multires=0
    ):
        super().__init__()
        self.neus_model = load_model()

    def normalize(self, x):
        return x * 2.0

    def forward(self, points, compute_grad=False):
        points = self.normalize(points)
        sdf, feature_vectors = self.neus_model.implicit_surface.forward(points, return_h=True)
        return torch.cat([sdf[..., None] / 2.0, feature_vectors], dim=-1)

    def color(self, points, normals, view_dirs, feature_vectors):
        points = self.normalize(points)
        return self.neus_model.radiance_net(points, normals, view_dirs, feature_vectors)

    def gradient(self, x):
        points = self.normalize(x)
        sdf, gradients, h = self.neus_model.implicit_surface.forward_with_nablas(points)
        return gradients


if __name__ == '__main__':
    # neus = load_model()
    x = torch.rand(1000, 3).cuda()
    v = torch.rand(1000, 3).cuda()
    # ret = neus(x, v)
    # print(*map(lambda t: t.shape, ret))
    # # radiance, sdf, normals
    model = ImplicitNetworkMy()
    y = model(x)
    print("[Forward]", y.shape)
    g = model.gradient(x)
    print("[Grad]", g.shape)
    c = model.color(x, g, v, y[..., 1:])
    print("[Color]", c.shape)


