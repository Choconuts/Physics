import os

import gin
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from cluster.third.utils_old import *
from cluster.third.schedule import *


@gin.register
@gin.configurable
class PE(nn.Module):
    def __init__(self, input_dims=3, num_freq=10, include_input=True, log_sampling=True, schedule=None):
        super(PE, self).__init__()
        self.kwargs = {
            'input_dims': input_dims,
            'include_input': include_input,
            'max_freq_log2': num_freq - 1,
            'num_freqs': num_freq,
            'log_sampling': log_sampling,
            'periodic_fns': [torch.sin, torch.cos],
        }
        self.create_embedding_fn()

        self.window_curve = None if schedule is None else Curve(schedule)

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = 2. ** torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2. ** 0., 2. ** max_freq, steps=N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)

    def forward(self, inputs):
        return self.embed(inputs)

    def feature_dim(self) -> int:
        return self.out_dim

    def windowed_embed(self, x):
        code = self.embed(x)
        if self.window_curve is None:
            return code
        start = 0
        if self.kwargs["include_input"]:
            start = 3
        init_shape = list(code.shape[:-1])
        w_code = code[..., start:].view(init_shape + [-1, 2, 3])
        window = self.get_cosine_easing_window()
        w_code = (window.view(-1, 1, 1) * w_code).view(init_shape + [-1])
        return torch.cat([code[..., :start], w_code], -1)

    def get_cosine_easing_window(self):
        alpha = self.window_curve()
        window = self.cosine_easing_window(0, self.kwargs['max_freq_log2'], self.kwargs['num_freqs'], alpha)
        return window

    @classmethod
    def cosine_easing_window(cls, min_freq_log2, max_freq_log2, num_bands, alpha):
        """Eases in each frequency one by one with a cosine.

        This is equivalent to taking a Tukey window and sliding it to the right
        along the frequency spectrum.

        Args:
          min_freq_log2: the lower frequency band.
          max_freq_log2: the upper frequency band.
          num_bands: the number of frequencies.
          alpha: will ease in each frequency as alpha goes from 0.0 to num_freqs.

        Returns:
          A 1-d numpy array with num_sample elements containing the window.
        """
        if max_freq_log2 is None:
            max_freq_log2 = num_bands - 1.0
        bands = torch.linspace(min_freq_log2, max_freq_log2, num_bands)
        x = torch.clip(alpha - bands, 0.0, 1.0)
        return 0.5 * (1 + torch.cos(torch.pi * x + torch.pi))


@gin.register
@gin.configurable
class Hash(nn.Module):

    def __init__(self, n_levels=16, n_features=2, in_dim=3, schedule=None, bbox=None):
        super(Hash, self).__init__()
        assert n_features in [2 ** i for i in range(6)]
        assert n_levels <= 16

        if n_features <= 8:
            pass
            # encodings = [tcnn_encoding(n_levels, n_features, in_dim)]
        else:
            encodings = []
            n_feat = 0
            while n_feat < n_features:
                # encodings.append(tcnn_encoding(n_levels, 8, in_dim))
                n_feat += 8

        self.encodings = nn.ModuleList(encodings)
        self.n_levels = n_levels
        self.n_features = min(n_features, 8)
        self.n_output_dims = n_levels * n_features

        self.window_curve = None if schedule is None else Curve(schedule)
        self.bbox = make_bbox(bbox)

    def forward(self, x):
        x = self.bbox(x)

        init_shape = x.shape
        x = x.view(-1, init_shape[-1])
        codes = torch.cat([enc(x).view(-1, self.n_levels, self.n_features) for enc in self.encodings], -1)
        shape = list(init_shape[:-1]) + [-1]
        return codes.view(*shape).float()

    def feature_dim(self) -> int:
        return self.n_output_dims

    def windowed_embed(self, x):
        code = self(x)
        if self.window_curve is None:
            return code
        init_shape = list(code.shape[:-1])
        w_code = code.view(init_shape + [self.n_levels, -1])
        window = self.get_cosine_easing_window()
        w_code = (window.view(-1, 1) * w_code).view(init_shape + [-1])
        return w_code

    def get_cosine_easing_window(self):
        alpha = self.window_curve()
        window = self.cosine_easing_window(0, self.n_levels, self.n_levels, alpha)
        return window

    @classmethod
    def cosine_easing_window(cls, min_freq_log2, max_freq_log2, num_bands, alpha):
        """Eases in each frequency one by one with a cosine.

        This is equivalent to taking a Tukey window and sliding it to the right
        along the frequency spectrum.

        Args:
          min_freq_log2: the lower frequency band.
          max_freq_log2: the upper frequency band.
          num_bands: the number of frequencies.
          alpha: will ease in each frequency as alpha goes from 0.0 to num_freqs.

        Returns:
          A 1-d numpy array with num_sample elements containing the window.
        """
        if max_freq_log2 is None:
            max_freq_log2 = num_bands - 1.0
        bands = torch.linspace(min_freq_log2, max_freq_log2, num_bands)
        x = torch.clip(alpha - bands, 0.0, 1.0)
        return 0.5 * (1 + torch.cos(torch.pi * x + torch.pi))


def get_embedder(multires, input_dims=3, Embedder=PE, windowed=False):
    embedder_obj = Embedder(input_dims=input_dims, num_freq=multires)
    def embed(x, eo=embedder_obj): return eo.embed(x) if not windowed else eo.windowed_embed(x)
    return embed, embedder_obj.out_dim


# This implementation is borrowed from IDR: https://github.com/lioryariv/idr
class SDFNetwork(nn.Module):
    def __init__(self,
                 d_in,
                 d_out,
                 d_hidden,
                 n_layers,
                 skip_in=(4,),
                 multires=10,
                 bias=0.5,
                 scale=1,
                 geometric_init=True,
                 weight_norm=True,
                 inside_outside=False):
        super(SDFNetwork, self).__init__()

        dims = [d_in] + [d_hidden for _ in range(n_layers)] + [d_out]

        self.embed_fn_fine = None

        if multires > 0:
            embed_fn, input_ch = get_embedder(multires, input_dims=d_in)
            self.embed_fn_fine = embed_fn
            dims[0] = input_ch

        self.num_layers = len(dims)
        self.skip_in = skip_in
        self.scale = scale

        for l in range(0, self.num_layers - 1):
            if l + 1 in self.skip_in:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]

            lin = nn.Linear(dims[l], out_dim)

            if geometric_init:
                if l == self.num_layers - 2:
                    if not inside_outside:
                        torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                        torch.nn.init.constant_(lin.bias, -bias)
                    else:
                        torch.nn.init.normal_(lin.weight, mean=-np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                        torch.nn.init.constant_(lin.bias, bias)
                elif multires > 0 and l == 0:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.constant_(lin.weight[:, 3:], 0.0)
                    torch.nn.init.normal_(lin.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(out_dim))
                elif multires > 0 and l in self.skip_in:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    torch.nn.init.constant_(lin.weight[:, -(dims[0] - 3):], 0.0)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        self.activation = nn.Softplus(beta=100)

    def forward(self, inputs):
        inputs = inputs * self.scale
        if self.embed_fn_fine is not None:
            inputs = self.embed_fn_fine(inputs)

        x = inputs
        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            if l in self.skip_in:
                x = torch.cat([x, inputs], 1) / np.sqrt(2)

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.activation(x)
        return torch.cat([x[:, :1] / self.scale, x[:, 1:]], dim=-1)

    def sdf(self, x):
        return self.forward(x)[:, :1]

    def sdf_hidden_appearance(self, x):
        return self.forward(x)

    def gradient(self, x):
        not_eval = torch.is_grad_enabled()
        with torch.enable_grad():
            x.requires_grad_(True)
            y = self.sdf(x)
            d_output = torch.ones_like(y, requires_grad=False, device=y.device)
            gradients = torch.autograd.grad(
                outputs=y,
                inputs=x,
                grad_outputs=d_output,
                create_graph=not_eval,
                retain_graph=not_eval,
                only_inputs=True)[0]
        return gradients.unsqueeze(1)


@gin.configurable
class HashSDFNetwork(nn.Module):
    def __init__(self,
                 d_in,
                 d_out,
                 multires=12,
                 dx_curve=0.1,
                 separated=False):
        super(HashSDFNetwork, self).__init__()

        self.hash = Hash(n_levels=multires, in_dim=d_in)
        input_ch = self.hash.feature_dim()
        embed_fn = self.hash.windowed_embed
        self.embed_fn_fine = embed_fn

        self.separated = separated
        # if separated:
        #     self.linear = TCNNLinear(input_ch, d_out - 1)
        #     self.sdf_linear = TCNNLinear(input_ch, 1)
        # else:
        #     self.linear = TCNNLinear(input_ch, d_out)
        self.dx = Curve(dx_curve)

    def forward(self, inputs):
        x = self.embed_fn_fine(inputs)
        if self.separated:
            feat = self.linear(x)
            sdf = self.sdf_linear(x)
            return torch.cat([sdf, feat], dim=-1)
        x = self.linear(x)
        return torch.cat([x[:, :1], x[:, 1:]], dim=-1)

    def sdf(self, x):
        if self.separated:
            x = self.embed_fn_fine(x)
            return self.sdf_linear(x)
        return self.forward(x)[:, :1]

    def sdf_hidden_appearance(self, x):
        return self.forward(x)

    def gradient(self, x, dx=None):
        if dx is None:
            dx = self.dx()
        return prox_gradients(self.sdf, x, dx)


# This implementation is borrowed from IDR: https://github.com/lioryariv/idr
class RenderingNetwork(nn.Module):
    def __init__(self,
                 d_feature,
                 mode,
                 d_in,
                 d_out,
                 d_hidden,
                 n_layers,
                 weight_norm=True,
                 multires_view=4,
                 squeeze_out=True):
        super().__init__()

        if "raw" in mode:
            squeeze_out = False

        self.mode = mode
        self.squeeze_out = squeeze_out

        if "no" in mode:
            d_in = d_in - 3

        dims = [d_in + d_feature] + [d_hidden for _ in range(n_layers)] + [d_out]

        self.embedview_fn = None
        if multires_view > 0:
            embedview_fn, input_ch = get_embedder(multires_view)
            self.embedview_fn = embedview_fn
            dims[0] += (input_ch - 3)

        if "tcnn" in mode:
            pass
            # self.linear = TCNNLinear(dims[0], d_out)
        else:
            self.num_layers = len(dims)

            for l in range(0, self.num_layers - 1):
                out_dim = dims[l + 1]
                lin = nn.Linear(dims[l], out_dim)

                if weight_norm:
                    lin = nn.utils.weight_norm(lin)

                setattr(self, "lin" + str(l), lin)

            self.relu = nn.ReLU()

    def forward(self, points, normals, view_dirs, feature_vectors):
        if self.embedview_fn is not None:
            view_dirs = self.embedview_fn(view_dirs)

        if 'no_view_dir' in self.mode:
            rendering_input = torch.cat([points, normals, feature_vectors], dim=-1)
        elif 'no_normal' in self.mode:
            rendering_input = torch.cat([points, view_dirs, feature_vectors], dim=-1)
        else:       # 'idr'
            rendering_input = torch.cat([points, view_dirs, normals, feature_vectors], dim=-1)
        x = rendering_input

        if "tcnn" in self.mode:
            x = self.linear(x)
        else:
            for l in range(0, self.num_layers - 1):
                lin = getattr(self, "lin" + str(l))

                x = lin(x)

                if l < self.num_layers - 2:
                    x = self.relu(x)

        if self.squeeze_out:
            x = torch.sigmoid(x)
        return x


class NeRF(nn.Module):
    def __init__(self,
                 D=8,
                 W=256,
                 d_in=3,
                 d_in_view=3,
                 multires=10,
                 multires_view=4,
                 output_ch=4,
                 skips=[4],
                 use_viewdirs=True):
        super(NeRF, self).__init__()
        self.D = D
        self.W = W
        self.d_in = d_in
        self.d_in_view = d_in_view
        self.input_ch = 3
        self.input_ch_view = 3
        self.embed_fn = None
        self.embed_fn_view = None

        if multires > 0:
            embed_fn, input_ch = get_embedder(multires, input_dims=d_in)
            self.embed_fn = embed_fn
            self.input_ch = input_ch

        if multires_view > 0:
            embed_fn_view, input_ch_view = get_embedder(multires_view, input_dims=d_in_view)
            self.embed_fn_view = embed_fn_view
            self.input_ch_view = input_ch_view

        self.skips = skips
        self.use_viewdirs = use_viewdirs

        self.pts_linears = nn.ModuleList(
            [nn.Linear(self.input_ch, W)] +
            [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + self.input_ch, W) for i in range(D - 1)])

        ### Implementation according to the official code release
        ### (https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py#L104-L105)
        self.views_linears = nn.ModuleList([nn.Linear(self.input_ch_view + W, W // 2)])

        ### Implementation according to the paper
        # self.views_linears = nn.ModuleList(
        #     [nn.Linear(input_ch_views + W, W//2)] + [nn.Linear(W//2, W//2) for i in range(D//2)])

        if use_viewdirs:
            self.feature_linear = nn.Linear(W, W)
            self.alpha_linear = nn.Linear(W, 1)
            self.rgb_linear = nn.Linear(W // 2, 3)
        else:
            self.output_linear = nn.Linear(W, output_ch)

    def forward(self, input_pts, input_views):
        if self.embed_fn is not None:
            input_pts = self.embed_fn(input_pts)
        if self.embed_fn_view is not None:
            input_views = self.embed_fn_view(input_views)

        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        if self.use_viewdirs:
            alpha = self.alpha_linear(h)
            feature = self.feature_linear(h)
            h = torch.cat([feature, input_views], -1)

            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = F.relu(h)

            rgb = self.rgb_linear(h)
            return alpha, rgb
        else:
            assert False


class SingleVarianceNetwork(nn.Module):
    def __init__(self, init_val):
        super(SingleVarianceNetwork, self).__init__()
        self.register_parameter('variance', nn.Parameter(torch.tensor(init_val)))

    def forward(self, x):
        return torch.ones([len(x), 1], device=x.device) * torch.exp(self.variance * 10.0)


def auto_flatten(f):
    from functools import wraps

    @wraps(f)
    def wrapper(self, x, *args, **kwargs):
        init_shape = list(x.shape[:-1]) + [-1]
        x = x.view(-1, 3)
        return f(self, x, *args, **kwargs).view(init_shape)

    return wrapper


def auto_flatten2(f):
    from functools import wraps

    @wraps(f)
    def wrapper(self, x, dirs, *args, **kwargs):
        init_shape = list(x.shape[:-1]) + [-1]
        if len(init_shape) > len(dirs.shape):
            dirs = dirs[:, None, :].expand(x.shape)

        x = x.view(-1, 3)
        dirs = dirs.reshape(-1, 3)
        rgb, a = f(self, x, dirs, *args, **kwargs)
        return rgb.view(init_shape), a.view(init_shape)

    return wrapper


@gin.register
@gin.configurable
class NeuSModel(nn.Module):

    def __init__(self, mode='idr', hashing=False, outside=True):
        """
        mode: { [no_view_dir/no_normal/-] + [tcnn/-] + [raw/-] (+ idr) } / sh / seg
        """
        super(NeuSModel, self).__init__()

        if mode == 'sh':
            # self.sh = SH()

            def wrap_color_net(x, gradients, dirs, feature_vector): return self.sh(feature_vector, dirs)

            self.color_network = wrap_color_net
            d_feat = self.sh.in_dim
        elif mode == 'seg':
            d_feat = 128
            # self.seg_net = TCNNLinear(2 * d_feat, 3)

            def wrap_color_net(x, gradients, dirs, feature_vector):
                # 512 x 128 x 3, assume sample num is 128
                feature_vector = feature_vector.view(-1, 128, feature_vector.size(-1))
                feat_pairs = torch.cat([feature_vector[..., :-1, :], feature_vector[..., 1:, :]], -1)
                rgb = self.seg_net(feat_pairs)
                rgb = torch.cat([rgb, rgb[:, -1:]], dim=1)
                return rgb.view(-1, 3)

            self.color_network = wrap_color_net
        else:
            d_feat = 256
            self.color_network = RenderingNetwork(d_feature=d_feat, mode=mode, d_in=9, d_out=3, d_hidden=256, n_layers=4)

        if outside:
            self.nerf_outside = NeRF(d_in=4)
        if hashing:
            self.sdf_network = HashSDFNetwork(d_in=3, d_out=d_feat + 1)
        else:
            self.sdf_network = SDFNetwork(d_in=3, d_out=d_feat + 1, d_hidden=256, n_layers=8)
        self.deviation_network = SingleVarianceNetwork(init_val=0.3)

    def sdf(self, x):
        return self.sdf_network.sdf(x)

    def sdf_and_feat(self, x):
        out = self.sdf_network(x)
        return out[..., :1], out[..., 1:]

    def color(self, x, gradients, dirs, feature_vector):
        return self.color_network(x, gradients, dirs, feature_vector)

    @auto_flatten
    def grad(self, x):
        return self.sdf_network.gradient(x)

    def dev(self, x):
        return self.deviation_network(x)

    def radius(self):
        return 2.0

    def background(self, x, dirs):
        return self.nerf_outside(x, dirs)

    @auto_flatten2
    def forward(self, pnts, dirs, **kwargs):
        a, feat = self.sdf_and_feat(pnts)
        return self.color(pnts, self.grad(pnts), dirs, feat), a


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
        NEUS_LOG_DIR = r"G:\WorkSpace\nerf\logs\lego-neus-return"

        neus_kwargs = {
            "mode": "idr",
            "hashing": False,
            "outside": False,
        }

        gin.parse_config_file(os.path.join(NEUS_LOG_DIR, "config.gin"))

        self.neus_model = NeuSModel(**neus_kwargs)
        state = torch.load(os.path.join(NEUS_LOG_DIR, "200000.tar"))
        step = state['global_step']
        self.neus_model.load_state_dict(state['model'])
        Curve.stepping(self.neus_model, step)

    def normalize(self, x):
        return x * 2.0

    def forward(self, points, compute_grad=False):
        points = self.normalize(points)
        return self.neus_model.sdf_network.forward(points) / 2.0

    def color(self, points, normals, view_dirs, feature_vectors):
        points = self.normalize(points)
        return self.neus_model.color(points, normals, view_dirs, feature_vectors)

    def gradient(self, x):
        if x.numel() == 0:
            return torch.ones_like(x)
        x.requires_grad_(True)
        y = self.forward(x)[:,:1]
        d_output = torch.ones_like(y, requires_grad=False, device=y.device)
        gradients = torch.autograd.grad(
            outputs=y,
            inputs=x,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]
        return gradients.unsqueeze(1)


if __name__ == '__main__':
    net = ImplicitNetworkMy()
    x = torch.rand(100, 3).to("cuda")
    net.cuda()
    print(net(x).shape)

    NeuSModel('', False, False)

