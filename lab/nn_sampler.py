import torch
from torch import nn
import torch.nn.functional as F
from lab.pref_encoder import PREF
import  matplotlib.pyplot as plt
from tqdm import tqdm, trange
import numpy as np


def grid_sample(image, optical, **kwargs):
    N, C, IH, IW = image.shape
    _, H, W, _ = optical.shape

    ix = optical[..., 0]
    iy = optical[..., 1]

    ix = ((ix + 1) / 2) * (IW - 1);
    iy = ((iy + 1) / 2) * (IH - 1);
    with torch.no_grad():
        ix_nw = torch.floor(ix);
        iy_nw = torch.floor(iy);
        ix_ne = ix_nw + 1;
        iy_ne = iy_nw;
        ix_sw = ix_nw;
        iy_sw = iy_nw + 1;
        ix_se = ix_nw + 1;
        iy_se = iy_nw + 1;

    nw = (ix_se - ix) * (iy_se - iy)
    ne = (ix - ix_sw) * (iy_sw - iy)
    sw = (ix_ne - ix) * (iy - iy_ne)
    se = (ix - ix_nw) * (iy - iy_nw)

    with torch.no_grad():
        torch.clamp(ix_nw, 0, IW - 1, out=ix_nw)
        torch.clamp(iy_nw, 0, IH - 1, out=iy_nw)

        torch.clamp(ix_ne, 0, IW - 1, out=ix_ne)
        torch.clamp(iy_ne, 0, IH - 1, out=iy_ne)

        torch.clamp(ix_sw, 0, IW - 1, out=ix_sw)
        torch.clamp(iy_sw, 0, IH - 1, out=iy_sw)

        torch.clamp(ix_se, 0, IW - 1, out=ix_se)
        torch.clamp(iy_se, 0, IH - 1, out=iy_se)

    image = image.view(N, C, IH * IW)

    nw_val = torch.gather(image, 2, (iy_nw * IW + ix_nw).long().view(N, 1, H * W).repeat(1, C, 1))
    ne_val = torch.gather(image, 2, (iy_ne * IW + ix_ne).long().view(N, 1, H * W).repeat(1, C, 1))
    sw_val = torch.gather(image, 2, (iy_sw * IW + ix_sw).long().view(N, 1, H * W).repeat(1, C, 1))
    se_val = torch.gather(image, 2, (iy_se * IW + ix_se).long().view(N, 1, H * W).repeat(1, C, 1))

    out_val = (nw_val.view(N, C, H, W) * nw.view(N, 1, H, W) +
               ne_val.view(N, C, H, W) * ne.view(N, 1, H, W) +
               sw_val.view(N, C, H, W) * sw.view(N, 1, H, W) +
               se_val.view(N, C, H, W) * se.view(N, 1, H, W))

    return out_val


def grid_sample_3d(feature_3d, grid):
    N, C, iD, iH, iW = feature_3d.shape
    _, D, H, W, _ = grid.shape

    ix = grid[..., 0]
    iy = grid[..., 1]
    iz = grid[..., 2]

    ix = ((ix + 1) / 2) * (iW - 1)
    iy = ((iy + 1) / 2) * (iH - 1)
    iz = ((iz + 1) / 2) * (iD - 1)

    with torch.no_grad():
        ix_tnw = torch.floor(ix)
        iy_tnw = torch.floor(iy)
        iz_tnw = torch.floor(ix)

        ix_tne = ix_tnw + 1
        iy_tne = iy_tnw
        iz_tne = iz_tnw

        ix_tsw = ix_tnw
        iy_tsw = iy_tnw + 1
        iz_tsw = iz_tnw

        ix_tse = ix_tnw + 1
        iy_tse = iy_tnw + 1
        iz_tse = iz_tnw

        ix_bnw = ix_tnw
        iy_bnw = iy_tnw
        iz_bnw = iz_tnw + 1

        ix_bne = ix_tnw + 1
        iy_bne = iy_tnw
        iz_bne = iz_tnw + 1

        ix_bsw = ix_tnw
        iy_bsw = iy_tnw + 1
        iz_bsw = iz_tnw + 1

        ix_bse = ix_tnw + 1
        iy_bse = iy_tnw + 1
        iz_bse = iz_tnw + 1

    bnw = (ix_tse - ix) * (iy_tse - iy) * (iz - iz_tse)
    bne = (ix - ix_tsw) * (iy_tsw - iy) * (iz - iz_tsw)
    bsw = (ix_tne - ix) * (iy - iy_tne) * (iz - iz_tne)
    bse = (ix - ix_tnw) * (iy - iy_tnw) * (iz - iz_tnw)

    tnw = (ix_bse - ix) * (iy_bse - iy) * (iz_bse - iz)
    tne = (ix - ix_bsw) * (iy_bsw - iy) * (iz_bsw - iz)
    tsw = (ix_bne - ix) * (iy - iy_bne) * (iz_bne - iz)
    tse = (ix - ix_bnw) * (iy - iy_bnw) * (iz_bnw - iz)

    with torch.no_grad():
        torch.clamp(ix_bnw, 0, iW - 1, out=ix_bnw)
        torch.clamp(iy_bnw, 0, iH - 1, out=iy_bnw)
        torch.clamp(iz_bnw, 0, iD - 1, out=iz_bnw)

        torch.clamp(ix_bne, 0, iW - 1, out=ix_bne)
        torch.clamp(iy_bne, 0, iH - 1, out=iy_bne)
        torch.clamp(iz_bne, 0, iD - 1, out=iz_bne)

        torch.clamp(ix_bsw, 0, iW - 1, out=ix_bsw)
        torch.clamp(iy_bsw, 0, iH - 1, out=iy_bsw)
        torch.clamp(iz_bsw, 0, iD - 1, out=iz_bsw)

        torch.clamp(ix_bse, 0, iW - 1, out=ix_bse)
        torch.clamp(iy_bse, 0, iH - 1, out=iy_bse)
        torch.clamp(iz_bse, 0, iD - 1, out=iz_bse)

        torch.clamp(ix_tnw, 0, iW - 1, out=ix_tnw)
        torch.clamp(iy_tnw, 0, iH - 1, out=iy_tnw)
        torch.clamp(iz_tnw, 0, iD - 1, out=iz_tnw)

        torch.clamp(ix_tne, 0, iW - 1, out=ix_tne)
        torch.clamp(iy_tne, 0, iH - 1, out=iy_tne)
        torch.clamp(iz_tne, 0, iD - 1, out=iz_tne)

        torch.clamp(ix_tsw, 0, iW - 1, out=ix_tsw)
        torch.clamp(iy_tsw, 0, iH - 1, out=iy_tsw)
        torch.clamp(iz_tsw, 0, iD - 1, out=iz_tsw)

        torch.clamp(ix_tse, 0, iW - 1, out=ix_tse)
        torch.clamp(iy_tse, 0, iH - 1, out=iy_tse)
        torch.clamp(iz_tse, 0, iD - 1, out=iz_tse)

    feature_3d = feature_3d.view(N, C, iH * iW * iD)

    bnw_val = torch.gather(feature_3d, 2, (iy_bnw * iW + ix_bnw * iH + iz_bnw * iD).long().view(N, 1, D * H * W).repeat(1, C, 1))
    bne_val = torch.gather(feature_3d, 2, (iy_bne * iW + ix_bne * iH + iz_bnw * iD).long().view(N, 1, D * H * W).repeat(1, C, 1))
    bsw_val = torch.gather(feature_3d, 2, (iy_bsw * iW + ix_bsw * iH + iz_bsw * iD).long().view(N, 1, D * H * W).repeat(1, C, 1))
    bse_val = torch.gather(feature_3d, 2, (iy_bse * iW + ix_bse * iH + iz_bse * iD).long().view(N, 1, D * H * W).repeat(1, C, 1))

    tnw_val = torch.gather(feature_3d, 2, (iy_tnw * iW + ix_tnw * iH + iz_tnw * iD).long().view(N, 1, D * H * W).repeat(1, C, 1))
    tne_val = torch.gather(feature_3d, 2, (iy_tne * iW + ix_tne * iH + iz_tnw * iD).long().view(N, 1, D * H * W).repeat(1, C, 1))
    tsw_val = torch.gather(feature_3d, 2, (iy_tsw * iW + ix_tsw * iH + iz_tsw * iD).long().view(N, 1, D * H * W).repeat(1, C, 1))
    tse_val = torch.gather(feature_3d, 2, (iy_tse * iW + ix_tse * iH + iz_tse * iD).long().view(N, 1, D * H * W).repeat(1, C, 1))

    out_val = (
        bnw_val.view(N, C, D, H, W) * bnw.view(N, 1, D, H, W) +
        bne_val.view(N, C, D, H, W) * bne.view(N, 1, D, H, W) +
        bsw_val.view(N, C, D, H, W) * bsw.view(N, 1, D, H, W) +
        bse_val.view(N, C, D, H, W) * bse.view(N, 1, D, H, W) +
        tnw_val.view(N, C, D, H, W) * tnw.view(N, 1, D, H, W) +
        tne_val.view(N, C, D, H, W) * tne.view(N, 1, D, H, W) +
        tsw_val.view(N, C, D, H, W) * tsw.view(N, 1, D, H, W) +
        tse_val.view(N, C, D, H, W) * tse.view(N, 1, D, H, W)
    )

    return out_val


class PE(nn.Module):
    def __init__(self, input_dims=3, num_freq=10, include_input=True, log_sampling=True):
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


class SDFNetwork(nn.Module):
    def __init__(self,
                 d_out=1,
                 d_in=3,
                 d_hidden=64,
                 n_layers=4,
                 skip_in=(2,),
                 multires=6,
                 bias=0.5,
                 scale=1,
                 geometric_init=True,
                 weight_norm=True,
                 inside_outside=False):
        super(SDFNetwork, self).__init__()

        dims = [d_in] + [d_hidden for _ in range(n_layers)] + [d_out]

        self.embed_fn_fine = None

        if multires > 0:
            embed_fn = PE(num_freq=multires, input_dims=d_in)
            self.embed_fn_fine = embed_fn
            dims[0] = embed_fn.feature_dim()

        self.num_layers = len(dims)
        self.skip_in = skip_in
        self.scale = scale

        for l in range(0, self.num_layers - 1):
            if l + 1 in self.skip_in:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]

            lin = nn.Linear(dims[l], out_dim)

            # if geometric_init:
            #     if l == self.num_layers - 2:
            #         if not inside_outside:
            #             torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
            #             torch.nn.init.constant_(lin.bias, -bias)
            #         else:
            #             torch.nn.init.normal_(lin.weight, mean=-np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
            #             torch.nn.init.constant_(lin.bias, bias)
            #     elif multires > 0 and l == 0:
            #         torch.nn.init.constant_(lin.bias, 0.0)
            #         torch.nn.init.normal_(lin.weight[:, ], 0.0, np.sqrt(2) / np.sqrt(out_dim))
            #     elif multires > 0 and l in self.skip_in:
            #         torch.nn.init.constant_(lin.bias, 0.0)
            #         torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
            #         torch.nn.init.constant_(lin.weight[:, -(dims[0] - 3):], 0.0)
            #     else:
            #         torch.nn.init.constant_(lin.bias, 0.0)
            #         torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
            #
            # if weight_norm:
            #     lin = nn.utils.weight_norm(lin)

            torch.nn.init.orthogonal_(lin.weight)
            torch.nn.init.normal_(lin.bias)

            setattr(self, "lin" + str(l), lin)

        self.activation = nn.LeakyReLU()

    def forward(self, z):
        inputs = z * self.scale
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
        return x

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


class NeuralField(nn.Module):
    def __init__(self, out_dim=1):
        super().__init__()
        self.encoder = PREF(linear_freqs=[128] * 3, reduced_freqs=[1] * 3, feature_dim=16)

        input_dim = self.encoder.output_dim
        hidden_dim = 64
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(hidden_dim, out_dim))

    def forward(self, x):
        x = self.encoder(x)
        x = self.mlp(x)
        return x


class Samp(nn.Module):

    def __init__(self):
        super(Samp, self).__init__()
        self.field = SDFNetwork(3, n_layers=2, multires=3)
        # lins = torch.linspace(0, 10, 10)
        # x, y, z = torch.meshgrid(lins, lins, lins, indexing="ij")
        # self.grid = torch.nn.Parameter(torch.stack([x, y, z], dim=0)[None])

    def forward(self, z):
        # z = z * 2 - 1
        # x = grid_sample_3d(self.grid, z.view(1, -1, 1, 1, 3)).view(-1, *z.shape[:1]).permute(1, 0)
        x = self.field(z)
        x = (torch.cos(x) + 1) / 2
        return x


class Disc(nn.Module):

    def __init__(self):
        super(Disc, self).__init__()
        self.field = NeuralField(1)

    def forward(self, x):
        return self.field(x)


# def prob_func(x):
#     dist = (x - 0.5).norm(dim=-1)
#     return torch.exp(-dist * 10)


def prob_func(x):
    a = torch.exp(-(x - 0.5).norm(dim=-1) * 20)
    b = torch.exp(-(x - 0.2).norm(dim=-1) * 40)
    c = torch.exp(-(x - 0.7).norm(dim=-1) * 30)
    return a + b + c


def vis(f):
    if not hasattr(vis, "i"):
        vis.i = 0
    plt.figure()
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    z = torch.rand(1000, 3).cuda()
    x = f(z)
    x = x.cpu().detach().numpy()
    plt.scatter(x[..., 0], x[..., 1])
    plt.savefig(f"logs/vis-{vis.i}.png")
    vis.i += 1


def gt_sampler(prob_f, n):
    x = torch.rand(n, 3).cuda()
    p0 = prob_f(x)

    for i in range(100):
        y = x + (torch.rand_like(x) * 2 - 1) * 0.15
        y[y < 0] = x[y < 0]
        y[y > 1] = x[y > 1]
        p = prob_f(y)

        h = torch.clip(p / p0, 0, 1)
        u = torch.rand_like(h)

        trans = u < h
        x[trans] = y[trans]
        p0[trans] = p[trans]

    return x


def gt_integral(prob_f, n=1000000):
    x = torch.rand(n, 3).cuda()
    return torch.sum(prob_f(x)) / n


def train():
    # vis(lambda x: gt_sampler(prob_func, x.shape[0]))

    model = Samp()
    disc = Disc()
    model.cuda()
    disc.cuda()
    batch_size = 100000

    optimizer = torch.optim.Adam(model.parameters(), 0.0002, (0.9, 0.99), weight_decay=0.001)
    optimizer_d = torch.optim.Adam(disc.parameters(), 0.0001, (0.5, 0.99))

    x_real_data = gt_sampler(prob_func, 100 * batch_size)

    # pre-train
    pbar0 = trange(500)
    for i in pbar0:
        z = torch.rand(batch_size, 3).cuda()
        z = z.requires_grad_(True)
        x = model(z)
        g = grad(x, z)
        loss = ((x - z) ** 2).mean() + torch.nn.functional.relu(0.5-g.det()).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 10 == 0:
            pbar0.set_postfix(Loss=loss.item(), Grad=g.det().mean().item())

    pbar = trange(30000)
    for i in pbar:

        for _ in range(1):
            idx = torch.randint(0, x_real_data.shape[0], (batch_size,))
            x_real = x_real_data[idx]
            pred_real = disc(x_real)
            loss_real = -pred_real.mean()

            z = torch.rand(batch_size, 3).cuda()
            x_fake = model(z)
            pred_fake = disc(x_fake)
            loss_fake = pred_fake.mean()

            loss_D = loss_real + loss_fake
            # optimize
            optimizer_d.zero_grad()
            loss_D.backward()
            optimizer_d.step()

        for _ in range(2):
            z = torch.rand(batch_size, 3).cuda()
            x = model(z)
            # p = prob_func(x)
            # loss = -torch.sum(p)
            loss = -disc(x).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if i % 10 == 0:
            pbar.set_postfix(D_Loss = loss_D.item(), G_Loss = loss.item())

        with torch.no_grad():
            for param in model.parameters():
                param.clamp_(-1, 1)

        if i > 0 and i % 1000 == 0:
            vis(model)


def grad(x, z):
    gradients = []
    for i in range(x.shape[-1]):
        d_output = torch.ones_like(x[..., i: i+1], requires_grad=False, device=x.device)
        gradients += torch.autograd.grad(
            outputs=x[..., i: i+1],
            inputs=z,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)

    gradients = torch.stack(gradients, -1)
    return gradients


def train_new():
    # logs(lambda x: gt_sampler(prob_func, x.shape[0]))

    model = Samp()
    model.cuda()
    batch_size = 100000

    optimizer = torch.optim.Adam(model.parameters(), 0.001, (0.5, 0.99))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda t: 1)

    F_gt = gt_integral(prob_func)
    print("[Int]", F_gt)

    # pre-train
    pbar0 = trange(100)
    for i in pbar0:
        z = torch.rand(batch_size, 3).cuda()
        z = z.requires_grad_(True)
        x = model(z)
        g = grad(x, z)
        loss = ((x - z) ** 2).mean() + torch.nn.functional.relu(0.5-g.det()).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 10 == 0:
            pbar0.set_postfix(Loss=loss.item(), Grad=g.det().mean().item())

    pbar = trange(3001)
    for i in pbar:

        z = torch.rand(batch_size, 3).cuda()
        z = z.requires_grad_(True)
        x = model(z)
        inv_p = grad(x, z).det()
        # inv_p = grad(x, z)
        # inv_p = inv_p[..., 0, 0] * inv_p[..., 1, 1] * inv_p[..., 2, 2] + \
        #         inv_p[..., 0, 1] * inv_p[..., 1, 2] * inv_p[..., 2, 0] + \
        #         inv_p[..., 0, 2] * inv_p[..., 1, 0] * inv_p[..., 2, 1] - \
        #         inv_p[..., 0, 0] * inv_p[..., 1, 2] * inv_p[..., 2, 1] - \
        #         inv_p[..., 0, 1] * inv_p[..., 1, 0] * inv_p[..., 2, 2] - \
        #         inv_p[..., 0, 2] * inv_p[..., 1, 1] * inv_p[..., 2, 0]


        neg_loss = torch.relu(0.0001-inv_p).mean()

        loss = ((prob_func(x) * inv_p - F_gt) ** 2).mean() + neg_loss * 10000

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 10 == 0:
            pbar.set_postfix(Loss=loss.item(), LR=scheduler.get_lr()[0])

        if i > 0 and i % 200 == 0:
            vis(model)

        # if i > 0 and i % 1000 == 0:
        #     scheduler.step()

    vis(lambda x: gt_sampler(prob_func, x.shape[0]))


if __name__ == '__main__':
    train_new()

