import cv2
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision.utils import make_grid, save_image
from functools import wraps
import numpy as np


class Data:

    def __init__(self):
        pass


class LabData(Data):

    def __init__(self):
        super(LabData, self).__init__()
        albedo = cv2.imread("data/abd.png") / 255.0
        gt = cv2.imread("data/gt.png") / 255.0
        shadow = cv2.imread("data/shadow.png") / 255.0

        albedo = cv2.resize(albedo, (216, 284))
        gt = cv2.resize(gt, (216, 284))
        shadow = cv2.resize(shadow, (218, 284))[:, 2:]

        ind_light = gt - shadow
        ind_light[ind_light > 0.22] = 0
        ind_light[ind_light < 0] = 0

        self.intensity = 2.0
        self.ind_light = ind_light
        self.light = (gt - ind_light)[..., :1]
        self.albedo = albedo
        self.color = albedo * (self.light * self.intensity + self.ind_light)
        self.h, self.w = albedo.shape[:2]
        self.size = (self.w, self.h)

        self.color = self.color / self.intensity

    def to_tensor(self, img):
        # 1, C, H, W
        return torch.tensor(img).permute(2, 0, 1).float().cuda()[None]

    def sample(self, n):
        x = torch.rand(1, 1, n, 2).cuda() * 2 - 1.0
        e = F.grid_sample(self.to_tensor(self.ind_light), x).permute(3, 0, 1, 2).reshape(-1, 3)
        c = F.grid_sample(self.to_tensor(self.color), x).permute(3, 0, 1, 2).reshape(-1, 3)
        a = F.grid_sample(self.to_tensor(self.albedo), x).permute(3, 0, 1, 2).reshape(-1, 3)
        l = F.grid_sample(self.to_tensor(self.light), x).permute(3, 0, 1, 2).reshape(-1, 1)
        i = torch.tensor(self.intensity).cuda()
        s = torch.zeros_like(a)

        return x.reshape(-1, 2), l, i, e, a, s, c

    def sample_image(self, x, img=None):
        x = x.view(1, 1, -1, 2)
        if img is None:
            img = self.color
        return F.grid_sample(self.to_tensor(img), x).permute(3, 0, 1, 2).reshape(-1, img.shape[-1])


class LabData(Data):

    def __init__(self):
        super(LabData, self).__init__()
        albedo = cv2.imread("data/albedo.png") / 255.0
        color = cv2.imread("data/color.png") / 255.0
        shadow = cv2.imread("data/light.png") / 255.0
        shadow = shadow.mean(-1, keepdims=True)

        self.intensity = 1 / 0.7

        ind_light = color - shadow * self.intensity * albedo
        ind_light[ind_light < 0] = 0
        ind_light = np.clip(ind_light / np.clip(albedo, 1e-5, 1.0), 0.0, 0.5)

        calc_color = (shadow * self.intensity + ind_light) * albedo

        # cv2.imshow("tst", calc_color)
        # cv2.waitKey()

        #     k = cv2.waitKey()
        # i = 0.7
        #
        # while True:
        #     cv2.imshow("tst", (shadow * albedo - color * i))
        #     k = cv2.waitKey()
        #     if k == 113:
        #         i += 0.01
        #     elif k == 97:
        #         i -= 0.01
        #         i = max(i, 0.0001)
        #     print(i)
        # shadow * albedo - color * i
        # ind_light[ind_light > 0.22] = 0
        # ind_light[ind_light < 0] = 0

        self.ind_light = ind_light
        self.light = shadow
        self.albedo = albedo
        self.color = calc_color
        self.h, self.w = albedo.shape[:2]
        self.size = (self.w, self.h)

        self.color = self.color / self.intensity

    def to_tensor(self, img):
        # 1, C, H, W
        return torch.tensor(img).permute(2, 0, 1).float().cuda()[None]

    def sample(self, n):
        x = torch.rand(1, 1, n, 2).cuda() * 2 - 1.0
        e = F.grid_sample(self.to_tensor(self.ind_light), x).permute(3, 0, 1, 2).reshape(-1, 3)
        c = F.grid_sample(self.to_tensor(self.color), x).permute(3, 0, 1, 2).reshape(-1, 3)
        a = F.grid_sample(self.to_tensor(self.albedo), x).permute(3, 0, 1, 2).reshape(-1, 3)
        l = F.grid_sample(self.to_tensor(self.light), x).permute(3, 0, 1, 2).reshape(-1, 1)
        i = torch.tensor(self.intensity).cuda()
        s = torch.zeros_like(a)

        return x.reshape(-1, 2), l, i, e, a, s, c

    def sample_image(self, x, img=None):
        x = x.view(1, 1, -1, 2)
        if img is None:
            img = self.color
        return F.grid_sample(self.to_tensor(img), x).permute(3, 0, 1, 2).reshape(-1, img.shape[-1])


def vis_img(f):
    if not hasattr(vis_img, "i"):
        vis_img.i = 0
    ind = torch.linspace(-1, 1, 400)
    z = torch.stack(torch.meshgrid(ind, ind), -1).cuda().view(-1, 2)
    x = f(z)
    x = x.view(400, 400, x.shape[-1])
    img = x.permute(1, 0, 2).detach().cpu().numpy()
    cv2.imwrite(f"logs/dsd/img-{vis_img.i}.png", img * 255)
    vis_img.i += 1


def vis_imgs(*img_fns, n_row=2):
    if not hasattr(vis_img, "i"):
        vis_img.i = 0

    ind = torch.linspace(-1, 1, 400)
    z = torch.stack(torch.meshgrid(ind, ind), -1).cuda().view(-1, 2)
    imgs = []
    for img_fn in img_fns:
        img = img_fn(z)
        if isinstance(img, torch.Tensor):
            img = img.view(400, 400, img.shape[-1]).permute(2, 1, 0).detach().expand(3, -1, -1)
            imgs += [img]
        else:
            imgs += [im.view(400, 400, im.shape[-1]).permute(2, 1, 0).detach().expand(3, -1, -1) for im in img]
    imgs = torch.stack(imgs, 0)
    imgs = torch.flip(imgs, (1,))
    imgs = make_grid(imgs, n_row)

    with open(f"logs/dsd/img-{vis_img.i}.png", "bw") as fp:
        save_image(imgs, fp)

    vis_img.i += 1


def batchify(chunk):

    def wrap(f):
        @wraps(f)
        def wrapper(x, *args, **kwargs):
            xs = torch.split(x, chunk)
            res = []
            is_tuples = False
            for bx in xs:
                tmp = f(bx)
                if isinstance(tmp, tuple):
                    is_tuples = True
                res += [tmp]

            if is_tuples:
                return tuple([torch.cat(r, 0) for r in zip(*res)])

            return torch.cat(res, 0)
        return wrapper

    return wrap


if __name__ == '__main__':
    labdata = LabData()

    def print_range(t):
        print("[Range]", t.min().item(), "~", t.max().item(), "[Mean]", t.mean().item())

    # for k in range(10):
    #     x, l, i, e, a, s, c = labdata.sample(1000)
    #     eq = (l * i + e) * a + s - c
    #     print(eq.mean())
    #     print_range(l)
    #     print_range(i)
    #     print_range(e)
    #     print_range(a)
    #     print_range(s)
    #     print_range(c)

    @batchify(5000)
    def dvd(x, val=2.0):
        x = labdata.sample_image(x, labdata.albedo)
        return x / 2.0, x * 2.0

    x = torch.rand(100000, 2).cuda()
    y, z = dvd(x)
    print(y.shape, z.shape)
