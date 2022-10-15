import builtins
import time

import torch

from cluster.observe import *
from octree import Octree, inside_box


class NeRFOctree:

    def __init__(self, radius=1.5, max_depth=8):
        self.thr = 4.0
        self.radius = radius
        self.min_depth = 3
        self.max_depth = max_depth
        self.octree = Octree([-radius] * 3, [radius * 2] * 3)
        self.nerf = MyNeRF()
        self.octree.build(self.divide, self.max_depth, self.min_depth)

    def divide(self, boxes):
        size = boxes[..., 3:]
        center = boxes[..., :3] + size * 0.5

        thr_radius = self.radius / (2 * 2 ** self.min_depth + 1)
        if size.min() < thr_radius:
            pad = center[None]
        else:
            pad_n = 20 * int((size.min() / thr_radius).item() ** 3)
            print(pad_n, *center.shape)
            pad = torch.rand(pad_n, *center.shape, device=center.device) * size * 3 + center - size

        x_k = torch.tensor([1, 0, 0.], device=size.device)
        y_k = torch.tensor([0, 1, 0.], device=size.device)
        z_k = torch.tensor([0, 0, 1.], device=size.device)
        centers = torch.stack([
            center + size * x_k,
            center + size * y_k,
            center + size * z_k,
            center - size * x_k,
            center - size * y_k,
            center - size * z_k,
        ], 0)
        centers = torch.cat([centers, pad], 0)

        def field(x):
            chunk = 8192
            res = []
            for j in range(0, x.shape[1], chunk):
                chunk_data = torch_tree_map(lambda r: r[:, j:j + chunk], x)
                with torch.no_grad():
                    a = self.nerf.density(chunk_data)
                res.append(a)
            return torch.cat(res, 1)

        alpha = field(centers) > self.thr
        return torch.logical_and(alpha.sum(0) > 0, alpha.prod(0) == 0)

    def cast(self, rays_o, rays_d):
        def hit_fn(boxes):
            size = boxes[..., 3:]
            thr_radius = self.radius / (2 ** self.max_depth / 2 - 1)
            return size < thr_radius
        t = self.octree.cast(rays_o, rays_d, hit_fn)
        return t


no = NeRFOctree(max_depth=9)
scene = ObserveScene(20, dh=True)


class A:
    @ui
    def show(self):

        def field(x):
            dirs = -x / (x.norm(dim=-1, keepdim=True) + 1e-5)
            rgb, a = no.nerf(x, dirs)
            return a

        scene.vis_field(field, 6)

    @ui(opt)
    def show_boxes(self):
        select_leaf = no.octree.boxes[..., 3] <= no.octree.boxes[..., 3].min() + 1e-4
        boxes = no.octree.boxes[select_leaf]
        box_min, box_size = torch.split(boxes, [3, 3], -1)
        x_mask = torch.tensor([1, 0, 0.], device=box_min.device)
        y_mask = torch.tensor([0, 1, 0.], device=box_min.device)
        z_mask = torch.tensor([0, 0, 1.], device=box_min.device)
        xy_mask = torch.tensor([1, 1, 0.], device=box_min.device)
        yz_mask = torch.tensor([0, 1, 1.], device=box_min.device)
        xz_mask = torch.tensor([1, 0, 1.], device=box_min.device)
        pnts = [
            box_min,
            box_min + box_size * x_mask,
            box_min + box_size * y_mask,
            box_min + box_size * z_mask,
            box_min + box_size * xy_mask,
            box_min + box_size * yz_mask,
            box_min + box_size * xz_mask,
            box_min + box_size,
        ]
        pnts = torch.cat(pnts, 0)


        rays_o = torch.tensor([
            [0.9, 0.9, 0.9],
            [0.7, 0.6, 0.5]
        ], device="cuda")

        rays_d = torch.tensor([
            [-1, -1, -1.]
        ], device="cuda")

        t = no.cast(rays_o, rays_d)

        while True:
            if opt.changed:
                if opt.show:
                    def field(x):
                        dirs = -x / (x.norm(dim=-1, keepdim=True) + 1e-5)
                        rgb, a = no.nerf(x, dirs)
                        return a

                    scene.vis_field(field, 6)
                visualize_field(rays_o, vectors=rays_d * t, len_limit=0)
            yield

    @ui(opt)
    def show_bad_case(self):
        def field(x):
            dirs = -x / (x.norm(dim=-1, keepdim=True) + 1e-5)
            rgb, a = no.nerf(x, dirs)
            return a

        scene.vis_field(field, 6)
        rays_o, rays_d, rgb, mask = scene.sample_image(4, 400)
        t = no.cast(rays_o, rays_d)

        visualize_field(rays_o, vectors=rays_d * t, len_limit=0)


def render():
    st = time.time()
    for i in range(20):
        rays_o, rays_d, rgb, mask = scene.sample_image(i, -1)
        # print(rays_o.shape, rays_d.shape, rgb.shape, mask.shape)
        t = no.cast(rays_o, rays_d)
        p = rays_o + t * rays_d
        with torch.no_grad():
            rgb, a = no.nerf(p, rays_d)
        rgb = torch.sigmoid(rgb)

        scene.save_image(rgb.view(*scene.data.img_res, 3), i)
        scene.save_gray_image(t.view(*scene.data.img_res) / 4, i)
    print("[Cost]", f"trace {20} images, cost {time.time() - st:.4f} seconds.")


def cost_time():
    st = time.time()
    n_it = 1
    n_res = 100
    rays_o, rays_d, rgb, mask = scene.sample_image(0, 800)
    for i in range(n_it):
        t = no.cast(rays_o, rays_d)
    print("[Cost]", f"trace {n_it} images of {n_res}x{n_res}, cost {time.time() - st:.4f} seconds.")


if __name__ == '__main__':
    A().show_boxes()
    octree = no.octree
    del no
    del scene
    # del octree
    torch.cuda.empty_cache()
    cost_time()
    # A().show_bad_case()



