import builtins
import time

import torch

from cluster.observe import *
from cluster.match import ImplicitNetworkMy
from octree import Octree, inside_box, OctreeSDF


class NeuSOctree:

    def __init__(self, radius=1.5, max_depth=8):
        self.thr = 4.0
        self.radius = radius
        self.min_depth = 3
        self.max_depth = max_depth
        self.octree = Octree([-radius] * 3, [radius * 2] * 3)
        self.nerf = ImplicitNetworkMy()
        self.nerf.cuda()
        self.octree.build(self.divide, self.max_depth, self.min_depth)

    def divide(self, boxes):
        size = boxes[..., 3:]
        center = boxes[..., :3] + size * 0.5

        def field(x):
            chunk = 8192
            res = []
            for j in range(0, x.shape[0], chunk):
                chunk_data = torch_tree_map(lambda r: r[j:j + chunk], x)
                with torch.no_grad():
                    a = self.nerf(chunk_data)
                res.append(a)
            return torch.cat(res, 0)[..., 0]

        inside_sphere = center.norm(-1) < 0.6
        alpha = field(center).abs() <= size.norm(dim=-1) * 0.3
        return torch.logical_and(inside_sphere, alpha)

    def cast(self, rays_o, rays_d):
        def hit_fn(boxes):
            size = boxes[..., 3:]
            thr_radius = self.radius / (2 ** self.max_depth / 2 - 1)
            return size < thr_radius
        t = self.octree.cast(rays_o, rays_d, hit_fn)
        return t


# no = NeuSOctree(max_depth=10)
scene = ObserveScene(20, dh=False)


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
                    visualize_field(pnts.cpu().numpy(), scalars=pnts)
                    # def field(x):
                    #     dirs = -x / (x.norm(dim=-1, keepdim=True) + 1e-5)
                    #     rgb, a = no.nerf(x, dirs)
                    #     return a
                    #
                    # scene.vis_field(field, 6)
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
            a = no.nerf(p, rays_d)
            rgb = no.nerf.color(p, rays_d, rays_d, a[..., 1:])

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


def render_sdf_octree():
    nerf = ImplicitNetworkMy()
    nerf.cuda()

    def sdf(x):
        return nerf(x)[..., 0]

    def color(x):
        nerf(x)[..., 1:]
        return nerf.color(x, dirs, )

    with torch.no_grad():
        osdf = OctreeSDF(sdf, [[-1.5] * 3, [1.5] * 3])

    st = time.time()
    for i in range(20):
        rays_o, rays_d, rgb, mask = scene.sample_image(i, -1)
        # print(rays_o.shape, rays_d.shape, rgb.shape, mask.shape)
        t = osdf.cast(rays_o, rays_d)
        p = rays_o + t * rays_d
        with torch.no_grad():
            a = nerf(p, rays_d)
            rgb = nerf.color(p, rays_d, rays_d, a[..., 1:])

        scene.save_image(rgb.view(*scene.data.img_res, 3), i)
        scene.save_gray_image(t.view(*scene.data.img_res) / 4, i)
    print("[Cost]", f"trace {20} images, cost {time.time() - st:.4f} seconds.")


if __name__ == '__main__':
    # A().show_boxes()
    render_sdf_octree()
    # A().show_bad_case()



