import imgui
import numpy as np
import torch
from octree import *
from interface import *
from visilab.utils.imgui_tool import input_jsoncollection


class Opt(Inputable):

    def __init__(self):
        self.p = (0.4, 0.4, 0.4)
        self.max_depth = 5
        self.min_depth = 2
        self.add = False

    def gui(self, label, *args, **kwargs):
        self.changed, self.max_depth = imgui.slider_int("max", self.max_depth, 1, 10)
        self.changed, self.min_depth = imgui.slider_int("min", self.min_depth, 1, self.max_depth)
        self.changed, self.p = imgui.input_float3('p', *self.p)
        self.add = imgui.button("add")


opt = Opt()


class Scene:

    def __init__(self):
        self.octree = Octree([0, 0, 0], [1, 1, 1])

    def vis_boxes(self, non_leaf=False):
        if non_leaf:
            boxes = self.octree.boxes[self.octree.non_leaf[..., 0].bool()]
        else:
            boxes = self.octree.boxes

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

        visualize_field(pnts.cpu().numpy(), scalars=pnts)

    def vis_field(self, field, thr=0.1, res=100, radius=1.5):
        with torch.no_grad():
            s = torch.linspace(-radius, radius, res).cuda()
            x = torch.stack(torch.meshgrid([s, s, s]), -1).view(-1, 3)
            pack = field(x)
            if isinstance(pack, tuple):
                a, rgb = pack
            else:
                a = pack
                rgb = x
            a = a.view(-1)
            rgb = rgb.view(-1 ,3)
            visualize_field(x[a > thr].cpu().detach(), scalars=rgb[a > thr].cpu().detach())

    @ui(opt)
    def show_boxes(self):

        points = []

        def div_fn(boxes):
            ps = torch.tensor(np.array(points), device=boxes.device)
            B = boxes.shape[0]
            N = ps.shape[0]
            box = boxes[:, None, :].expand(B, N, -1).reshape(-1, 6)
            ps = ps[None, :, :].expand(B, N, -1).reshape(-1, 3)
            mask = inside_box(box, ps, False).view(B, N)
            mask = mask.sum(1)
            return mask > 0

        points = [
            [0.45, 0.45, 0.45],
            [0.6, 0.6, 0.4]
        ]
        self.octree.build(div_fn, opt.max_depth, opt.min_depth)

        self.vis_boxes(non_leaf=True)
        while True:
            if opt.add:
                points += [opt.p]
                new_octree = Octree(self.octree.boxes[0][:3], self.octree.boxes[0][3:])
                new_octree.build(div_fn, opt.max_depth, opt.min_depth)
                self.octree = new_octree
                self.vis_boxes(non_leaf=True)

                rays_o = torch.tensor([
                    [0.9, 0.9, 0.9],
                    [0.7, 0.6, 0.5]
                ], device="cuda")

                rays_d = torch.tensor([
                    [-1, -1, -1.]
                ], device="cuda")

                def hit_fn(boxes):
                    print(boxes[..., 3], 1 / 2 ** opt.max_depth)
                    return boxes[..., 3] < 1 / 2 ** opt.max_depth + 1e-4

                out = self.octree.cast(rays_o, rays_d, hit_fn)

                visualize_field(rays_o, vectors=rays_d * out, len_limit=0)
                yield
            else:
                yield


if __name__ == '__main__':
    scene = Scene()
    scene.show_boxes()
